from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np
import torch

import jax
import jax.numpy as jp

from mujoco import mjx
from mujoco_playground._src import mjx_env
from panda_kinematics import compute_franka_fk, compute_franka_ik


def fk_pos(q: jp.ndarray) -> jp.ndarray:
    return compute_franka_fk(q)[:3, 3]


def fk_rot(q: jp.ndarray) -> jp.ndarray:
    return compute_franka_fk(q)[:3, :3]


def make_pose(R: jp.ndarray, p: jp.ndarray) -> jp.ndarray:
    T = jp.eye(4, dtype=jp.float32)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(p)
    return T


Q_MIN = jp.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=jp.float32,
)
Q_MAX = jp.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=jp.float32,
)


def clip_q(q: jp.ndarray) -> jp.ndarray:
    return jp.clip(q, Q_MIN, Q_MAX)


# obstacles 
OBSTACLE_CENTERS = jp.array(
    [
        [0.5, -0.10],
        [0.425, 0.08],
        [0.575, 0.08],
        [0.35, 0.26],
        [0.5, 0.26],
        [0.65, 0.26],
    ],
    dtype=jp.float32,
)  # (N,2)

OBSTACLE_RADII = jp.array(
    [0.03, 0.025, 0.025, 0.025, 0.025, 0.025],
    dtype=jp.float32,
)  # (N,)


def obstacle_penalty_xy(points_xy: jp.ndarray, safety_margin: float = 0.03) -> jp.ndarray:
    diff = points_xy[:, None, :] - OBSTACLE_CENTERS[None, :, :]
    dist = jp.linalg.norm(diff, axis=-1)
    safe_r = (OBSTACLE_RADII + safety_margin)[None, :]
    pen = jp.maximum(0.0, safe_r - dist)
    return jp.sum(pen * pen)


def rollout_track_waypoints_mjx(
    mjx_model: mjx.Model,
    data0: mjx.Data,
    xy_waypoints: jp.ndarray,  # (H,2)
    z_fixed: float,
    ctrl_lower: jp.ndarray,
    ctrl_upper: jp.ndarray,
    n_substeps: int,
):
    """
    Track waypoints by IK each step (NO stop_gradient).
    This makes ee_real depend on xy_waypoints, so gradients are meaningful.
    """
    q0 = data0.qpos[:7]
    R_fixed = fk_rot(q0)

    def body(carry_data, xy_des):
        p_des = jp.array([xy_des[0], xy_des[1], z_fixed], dtype=jp.float32)
        T_des = make_pose(R_fixed, p_des)

        q_prev = carry_data.qpos[:7]
        q7_guess = q_prev[6]
        q_target = compute_franka_ik(T_des, q7_guess, q_prev)
        q_target = clip_q(q_target)

        ctrl = carry_data.ctrl
        ctrl = ctrl.at[:7].set(q_target)
        ctrl = jp.clip(ctrl, ctrl_lower, ctrl_upper)

        data_next = mjx_env.step(mjx_model, carry_data, ctrl, n_substeps)

        q_next = data_next.qpos[:7]
        ee_pos_next = fk_pos(q_next)
        return data_next, (q_next, ee_pos_next, p_des)

    data_T, (q_hist, ee_real, desired_traj) = jax.lax.scan(
        body,
        data0,
        xy_waypoints,
    )
    return data_T, q_hist, ee_real, desired_traj


@dataclass
class TorchAffineScalerAdapter:
    """
    Generic adapter for agent.scaler.inverse_scale_output (torch) so guidance can operate in
    real units while diffusion samples in scaled space.

    We estimate Jacobian dy/dx by finite difference once (assumes approximately affine scaling).
    """
    inverse_scale_output: Callable[[torch.Tensor], torch.Tensor]
    action_dim: int
    eps: float = 1e-3
    device: str = "cpu"

    def __post_init__(self):
        with torch.no_grad():
            x0 = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
            y0 = self.inverse_scale_output(x0).detach().cpu().numpy()[0]  # (D,)
            J = np.zeros((self.action_dim, self.action_dim), dtype=np.float32)
            for k in range(self.action_dim):
                x1 = x0.clone()
                x1[0, k] += self.eps
                y1 = self.inverse_scale_output(x1).detach().cpu().numpy()[0]
                J[:, k] = (y1 - y0) / self.eps
        self.J = J 

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        flat = x_scaled.reshape(-1, self.action_dim)
        y = self.inverse_scale_output(flat)
        return y.reshape_as(x_scaled)

    def grad_to_scaled(self, grad_unscaled: torch.Tensor) -> torch.Tensor:
        # grad_x = grad_y @ (dy/dx)
        J = torch.as_tensor(self.J, dtype=grad_unscaled.dtype, device=grad_unscaled.device)
        return torch.einsum("...j,jk->...k", grad_unscaled, J)


class MJXDeltaXYGuidance:
    """
    Callable guidance_fn for diffusion:
      grad_J = guidance_fn(state, action, t)  (returns dJ/da in scaled action space)

    action is the diffusion model_mean in *scaled* space.
    We:
      1) unscale to real dxy (meters)
      2) integrate to xy waypoints (meters)
      3) MJX rollout + loss
      4) grad wrt dxy
      5) map grad back to scaled space via scaler Jacobian
    """

    def __init__(
        self,
        mjx_model: mjx.Model,
        ctrl_lower: jp.ndarray,
        ctrl_upper: jp.ndarray,
        *,
        sim_dt: float,
        ctrl_dt: float,
        w_track_xyz: float = 5.0,
        w_obs: float = 5.0,
        w_z_track: float = 1.0,
        w_dxy_l2: float = 1e-4,
        w_dxy_smooth: float = 1e-4,
        w_ref_xy: float = 0.0,     
        scaler: Optional[TorchAffineScalerAdapter] = None,
    ):
        self.mjx_model = mjx_model
        self.ctrl_lower = ctrl_lower
        self.ctrl_upper = ctrl_upper
        self.sim_dt = float(sim_dt)
        self.ctrl_dt = float(ctrl_dt)
        self.n_substeps = int(round(self.ctrl_dt / self.sim_dt))

        self.w_track_xyz = float(w_track_xyz)
        self.w_obs = float(w_obs)
        self.w_z_track = float(w_z_track)
        self.w_dxy_l2 = float(w_dxy_l2)
        self.w_dxy_smooth = float(w_dxy_smooth)
        self.w_ref_xy = float(w_ref_xy)

        self.scaler = scaler

        # context (set per control step)
        self._data0: Optional[mjx.Data] = None
        self._xy0: Optional[jp.ndarray] = None          # (2,)
        self._xy_ref: Optional[jp.ndarray] = None       # (H,2) or None

        # jit compile value_and_grad
        self._loss_and_grad = jax.jit(jax.value_and_grad(self._loss_fn))

    def set_context(
        self,
        *,
        data0: mjx.Data,
        current_xy: np.ndarray | jp.ndarray,
        xy_ref: Optional[np.ndarray | jp.ndarray] = None,
    ):
        self._data0 = data0
        self._xy0 = jp.array(np.asarray(current_xy, dtype=np.float32))
        if xy_ref is None:
            self._xy_ref = None
        else:
            self._xy_ref = jp.array(np.asarray(xy_ref, dtype=np.float32))

    def _loss_fn(self, dxy_seq: jp.ndarray) -> jp.ndarray:
        """
        dxy_seq: (H,2) in meters
        """
        assert self._data0 is not None and self._xy0 is not None, "Call set_context() before guidance."

        data0 = self._data0
        xy0 = self._xy0
        xy_ref = self._xy_ref

        # Δxy -> xy waypoints
        xy_waypoints = xy0[None, :] + jp.cumsum(dxy_seq, axis=0)

        q0 = data0.qpos[:7]
        # z_fixed = jp.asarray(fk_pos(q0)[2])

        # _, q_hist, ee_real, desired_traj = rollout_track_waypoints_mjx(
        #     self.mjx_model,
        #     data0,
        #     xy_waypoints,
        #     float(z_fixed),
        #     self.ctrl_lower,
        #     self.ctrl_upper,
        #     self.n_substeps,
        # )
        
        z_fixed = fk_pos(q0)[2] 
        _, q_hist, ee_real, desired_traj = rollout_track_waypoints_mjx(
            self.mjx_model,
            data0,
            xy_waypoints,
            z_fixed,               
            self.ctrl_lower,
            self.ctrl_upper,
            self.n_substeps,
        )


        track_xyz = jp.sum((ee_real - desired_traj) ** 2)

        obs_cost = obstacle_penalty_xy(ee_real[:, :2])

        z_track = jp.sum((ee_real[:, 2] - z_fixed) ** 2)

        dxy_l2 = jp.sum(dxy_seq ** 2)
        dxy_smooth = jp.sum((dxy_seq[1:] - dxy_seq[:-1]) ** 2) if dxy_seq.shape[0] > 1 else 0.0

        ref_cost = 0.0
        if (xy_ref is not None) and (xy_ref.shape == xy_waypoints.shape):
            ref_cost = jp.sum((xy_waypoints - xy_ref) ** 2)

        loss = (
            self.w_track_xyz * track_xyz
            + self.w_obs * obs_cost
            + self.w_z_track * z_track
            + self.w_dxy_l2 * dxy_l2
            + self.w_dxy_smooth * dxy_smooth
            + self.w_ref_xy * ref_cost
        )
        return loss

    def __call__(self, state: torch.Tensor, action_scaled: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        This matches gc_diffusion.guided_p_sample_external calling convention.

        Returns grad_J w.r.t action_scaled (same shape).
        """
        if self._data0 is None or self._xy0 is None:
            raise RuntimeError("MJXDeltaXYGuidance: set_context() must be called before sampling.")

        with torch.no_grad():
            if self.scaler is not None:
                action_unscaled = self.scaler.unscale(action_scaled)
            else:
                action_unscaled = action_scaled

        a_np = action_unscaled.detach().cpu().float().numpy()

        # ensure shape (B,H,2)
        if a_np.ndim == 2:
            a_np = a_np[:, None, :]  

        B, H, D = a_np.shape
        assert D == 2, f"Expected Δdes_xy dim=2, got {D}"

        grads_unscaled = np.zeros_like(a_np, dtype=np.float32)

        for b in range(B):
            dxy_seq = jp.array(a_np[b], dtype=jp.float32)  # (H,2)
            _, g = self._loss_and_grad(dxy_seq)            # g: (H,2)
            grads_unscaled[b] = np.array(g, dtype=np.float32)

        grad_unscaled_t = torch.from_numpy(grads_unscaled).to(device=action_scaled.device, dtype=action_scaled.dtype)

        if self.scaler is not None:
            grad_scaled = self.scaler.grad_to_scaled(grad_unscaled_t)
        else:
            grad_scaled = grad_unscaled_t

        if action_scaled.ndim == 2:
            grad_scaled = grad_scaled[:, 0, :]

        return grad_scaled
