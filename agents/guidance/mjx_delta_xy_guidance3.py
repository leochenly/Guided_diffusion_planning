from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any

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


# Franka joint limits
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


def rollout_track_waypoints_mjx(
    mjx_model: mjx.Model,
    data0: mjx.Data,
    xy_waypoints: jp.ndarray,  # (H,2)
    z_fixed: jp.ndarray,        # scalar
    ctrl_lower: jp.ndarray,
    ctrl_upper: jp.ndarray,
    n_substeps: int,
):
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
    Adapter to map gradients from unscaled action space back to scaled space.
    """
    inverse_scale_output: Callable[[torch.Tensor], torch.Tensor]
    action_dim: int
    eps: float = 1e-3
    device: str = "cpu"

    def __post_init__(self):
        with torch.no_grad():
            x0 = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
            y0 = self.inverse_scale_output(x0).detach().cpu().numpy()[0]
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
        J = torch.as_tensor(self.J, dtype=grad_unscaled.dtype, device=grad_unscaled.device)
        return torch.einsum("...j,jk->...k", grad_unscaled, J)


class MJXDeltaXYGuidance:
    """
    Goal-conditioned MJX guidance with dynamic XML obstacles.

    Interpretation:
      dxy_seq: (H,2)
      xy_waypoints = xy_base + cumsum(dxy_seq)

    Key fix vs hinge penalty:
      obstacle penalty uses a softplus barrier => has nonzero gradient
      when approaching obstacles, not only when penetrating.
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
        w_obs: float = 20.0,
        obstacle_margin: float = 0.06,
        barrier_band: float = 0.06,
        barrier_temp: float = 20.0,
        w_z_track: float = 1.0,
        w_dxy_l2: float = 1e-4,
        w_dxy_smooth: float = 1e-4,
        w_goal_path: float = 0.5,
        w_goal_terminal: float = 10.0,
        scaler: Optional[TorchAffineScalerAdapter] = None,
    ):
        self.mjx_model = mjx_model
        self.ctrl_lower = ctrl_lower
        self.ctrl_upper = ctrl_upper
        self.sim_dt = float(sim_dt)
        self.ctrl_dt = float(ctrl_dt)
        self.n_substeps = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.w_track_xyz = float(w_track_xyz)
        self.w_obs = float(w_obs)
        self.obstacle_margin = float(obstacle_margin)
        self.barrier_band = float(barrier_band)
        self.barrier_temp = float(barrier_temp)

        self.w_z_track = float(w_z_track)
        self.w_dxy_l2 = float(w_dxy_l2)
        self.w_dxy_smooth = float(w_dxy_smooth)

        self.w_goal_path = float(w_goal_path)
        self.w_goal_terminal = float(w_goal_terminal)

        self.scaler = scaler

        self._data0: Optional[mjx.Data] = None
        self._xy_base: Optional[jp.ndarray] = None
        self._goal_xy: Optional[jp.ndarray] = None

        self._obs_centers: Optional[jp.ndarray] = None  # (N,2)
        self._obs_radii: Optional[jp.ndarray] = None    # (N,)

        self._loss_and_grad = jax.jit(jax.value_and_grad(self._loss_fn))

    def set_obstacles(self, obstacles: List[Dict[str, Any]]):
        if obstacles is None or len(obstacles) == 0:
            self._obs_centers = None
            self._obs_radii = None
            return
        centers = np.stack([np.asarray(o["center_xy"], dtype=np.float32) for o in obstacles], axis=0)
        radii = np.asarray([float(o["radius"]) for o in obstacles], dtype=np.float32)
        self._obs_centers = jp.array(centers, dtype=jp.float32)
        self._obs_radii = jp.array(radii, dtype=jp.float32)

    def set_context(
        self,
        *,
        data0: mjx.Data,
        current_xy: np.ndarray | jp.ndarray,
        goal_xy: np.ndarray | jp.ndarray,
    ):
        self._data0 = data0
        self._xy_base = jp.array(np.asarray(current_xy, dtype=np.float32))
        self._goal_xy = jp.array(np.asarray(goal_xy, dtype=np.float32))

    def _obstacle_penalty_xy(self, points_xy: jp.ndarray) -> jp.ndarray:
        """
        Softplus barrier:
          - For dist > safe + band: penalty ~ 0
          - As approaching within band: grows smoothly with nonzero gradient
          - Inside safe: grows more
        """
        if self._obs_centers is None or self._obs_radii is None:
            return jp.array(0.0, dtype=jp.float32)

        diff = points_xy[:, None, :] - self._obs_centers[None, :, :]  # (H,N,2)
        dist = jp.linalg.norm(diff, axis=-1)                          # (H,N)

        safe = (self._obs_radii + self.obstacle_margin)[None, :]      # (1,N)
        band = jp.array(self.barrier_band, dtype=jp.float32)
        temp = jp.array(self.barrier_temp, dtype=jp.float32)

        x = (safe + band) - dist                                      # >0 => within influence region
        pen = jp.log1p(jp.exp(temp * x)) / temp                       # softplus(temp*x)/temp
        return jp.sum(pen * pen)

    def _loss_fn(self, dxy_seq: jp.ndarray) -> jp.ndarray:
        assert self._data0 is not None and self._xy_base is not None and self._goal_xy is not None, \
            "Call set_context() before guidance."

        data0 = self._data0
        xy_base = self._xy_base
        goal_xy = self._goal_xy

        xy_waypoints = xy_base[None, :] + jp.cumsum(dxy_seq, axis=0)

        q0 = data0.qpos[:7]
        z_fixed = fk_pos(q0)[2]

        _, _, ee_real, desired_traj = rollout_track_waypoints_mjx(
            self.mjx_model,
            data0,
            xy_waypoints,
            z_fixed,
            self.ctrl_lower,
            self.ctrl_upper,
            self.n_substeps,
        )

        track_xyz = jp.sum((ee_real - desired_traj) ** 2)

        ee_xy = ee_real[:, :2]
        obs_cost = self._obstacle_penalty_xy(ee_xy)

        # goal shaping (kept, but not too dominant)
        goal_path = jp.sum((ee_xy - goal_xy[None, :]) ** 2)
        goal_terminal = jp.sum((ee_xy[-1] - goal_xy) ** 2)

        z_track = jp.sum((ee_real[:, 2] - z_fixed) ** 2)

        dxy_l2 = jp.sum(dxy_seq ** 2)
        dxy_smooth = jp.sum((dxy_seq[1:] - dxy_seq[:-1]) ** 2) if dxy_seq.shape[0] > 1 else 0.0

        loss = (
            self.w_track_xyz * track_xyz
            + self.w_obs * obs_cost
            + self.w_goal_path * goal_path
            + self.w_goal_terminal * goal_terminal
            + self.w_z_track * z_track
            + self.w_dxy_l2 * dxy_l2
            + self.w_dxy_smooth * dxy_smooth
        )
        return loss

    def __call__(self, state: torch.Tensor, action_scaled: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        if self._data0 is None or self._xy_base is None or self._goal_xy is None:
            raise RuntimeError("MJXDeltaXYGuidance: set_context() must be called before sampling.")

        with torch.no_grad():
            action_unscaled = self.scaler.unscale(action_scaled) if self.scaler is not None else action_scaled

        a_np = action_unscaled.detach().cpu().float().numpy()
        if a_np.ndim == 2:
            a_np = a_np[:, None, :]

        B, H, D = a_np.shape
        if D != 2:
            raise ValueError(f"Expected Δxy dim=2, got {D}")

        grads_unscaled = np.zeros_like(a_np, dtype=np.float32)
        for b in range(B):
            dxy_seq = jp.array(a_np[b], dtype=jp.float32)
            _, g = self._loss_and_grad(dxy_seq)
            grads_unscaled[b] = np.array(g, dtype=np.float32)

        grad_unscaled_t = torch.from_numpy(grads_unscaled).to(
            device=action_scaled.device, dtype=action_scaled.dtype
        )

        grad_scaled = self.scaler.grad_to_scaled(grad_unscaled_t) if self.scaler is not None else grad_unscaled_t

        if action_scaled.ndim == 2:
            grad_scaled = grad_scaled[:, 0, :]

        return grad_scaled
