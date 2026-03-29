# agents/guidance/d3il_mjx_delta_xy_guidance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable

import numpy as np
import torch

import jax
import jax.numpy as jp
from mujoco import mjx


@dataclass
class D3ILGuidanceWeights:
    w_goal_terminal: float = 5.0
    w_obs: float = 20.0
    obstacle_margin: float = 0.10
    w_dxy_l2: float = 1e-4
    w_dxy_smooth: float = 1e-4


class TorchAffineScalerAdapter:
    """
    跟你案例一致：用 inverse_scale_output 估一个近似线性 Jacobian，把 grad_unscaled -> grad_scaled
    （你案例里已经有同名类） :contentReference[oaicite:7]{index=7}
    """
    def __init__(self, inverse_scale_output: Callable[[torch.Tensor], torch.Tensor],
                 action_dim: int, eps: float = 1e-3, device: str = "cpu"):
        self.inverse_scale_output = inverse_scale_output
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self.device = device

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


def _hinge_sq(x: jp.ndarray) -> jp.ndarray:
    # max(0, x)^2
    return jp.square(jp.maximum(0.0, x))


class D3ILMJXDeltaXYGuidance:
    """
    你项目链路版 guidance：

    - dxy_seq interpreted as Δxy from current_xy(base)
      xy_waypoints = base + cumsum(dxy_seq)
    - rollout 用你项目的 controller_compiled + mjx.step
    - loss = goal_terminal + obstacle + regularizers
    - 返回 grad wrt action_scaled（与 d3il diffusion sampler 对接）
    """

    def __init__(
        self,
        *,
        mjx_model: mjx.Model,
        controller_compiled: Callable[[mjx.Data, jp.ndarray, jp.ndarray], jp.ndarray],
        joint_act_idx: jp.ndarray,
        tcp_body_id: int,
        fixed_quat_xyzw: np.ndarray,
        n_substeps: int,
        weights: D3ILGuidanceWeights,
        scaler: Optional[TorchAffineScalerAdapter] = None,
    ):
        self.mjx_model = mjx_model
        self.controller_compiled = controller_compiled
        self.joint_act_idx = joint_act_idx
        self.tcp_body_id = int(tcp_body_id)
        self.fixed_quat = jp.array(np.asarray(fixed_quat_xyzw, dtype=np.float32))
        self.n_substeps = int(n_substeps)

        self.w = weights
        self.scaler = scaler

        self._data0: Optional[mjx.Data] = None
        self._xy_base: Optional[jp.ndarray] = None
        self._goal_xy: Optional[jp.ndarray] = None

        self._obs_centers: Optional[jp.ndarray] = None  # (K,2)

        self._loss_and_grad = jax.jit(jax.value_and_grad(self._loss_fn))

    def set_obstacles_xy(self, obs_xy_list: np.ndarray):
        # obs_xy_list: (K,2)
        if obs_xy_list is None or len(obs_xy_list) == 0:
            self._obs_centers = None
        else:
            self._obs_centers = jp.array(np.asarray(obs_xy_list, dtype=np.float32))

    def set_context(self, *, data0: mjx.Data, current_xy: np.ndarray, goal_xy: np.ndarray):
        self._data0 = data0
        self._xy_base = jp.array(np.asarray(current_xy, dtype=np.float32))
        self._goal_xy = jp.array(np.asarray(goal_xy, dtype=np.float32))

    # def _step_mjx(self, data: mjx.Data, target_xyz: jp.ndarray) -> mjx.Data:
    #     # one macro-step with n_substeps of mjx.step
    #     def substep(d, _):
    #         ctrl7 = self.controller_compiled(d, target_xyz, self.fixed_quat)
    #         d = d.replace(ctrl=d.ctrl.at[self.joint_act_idx].set(ctrl7))
    #         d = mjx.step(self.mjx_model, d)
    #         return d, None
    #     data, _ = jax.lax.scan(substep, data, xs=None, length=self.n_substeps)
    #     return data
    
    def _step_mjx(self, data, target_xyz):
        tgt_quat = self.fixed_quat

        def substep(d_in, _):
            d_ctrl = self.controller_compiled(self.mjx_model, d_in, target_xyz, tgt_quat)
            d_next = mjx.step(self.mjx_model, d_ctrl)
            return d_next, None

        data_out, _ = jax.lax.scan(substep, data, xs=None, length=self.n_substeps)
        return data_out

    def _rollout_tcp_xy(self, data0: mjx.Data, xy_waypoints: jp.ndarray, z_fixed: jp.ndarray) -> jp.ndarray:
        # returns tcp_xy traj: (H,2)
        def body(d, xy):
            target_xyz = jp.array([xy[0], xy[1], z_fixed], dtype=jp.float32)
            d2 = self._step_mjx(d, target_xyz)
            tcp_xyz = d2.xpos[self.tcp_body_id]
            return d2, tcp_xyz[:2]
        _, tcp_xy = jax.lax.scan(body, data0, xy_waypoints)
        return tcp_xy

    def _obs_cost(self, tcp_xy: jp.ndarray) -> jp.ndarray:
        if self._obs_centers is None:
            return jp.array(0.0, dtype=jp.float32)
        diff = tcp_xy[:, None, :] - self._obs_centers[None, :, :]
        dist = jp.linalg.norm(diff, axis=-1)  # (H,K)
        pen = self.w.obstacle_margin - dist
        return jp.sum(_hinge_sq(pen))

    def _loss_fn(self, dxy_seq: jp.ndarray) -> jp.ndarray:
        assert self._data0 is not None and self._xy_base is not None and self._goal_xy is not None, \
            "Call set_context() before guidance."

        xy_base = self._xy_base
        goal_xy = self._goal_xy
        data0 = self._data0

        # integrate Δxy
        xy_waypoints = xy_base[None, :] + jp.cumsum(dxy_seq, axis=0)  # (H,2)

        # z_fixed from current tcp body
        z_fixed = data0.xpos[self.tcp_body_id][2]

        tcp_xy = self._rollout_tcp_xy(data0, xy_waypoints, z_fixed)

        # goal terminal
        goal_terminal = jp.sum((tcp_xy[-1] - goal_xy) ** 2)

        # obstacle
        obs_cost = self._obs_cost(tcp_xy)

        # regularize dxy
        dxy_l2 = jp.sum(dxy_seq ** 2)
        dxy_smooth = jp.sum((dxy_seq[1:] - dxy_seq[:-1]) ** 2) if dxy_seq.shape[0] > 1 else 0.0

        loss = (
            self.w.w_goal_terminal * goal_terminal
            + self.w.w_obs * obs_cost
            + self.w.w_dxy_l2 * dxy_l2
            + self.w.w_dxy_smooth * dxy_smooth
        )
        return loss

    def __call__(self, state: torch.Tensor, action_scaled: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # action_scaled: (B,H,2) in diffusion space
        with torch.no_grad():
            if self.scaler is not None:
                action_unscaled = self.scaler.unscale(action_scaled)  # meters
            else:
                action_unscaled = action_scaled

        a_np = action_unscaled.detach().cpu().float().numpy()
        if a_np.ndim == 2:
            a_np = a_np[:, None, :]

        B, H, D = a_np.shape
        assert D == 2, f"expected Δxy dim=2, got {D}"
        grads_unscaled = np.zeros_like(a_np, dtype=np.float32)

        for b in range(B):
            dxy = jp.array(a_np[b], dtype=jp.float32)  # (H,2)
            _, g = self._loss_and_grad(dxy)
            grads_unscaled[b] = np.array(g, dtype=np.float32)

        grad_unscaled_t = torch.from_numpy(grads_unscaled).to(action_scaled.device, dtype=action_scaled.dtype)

        if self.scaler is not None:
            grad_scaled = self.scaler.grad_to_scaled(grad_unscaled_t)
        else:
            grad_scaled = grad_unscaled_t

        if action_scaled.ndim == 2:
            grad_scaled = grad_scaled[:, 0, :]

        return grad_scaled