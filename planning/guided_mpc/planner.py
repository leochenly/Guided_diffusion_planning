from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import mujoco
import numpy as np
import torch

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

from .constants import DEFAULT_FIXED_QUAT, DEFAULT_OBSTACLES_XY, FINISH_Y
from .costs import AvoidingCostWeights, cost_from_rollout, decompose_cost_from_rollout


@dataclass
class GuidanceConfig:
    enabled: bool = True
    scale_alpha: float = 1.0
    start_ratio: float = 0.0
    interval: int = 1
    grad_clip: float = 100.0
    use_jit: bool = True


class GuidedDiffusionPlannerD3IL:
    """Planner only depends on injected reference trajectory and simulator.

    This is the key refactor: no hard-coded npy loading inside the module.
    """

    def __init__(
        self,
        agent,
        env,
        sim,
        xy_ref: np.ndarray,
        device: str = "cuda",
        weights: AvoidingCostWeights = AvoidingCostWeights(),
        guidance: GuidanceConfig = GuidanceConfig(),
        fixed_quat: np.ndarray = DEFAULT_FIXED_QUAT,
        obstacles_xy: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        settle_steps: int = 10,
        warmstart_steps: int = 3,
        tcp_name_key: str = "tcp",
        debug_steps: int = 100,
        guidance_print_every: int = 20,
        ref_update_mode: str = "inc",
        ref_nearest_window: int = 40,
        max_delta_xy: float = 0.05,
    ):
        if jax is None:
            raise RuntimeError("JAX is required for MJX guidance.")

        self.agent = agent
        self.env = env
        self.sim = sim
        self.device = device
        self.weights = weights
        self.guidance = guidance
        self.fixed_quat = np.asarray(fixed_quat, dtype=np.float32).reshape(4,)
        self.obstacles_xy = jnp.asarray(
            DEFAULT_OBSTACLES_XY if obstacles_xy is None else obstacles_xy,
            dtype=jnp.float32,
        )
        self.xy_ref_np = np.asarray(xy_ref, dtype=np.float32)
        self.xy_ref = jnp.asarray(self.xy_ref_np)
        self.xy_ref_len = int(self.xy_ref_np.shape[0])
        self.max_delta_xy = float(max_delta_xy)

        self.settle_steps = int(settle_steps)
        self.warmstart_steps = int(warmstart_steps)
        self.debug_steps = int(debug_steps)
        self.guidance_print_every = int(guidance_print_every)
        self.ref_update_mode = str(ref_update_mode)
        self.ref_nearest_window = int(ref_nearest_window)

        self.H = int(self.agent.action_seq_size)
        self._tcp_key = self.env.robot.add_id2model_key(tcp_name_key)

        if q_init is None:
            q_init = np.array(
                [-0.35640868, 0.42944545, -0.1350106, -2.05465189, 0.09357822, 2.48071794, 0.23250676],
                dtype=np.float32,
            )
        self.q_init = np.asarray(q_init, dtype=np.float32)

        self.ref_t = 0
        self.step_debug_count = 0
        self._dbg_guidance_calls = 0
        self._guidance_step_logs = []
        self._guidance_plan_logs = []
        self._current_plan_meta = None
        self._plan_counter = 0

        max_dxy = jnp.asarray(self.max_delta_xy, dtype=jnp.float32)

        def _cost_grad(s0, dxy):
            def _J(dxy_local):
                dxy_local = jnp.clip(dxy_local, -max_dxy, max_dxy)
                xy_traj_local = self.sim.rollout_xy_traj(s0, dxy_local)
                xy_traj_local = jnp.nan_to_num(xy_traj_local, nan=1e3, posinf=1e3, neginf=-1e3)
                return cost_from_rollout(
                    xy_traj_local,
                    dxy_local,
                    self.weights,
                    self.obstacles_xy,
                    self.xy_ref,
                    s0["xy_ref0"],
                    s0["ref_t"],
                )

            J, g = jax.value_and_grad(_J)(dxy)
            return jnp.nan_to_num(J, nan=1e3, posinf=1e3, neginf=1e3), jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        self._cost_grad_fn = jax.jit(_cost_grad) if guidance.use_jit else _cost_grad

    def _mj_forward(self):
        mujoco.mj_forward(self.env.scene.model, self.env.scene.data)

    def _get_tcp_xyz(self) -> np.ndarray:
        self._mj_forward()
        m = self.env.scene.model
        d = self.env.scene.data
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self._tcp_key)
        if sid >= 0:
            return d.site_xpos[sid].copy()
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._tcp_key)
        if bid >= 0:
            return d.xpos[bid].copy()
        raise RuntimeError(f"TCP '{self._tcp_key}' not found as site or body")

    def _get_cur_xy_z(self) -> Tuple[np.ndarray, float]:
        p = self._get_tcp_xyz()
        return p[:2].astype(np.float32), float(p[2])

    def _get_cur_quat(self) -> Optional[np.ndarray]:
        if hasattr(self.env.robot, "current_c_quat"):
            self.env.robot.receiveState()
            q = np.asarray(self.env.robot.current_c_quat, dtype=np.float32).copy()
            if q.shape == (4,):
                return q
        return None

    def _start_new_plan_log(self, des_xy_exec: np.ndarray, cur_xy: np.ndarray) -> None:
        self._guidance_step_logs = []
        self._current_plan_meta = {
            "plan_id": int(self._plan_counter),
            "episode_step_idx": int(self.step_debug_count),
            "ref_t_before_sampling": int(self.ref_t),
            "des_xy_exec": np.asarray(des_xy_exec, dtype=np.float32).copy(),
            "cur_xy": np.asarray(cur_xy, dtype=np.float32).copy(),
            "guidance_scale": float(self.guidance.scale_alpha),
            "grad_clip": float(self.guidance.grad_clip) if self.guidance.grad_clip else None,
        }
        self._plan_counter += 1

    def _finalize_current_plan_log(self, pred_delta: Optional[np.ndarray] = None, target_xy: Optional[np.ndarray] = None) -> None:
        if self._current_plan_meta is None:
            return
        item = dict(self._current_plan_meta)
        item["logs"] = list(self._guidance_step_logs)
        item["n_denoising_logs"] = len(self._guidance_step_logs)
        if pred_delta is not None:
            item["pred_delta"] = np.asarray(pred_delta, dtype=np.float32).copy()
        if target_xy is not None:
            item["target_xy"] = np.asarray(target_xy, dtype=np.float32).copy()
        self._guidance_plan_logs.append(item)
        self._guidance_step_logs = []
        self._current_plan_meta = None

    def _make_guidance_fn(self, s0_jax: Dict[str, Any]) -> Callable:
        def guidance_fn(s_torch: torch.Tensor, model_mean: torch.Tensor, t_torch: torch.Tensor, **kwargs):
            del s_torch, kwargs
            x = model_mean
            x_np = x.detach().float().cpu().numpy()
            self._dbg_guidance_calls += 1
            do_print = self._dbg_guidance_calls % self.guidance_print_every == 0
            diffusion_t = int(t_torch[0].detach().cpu().item()) if torch.is_tensor(t_torch) else int(t_torch)

            grads = []
            first_log = None
            if x_np.ndim == 2:
                iterable = [x_np[b : b + 1, :] for b in range(x_np.shape[0])]
            elif x_np.ndim == 3:
                iterable = [x_np[b] for b in range(x_np.shape[0])]
            else:
                raise ValueError(f"Unexpected model_mean shape: {x_np.shape}")

            for b, sample in enumerate(iterable):
                dxy = jnp.asarray(sample, dtype=jnp.float32)
                J, g = self._cost_grad_fn(s0_jax, dxy)
                grads.append(np.array(g[0] if x_np.ndim == 2 else g, dtype=np.float32))

                if b == 0:
                    dxy_used = jnp.clip(dxy, -self.max_delta_xy, self.max_delta_xy)
                    xy_traj = self.sim.rollout_xy_traj(s0_jax, dxy_used)
                    parts = decompose_cost_from_rollout(
                        xy_traj=xy_traj,
                        dxy_seq=dxy_used,
                        weights=self.weights,
                        obstacles_xy=self.obstacles_xy,
                        xy_ref=self.xy_ref,
                        xy_ref0=s0_jax["xy_ref0"],
                        ref_t=s0_jax["ref_t"],
                    )
                    g_np = np.array(g[0] if x_np.ndim == 2 else g, dtype=np.float32)
                    first_log = {
                        "denoise_step_idx": len(self._guidance_step_logs),
                        "diffusion_t": diffusion_t,
                        "J_total": float(np.array(parts["J_total"])),
                        "centerx_cost": float(np.array(parts["centerx_cost"])),
                        "smooth_cost": float(np.array(parts["smooth_cost"])),
                        "track_cost": float(np.array(parts["track_cost"])),
                        "barrier_cost": float(np.array(parts["barrier_cost"])),
                        "weighted_centerx_cost": float(np.array(parts["weighted_centerx_cost"])),
                        "weighted_smooth_cost": float(np.array(parts["weighted_smooth_cost"])),
                        "weighted_track_cost": float(np.array(parts["weighted_track_cost"])),
                        "weighted_barrier_cost": float(np.array(parts["weighted_barrier_cost"])),
                        "grad_norm": float(np.linalg.norm(g_np.reshape(-1))),
                        "grad_max_abs": float(np.max(np.abs(g_np))),
                        "model_mean_norm": float(np.linalg.norm(x_np[0].reshape(-1))),
                        "dxy_norm": float(np.linalg.norm(np.array(dxy_used).reshape(-1))),
                        "rollout_xy_traj": np.array(xy_traj, dtype=np.float32),
                        "sp_xy_traj": np.array(parts["sp_xy_traj"], dtype=np.float32),
                    }
                if do_print and b == 0:
                    self._print_guidance_debug(J, g, dxy, s0_jax)

            grad_np = np.stack(grads, axis=0)
            if self.guidance.grad_clip and self.guidance.grad_clip > 0:
                flat = grad_np.reshape(grad_np.shape[0], -1)
                gn = np.linalg.norm(flat, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.guidance.grad_clip / (gn + 1e-6))
                grad_np_clipped = grad_np * scale.reshape((-1,) + (1,) * (grad_np.ndim - 1))
            else:
                grad_np_clipped = grad_np

            if first_log is not None:
                g0_clipped = grad_np_clipped[0]
                first_log["grad_norm_clipped"] = float(np.linalg.norm(g0_clipped.reshape(-1)))
                first_log["grad_max_abs_clipped"] = float(np.max(np.abs(g0_clipped)))
                first_log["alpha_grad_norm"] = float(self.guidance.scale_alpha) * first_log["grad_norm_clipped"]
                self._guidance_step_logs.append(first_log)

            return torch.from_numpy(grad_np_clipped).to(x.device).type_as(x)

        return guidance_fn

    def _print_guidance_debug(self, J, g, dxy, s0_jax) -> None:
        J_val = float(np.array(J))
        g_np = np.array(g)
        g_norm = float(np.linalg.norm(g_np.reshape(-1)))
        print(
            f"[guidance] call={self._dbg_guidance_calls} ref_t={int(self.ref_t)} "
            f"J={J_val:.6f} ||grad||={g_norm:.3e} approx_step={float(self.guidance.scale_alpha)*g_norm:.3e}"
        )
        eps = 1e-3
        J2, _ = self._cost_grad_fn(s0_jax, dxy - eps * g)
        print(f"    J_after_eps={float(np.array(J2)):.6f}  delta={float(np.array(J2)) - J_val:+.6f}")

    def _build_extra_args(self, des_xy_exec: np.ndarray, cur_xy: np.ndarray) -> Dict[str, Any]:
        if (not self.guidance.enabled) or self.guidance.scale_alpha is None or float(self.guidance.scale_alpha) == 0.0:
            return {}
        if hasattr(self.sim, "sync_from_env"):
            self.sim.sync_from_env()

        self._start_new_plan_log(des_xy_exec=des_xy_exec, cur_xy=cur_xy)
        _, cur_z = self._get_cur_xy_z()
        s0_jax = {
            "d0": self.env.scene.mjx_data,
            "z_fixed": jnp.asarray(cur_z, dtype=jnp.float32),
            "quat_fixed": jnp.asarray(self.fixed_quat, dtype=jnp.float32),
            "xy_ref0": jnp.asarray(des_xy_exec.astype(np.float32), dtype=jnp.float32),
            "ref_t": jnp.asarray(self.ref_t, dtype=jnp.int32),
        }
        return {
            "guidance_fn": self._make_guidance_fn(s0_jax),
            "guidance_scale": float(self.guidance.scale_alpha),
            "guidance_start": float(self.guidance.start_ratio),
            "guidance_every": int(self.guidance.interval),
            "grad_clip_norm": float(self.guidance.grad_clip) if self.guidance.grad_clip else None,
            "grad_normalize": False,
            "guidance_kwargs": None,
        }

    def _init_episode_state(self, random_reset: bool):
        self.agent.reset()
        obs = self.env.reset(random=True) if random_reset else self.env.reset()
        obs = np.asarray(obs, dtype=np.float32).copy()
        if obs.shape[0] != 2:
            raise ValueError(f"Expected env.reset() to return tcp_xy shape (2,), got {obs.shape}")

        self.env.robot.beam_to_joint_pos(self.q_init)
        if hasattr(self.sim, "sync_from_env"):
            self.sim.sync_from_env()
        for _ in range(self.settle_steps):
            self.env.scene.next_step()

        cur_xy = np.asarray(self.env.get_observation(), dtype=np.float32).copy()
        if cur_xy.shape[0] != 2:
            cur_xy, _ = self._get_cur_xy_z()
        _, cur_z = self._get_cur_xy_z()
        self.z_fixed = float(cur_z)

        cur_quat = self._get_cur_quat()
        if self.warmstart_steps > 0 and cur_quat is not None:
            des_xy_exec = cur_xy.copy()
            full_action0 = np.concatenate([des_xy_exec, [self.z_fixed], cur_quat], axis=0).astype(np.float32)
            for _ in range(self.warmstart_steps):
                obs_next, *_ = self.env.step(full_action0)
                cur_xy = np.asarray(obs_next, dtype=np.float32).copy()

        return cur_xy, cur_xy.copy()

    def _update_ref_t(self, xy_for_alignment: np.ndarray) -> None:
        if self.ref_update_mode == "inc":
            self.ref_t = min(int(self.ref_t) + 1, self.xy_ref_len - 1)
            return
        if self.ref_update_mode == "nearest":
            window = int(self.ref_nearest_window)
            t0 = max(int(self.ref_t) - window // 2, 0)
            t1 = min(int(self.ref_t) + window // 2, self.xy_ref_len)
            seg = self.xy_ref_np[t0:t1]
            d2 = np.sum((seg - xy_for_alignment[None, :]) ** 2, axis=1)
            self.ref_t = t0 + int(np.argmin(d2))
            return
        raise ValueError(f"Unknown ref_update_mode: {self.ref_update_mode}")

    def step_once(self, des_xy_exec: np.ndarray, cur_xy: np.ndarray):
        if self.ref_update_mode in ("nearest", "inc"):
            self._update_ref_t(des_xy_exec)

        obs_4d = np.concatenate([des_xy_exec, cur_xy]).astype(np.float32)
        extra_args = self._build_extra_args(des_xy_exec, cur_xy)
        pred_delta = np.asarray(self.agent.predict(obs_4d, extra_args=extra_args), dtype=np.float32)

        if not np.isfinite(pred_delta).all():
            pred_delta = np.zeros((2,), dtype=np.float32)
        if pred_delta.ndim == 2:
            pred_delta = pred_delta[0]
        elif pred_delta.ndim > 2:
            pred_delta = pred_delta.reshape(-1, 2)[0]
        if pred_delta.shape != (2,):
            raise ValueError(f"pred_delta shape should be (2,), got {pred_delta.shape}")

        pred_delta = np.clip(pred_delta, -self.max_delta_xy, self.max_delta_xy)
        target_xy = des_xy_exec + pred_delta
        if not np.isfinite(target_xy).all():
            target_xy = des_xy_exec.copy()

        target_xy = target_xy.astype(np.float32).reshape(2,)
        target_xy[0] = np.clip(target_xy[0], 0.0, 0.9)
        target_xy[1] = np.clip(target_xy[1], -0.4, 0.6)

        z_use = getattr(self, "z_fixed", None)
        if z_use is None:
            _, z_use = self._get_cur_xy_z()
        full_action = np.concatenate([target_xy, [float(z_use)], self.fixed_quat], axis=0).astype(np.float32)

        self._finalize_current_plan_log(pred_delta=np.asarray(pred_delta, dtype=np.float32), target_xy=target_xy)

        if not np.isfinite(full_action).all():
            full_action = np.nan_to_num(full_action, nan=0.0, posinf=0.0, neginf=0.0)
        obs_next, reward, done, info = self.env.step(full_action)
        obs_next = np.asarray(obs_next)
        if not np.isfinite(obs_next).all():
            done = True

        cur_xy_next = np.asarray(obs_next, dtype=np.float32).copy()
        if cur_xy_next.shape[0] != 2:
            cur_xy_next, _ = self._get_cur_xy_z()

        if self.step_debug_count < self.debug_steps:
            ref_xy = self.xy_ref_np[int(self.ref_t)]
            print(f"[step {self.step_debug_count}] ref_t={int(self.ref_t)} target_xy={target_xy} ref_xy={ref_xy}")
        self.step_debug_count += 1
        return cur_xy_next, float(0.0 if reward is None else reward), bool(done), info, target_xy.copy()

    def run_episode(self, max_steps: int = 250, random_reset: bool = True) -> Dict[str, Any]:
        self._guidance_plan_logs = []
        self._guidance_step_logs = []
        self._current_plan_meta = None
        self._plan_counter = 0
        self.step_debug_count = 0

        cur_xy, des_xy_exec = self._init_episode_state(random_reset=random_reset)
        self.ref_t = 0
        traj = [cur_xy.copy()]
        sp_traj = [des_xy_exec.copy()]
        ref_idx = [int(self.ref_t)]
        rewards = []
        last_info = None
        done = False

        for _ in range(max_steps):
            cur_xy, reward, done, info, des_xy_exec = self.step_once(des_xy_exec, cur_xy)
            traj.append(cur_xy.copy())
            sp_traj.append(des_xy_exec.copy())
            ref_idx.append(int(self.ref_t))
            rewards.append(reward)
            last_info = info
            self.agent.action_counter = self.agent.action_seq_size
            self.agent.curr_action_seq = None
            if done:
                break

        traj = np.asarray(traj, dtype=np.float32)
        if last_info is not None and isinstance(last_info, (list, tuple)) and len(last_info) > 1:
            success = bool(last_info[1])
        else:
            success = bool(traj[-1, 1] >= FINISH_Y)

        return {
            "traj_xy": traj,
            "sp_xy": np.asarray(sp_traj, dtype=np.float32),
            "ref_idx": np.asarray(ref_idx, dtype=np.int32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "done": bool(done),
            "success": success,
            "steps": int(len(traj) - 1),
            "guidance_logs": self._guidance_plan_logs,
        }
