# guided_diffusion_planner.py
# ------------------------------------------------------------
# Debuggable D3IL-compatible Guided Diffusion Planner
# - Adds strong diagnostics for tracking-guidance
# - Fixes duplicated grad computation / duplicated track cost accumulation
# - Adds setpoint trajectory (sp_xy) and ref_idx logging for plotting/debug
# - Adds optional ref_t update via nearest-point (recommended)
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, Tuple, Optional

import numpy as np
import torch
import mujoco

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None


from jax.nn import softplus

def _dist_point_to_segment(xy, a, b, eps=1e-9):
    """
    xy: (..., 2)
    a,b: (2,) segment endpoints
    returns: (...,) Euclidean distance to segment
    """
    ab = b - a                       # (2,)
    ap = xy - a                      # (...,2)
    ab2 = jnp.sum(ab * ab) + eps     # scalar
    t = jnp.sum(ap * ab, axis=-1) / ab2   # (...,)
    t = jnp.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab     # (...,2)
    d = jnp.linalg.norm(xy - proj, axis=-1)
    return d

def barrier_segment_cost(
    xy: jnp.ndarray,          # [H,2] or [H+1,2]
    x0: float, x1: float, y0: float, y1: float,
    margin: float,
    temperature: float,
) -> jnp.ndarray:
    """
    Soft barrier around a horizontal segment from (x0,y0) to (x1,y0).
    Penalize points whose distance to the segment is < margin.
    Smooth with softplus((margin - dist)/temperature).
    """
    x = xy[..., 0]
    y = xy[..., 1]

    # distance in x to the interval [x0, x1]
    dx = jax.nn.relu(x0 - x) + jax.nn.relu(x - x1)
    # dy = jax.nn.relu(y0 - y) + jax.nn.relu(y - y1)
    dy = y - y0

    # distance to the segment
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-12)

    # soft penalty when inside margin
    z = (margin - dist) / (temperature + 1e-12)
    penalty = jax.nn.softplus(z)

    return jnp.mean(penalty)

# ----------------------------
# Load reference trajectory
# ----------------------------
# IMPORTANT: make sure this path points to the intended reference file in YOUR repo.
# If you want, change to an absolute path.
xy_ref_np = np.load("avoiding/plans/xy_plan_env_065_00.npy").astype(np.float32)  # [T,2]
xy_ref = jnp.asarray(xy_ref_np)

FINISH_Y = 0.35
CENTER_X = 0.5

OBSTACLES_XY = np.array(
    [
        [0.5,   -0.1],
        [0.425,  0.08],
        [0.575,  0.08],
        [0.35,   0.26],
        [0.5,    0.26],
        [0.65,   0.26],
    ],
    dtype=np.float32
)

DEFAULT_FIXED_QUAT = np.array([0, 1, 0, 0], dtype=np.float32)


def huber(x, delta):
    absx = jnp.abs(x)
    quad = jnp.minimum(absx, delta)
    lin = absx - quad
    return 0.5 * quad**2 + delta * lin


# ============================================================
# Weights
# ============================================================

@dataclass
class AvoidingCostWeights:
    # tracking
    w_track: float = 0.0
    track_huber_delta: float = 0.01

    # point obstacles (Gaussian)
    w_obs: float = 0.0
    obs_sigma: float = 0.03

    # regularization
    # NOTE: for center-x experiment, set w_centerx > 0 and w_track = 0
    w_centerx: float = 1.0
    w_smooth: float = 0.00

    # bounds
    w_bound: float = 0.0
    x_min: float = 0.00
    x_max: float = 0.90
    y_min: float = -0.35
    y_max: float = 0.35

    # barrier / obstacle (optional)
    # mode: "segment" -> horizontal segment at y=barrier_y0, x in [x0,x1]
    #       "rect"    -> rectangle [x0,x1] x [y0,y1]
    barrier_mode: str = "rect"
    barrier_x0: float = 0.45
    barrier_x1: float = 0.55
    barrier_y0: float = 0.08
    barrier_y1: float = 0.08

    w_barrier: float = 0.0
    barrier_margin: float = 0.02
    barrier_temp: float = 0.01


# ============================================================
# JAX cost and grad
# ============================================================

def barrier_rect_cost(
    xy: "jnp.ndarray",
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    margin: float,
    temperature: float,
) -> "jnp.ndarray":
    """Soft barrier for an axis-aligned rectangle."""
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    hx = 0.5 * (x_max - x_min)
    hy = 0.5 * (y_max - y_min)
    p = xy - jnp.asarray([cx, cy], dtype=xy.dtype)
    q = jnp.abs(p) - jnp.asarray([hx, hy], dtype=xy.dtype)
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
    sdf = outside + inside
    z = (margin - sdf) / jnp.maximum(temperature, 1e-6)
    return jnp.mean(jax.nn.softplus(z))


def cost_from_rollout(
    xy_traj: "jnp.ndarray",      # [H+1,2] TCP rollout from differentiable simulator S
    dxy_seq: "jnp.ndarray",      # [H,2]   diffusion variable
    weights: AvoidingCostWeights,
    obstacles_xy: "jnp.ndarray", # [K,2]
    xy_ref: "jnp.ndarray",       # [T,2]
    xy_ref0: "jnp.ndarray",      # [2]
    ref_t: "jnp.ndarray",        # scalar int32
) -> "jnp.ndarray":
    """Cost J(τ) used by Algorithm 1.1 Guided Diffusion Planning."""
    H = dxy_seq.shape[0]

    # setpoint reconstruction (optional for tracking)
    xy_sp = xy_ref0[None, :] + jnp.cumsum(dxy_seq, axis=0)  # [H,2]

    # reference window
    pad = H
    last = xy_ref[-1:, :]
    xy_ref_pad = jnp.concatenate([xy_ref, jnp.repeat(last, pad, axis=0)], axis=0)
    idx_dtype = jnp.int32
    t0 = jnp.asarray(ref_t, dtype=idx_dtype)
    t0 = jnp.clip(t0, jnp.asarray(0, idx_dtype), jnp.asarray(xy_ref.shape[0] - 1, idx_dtype))
    xy_tgt = jax.lax.dynamic_slice(xy_ref_pad, (t0, jnp.asarray(0, idx_dtype)), (H, 2))

    # center-x objective (ROLL-OUT based)
    # x_roll = xy_sp[:, 0]
    x_roll = xy_traj[1:, 0]
    centerx_cost = jnp.mean((x_roll - jnp.asarray(0.5, jnp.float32)) ** 2)

    # smoothness
    smooth_cost = jnp.mean(jnp.sum(dxy_seq * dxy_seq, axis=-1))

    total = weights.w_centerx * centerx_cost + weights.w_smooth * smooth_cost

    # optional tracking
    if getattr(weights, "w_track", 0.0) and weights.w_track > 0.0:
        err = xy_sp - xy_tgt
        w = jnp.power(jnp.asarray(0.7, jnp.float32), jnp.arange(H, dtype=jnp.float32))
        w = w / jnp.sum(w)
        track_cost = jnp.sum(w * jnp.sum(huber(err, weights.track_huber_delta), axis=-1))
        total = total + weights.w_track * track_cost

    # optional barrier
    if getattr(weights, "w_barrier", 0.0) and weights.w_barrier > 0.0:
        mode = getattr(weights, "barrier_mode", "segment")
        if mode == "segment":
            barrier = barrier_segment_cost(
                xy_traj[1:, :],
                x0=weights.barrier_x0, x1=weights.barrier_x1, y0=weights.barrier_y0,y1=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )
        else:
            barrier = barrier_rect_cost(
                xy_traj[1:, :],
                x_min=weights.barrier_x0, x_max=weights.barrier_x1,
                y_min=weights.barrier_y0, y_max=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )
        total = total + weights.w_barrier * barrier

    return total

def decompose_cost_from_rollout(
    xy_traj: "jnp.ndarray",      # [H+1,2]
    dxy_seq: "jnp.ndarray",      # [H,2]
    weights: AvoidingCostWeights,
    obstacles_xy: "jnp.ndarray", # [K,2]
    xy_ref: "jnp.ndarray",       # [T,2]
    xy_ref0: "jnp.ndarray",      # [2]
    ref_t: "jnp.ndarray",        # scalar int32
) -> Dict[str, "jnp.ndarray"]:
    """
    Return detailed cost terms for logging / plotting.
    """
    H = dxy_seq.shape[0]

    xy_sp = xy_ref0[None, :] + jnp.cumsum(dxy_seq, axis=0)  # [H,2]

    pad = H
    last = xy_ref[-1:, :]
    xy_ref_pad = jnp.concatenate([xy_ref, jnp.repeat(last, pad, axis=0)], axis=0)
    idx_dtype = jnp.int32
    t0 = jnp.asarray(ref_t, dtype=idx_dtype)
    t0 = jnp.clip(t0, jnp.asarray(0, idx_dtype), jnp.asarray(xy_ref.shape[0] - 1, idx_dtype))
    xy_tgt = jax.lax.dynamic_slice(xy_ref_pad, (t0, jnp.asarray(0, idx_dtype)), (H, 2))

    # 1) center-x
    x_roll = xy_traj[1:, 0]
    centerx_cost = jnp.mean((x_roll - jnp.asarray(0.5, jnp.float32)) ** 2)

    # 2) smoothness
    smooth_cost = jnp.mean(jnp.sum(dxy_seq * dxy_seq, axis=-1))

    # 3) track
    track_cost = jnp.asarray(0.0, dtype=jnp.float32)
    if getattr(weights, "w_track", 0.0) and weights.w_track > 0.0:
        err = xy_sp - xy_tgt
        w = jnp.power(jnp.asarray(0.7, jnp.float32), jnp.arange(H, dtype=jnp.float32))
        w = w / jnp.sum(w)
        track_cost = jnp.sum(w * jnp.sum(huber(err, weights.track_huber_delta), axis=-1))

    # 4) barrier
    barrier_cost = jnp.asarray(0.0, dtype=jnp.float32)
    if getattr(weights, "w_barrier", 0.0) and weights.w_barrier > 0.0:
        mode = getattr(weights, "barrier_mode", "segment")
        if mode == "segment":
            barrier_cost = barrier_segment_cost(
                xy_traj[1:, :],
                x0=weights.barrier_x0,
                x1=weights.barrier_x1,
                y0=weights.barrier_y0,
                y1=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )
        else:
            barrier_cost = barrier_rect_cost(
                xy_traj[1:, :],
                x_min=weights.barrier_x0,
                x_max=weights.barrier_x1,
                y_min=weights.barrier_y0,
                y_max=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )

    weighted_centerx = weights.w_centerx * centerx_cost
    weighted_smooth = weights.w_smooth * smooth_cost
    weighted_track = weights.w_track * track_cost
    weighted_barrier = weights.w_barrier * barrier_cost

    total = weighted_centerx + weighted_smooth + weighted_track + weighted_barrier

    return {
        "J_total": total,
        "centerx_cost": centerx_cost,
        "smooth_cost": smooth_cost,
        "track_cost": track_cost,
        "barrier_cost": barrier_cost,
        "weighted_centerx_cost": weighted_centerx,
        "weighted_smooth_cost": weighted_smooth,
        "weighted_track_cost": weighted_track,
        "weighted_barrier_cost": weighted_barrier,
        "sp_xy_traj": xy_sp,
        "xy_ref_tgt": xy_tgt,
    }

# ============================================================
# Guidance config
# ============================================================

@dataclass
class GuidanceConfig:
    enabled: bool = True
    scale_alpha: float = 1.0
    start_ratio: float = 0.0
    interval: int = 1
    grad_clip: float = 100.0
    use_jit: bool = True


# ============================================================
# Planner
# ============================================================

class GuidedDiffusionPlannerD3IL:

    def __init__(
        self,
        agent,
        env,
        sim,
        device: str = "cuda",
        weights: AvoidingCostWeights = AvoidingCostWeights(),
        guidance: GuidanceConfig = GuidanceConfig(),
        fixed_quat: np.ndarray = DEFAULT_FIXED_QUAT,
        q_init: Optional[np.ndarray] = None,
        settle_steps: int = 10,
        warmstart_steps: int = 3,
        tcp_name_key: str = "tcp",

        # debug + ref update
        debug_steps: int = 100,
        guidance_print_every: int = 20,
        ref_update_mode: str = "inc",   # "inc" or "nearest"
        ref_nearest_window: int = 40,
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

        self.settle_steps = int(settle_steps)
        self.warmstart_steps = int(warmstart_steps)

        # ddpm-encdec sequence length
        self.H = int(self.agent.action_seq_size)
        print(f"Using horizon H={self.H} from agent.action_seq_size={self.agent.action_seq_size}")

        # init pose
        if q_init is None:
            q_init = np.array([
                -0.35640868, 0.42944545, -0.1350106, -2.05465189,
                 0.09357822, 2.48071794,  0.23250676
            ], dtype=np.float32)
        self.q_init = np.asarray(q_init, dtype=np.float32)

        # TCP key
        self._tcp_key = self.env.robot.add_id2model_key(tcp_name_key)

        # reference traj
        self.xy_ref = xy_ref
        self._xy_ref_np = xy_ref_np
        self._xy_ref_len = int(xy_ref_np.shape[0])

        # obstacle list
        self.obstacles_xy = jnp.asarray(OBSTACLES_XY, dtype=jnp.float32)

        # debug
        self.debug_steps = int(debug_steps)
        self.guidance_print_every = int(guidance_print_every)
        self.ref_update_mode = str(ref_update_mode)
        self.ref_nearest_window = int(ref_nearest_window)

        self.ref_t = 0
        self.step_debug_count = 0
        self._dbg_guidance_calls = 0
        
        # ---- denoising-step logging ----
        self._guidance_step_logs = []   # 当前一次 planning 内，每个 denoising step 的日志
        self._guidance_plan_logs = []   # 当前 episode 内，每次 planning 的日志
        self._current_plan_meta = None  # 当前 planning 的元信息
        self._plan_counter = 0
        
        MAX_DXY = jnp.asarray(0.05, dtype=jnp.float32)  # 和 step_once 里 pred_delta clip 保持一致

        def _cg(s0, dxy):
            """Return (J, grad_dxy J) following Algorithm 1.1.
            Rollout must be inside the closure for correct grad.
            """

            def _J(dxy_local):
                # 1) 关键：在 guidance 内也要 clip，否则 rollout 会被 early diffusion 的大跳变拉爆
                dxy_local = jnp.clip(dxy_local, -MAX_DXY, MAX_DXY)
                # jax.debug.print(
                #     "[debug] inside _J: min={minv:.3f} max={maxv:.3f} mean={meanv:.3f}",
                #     minv=jnp.min(dxy_local),
                #     maxv=jnp.max(dxy_local),
                #     meanv=jnp.mean(dxy_local),
                # )
                # 2) 可微 rollout
                xy_traj_local = self.sim.rollout_xy_traj(s0, dxy_local)  # [H+1,2]

                # 3) NaN/Inf 防护：一旦 rollout 爆了，直接给一个很大代价（梯度会变弱/不稳定，但比 NaN 强）
                xy_traj_local = jnp.nan_to_num(
                    xy_traj_local, nan=1e3, posinf=1e3, neginf=-1e3
                )

                return cost_from_rollout(
                    xy_traj_local,
                    dxy_local,                # 注意：传 clip 后的 dxy
                    self.weights,
                    self.obstacles_xy,
                    self.xy_ref,
                    s0["xy_ref0"],
                    s0["ref_t"],
                )

            J, g = jax.value_and_grad(_J)(dxy)
            # jax.debug.print(
            #     "[debug] after _J: J={J:.6f} grad norm={grad_norm:.3e}",
            #     J=J,
            #     grad_norm=jnp.linalg.norm(g),
            # )

            g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            J = jnp.nan_to_num(J, nan=1e3, posinf=1e3, neginf=1e3)

            # jax.debug.print(
            #     "[debug] after NaN/Inf protection J={J:.6f} grad norm={grad_norm:.3e}",
            #     J=J,
            #     grad_norm=jnp.linalg.norm(g),
            # )


            return J, g

        # def _cg(s0, dxy):
        #     """Return (J, grad_dxy J) following Algorithm 1.1.
        #     MUST compute rollout inside value_and_grad closure.
        #     """
        #     def _J(dxy_local):
        #         xy_traj_local = self.sim.rollout_xy_traj(s0, dxy_local)   # ✅ inside closure
        #         return cost_from_rollout(
        #             xy_traj_local,
        #             dxy_local,
        #             self.weights,
        #             self.obstacles_xy,
        #             self.xy_ref,
        #             s0["xy_ref0"],
        #             s0["ref_t"],
        #         )

        #     J, g = jax.value_and_grad(_J)(dxy)
        #     return J, g

        self._cost_grad_fn = jax.jit(_cg) if guidance.use_jit else _cg

    # ---------------- TCP reading ----------------

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
    
    def _start_new_plan_log(self, des_xy_exec: np.ndarray, cur_xy: np.ndarray):
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

    def _finalize_current_plan_log(self, pred_delta: Optional[np.ndarray] = None, target_xy: Optional[np.ndarray] = None):
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

    # ---------------- guidance fn ----------------

    # def _make_guidance_fn(self, s0_jax: Dict[str, Any]) -> Callable:

    #     def guidance_fn(s_torch: torch.Tensor, model_mean: torch.Tensor, t_torch: torch.Tensor, **kwargs):
    #         x = model_mean
    #         x_np = x.detach().float().cpu().numpy()  # [B,H,2] typically

    #         self._dbg_guidance_calls += 1
    #         do_print = (self._dbg_guidance_calls % self.guidance_print_every == 0)

    #         if x_np.ndim == 2:
    #             # [B,2] (rare)
    #             grads = []
    #             for b in range(x_np.shape[0]):
    #                 dxy = jnp.asarray(x_np[b:b+1, :], dtype=jnp.float32)  # [1,2]
    #                 J, g = self._cost_grad_fn(s0_jax, dxy)
    #                 grads.append(np.array(g[0], dtype=np.float32))
    #                 if do_print and b == 0:
    #                     self._print_guidance_debug(J, g, dxy, s0_jax)
    #             grad_np = np.stack(grads, axis=0)  # [B,2]

    #         elif x_np.ndim == 3:
    #             # [B,H,2]
    #             grads = []
    #             for b in range(x_np.shape[0]):
    #                 dxy = jnp.asarray(x_np[b], dtype=jnp.float32)  # [H,2]
    #                 J, g = self._cost_grad_fn(s0_jax, dxy)
    #                 grads.append(np.array(g, dtype=np.float32))
    #                 if do_print and b == 0:
    #                     self._print_guidance_debug(J, g, dxy, s0_jax)
    #             grad_np = np.stack(grads, axis=0)  # [B,H,2]
    #         else:
    #             raise ValueError(f"Unexpected model_mean shape: {x_np.shape}")

    #         # clip
    #         if self.guidance.grad_clip and self.guidance.grad_clip > 0:
    #             flat = grad_np.reshape(grad_np.shape[0], -1)
    #             gn = np.linalg.norm(flat, axis=1, keepdims=True)
    #             scale = np.minimum(1.0, self.guidance.grad_clip / (gn + 1e-6))
    #             grad_np = grad_np * scale.reshape((-1,) + (1,) * (grad_np.ndim - 1))

    #         return torch.from_numpy(grad_np).to(x.device).type_as(x)

    #     return guidance_fn
    
    def _make_guidance_fn(self, s0_jax: Dict[str, Any]) -> Callable:

        def guidance_fn(s_torch: torch.Tensor, model_mean: torch.Tensor, t_torch: torch.Tensor, **kwargs):
            x = model_mean
            x_np = x.detach().float().cpu().numpy()

            self._dbg_guidance_calls += 1
            do_print = (self._dbg_guidance_calls % self.guidance_print_every == 0)

            # 当前 diffusion t
            diffusion_t = int(t_torch[0].detach().cpu().item()) if torch.is_tensor(t_torch) else int(t_torch)

            # 先计算 raw grads
            if x_np.ndim == 2:
                grads = []
                first_log = None
                for b in range(x_np.shape[0]):
                    dxy = jnp.asarray(x_np[b:b+1, :], dtype=jnp.float32)  # [1,2]
                    J, g = self._cost_grad_fn(s0_jax, dxy)
                    grads.append(np.array(g[0], dtype=np.float32))

                    if b == 0:
                        dxy_used = jnp.clip(dxy, -0.05, 0.05)
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
                        g_np = np.array(g[0], dtype=np.float32)
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

            elif x_np.ndim == 3:
                grads = []
                first_log = None
                for b in range(x_np.shape[0]):
                    dxy = jnp.asarray(x_np[b], dtype=jnp.float32)  # [H,2]
                    J, g = self._cost_grad_fn(s0_jax, dxy)
                    grads.append(np.array(g, dtype=np.float32))

                    if b == 0:
                        dxy_used = jnp.clip(dxy, -0.05, 0.05)
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
                        g_np = np.array(g, dtype=np.float32)
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

            else:
                raise ValueError(f"Unexpected model_mean shape: {x_np.shape}")

            # ---- clip grad，并把 clip 后范数也记录进去 ----
            if self.guidance.grad_clip and self.guidance.grad_clip > 0:
                flat = grad_np.reshape(grad_np.shape[0], -1)
                gn = np.linalg.norm(flat, axis=1, keepdims=True)
                scale = np.minimum(1.0, self.guidance.grad_clip / (gn + 1e-6))
                grad_np_clipped = grad_np * scale.reshape((-1,) + (1,) * (grad_np.ndim - 1))
            else:
                grad_np_clipped = grad_np

            if first_log is not None:
                g0_clipped = grad_np_clipped[0]
                g0_clipped_norm = float(np.linalg.norm(g0_clipped.reshape(-1)))
                first_log["grad_norm_clipped"] = g0_clipped_norm
                first_log["grad_max_abs_clipped"] = float(np.max(np.abs(g0_clipped)))
                first_log["alpha_grad_norm"] = float(self.guidance.scale_alpha) * g0_clipped_norm
                self._guidance_step_logs.append(first_log)

            return torch.from_numpy(grad_np_clipped).to(x.device).type_as(x)

        return guidance_fn

    def _print_guidance_debug(self, J, g, dxy, s0_jax):
        J_val = float(np.array(J))
        g_np = np.array(g)
        g_norm = float(np.linalg.norm(g_np.reshape(-1)))
        alpha = float(self.guidance.scale_alpha)

        print(f"[guidance] call={self._dbg_guidance_calls} ref_t={int(self.ref_t)} "
              f"J={J_val:.6f} ||grad||={g_norm:.3e} approx_step={alpha*g_norm:.3e}")

        # local descent check
        eps = 1e-3
        dxy2 = dxy - eps * g
        J2, _ = self._cost_grad_fn(s0_jax, dxy2)
        J2_val = float(np.array(J2))
        print(f"    J_after_eps={J2_val:.6f}  delta={J2_val - J_val:+.6f}")
        
    # def _build_extra_args(self, des_xy_exec: np.ndarray) -> Dict[str, Any]:
    #     if (not self.guidance.enabled) or (self.guidance.scale_alpha is None) or (float(self.guidance.scale_alpha) == 0.0):
    #         return {}

    #     if hasattr(self.sim, "sync_from_env"):
    #         self.sim.sync_from_env()

    #     # ---- 强制 mjx_data pytree 里 float64 -> float32 ----
    #     from jax import tree_util
    #     def _to_f32(x):
    #         if hasattr(x, "dtype") and x.dtype == jnp.float64:
    #             return x.astype(jnp.float32)
    #         return x
    #     d0 = tree_util.tree_map(_to_f32, self.env.scene.mjx_data)

    #     cur_xy, cur_z = self._get_cur_xy_z()

    #     s0_jax = dict(
    #         d0=d0,
    #         z_fixed=jnp.asarray(cur_z, dtype=jnp.float32),
    #         quat_fixed=jnp.asarray(self.fixed_quat, dtype=jnp.float32),
    #         xy_ref0=jnp.asarray(des_xy_exec.astype(np.float32), dtype=jnp.float32),
    #         ref_t=jnp.asarray(self.ref_t, dtype=jnp.int32),
    #     )

    #     guidance_fn = self._make_guidance_fn(s0_jax)

    #     return dict(
    #         guidance_fn=guidance_fn,
    #         guidance_scale=float(self.guidance.scale_alpha),
    #         guidance_start=float(self.guidance.start_ratio),
    #         guidance_every=int(self.guidance.interval),
    #         grad_clip_norm=float(self.guidance.grad_clip) if self.guidance.grad_clip else None,
    #         grad_normalize=False,
    #         guidance_kwargs=None,
    #     )

    def _build_extra_args(self, des_xy_exec: np.ndarray) -> Dict[str, Any]:
        if (not self.guidance.enabled) or (self.guidance.scale_alpha is None) or (float(self.guidance.scale_alpha) == 0.0):
            return {}

        if hasattr(self.sim, "sync_from_env"):
            self.sim.sync_from_env()

        d0 = self.env.scene.mjx_data
        cur_xy, cur_z = self._get_cur_xy_z()
        self._start_new_plan_log(des_xy_exec=des_xy_exec, cur_xy=cur_xy)

        s0_jax = dict(
            d0=d0,
            z_fixed=jnp.asarray(cur_z, dtype=jnp.float32),
            quat_fixed=jnp.asarray(self.fixed_quat, dtype=jnp.float32),
            xy_ref0=jnp.asarray(des_xy_exec.astype(np.float32), dtype=jnp.float32),
            ref_t=jnp.asarray(self.ref_t, dtype=jnp.int32),
        )

        guidance_fn = self._make_guidance_fn(s0_jax)

        return dict(
            guidance_fn=guidance_fn,
            guidance_scale=float(self.guidance.scale_alpha),
            guidance_start=float(self.guidance.start_ratio),
            guidance_every=int(self.guidance.interval),
            grad_clip_norm=float(self.guidance.grad_clip) if self.guidance.grad_clip else None,
            grad_normalize=False,
            guidance_kwargs=None,
        )

    # ---------------- init / step ----------------

    def _init_episode_state(self, random_reset: bool):
        self.agent.reset()
        if random_reset:
            obs = self.env.reset(random=True)
        else:
            obs = self.env.reset()

        obs = np.asarray(obs, dtype=np.float32).copy()
        if obs.shape[0] != 2:
            raise ValueError(f"Expected env.reset() to return tcp_xy shape (2,), got {obs.shape}")

        # force q_init
        self.env.robot.beam_to_joint_pos(self.q_init)
        if hasattr(self.sim, "sync_from_env"):
            self.sim.sync_from_env()

        for _ in range(self.settle_steps):
            self.env.scene.next_step()

        # use env observation as cur_xy (avoiding_sim)
        cur_xy = np.asarray(self.env.get_observation(), dtype=np.float32).copy()
        if cur_xy.shape[0] != 2:
            cur_xy, _ = self._get_cur_xy_z()

        _, cur_z = self._get_cur_xy_z()
        self.z_fixed = float(cur_z)

        # warm-start
        cur_quat = self._get_cur_quat()
        if self.warmstart_steps > 0 and cur_quat is not None:
            des_xy_exec = cur_xy.copy()
            full_action0 = np.concatenate([des_xy_exec, [self.z_fixed], cur_quat], axis=0).astype(np.float32)
            for _ in range(self.warmstart_steps):
                obs_next, *_ = self.env.step(full_action0)
                cur_xy = np.asarray(obs_next, dtype=np.float32).copy()

        des_xy_exec = cur_xy.copy()
        return cur_xy, des_xy_exec

    def _update_ref_t(self, xy_for_alignment: np.ndarray):
        """
        Update ref index self.ref_t.
        Use SETPOINT (des_xy_exec / target_xy) for alignment, not TCP.

        nearest mode uses a BI-DIRECTIONAL window around current ref_t,
        so it can both advance and retreat (prevents phase lock/drift).
        """
        if self.ref_update_mode == "inc":
            self.ref_t = min(int(self.ref_t) + 1, self._xy_ref_len - 1)
            return

        if self.ref_update_mode == "nearest":
            W = int(self.ref_nearest_window)
            t0 = max(int(self.ref_t) - W // 2, 0)
            t1 = min(int(self.ref_t) + W // 2, self._xy_ref_len)
            seg = self._xy_ref_np[t0:t1]  # [M,2]
            d2 = np.sum((seg - xy_for_alignment[None, :]) ** 2, axis=1)
            self.ref_t = t0 + int(np.argmin(d2))
            return

        raise ValueError(f"Unknown ref_update_mode: {self.ref_update_mode}")

    def step_once(self, des_xy_exec: np.ndarray, cur_xy: np.ndarray):
        """
        One MPC step:
        - Align ref_t to CURRENT setpoint (des_xy_exec) BEFORE sampling
        - Build obs_4d = [des_xy_exec, cur_xy]
        - Sample delta from diffusion (with guidance)
        - Execute env.step(target_xy)
        """

        # (A) Align ref_t BEFORE sampling/guidance (critical)
        if self.ref_update_mode in ("nearest", "inc"):
            self._update_ref_t(des_xy_exec)   # use setpoint for alignment

        # d3il observation
        obs_4d = np.concatenate([des_xy_exec, cur_xy]).astype(np.float32)

        # guidance kwargs uses aligned self.ref_t
        extra_args = self._build_extra_args(des_xy_exec)

        # pred_delta = self.agent.predict(obs_4d, extra_args=extra_args)
        # pred_delta = np.asarray(pred_delta, dtype=np.float32)
        # if pred_delta.ndim == 2:
        #     pred_delta = pred_delta[0]

        # target_xy = des_xy_exec + pred_delta

        # z_use = getattr(self, "z_fixed", None)
        # if z_use is None:
        #     _, z_use = self._get_cur_xy_z()

        # full_action = np.concatenate([target_xy, [float(z_use)], self.fixed_quat], axis=0).astype(np.float32)
        # print("action stats:", np.min(full_action), np.max(full_action), np.any(np.isnan(full_action)), np.any(np.isinf(full_action)))
        # obs_next, reward, done, info = self.env.step(full_action)
        pred_delta = self.agent.predict(obs_4d, extra_args=extra_args)
        pred_delta = np.asarray(pred_delta, dtype=np.float32)

        if not np.isfinite(pred_delta).all():
            print("[ERR] pred_delta has NaN/Inf:", pred_delta)
            print("[ERR] obs_4d:", obs_4d)
            pred_delta = np.zeros((2,), dtype=np.float32)

        if pred_delta.ndim == 2:
            pred_delta = pred_delta[0]
        elif pred_delta.ndim > 2:
            pred_delta = pred_delta.reshape(-1, 2)[0]

        if pred_delta.shape != (2,):
            raise ValueError(f"pred_delta shape should be (2,), got {pred_delta.shape}")

        pred_delta = np.clip(pred_delta, -0.05, 0.05)
        target_xy = des_xy_exec + pred_delta

        if not np.isfinite(target_xy).all():
            print("[ERR] target_xy NaN/Inf. des_xy_exec=", des_xy_exec, " pred_delta=", pred_delta)
            target_xy = des_xy_exec.copy()

        target_xy = target_xy.astype(np.float32).reshape(2,)
        target_xy[0] = np.clip(target_xy[0], 0.0, 0.9)
        target_xy[1] = np.clip(target_xy[1], -0.4, 0.6)

        z_use = getattr(self, "z_fixed", None)
        if z_use is None:
            _, z_use = self._get_cur_xy_z()

        full_action = np.concatenate(
            [target_xy, [float(z_use)], self.fixed_quat], axis=0
        ).astype(np.float32)
        
        if isinstance(pred_delta, np.ndarray):
            pred_delta_np = np.asarray(pred_delta, dtype=np.float32)
        else:
            pred_delta_np = np.asarray(pred_delta, dtype=np.float32)

        # 这里 target_xy 可以按你当前 step_once / planning 的定义传
        self._finalize_current_plan_log(
            pred_delta=pred_delta_np,
            target_xy=None,
        )

        if not np.isfinite(full_action).all():
            print("[ERR] full_action has NaN/Inf:", full_action)
            print("[ERR] target_xy:", target_xy)
            full_action = np.nan_to_num(full_action, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("action stats:", np.min(full_action), np.max(full_action), np.any(np.isnan(full_action)), np.any(np.isinf(full_action)))
        obs_next, r, done, info = self.env.step(full_action)

        obs_next = np.asarray(obs_next)
        if not np.isfinite(obs_next).all():
            print("[ERR] obs_next has NaN/Inf (sim unstable).")
            print("[ERR] full_action:", full_action)
            print("[ERR] target_xy:", target_xy)
            done = True

        cur_xy_next = np.asarray(obs_next, dtype=np.float32).copy()
        if cur_xy_next.shape[0] != 2:
            cur_xy_next, _ = self._get_cur_xy_z()

        r = 0.0 if r is None else float(r)
        des_xy_next = target_xy.copy()

        # debug print (ref_t here is already aligned for this step)
        if self.step_debug_count < self.debug_steps:
            ref_xy = self._xy_ref_np[int(self.ref_t)]
            print("des_xy_exec=", des_xy_exec, "ref_xy=", ref_xy, "||diff||=", np.linalg.norm(des_xy_exec-ref_xy))
            print(f"[step {self.step_debug_count}] ref_t={int(self.ref_t)} "
                f"target_xy={target_xy} ref_xy={ref_xy} err={target_xy - ref_xy}")
        self.step_debug_count += 1

        return cur_xy_next, r, bool(done), info, des_xy_next
    # ---------------- episode ----------------

    def run_episode(self, max_steps: int = 250, random_reset: bool = True) -> Dict[str, Any]:
        self._guidance_plan_logs = []
        self._guidance_step_logs = []
        self._current_plan_meta = None
        self._plan_counter = 0
        # reset debug counters
        self.step_debug_count = 0

        cur_xy, des_xy_exec = self._init_episode_state(random_reset=random_reset)

        # reset ref_t (optionally nearest alignment at start)
        self.ref_t = 0

        print("xy_ref[0]=", self._xy_ref_np[0])
        print("Initial TCP position (x,y):", cur_xy)
        print("cur_xy=", cur_xy, "xy_ref[0]=", np.array(self._xy_ref_np[0]))

        # print reference step directions for first 10 points
        if self._xy_ref_len >= 11:
            dref = self._xy_ref_np[1:11] - self._xy_ref_np[:10]
            print("ref dx/dy first 10:", dref)

        traj = [cur_xy.copy()]
        sp_traj = [des_xy_exec.copy()]
        ref_idx = [int(self.ref_t)]
        rewards = []
        last_info = None
        done = False

        for k in range(max_steps):
            cur_xy, r, done, info, des_xy_exec = self.step_once(des_xy_exec, cur_xy)

            traj.append(cur_xy.copy())
            sp_traj.append(des_xy_exec.copy())
            rewards.append(float(r))
            last_info = info

            # force replanning each env step (so guidance applies each step)
            self.agent.action_counter = self.agent.action_seq_size
            self.agent.curr_action_seq = None

            # update ref index
            # self._update_ref_t(cur_xy)
            # self._update_ref_t(des_xy_exec)  # 用 setpoint
            
            # d2 = np.sum((self._xy_ref_np - des_xy_exec[None, :])**2, axis=1)
            # self.ref_t = int(np.argmin(d2))
            # if self.step_debug_count < 10:
            #     print(f"[ref_global] ref_t={self.ref_t}")
            # ref_idx.append(int(self.ref_t))
            
            if self.step_debug_count < 60:
                d2 = np.sum((self._xy_ref_np - des_xy_exec[None, :])**2, axis=1)
                ref_global = int(np.argmin(d2))
                print(f"[ref_global_check] global={ref_global} current_ref_t={self.ref_t}")
            ref_idx.append(int(self.ref_t))

            if done:
                break

        traj = np.asarray(traj, dtype=np.float32)

        if last_info is not None and isinstance(last_info, (list, tuple)) and len(last_info) > 1:
            success = bool(last_info[1])
        else:
            success = bool(traj[-1, 1] >= FINISH_Y)

        return dict(
            traj_xy=traj,
            sp_xy=np.asarray(sp_traj, dtype=np.float32),
            ref_idx=np.asarray(ref_idx, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            done=bool(done),
            success=success,
            steps=int(len(traj) - 1),
            guidance_logs=self._guidance_plan_logs,
        )