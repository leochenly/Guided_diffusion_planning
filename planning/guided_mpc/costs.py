from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

from .constants import CENTER_X


@dataclass
class AvoidingCostWeights:
    # reference tracking
    w_track: float = 0.0
    track_huber_delta: float = 0.01

    # point obstacles (reserved for future use)
    w_obs: float = 0.0
    obs_sigma: float = 0.03

    # regularization / task terms
    w_centerx: float = 1.0
    w_smooth: float = 0.0

    # corridor bounds (reserved for future use)
    w_bound: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.9
    y_min: float = -0.35
    y_max: float = 0.35

    # optional barrier region
    barrier_mode: str = "rect"  # "segment" or "rect"
    barrier_x0: float = 0.5
    barrier_x1: float = 0.5
    barrier_y0: float = -0.05
    barrier_y1: float = 0.2
    w_barrier: float = 0.0
    barrier_margin: float = 0.02
    barrier_temp: float = 0.01


def huber(x: jnp.ndarray, delta: float) -> jnp.ndarray:
    absx = jnp.abs(x)
    quad = jnp.minimum(absx, delta)
    lin = absx - quad
    return 0.5 * quad**2 + delta * lin


def barrier_segment_cost(
    xy: jnp.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    margin: float,
    temperature: float,
) -> jnp.ndarray:
    x = xy[..., 0]
    y = xy[..., 1]
    dx = jax.nn.relu(x0 - x) + jax.nn.relu(x - x1)
    dy = y - y0
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-12)
    z = (margin - dist) / (temperature + 1e-12)
    penalty = jax.nn.softplus(z)
    return jnp.mean(penalty)


def barrier_rect_cost(
    xy: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    margin: float,
    temperature: float,
) -> jnp.ndarray:
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


def _reference_window(xy_ref: jnp.ndarray, ref_t: jnp.ndarray, horizon: int) -> jnp.ndarray:
    pad = horizon
    last = xy_ref[-1:, :]
    xy_ref_pad = jnp.concatenate([xy_ref, jnp.repeat(last, pad, axis=0)], axis=0)
    idx_dtype = jnp.int32
    t0 = jnp.asarray(ref_t, dtype=idx_dtype)
    t0 = jnp.clip(t0, jnp.asarray(0, idx_dtype), jnp.asarray(xy_ref.shape[0] - 1, idx_dtype))
    return jax.lax.dynamic_slice(xy_ref_pad, (t0, jnp.asarray(0, idx_dtype)), (horizon, 2))


def cost_from_rollout(
    xy_traj: jnp.ndarray,
    dxy_seq: jnp.ndarray,
    weights: AvoidingCostWeights,
    obstacles_xy: jnp.ndarray,
    xy_ref: jnp.ndarray,
    xy_ref0: jnp.ndarray,
    ref_t: jnp.ndarray,
) -> jnp.ndarray:
    """Main scalar objective used by guided diffusion planning."""
    del obstacles_xy  # kept for API compatibility and future extension
    horizon = dxy_seq.shape[0]
    xy_sp = xy_ref0[None, :] + jnp.cumsum(dxy_seq, axis=0)
    xy_tgt = _reference_window(xy_ref, ref_t, horizon)

    x_roll = xy_traj[1:, 0]
    centerx_cost = jnp.mean((x_roll - jnp.asarray(CENTER_X, jnp.float32)) ** 2)
    smooth_cost = jnp.mean(jnp.sum(dxy_seq * dxy_seq, axis=-1))

    total = weights.w_centerx * centerx_cost + weights.w_smooth * smooth_cost

    if weights.w_track > 0.0:
        err = xy_sp - xy_tgt
        w = jnp.power(jnp.asarray(0.7, jnp.float32), jnp.arange(horizon, dtype=jnp.float32))
        w = w / jnp.sum(w)
        track_cost = jnp.sum(w * jnp.sum(huber(err, weights.track_huber_delta), axis=-1))
        total = total + weights.w_track * track_cost

    if weights.w_barrier > 0.0:
        if weights.barrier_mode == "segment":
            barrier = barrier_segment_cost(
                xy_traj[1:, :],
                x0=weights.barrier_x0,
                x1=weights.barrier_x1,
                y0=weights.barrier_y0,
                y1=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )
        else:
            barrier = barrier_rect_cost(
                xy_traj[1:, :],
                x_min=weights.barrier_x0,
                x_max=weights.barrier_x1,
                y_min=weights.barrier_y0,
                y_max=weights.barrier_y1,
                margin=weights.barrier_margin,
                temperature=weights.barrier_temp,
            )
        total = total + weights.w_barrier * barrier

    return total


def decompose_cost_from_rollout(
    xy_traj: jnp.ndarray,
    dxy_seq: jnp.ndarray,
    weights: AvoidingCostWeights,
    obstacles_xy: jnp.ndarray,
    xy_ref: jnp.ndarray,
    xy_ref0: jnp.ndarray,
    ref_t: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    del obstacles_xy
    horizon = dxy_seq.shape[0]
    xy_sp = xy_ref0[None, :] + jnp.cumsum(dxy_seq, axis=0)
    xy_tgt = _reference_window(xy_ref, ref_t, horizon)

    x_roll = xy_traj[1:, 0]
    centerx_cost = jnp.mean((x_roll - jnp.asarray(CENTER_X, jnp.float32)) ** 2)
    smooth_cost = jnp.mean(jnp.sum(dxy_seq * dxy_seq, axis=-1))

    track_cost = jnp.asarray(0.0, dtype=jnp.float32)
    if weights.w_track > 0.0:
        err = xy_sp - xy_tgt
        w = jnp.power(jnp.asarray(0.7, jnp.float32), jnp.arange(horizon, dtype=jnp.float32))
        w = w / jnp.sum(w)
        track_cost = jnp.sum(w * jnp.sum(huber(err, weights.track_huber_delta), axis=-1))

    barrier_cost = jnp.asarray(0.0, dtype=jnp.float32)
    if weights.w_barrier > 0.0:
        if weights.barrier_mode == "segment":
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

    return {
        "J_total": weighted_centerx + weighted_smooth + weighted_track + weighted_barrier,
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
