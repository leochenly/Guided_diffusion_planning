#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

# ---- hard force CPU to avoid JAX CUDA plugin / cuSPARSE issues ----
os.environ.setdefault("JAX_DISABLE_PJRT_PLUGINS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# if no display, use non-interactive backend
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import wandb
wandb.init(mode="disabled")

import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

import xml.etree.ElementTree as ET

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env

from agents.guidance.mjx_delta_xy_guidance import MJXDeltaXYGuidance, TorchAffineScalerAdapter
from panda_kinematics import compute_franka_fk, compute_franka_ik


# ====== Franka joint limits ======
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

def fk_pos(q: jp.ndarray) -> jp.ndarray:
    return compute_franka_fk(q)[:3, 3]

def fk_rot(q: jp.ndarray) -> jp.ndarray:
    return compute_franka_fk(q)[:3, :3]

def make_pose(R: jp.ndarray, p: jp.ndarray) -> jp.ndarray:
    T = jp.eye(4, dtype=jp.float32)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(p)
    return T


def _register_resolvers():
    def add_resolver(*args):
        vals = []
        for a in args:
            try:
                vals.append(float(a))
            except Exception:
                vals.append(a)
        if all(isinstance(v, (int, float)) for v in vals):
            s = sum(vals)
            if abs(s - int(s)) < 1e-9:
                return int(s)
            return s
        return vals
    try:
        OmegaConf.register_new_resolver("add", add_resolver)
    except Exception:
        try:
            OmegaConf.register_resolver("add", add_resolver)
        except Exception:
            pass


def _maybe_set(cfg, key: str, value):
    try:
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, value, merge=False)
    except Exception:
        pass


def extract_obstacles_from_xml_model(m: mujoco.MjModel, d: mujoco.MjData,
                                    name_keywords=("obs", "barrier", "wall"),
                                    box_samples=9):
    """
    Extract obstacles from MuJoCo geoms whose BODY name contains keywords.
    Supports:
      - cylinder: one circle
      - capsule: one circle (radius = size[0])
      - box: approximated by multiple circles along its long axis in XY plane
    Returns: list of {"center_xy": np.array(2,), "radius": float}
    """
    obstacles = []

    def body_name_for_geom(gi: int):
        bid = int(m.geom_bodyid[gi])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        return bname

    def geom_world_xy(gi: int):
        p = np.array(d.geom_xpos[gi], dtype=np.float32)
        return p[:2]

    def geom_world_R(gi: int):
        R = np.array(d.geom_xmat[gi], dtype=np.float32).reshape(3, 3)
        return R

    for gi in range(m.ngeom):
        bname = body_name_for_geom(gi)
        if bname is None:
            continue
        if not any(k in bname for k in name_keywords):
            continue

        gtype = int(m.geom_type[gi])
        size = np.array(m.geom_size[gi], dtype=np.float32)

        # cylinder -> circle
        if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            center_xy = geom_world_xy(gi)
            radius = float(size[0])
            obstacles.append({"center_xy": center_xy, "radius": radius})
            continue

        # capsule -> circle (rough)
        if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            center_xy = geom_world_xy(gi)
            radius = float(size[0])
            obstacles.append({"center_xy": center_xy, "radius": radius})
            continue

        # box -> multiple circles along long axis in XY
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            hx, hy, _hz = float(size[0]), float(size[1]), float(size[2])

            center = np.array(d.geom_xpos[gi], dtype=np.float32)  # (3,)
            R = geom_world_R(gi)

            ax_x = R[:, 0]
            ax_y = R[:, 1]

            proj_x = np.linalg.norm(ax_x[:2])
            proj_y = np.linalg.norm(ax_y[:2])

            if hx * proj_x >= hy * proj_y:
                long_half = hx
                long_dir = ax_x
                thick_half = hy
            else:
                long_half = hy
                long_dir = ax_y
                thick_half = hx

            dir_xy = long_dir[:2].astype(np.float32)
            norm = float(np.linalg.norm(dir_xy))
            if norm < 1e-8:
                obstacles.append({"center_xy": center[:2].copy(), "radius": max(hx, hy)})
                continue
            dir_xy = dir_xy / norm

            radius = float(thick_half)

            n = int(max(2, box_samples))
            ts = np.linspace(-long_half, long_half, n, dtype=np.float32)
            for tt in ts:
                cxy = center[:2] + dir_xy * float(tt)
                obstacles.append({"center_xy": cxy.astype(np.float32), "radius": radius})
            continue

        # mesh etc. ignored
        continue

    obstacles = sorted(obstacles, key=lambda o: (float(o["center_xy"][0]), float(o["center_xy"][1]), float(o["radius"])))
    return obstacles


def build_mjx_world_with_eef_init(
    xml_path: str,
    eef_site_name: str = "eef_site",
    goal_site_name: str = "goal_site",
):
    """
    Load MuJoCo, read sites, IK init to eef_site, sync ctrl, convert to MJX.
    """
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    eef_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, eef_site_name)
    goal_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, goal_site_name)

    eef_xyz = np.array(d.site_xpos[eef_sid], dtype=np.float32)
    goal_xyz = np.array(d.site_xpos[goal_sid], dtype=np.float32)

    # ---- IK init ----
    q_guess = jp.array(np.array(d.qpos[:7], dtype=np.float32), dtype=jp.float32)
    R_fixed = fk_rot(q_guess)
    T_des = make_pose(R_fixed, jp.array(eef_xyz, dtype=jp.float32))

    q_init = compute_franka_ik(T_des, q_guess[6], q_guess)
    q_init = clip_q(q_init)

    d.qpos[:7] = np.array(q_init, dtype=np.float32)
    if m.nu >= 7:
        d.ctrl[:7] = d.qpos[:7]
    mujoco.mj_forward(m, d)

    eef_real_xyz = np.array(fk_pos(jp.array(d.qpos[:7], dtype=jp.float32)), dtype=np.float32)
    obstacles = extract_obstacles_from_xml_model(m, d)

    mjx_model = mjx.put_model(m)
    data0 = mjx.make_data(mjx_model).replace(
        qpos=jp.array(d.qpos, dtype=jp.float32),
        qvel=jp.array(d.qvel, dtype=jp.float32),
        ctrl=jp.array(d.ctrl, dtype=jp.float32) if m.nu > 0 else jp.zeros((m.nu,), dtype=jp.float32),
    )
    ctrl_lower = jp.array(m.actuator_ctrlrange[:, 0], dtype=jp.float32)
    ctrl_upper = jp.array(m.actuator_ctrlrange[:, 1], dtype=jp.float32)

    meta = dict(
        eef_site_xyz=eef_xyz,
        goal_site_xyz=goal_xyz,
        eef_real_xyz_after_ik=eef_real_xyz,
        obstacles=obstacles,
    )
    print("obstacles extracted:", len(obstacles), flush=True)
    for o in obstacles[:20]:
        print(o, flush=True)
    if len(obstacles) > 20:
        print(f"... ({len(obstacles)-20} more)", flush=True)

    return mjx_model, data0, ctrl_lower, ctrl_upper, meta


def make_single_step_executor_safe(mjx_model, ctrl_lower, ctrl_upper, sim_dt: float, ctrl_dt: float):
    n_substeps = int(round(ctrl_dt / sim_dt))
    n_substeps = max(1, n_substeps)

    def _step(data: mjx.Data, des_xy: jp.ndarray):
        q_prev = data.qpos[:7]
        ee_prev = fk_pos(q_prev)

        R_fixed = fk_rot(q_prev)
        z_fixed = ee_prev[2]
        T_des = make_pose(R_fixed, jp.array([des_xy[0], des_xy[1], z_fixed], dtype=jp.float32))

        q_target = compute_franka_ik(T_des, q_prev[6], q_prev)
        q_target = clip_q(q_target)

        ok_ik = jp.all(jp.isfinite(q_target))
        q_target = jp.where(ok_ik, q_target, q_prev)
        q_target = jax.lax.stop_gradient(q_target)

        ctrl = data.ctrl.at[:7].set(q_target)
        ctrl = jp.clip(ctrl, ctrl_lower, ctrl_upper)

        data_next = mjx_env.step(mjx_model, data, ctrl, n_substeps)

        ok_state = jp.all(jp.isfinite(data_next.qpos)) & jp.all(jp.isfinite(data_next.qvel))
        data_next = jax.lax.cond(ok_state, lambda _: data_next, lambda _: data, operand=None)

        ee_next = fk_pos(data_next.qpos[:7])
        ee_next = jp.where(jp.all(jp.isfinite(ee_next)), ee_next, ee_prev)
        return data_next, ee_next

    return jax.jit(_step)


def build_obs(cfg, des_xy_np: np.ndarray, cur_xy_np: np.ndarray):
    """
    Avoiding_Dataset convention: obs = [des_xy(2), cur_xy(2), zeros...]
    """
    obs = np.zeros((cfg.obs_dim,), dtype=np.float32)
    obs[0:2] = des_xy_np.astype(np.float32)
    obs[2:4] = cur_xy_np.astype(np.float32)
    return obs


def load_xy_plan(path: str, stride: int = 1, offset: int = 0, align_to_start_xy=None):
    plan = np.load(path).astype(np.float32)
    if plan.ndim != 2:
        raise ValueError(f"xy_plan must be 2D, got shape={plan.shape}")
    if plan.shape[1] >= 3:
        plan = plan[:, :2]
    plan = plan[offset::stride]
    if align_to_start_xy is not None:
        align_to_start_xy = np.asarray(align_to_start_xy, dtype=np.float32)
        plan = plan + (align_to_start_xy - plan[0])
    return plan


def append_tail_to_goal(xy_plan: np.ndarray, goal_xy: np.ndarray, n_tail: int):
    if n_tail <= 0:
        return xy_plan
    p0 = xy_plan[-1]
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    tail = np.linspace(p0, goal_xy, n_tail + 1, dtype=np.float32)[1:]
    return np.concatenate([xy_plan, tail], axis=0)


# ----------------- DEBUG LOSSES (fix NaN prints) -----------------
def compute_debug_losses(cur_xy, cmd_xy, des_base, goal_xy, obstacles, margin=0.0):
    """
    Stable debug losses (does not depend on MJXDeltaXYGuidance internal attribute names).
    """
    cur_xy = np.asarray(cur_xy, np.float32)
    cmd_xy = np.asarray(cmd_xy, np.float32)
    des_base = np.asarray(des_base, np.float32)
    goal_xy = np.asarray(goal_xy, np.float32)

    track_plan = float(np.linalg.norm(cur_xy - des_base))
    cmd_error = float(np.linalg.norm(cur_xy - cmd_xy))
    goal_dist = float(np.linalg.norm(cur_xy - goal_xy))

    obs_pen = 0.0
    for o in obstacles:
        c = np.asarray(o["center_xy"], np.float32)
        r = float(o["radius"])
        d = float(np.linalg.norm(cur_xy - c))
        viol = (r + margin) - d
        if viol > 0:
            obs_pen += viol * viol

    return dict(track_plan=track_plan, cmd_error=cmd_error, goal_dist=goal_dist, obs_pen=obs_pen)


# ----------------- XML: add barrier (as many tiny cylinders) -----------------
def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _find_worldbody(root: ET.Element) -> ET.Element:
    for child in root.iter():
        if _strip_ns(child.tag) == "worldbody":
            return child
    raise RuntimeError("Cannot find <worldbody> in xml.")

def add_barrier_cylinders_to_xml(
    base_xml_path: str,
    out_xml_path: str,
    p0_xy: np.ndarray,
    p1_xy: np.ndarray,
    n_cyl: int = 15,
    radius: float = 0.010,
    halfheight: float = 0.12,
    z: float = 0.12,
    name_prefix: str = "barrier_obs",
    rgba: str = "0.2 0.2 0.2 1",
):
    """
    Add a 'wall' approximated by n_cyl vertical cylinders.
    IMPORTANT: body name contains 'obs' so obstacle extraction will include it.
    """
    p0_xy = np.asarray(p0_xy, dtype=np.float32).reshape(2,)
    p1_xy = np.asarray(p1_xy, dtype=np.float32).reshape(2,)
    if n_cyl <= 0:
        raise ValueError("n_cyl must be > 0")

    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = _find_worldbody(root)

    for i in range(n_cyl):
        a = (i + 0.5) / float(n_cyl)
        xy = (1.0 - a) * p0_xy + a * p1_xy
        b = ET.Element("body", attrib={
            "name": f"{name_prefix}_{i:02d}",
            "pos": f"{float(xy[0]):.6f} {float(xy[1]):.6f} {float(z):.6f}",
        })
        g = ET.Element("geom", attrib={
            "type": "cylinder",
            "size": f"{float(radius):.6f} {float(halfheight):.6f}",
            "rgba": rgba,
            "contype": "0",
            "conaffinity": "0",
        })
        b.append(g)
        worldbody.append(b)

    tree.write(out_xml_path)
    return out_xml_path


# ----------------- Guidance wrapper (print guidance_call 1/2/3...) -----------------
class GuidancePrintWrapper:
    """
    Wrap guidance_fn to:
      - print [traj k][step t] guidance_call j
      - accumulate timing of guidance calls
    """
    def __init__(self, base_guide):
        self.base = base_guide
        self.reset()
        self._traj_idx = None
        self._step_idx = None

    def reset(self):
        self.call_count = 0
        self.time_sum = 0.0

    def set_step_tag(self, traj_idx: int, step_idx: int):
        self._traj_idx = traj_idx
        self._step_idx = step_idx
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def set_context(self, *args, **kwargs):
        return self.base.set_context(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self._traj_idx is not None and self._step_idx is not None:
            print(f"[traj {self._traj_idx}] [step {self._step_idx}] guidance_call {self.call_count}", flush=True)

        t0 = time.perf_counter()
        out = self.base(*args, **kwargs)
        self.time_sum += (time.perf_counter() - t0)
        return out


@torch.no_grad()
def rollout_cloud_guided_with_barrier(
    cfg,
    agent,
    mjx_model,
    data0,
    ctrl_lower,
    ctrl_upper,
    guide_wrapped: GuidancePrintWrapper,
    xy_plan: np.ndarray,
    n_traj=80,
    T_steps=200,
    sim_dt=0.0025,
    ctrl_dt=0.08,
    seed0=0,
    # guidance
    guidance_scale=0.2,
    guidance_start=0.2,
    guidance_every=1,
    grad_clip_norm=1.0,
    grad_normalize=False,
    # safety
    dxy_clip=0.05,
    # termination
    goal_xy=None,
    goal_eps=0.03,
    # print controls
    print_every_step=1,
):
    executor = make_single_step_executor_safe(mjx_model, ctrl_lower, ctrl_upper, sim_dt, ctrl_dt)

    paths = []
    reached_goal = 0

    n_timesteps = int(getattr(cfg.agents.model, "n_timesteps", 0) or getattr(cfg.agents, "n_timesteps", 0) or 0)
    n_substeps = int(round(ctrl_dt / sim_dt))

    print("\n===== ITERATION COUNTS (expected) =====", flush=True)
    print(f"Diffusion inner loop (per agent.predict): n_timesteps = {n_timesteps}", flush=True)
    print(f"Outer loops: n_traj={n_traj}, T_steps(max)={T_steps}, window_size={int(cfg.window_size)}", flush=True)
    print(f"MJX inner loop per control step: n_substeps = round(ctrl_dt/sim_dt) = round({ctrl_dt}/{sim_dt}) = {n_substeps}\n", flush=True)

    # timing totals
    t_rollout0 = time.perf_counter()
    t_setctx_sum = 0.0
    t_pred_sum = 0.0
    t_mjx_sum = 0.0
    predict_calls_total = 0
    mjx_steps_total = 0

    for k in range(n_traj):
        print(f"\n========== Trajectory {k+1}/{n_traj} ==========", flush=True)
        torch.manual_seed(seed0 + k)
        np.random.seed(seed0 + k)

        agent.reset()
        data = data0

        ee0 = np.array(fk_pos(data.qpos[:7]), dtype=np.float32)
        cur_xy = ee0[:2].copy()
        xy_path = [cur_xy.copy()]

        for t in range(T_steps):
            step_id = t + 1
            if print_every_step == 1 or (step_id % print_every_step) == 0:
                print(f"  -- Step {step_id}/{T_steps}", flush=True)

            # policy "desired" from plan
            des_base = xy_plan[min(t, len(xy_plan) - 1)].copy()

            obs = build_obs(cfg, des_base, cur_xy)

            # set guide context (NO xy_ref)
            guide_wrapped.set_step_tag(k+1, step_id)

            t0 = time.perf_counter()
            guide_wrapped.set_context(
                data0=data,
                current_xy=des_base.astype(np.float32),
                xy_ref=None,
            )
            t_setctx_sum += (time.perf_counter() - t0)

            extra_args = dict(
                guidance_fn=guide_wrapped,
                guidance_scale=float(guidance_scale),
                guidance_start=float(guidance_start),
                guidance_every=int(guidance_every),
                grad_clip_norm=float(grad_clip_norm),
                grad_normalize=bool(grad_normalize),
            )

            # agent.predict
            t1 = time.perf_counter()
            dxy = agent.predict(obs, extra_args=extra_args)
            t_pred = time.perf_counter() - t1

            predict_calls_total += 1
            t_pred_sum += t_pred

            dxy = np.asarray(dxy, dtype=np.float32).reshape(-1)[:2]
            if not np.isfinite(dxy).all():
                dxy = np.zeros((2,), dtype=np.float32)
            dxy = np.clip(dxy, -dxy_clip, dxy_clip)

            cmd_xy = des_base + dxy

            # MJX step
            t2 = time.perf_counter()
            data, ee_next = executor(data, jp.array(cmd_xy, dtype=jp.float32))
            t_mjx = time.perf_counter() - t2

            mjx_steps_total += 1
            t_mjx_sum += t_mjx

            ee_next = np.array(ee_next, dtype=np.float32)
            cur_xy = ee_next[:2]
            xy_path.append(cur_xy.copy())

            # ---------- FIX: print real losses (no NaN) ----------
            goal_xy_for_loss = des_base if goal_xy is None else goal_xy
            obstacles_for_loss = getattr(guide_wrapped.base, "obstacles", [])

            losses = compute_debug_losses(
                cur_xy=cur_xy,
                cmd_xy=cmd_xy,
                des_base=des_base,
                goal_xy=goal_xy_for_loss,
                obstacles=obstacles_for_loss,
                margin=0.0,
            )

            print(f"     des_base={des_base}, dxy={dxy}, cmd_xy={cmd_xy}, cur_xy={cur_xy}", flush=True)
            print(
                f"     loss(track_plan)={losses['track_plan']:.4f}  "
                f"loss(cmd_error)={losses['cmd_error']:.4f}  "
                f"loss(obs_pen)={losses['obs_pen']:.6f}  "
                f"goal_dist={losses['goal_dist']:.4f}",
                flush=True
            )
            print(
                f"     time: predict={t_pred*1000:.2f}ms  mjx_step={t_mjx*1000:.2f}ms  "
                f"guidance_calls={guide_wrapped.call_count} (sum {guide_wrapped.time_sum*1000:.2f}ms)",
                flush=True
            )

            # termination
            if goal_xy is not None and float(np.linalg.norm(cur_xy - goal_xy)) < goal_eps:
                reached_goal += 1
                print(f"  Reached goal at step {step_id} (eps={goal_eps})", flush=True)
                break

        paths.append(np.stack(xy_path, axis=0))

    total_rollout_time = time.perf_counter() - t_rollout0

    max_len = max(p.shape[0] for p in paths)
    out = np.full((n_traj, max_len, 2), np.nan, dtype=np.float32)
    for i, p in enumerate(paths):
        out[i, :p.shape[0]] = p

    # summary
    len_g = [int(np.isfinite(out[i, :, 0]).sum()) for i in range(n_traj)]
    std_g = np.nanmean(np.nanstd(out, axis=0), axis=0)

    print("\nlen_g min/mean/max:", min(len_g), float(np.mean(len_g)), max(len_g), flush=True)
    print("mean std guided (x,y):", std_g, flush=True)

    print("\n===== TIMING BREAKDOWN =====", flush=True)
    print(f"Total rollout time: {total_rollout_time:.3f} s", flush=True)
    print(f"  set_context total: {t_setctx_sum:.3f} s  | avg per call: {(t_setctx_sum/max(1,predict_calls_total))*1000:.3f} ms", flush=True)
    print(f"  agent.predict total: {t_pred_sum:.3f} s | avg per call: {(t_pred_sum/max(1,predict_calls_total))*1000:.3f} ms", flush=True)
    print(f"  MJX executor total: {t_mjx_sum:.3f} s    | avg per step: {(t_mjx_sum/max(1,mjx_steps_total))*1000:.3f} ms", flush=True)

    print("\n===== ACTUAL ITERATION COUNTS =====", flush=True)
    print(f"Trajectories: {n_traj}", flush=True)
    print(f"Executed control steps total: {sum(len_g)}  (<= n_traj*T_steps)", flush=True)
    print(f"MJX substeps per control step: {int(round(ctrl_dt/sim_dt))}", flush=True)
    print(f"Executed MJX substeps total: {sum(len_g) * int(round(ctrl_dt/sim_dt))}", flush=True)
    print(f"agent.predict calls total (printed): {predict_calls_total}", flush=True)
    print(f"Reached goal trajectories: {reached_goal} / {n_traj}", flush=True)

    return out


def plot_guided_cloud(trajs_g, obstacles, goal_xy, xy_plan, out_png):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.4))
    ang = np.linspace(0, 2*np.pi, 140)

    # obstacles
    for o in obstacles:
        c = np.array(o["center_xy"], dtype=np.float32)
        r = float(o["radius"])
        ax.plot(c[0] + r*np.cos(ang), c[1] + r*np.sin(ang), linewidth=1.2)
        ax.scatter(c[0], c[1], s=70)

    # plan (visual only)
    if xy_plan is not None:
        ax.plot(xy_plan[:, 0], xy_plan[:, 1], linestyle="--", linewidth=2.0)

    # trajectories
    for i in range(trajs_g.shape[0]):
        xy = trajs_g[i]
        valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        xy = xy[valid]
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], linewidth=1.0, alpha=0.22)

    # start marker
    start_xy = trajs_g[0, 0]
    if np.isfinite(start_xy).all():
        ax.scatter(start_xy[0], start_xy[1], s=90, marker="o")

    # goal marker
    if goal_xy is not None:
        ax.scatter(goal_xy[0], goal_xy[1], s=150, marker="*", color="gray")

    ax.set_title("GUIDED (barrier in obstacles + debug losses printed)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[Saved] {out_png}", flush=True)
    if os.environ.get("DISPLAY", "") != "":
        plt.show()


def safe_load_pretrained(agent, weights_dir: str, sv_name: str):
    """
    Fix: weights saved on CUDA but running on CPU-only machine.
    """
    weights_path = os.path.join(weights_dir, sv_name)
    if torch.cuda.is_available():
        agent.load_pretrained_model(weights_dir, sv_name=sv_name)
        return

    sd = torch.load(weights_path, map_location=torch.device("cpu"))
    agent.model.load_state_dict(sd)
    try:
        agent.model.to(torch.device("cpu"))
    except Exception:
        pass
    try:
        agent.device = torch.device("cpu")
    except Exception:
        pass


def parse_xy(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y' got: {s}")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)


def main():
    _register_resolvers()

    ap = argparse.ArgumentParser()

    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--sv_name", default="eval_best_ddpm.pth")

    ap.add_argument("--xml", default="avoiding/xmls/avoiding_mjx_nocollide.xml")
    ap.add_argument("--xyplan", required=True)
    ap.add_argument("--plan_stride", type=int, default=1)
    ap.add_argument("--plan_offset", type=int, default=0)
    ap.add_argument("--align_plan_to_start", action="store_true")
    ap.add_argument("--append_goal_steps", type=int, default=40)

    # rollout
    ap.add_argument("--n_traj", type=int, default=80)
    ap.add_argument("--T_steps", type=int, default=80)
    ap.add_argument("--goal_eps", type=float, default=0.03)
    ap.add_argument("--dxy_clip", type=float, default=0.05)

    # guidance
    ap.add_argument("--guidance_scale", type=float, default=0.2)
    ap.add_argument("--guidance_start", type=float, default=0.2)
    ap.add_argument("--guidance_every", type=int, default=1)
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--grad_normalize", action="store_true")

    # IMPORTANT: turn off ref-loss (still keep ref in policy obs)
    ap.add_argument("--w_ref_xy", type=float, default=0.0)

    # barrier settings
    ap.add_argument("--add_barrier", action="store_true", help="insert barrier (many small cylinders) into xml")
    ap.add_argument("--barrier_p0", type=str, default="0.52,-0.05", help="format 'x,y'")
    ap.add_argument("--barrier_p1", type=str, default="0.52,0.10", help="format 'x,y'")
    ap.add_argument("--barrier_n", type=int, default=18)
    ap.add_argument("--barrier_radius", type=float, default=0.010)
    ap.add_argument("--barrier_halfheight", type=float, default=0.12)
    ap.add_argument("--barrier_z", type=float, default=0.12)

    # output
    ap.add_argument("--out", default="guided_barrier_mjx.png")
    args = ap.parse_args()

    # ====== Hydra config ======
    with initialize(config_path="configs"):
        cfg = compose(
            config_name="avoiding_config",
            overrides=[
                "agents=ddpm_transformer_agent",
                "agent_name=ddpm_transformer",
                "window_size=5",
                "agents.model.n_timesteps=8",
            ],
        )

    # force cpu in config if keys exist
    _maybe_set(cfg, "device", "cpu")
    _maybe_set(cfg, "agents.device", "cpu")
    _maybe_set(cfg, "agents.model.device", "cpu")

    agent = hydra.utils.instantiate(cfg.agents)
    safe_load_pretrained(agent, args.weights_dir, args.sv_name)

    # If adding barrier: create modified xml file
    xml_to_use = args.xml
    if args.add_barrier:
        p0 = parse_xy(args.barrier_p0)
        p1 = parse_xy(args.barrier_p1)
        out_xml = os.path.splitext(args.xml)[0] + "_with_barrier.xml"
        xml_to_use = add_barrier_cylinders_to_xml(
            base_xml_path=args.xml,
            out_xml_path=out_xml,
            p0_xy=p0,
            p1_xy=p1,
            n_cyl=args.barrier_n,
            radius=args.barrier_radius,
            halfheight=args.barrier_halfheight,
            z=args.barrier_z,
        )
        print(f"[Barrier] inserted into xml: {xml_to_use}", flush=True)
        print(f"[Barrier] p0={p0}, p1={p1}, n={args.barrier_n}, r={args.barrier_radius}", flush=True)

    # ====== MJX world + IK init ======
    mjx_model, data0, ctrl_lower, ctrl_upper, meta = build_mjx_world_with_eef_init(xml_to_use)

    start_xy = meta["eef_real_xyz_after_ik"][:2].astype(np.float32)
    goal_xy = meta["goal_site_xyz"][:2].astype(np.float32)
    obstacles = meta["obstacles"]

    print("EEF site xyz:", meta["eef_site_xyz"], " | EEF real xyz:", meta["eef_real_xyz_after_ik"], flush=True)
    print("Goal xyz:", meta["goal_site_xyz"], flush=True)
    print("Num obstacles:", len(obstacles), flush=True)
    print("Start XY:", start_xy, "| Goal XY:", goal_xy, flush=True)

    # ====== load xyplan (policy obs & plot) ======
    align_to = start_xy if args.align_plan_to_start else None
    xy_plan = load_xy_plan(args.xyplan, stride=args.plan_stride, offset=args.plan_offset, align_to_start_xy=align_to)
    xy_plan = append_tail_to_goal(xy_plan, goal_xy, args.append_goal_steps)
    print("xy_plan shape:", xy_plan.shape, "| first:", xy_plan[0], "| last:", xy_plan[-1], flush=True)

    # ====== Guidance module ======
    scaler_adapter = TorchAffineScalerAdapter(
        inverse_scale_output=agent.scaler.inverse_scale_output,
        action_dim=2,
        device=agent.device,
    )
    base_guide = MJXDeltaXYGuidance(
        mjx_model=mjx_model,
        ctrl_lower=ctrl_lower,
        ctrl_upper=ctrl_upper,
        sim_dt=0.0025,
        ctrl_dt=0.08,
        w_track_xyz=5.0,
        w_obs=5.0,                   # ✅ obstacle loss
        w_z_track=1.0,
        w_dxy_l2=1e-4,
        w_dxy_smooth=1e-4,
        w_ref_xy=float(args.w_ref_xy),  # ✅ 0 => no reference tracking in guidance loss
        scaler=scaler_adapter,
    )

    # ✅关键：把提取到的障碍（包含挡板）明确传入 guide，确保 w_obs 真的在算挡板
    base_guide.obstacles = obstacles
    print(f"[Guide] obstacles passed into guide: {len(base_guide.obstacles)}", flush=True)

    guide = GuidancePrintWrapper(base_guide)

    # ====== guided rollouts ======
    trajs_g = rollout_cloud_guided_with_barrier(
        cfg=cfg,
        agent=agent,
        mjx_model=mjx_model,
        data0=data0,
        ctrl_lower=ctrl_lower,
        ctrl_upper=ctrl_upper,
        guide_wrapped=guide,
        xy_plan=xy_plan,
        n_traj=args.n_traj,
        T_steps=args.T_steps,
        seed0=0,
        guidance_scale=args.guidance_scale,
        guidance_start=args.guidance_start,
        guidance_every=args.guidance_every,
        grad_clip_norm=args.grad_clip_norm,
        grad_normalize=args.grad_normalize,
        dxy_clip=args.dxy_clip,
        goal_xy=goal_xy,
        goal_eps=args.goal_eps,
        print_every_step=1,
    )

    # ====== plot guided only ======
    plot_guided_cloud(trajs_g, obstacles, goal_xy, xy_plan, out_png=args.out)


if __name__ == "__main__":
    main()
