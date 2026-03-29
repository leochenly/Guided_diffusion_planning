#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test 1) Differentiability of MJX rollout: d(cost)/d(delta_xy_seq) exists and finite.
Test 2) Alignment with d3il original env.step chain: same plan -> same tcp xy trajectory.

Usage:
  python test_mjx_rollout.py \
      --plan avoiding/plans/xy_plan_euler_substeps2.npy \
      --horizon 80 \
      --n_substeps 35 \
      --tol 1e-3
"""

import os
import argparse
import numpy as np

import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt

# run  python test_mjx_rollout.py   --plan avoiding/plans/xy_plan_euler_substeps2.npy   --n_substeps 35   --tol 1e-3
# -----------------------
# Helpers: from plot_d3il_plan.py (slightly simplified)
# -----------------------
def sync_cpu_state_to_mjx(scene):
    """Copy qpos/qvel/act/ctrl from CPU mujoco data to mjx_data."""
    if getattr(scene, "mjx_data", None) is None:
        return
    scene.mjx_data = scene.mjx_data.replace(
        qpos=jnp.asarray(scene.data.qpos),
        qvel=jnp.asarray(scene.data.qvel),
        act=jnp.asarray(scene.data.act) if scene.model.na > 0 else scene.mjx_data.act,
        ctrl=jnp.asarray(scene.data.ctrl) if scene.model.nu > 0 else scene.mjx_data.ctrl,
    )


def set_newton_solver_for_mjx(scene, iterations=1, ls_iterations=4):
    """Match your plot script behavior: set Newton solver then rebuild mjx model/data."""
    scene.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    scene.model.opt.iterations = int(iterations)
    scene.model.opt.ls_iterations = int(ls_iterations)

    mujoco.mj_forward(scene.model, scene.data)

    scene.mjx_model = mjx.put_model(scene.model)
    scene.mjx_data = mjx.make_data(scene.mjx_model)
    sync_cpu_state_to_mjx(scene)


# -----------------------
# MJX rollout (same as I gave you earlier, but embedded here for a self-contained test)
# -----------------------
def _resolve_tcp_site_id(scene, robot) -> int:
    tcp_name = robot.add_id2model_key("tcp")
    sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
    if sid == -1:
        raise RuntimeError(f"TCP site '{tcp_name}' not found in model.")
    return int(sid)


def _tcp_xy_from_forward(m: mjx.Model, d: mjx.Data, tcp_site_id: int) -> jnp.ndarray:
    d_f = mjx.forward(m, d)
    pos = d_f.site_xpos[tcp_site_id]
    return pos[:2]

def make_mjx_rollout_fn(scene, robot, n_substeps: int):
    ctrl = robot.cartesianPosQuatTrackingController

    # Ensure controller builds compiled JAX functions (creates _compiled_stateful)
    if getattr(ctrl, "_compiled_stateful", None) is None or not getattr(ctrl, "_mjx_cache_ready", False):
        ctrl._build(scene, robot)

    # Ensure mjx model/data exist
    if getattr(scene, "mjx_model", None) is None:
        scene.mjx_model = mjx.put_model(scene.model)
    if getattr(scene, "mjx_data", None) is None:
        scene.mjx_data = mjx.make_data(scene.mjx_model)
        sync_cpu_state_to_mjx(scene)

    m = scene.mjx_model

    # 推荐用 BODY xpos（你工程里 tcp 是 body，这样更一致）
    tcp_name = robot.add_id2model_key("tcp")
    tcp_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)
    if tcp_bid == -1:
        raise RuntimeError(f"TCP body '{tcp_name}' not found.")

    step_jit = jax.jit(lambda d: mjx.step(m, d))

    def tcp_xy(d):
        d_f = mjx.forward(m, d)
        return d_f.xpos[tcp_bid][:2]

    @jax.jit
    def rollout(d0, delta_xy_seq, z_fixed, quat_fixed):
        delta_xy_seq = jnp.asarray(delta_xy_seq, dtype=jnp.float32)
        z_fixed = jnp.asarray(z_fixed, dtype=jnp.float32).reshape(())
        quat_fixed = jnp.asarray(quat_fixed, dtype=jnp.float32)
        quat_fixed = quat_fixed / (jnp.linalg.norm(quat_fixed) + 1e-12)

        H = delta_xy_seq.shape[0]
        xy0 = tcp_xy(d0)

        # 目标累加：target_xy[t+1]=target_xy[t]+delta[t]
        def cum_step(carry_xy, dxy):
            nxt = carry_xy + dxy
            return nxt, nxt
        _, target_xy_tail = jax.lax.scan(cum_step, xy0, delta_xy_seq)
        target_xy_seq = jnp.vstack([xy0[None, :], target_xy_tail])  # (H+1,2)

        # controller state init（对齐 CPU controller 的 old_q nan / old_v 0）
        old_q0 = jnp.full((7,), jnp.nan, dtype=jnp.float32)
        old_v0 = jnp.zeros((7,), dtype=jnp.float32)
        state0 = (old_q0, old_v0)

        def macro_step(carry, target_xy_next):
            d, state = carry
            target_pos = jnp.array([target_xy_next[0], target_xy_next[1], z_fixed], dtype=jnp.float32)
            target_quat = quat_fixed

            def micro_step(mcarry, _):
                d_in, st_in = mcarry
                # 强制 float32（防止 state 漂成 float64，scan 会炸）
                st_in = (st_in[0].astype(jnp.float32), st_in[1].astype(jnp.float32))

                d_ctrl, st_out = ctrl._compiled_stateful(m, d_in, target_pos, target_quat, st_in)
                st_out = (st_out[0].astype(jnp.float32), st_out[1].astype(jnp.float32))

                d_out = step_jit(d_ctrl)
                return (d_out, st_out), None

            (dN, stN), _ = jax.lax.scan(micro_step, (d, state), xs=None, length=n_substeps)
            xyN = tcp_xy(dN)
            return (dN, stN), xyN

        (dT, stT), xy_tail = jax.lax.scan(macro_step, (d0, state0), target_xy_seq[1:])
        tcp_xy_traj = jnp.vstack([xy0[None, :], xy_tail])
        return tcp_xy_traj, dT, stT

    return rollout

# def make_mjx_rollout_fn(scene, robot, n_substeps: int):
#     ctrl = robot.cartesianPosQuatTrackingController

#     # Ensure controller builds compiled JAX functions (creates _compiled)
#     if getattr(ctrl, "_compiled", None) is None or not getattr(ctrl, "_mjx_cache_ready", False):
#         ctrl._build(scene, robot)

#     # Ensure mjx model/data exist
#     if getattr(scene, "mjx_model", None) is None:
#         scene.mjx_model = mjx.put_model(scene.model)
#     if getattr(scene, "mjx_data", None) is None:
#         scene.mjx_data = mjx.make_data(scene.mjx_model)
#         sync_cpu_state_to_mjx(scene)

#     m = scene.mjx_model
#     tcp_site_id = _resolve_tcp_site_id(scene, robot)
#     step_jit = jax.jit(lambda d: mjx.step(m, d))

#     @jax.jit
#     def rollout(
#         d0: mjx.Data,
#         delta_xy_seq: jnp.ndarray,          # (H,2)
#         z_fixed: jnp.ndarray,               # scalar or shape (1,)
#         quat_fixed: jnp.ndarray,            # (4,) [w,x,y,z]
#     ):
#         delta_xy_seq = jnp.asarray(delta_xy_seq, dtype=jnp.float32)
#         xy0 = _tcp_xy_from_forward(m, d0, tcp_site_id)

#         def cum_step(carry_xy, dxy):
#             next_xy = carry_xy + dxy
#             return next_xy, next_xy

#         _, target_xy_tail = jax.lax.scan(cum_step, xy0, delta_xy_seq)
#         target_xy_seq = jnp.vstack([xy0[None, :], target_xy_tail])  # (H+1,2)

#         quat_fixed_n = jnp.asarray(quat_fixed, dtype=jnp.float32)
#         quat_fixed_n = quat_fixed_n / (jnp.linalg.norm(quat_fixed_n) + 1e-12)
#         z_fixed_s = jnp.asarray(z_fixed, dtype=jnp.float32).reshape(())

#         def macro_step(d, target_xy_next):
#             target_pos = jnp.array([target_xy_next[0], target_xy_next[1], z_fixed_s], dtype=jnp.float32)
#             target_quat = quat_fixed_n

#             def micro_step(d_in, _):
#                 # IMPORTANT: ctrl._compiled takes 4 args, stateless
#                 d_ctrl = ctrl._compiled(m, d_in, target_pos, target_quat)
#                 d_out = step_jit(d_ctrl)
#                 return d_out, None

#             dN, _ = jax.lax.scan(micro_step, d, xs=None, length=n_substeps)
#             xyN = _tcp_xy_from_forward(m, dN, tcp_site_id)
#             return dN, xyN

#         dT, xy_tail = jax.lax.scan(macro_step, d0, target_xy_seq[1:])
#         tcp_xy = jnp.vstack([xy0[None, :], xy_tail])  # (H+1,2)
#         return tcp_xy, dT

#     return rollout


# -----------------------
# Main tests
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, required=True, help="Absolute XY plan .npy (N,2)")
    parser.add_argument("--horizon", type=int, default=None, help="H steps to test; default uses plan length-1")
    parser.add_argument("--n_substeps", type=int, default=35, help="Macro-step substeps (match env config)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Alignment tolerance for max error")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- Load plan (absolute XY)
    if not os.path.exists(args.plan):
        raise FileNotFoundError(args.plan)
    target_xy_seq = np.load(args.plan).astype(np.float32)
    if target_xy_seq.ndim == 1:
        target_xy_seq = target_xy_seq.reshape(-1, 2)
    if target_xy_seq.ndim != 2 or target_xy_seq.shape[1] != 2:
        raise ValueError(f"plan must be (N,2), got {target_xy_seq.shape}")

    H_plan = target_xy_seq.shape[0] - 1
    H = H_plan if args.horizon is None else min(args.horizon, H_plan)
    target_xy_seq = target_xy_seq[: H + 1]  # keep (H+1,2)

    # ---- Build env (same as plot_d3il_plan.py)
    from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

    env = ObstacleAvoidanceEnv(render=args.render)
    env.start()
    env.reset()

    # Keep consistent with your plot script initial joints
    q_init = np.array([
        -0.35640868, 0.42944545, -0.1350106, -2.05465189,
        0.09357822, 2.48071794, 0.23250676
    ], dtype=np.float32)

    # Disable collisions like plot script (optional but recommended for strict matching)
    m = env.scene.model
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.jnt_limited[:] = 0

    # Prepare mjx artifacts on the same model (Newton solver as in your plot script)
    set_newton_solver_for_mjx(env.scene, iterations=1, ls_iterations=4)

    # Reset to deterministic initial state
    env.robot.beam_to_joint_pos(q_init)
    sync_cpu_state_to_mjx(env.scene)
    for _ in range(10):
        env.scene.next_step()
    env.robot.receiveState()
    print("Initial TCP position (x,y,z):", env.robot.current_c_pos[:3])
    # Read initial TCP z + quat convention (your plot script uses quat=[0,1,0,0]) :contentReference[oaicite:1]{index=1}
    tcp0 = env.robot.current_c_pos.copy()
    z_fixed = float(tcp0[2])
    quat_fixed = np.array([0, 1, 0, 0], dtype=np.float32)

    # Build delta_xy from absolute plan: delta[t] = target[t+1]-target[t]
    delta_xy_seq = (target_xy_seq[1:] - target_xy_seq[:-1]).astype(np.float32)  # (H,2)



    # =========================================================
    # TEST (1): Differentiability of MJX rollout
    # =========================================================
    rollout_fn = make_mjx_rollout_fn(env.scene, env.robot, n_substeps=args.n_substeps)

    # Sync mjx initial data to the same CPU init
    env.robot.beam_to_joint_pos(q_init)
    sync_cpu_state_to_mjx(env.scene)
    for _ in range(10):
        env.scene.next_step()
    mujoco.mj_forward(env.scene.model, env.scene.data)
    env.robot.receiveState()
    print("Initial TCP position (x,y,z):", env.robot.current_c_pos[:3])
    sync_cpu_state_to_mjx(env.scene)

    d0 = env.scene.mjx_data

    delta0 = jnp.asarray(delta_xy_seq)              # (H,2)
    z0 = jnp.asarray(z_fixed, dtype=jnp.float32)
    q0 = jnp.asarray(quat_fixed, dtype=jnp.float32)

    # Define a simple cost: final position to final target + small smoothness regularizer
    goal_xy = jnp.asarray(target_xy_seq[-1], dtype=jnp.float32)

    def cost_fn(dxy):
        # tcp_xy, _dT, _stT = rollout_fn(d0, dxy, z0, q0, ctrl_state0=None)
        tcp_xy, _dT, _stT = rollout_fn(d0, dxy, z0, q0)
        # tcp_xy, _dT = rollout_fn(d0, dxy, z0, q0)
        final_err = jnp.sum((tcp_xy[-1] - goal_xy) ** 2)
        smooth = 1e-2 * jnp.sum((dxy[1:] - dxy[:-1]) ** 2)
        return final_err + smooth

    grad_fn = jax.jit(jax.grad(cost_fn))
    g = grad_fn(delta0)  # (H,2)

    g_np = np.array(g)
    print("\n[TEST 1] Differentiability")
    print("  grad shape:", g_np.shape)
    print("  grad finite:", np.isfinite(g_np).all())
    print("  grad norm:", float(np.linalg.norm(g_np)))
    if not np.isfinite(g_np).all():
        raise RuntimeError("Gradient contains NaN/Inf -> rollout not properly differentiable.")
    if float(np.linalg.norm(g_np)) == 0.0:
        print("  [warn] grad norm is 0. This *can* happen if cost is flat / controller saturates, "
              "but usually indicates something is wrong. Try different plan or cost.")

    # =========================================================
    # MJX rollout trajectory (for alignment comparison)
    # =========================================================
    # tcp_xy_mjx, _dT, _stT = rollout_fn(d0, delta0, z0, q0, ctrl_state0=None)
    tcp_xy_mjx, _dT, _stT = rollout_fn(d0, delta0, z0, q0)
    # tcp_xy_mjx, _dT = rollout_fn(d0, delta0, z0, q0)
    traj_mjx = np.array(tcp_xy_mjx)  # (H+1,2)
    
        # =========================================================
    # TEST (2): Alignment with d3il env.step chain (CPU mujoco)
    # =========================================================
    # Rollout using env.step on CPU chain with absolute targets:
    # des_pose = [x, y, z_fixed, quat_fixed]
    env.robot.beam_to_joint_pos(q_init)
    sync_cpu_state_to_mjx(env.scene)
    for _ in range(10):
        env.scene.next_step()
    env.robot.receiveState()
    print("CPU mujoco initial TCP position (x,y,z):", env.robot.current_c_pos[:3])
    
    # env.scene.use_mjx = False
    # env.scene.use_mjx_runtime_step = False

    traj_cpu = [env.robot.current_c_pos[:2].copy()]
    for k in range(H):
        cmd_xy = target_xy_seq[k + 1]
        des_pos = np.array([cmd_xy[0], cmd_xy[1], z_fixed], dtype=np.float32)
        des_pose = np.concatenate([des_pos, quat_fixed], axis=0)
        _obs, _reward, _done, _info = env.step(des_pose)
        # make sure cpu forward updates site_xpos, etc
        mujoco.mj_forward(env.scene.model, env.scene.data)
        env.robot.receiveState()
        traj_cpu.append(env.robot.current_c_pos[:2].copy())
    traj_cpu = np.asarray(traj_cpu, dtype=np.float32)  # (H+1,2)

    # =========================================================
    # Compare trajectories
    # =========================================================
    err = np.linalg.norm(traj_mjx - traj_cpu, axis=1)  # per-step error
    print("\n[TEST 2] Alignment with d3il env.step (CPU mujoco)")
    print("  traj length:", traj_cpu.shape[0])
    print("  max |xy_mjx-xy_cpu|:", float(err.max()))
    print("  mean |xy_mjx-xy_cpu|:", float(err.mean()))
    print("  final |xy_mjx-xy_cpu|:", float(err[-1]))
    
    # -----------------------
    # Save + Plot trajectories
    # -----------------------
    out_dir = os.path.join(os.getcwd(), "mjx_rollout_debug")
    os.makedirs(out_dir, exist_ok=True)

    traj_oop = traj_cpu  
    traj_rollout = traj_mjx

    np.save(os.path.join(out_dir, "traj_oop.npy"), traj_oop)
    np.save(os.path.join(out_dir, "traj_rollout.npy"), traj_rollout)

    # Per-step Euclidean error
    err = np.linalg.norm(traj_rollout - traj_oop, axis=1)

    # 1) XY trajectory overlay
    plt.figure()
    plt.plot(traj_oop[:, 0], traj_oop[:, 1], marker="o", markersize=2, linewidth=1, label="OOP env.step (baseline)")
    plt.plot(traj_rollout[:, 0], traj_rollout[:, 1], marker="x", markersize=2, linewidth=1, label="MJX rollout (scan)")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory overlay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traj_overlay.png"), dpi=200)
    plt.close()

    # 2) Error curve
    plt.figure()
    plt.plot(err, linewidth=1)
    plt.xlabel("t")
    plt.ylabel("||xy_rollout - xy_baseline||")
    plt.title("Per-step XY error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traj_error.png"), dpi=200)
    plt.close()

    # 3) Optional: save a quick text summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"T = {traj_oop.shape[0]}\n")
        f.write(f"max_err = {err.max()}\n")
        f.write(f"mean_err = {err.mean()}\n")
        f.write(f"final_err = {err[-1]}\n")

    print(f"[Saved] npy/png to: {out_dir}")

    if float(err.max()) > args.tol:
        print(f"  [FAIL] max error {float(err.max())} > tol {args.tol}")
        print("  Tips:")
        print("   - Ensure n_substeps matches env internal stepping.")
        print("   - Ensure collisions are disabled in both paths (as in plot_d3il_plan).")
        print("   - Ensure same solver/iterations, same initial state, and sync_cpu_state_to_mjx is called.")
        raise SystemExit(2)

    print(f"  [PASS] Alignment within tol={args.tol}")

    env.close()


if __name__ == "__main__":
    main()