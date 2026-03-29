from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def sync_cpu_state_to_mjx(scene) -> None:
    if getattr(scene, "mjx_data", None) is None:
        return
    scene.mjx_data = scene.mjx_data.replace(
        qpos=jnp.asarray(scene.data.qpos),
        qvel=jnp.asarray(scene.data.qvel),
        act=jnp.asarray(scene.data.act) if scene.model.na > 0 else scene.mjx_data.act,
        ctrl=jnp.asarray(scene.data.ctrl) if scene.model.nu > 0 else scene.mjx_data.ctrl,
    )


def set_newton_solver_for_mjx(scene, iterations: int = 1, ls_iterations: int = 4) -> None:
    scene.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    scene.model.opt.iterations = int(iterations)
    scene.model.opt.ls_iterations = int(ls_iterations)
    mujoco.mj_forward(scene.model, scene.data)
    scene.mjx_model = mjx.put_model(scene.model)
    scene.mjx_data = mjx.make_data(scene.mjx_model)
    sync_cpu_state_to_mjx(scene)


def print_solver_config(model, prefix: str = "[solver]") -> None:
    print(
        f"{prefix} solver={model.opt.solver} "
        f"iterations={int(model.opt.iterations)} "
        f"ls_iterations={int(model.opt.ls_iterations)}"
    )


def configure_scene_for_stable_mjx(scene) -> None:
    m = scene.model
    m.jnt_limited[:] = 0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    print_solver_config(scene.model, prefix="[before]")
    set_newton_solver_for_mjx(scene, iterations=1, ls_iterations=4)
    print_solver_config(scene.model, prefix="[after ]")


def make_mjx_rollout_fn(scene, robot, n_substeps: int):
    ctrl = robot.cartesianPosQuatTrackingController

    if getattr(ctrl, "_compiled_stateful", None) is None or not getattr(ctrl, "_mjx_cache_ready", False):
        ctrl._build(scene, robot)

    if getattr(scene, "mjx_model", None) is None:
        scene.mjx_model = mjx.put_model(scene.model)
    if getattr(scene, "mjx_data", None) is None:
        scene.mjx_data = mjx.make_data(scene.mjx_model)
        sync_cpu_state_to_mjx(scene)

    m = scene.mjx_model
    tcp_name = robot.add_id2model_key("tcp")
    tcp_bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, tcp_name)
    if tcp_bid == -1:
        raise RuntimeError(f"TCP body '{tcp_name}' not found.")

    step_jit = jax.jit(lambda d: mjx.step(m, d))

    def tcp_xy(d):
        d_f = mjx.forward(m, d)
        return d_f.xpos[tcp_bid][:2]

    @jax.jit
    def rollout(d0, delta_xy_seq, z_fixed, quat_fixed, xy_ref0):
        delta_xy_seq = jnp.asarray(delta_xy_seq, dtype=jnp.float32)
        z_fixed = jnp.asarray(z_fixed, dtype=jnp.float32).reshape(())
        quat_fixed = jnp.asarray(quat_fixed, dtype=jnp.float32)
        quat_fixed = quat_fixed / (jnp.linalg.norm(quat_fixed) + 1e-12)
        xy_ref0 = jnp.asarray(xy_ref0, dtype=jnp.float32).reshape((2,))

        xy0 = tcp_xy(d0)

        def cum_step(carry_xy, dxy):
            nxt = carry_xy + dxy
            return nxt, nxt

        _, target_xy_tail = jax.lax.scan(cum_step, xy_ref0, delta_xy_seq)
        target_xy_seq = jnp.vstack([xy_ref0[None, :], target_xy_tail])

        old_q0 = jnp.full((7,), jnp.nan, dtype=jnp.float32)
        old_v0 = jnp.zeros((7,), dtype=jnp.float32)
        state0 = (old_q0, old_v0)

        def macro_step(carry, target_xy_next):
            d, state = carry
            target_pos = jnp.array([target_xy_next[0], target_xy_next[1], z_fixed], dtype=jnp.float32)
            target_quat = quat_fixed

            def micro_step(mcarry, _):
                d_in, st_in = mcarry
                st_in = (st_in[0].astype(jnp.float32), st_in[1].astype(jnp.float32))
                d_ctrl, st_out = ctrl._compiled_stateful(m, d_in, target_pos, target_quat, st_in)
                st_out = (st_out[0].astype(jnp.float32), st_out[1].astype(jnp.float32))
                d_out = step_jit(d_ctrl)
                return (d_out, st_out), None

            (dN, stN), _ = jax.lax.scan(micro_step, (d, state), xs=None, length=n_substeps)
            xyN = tcp_xy(dN)
            return (dN, stN), xyN

        (_, _), xy_tail = jax.lax.scan(macro_step, (d0, state0), target_xy_seq[1:])
        return jnp.vstack([xy0[None, :], xy_tail])

    return rollout


@dataclass
class MjxRolloutSim:
    scene: Any
    robot: Any
    n_substeps: int = 35
    newton_iterations: int = 1
    newton_ls_iterations: int = 4

    def __post_init__(self):
        configure_scene_for_stable_mjx(self.scene)
        set_newton_solver_for_mjx(
            self.scene,
            iterations=self.newton_iterations,
            ls_iterations=self.newton_ls_iterations,
        )
        self.rollout_fn = make_mjx_rollout_fn(self.scene, self.robot, n_substeps=self.n_substeps)

    def sync_from_env(self) -> None:
        mujoco.mj_forward(self.scene.model, self.scene.data)
        sync_cpu_state_to_mjx(self.scene)

    def rollout_xy_traj(self, s0: Dict[str, Any], delta_xy_seq: jnp.ndarray) -> jnp.ndarray:
        return self.rollout_fn(
            s0["d0"],
            delta_xy_seq,
            s0["z_fixed"],
            s0["quat_fixed"],
            s0["xy_ref0"],
        )
