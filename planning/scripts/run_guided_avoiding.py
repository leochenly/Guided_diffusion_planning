"""Example experiment script.

This file is intentionally the *only* place that contains experiment-specific paths.
Library code under guided_mpc/ stays reusable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

wandb.init(mode="disabled")

# Keep your original project path dependency explicit in the experiment script only.
sys.path.append("/home/liangyuchen/project/d3il_4_diff/d3il")

from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

from planning.guided_mpc import (
    AvoidingCostWeights,
    GuidanceConfig,
    GuidedDiffusionPlannerD3IL,
    MjxRolloutSim,
    load_agent,
)
from planning.guided_mpc.plots import (
    build_flattened_episode_tensor,
    flatten_episode_all_denoising_logs,
    plot_cost_vs_denoising_step,
    plot_episode_all_denoising_cost,
    plot_episode_all_denoising_cost_mean_std,
    plot_episode_all_denoising_grad,
    plot_episode_all_denoising_grad_mean_std,
    plot_episode_cost,
    plot_episode_grad,
    plot_flattened_mean_std_with_boundaries,
    plot_grad_vs_denoising_step,
    plot_obstacles,
    plot_rollout_snapshots,
    summarize_episode_guidance_logs,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ddpm_exp_dir = "/home/liangyuchen/project/d3il_4_raw/d3il/logs/avoiding/sweeps/ddpm_encdec/2026-02-21/2500/agent_name=ddpm_encdec,agents.model.n_timesteps=16,agents=ddpm_encdec_agent,group=avoiding_ddpm_encdec_seeds,seed=0,simulation.n_cores=15,simulation.n_trajectories=480,window_size=8"
    checkpoint_name = "eval_best_ddpm.pth"
    xy_ref_path = Path("avoiding/plans/xy_plan_env_065_00.npy")

    agent = load_agent(ddpm_exp_dir, checkpoint_name, device=device)
    xy_ref = np.load(xy_ref_path).astype(np.float32)

    env = ObstacleAvoidanceEnv(render=False)
    env.start()
    sim = MjxRolloutSim(scene=env.scene, robot=env.robot, n_substeps=35, newton_iterations=1, newton_ls_iterations=4)

    weights = AvoidingCostWeights(
        w_centerx=10.0,
        w_smooth=0.0,
        w_bound=0.0,
        barrier_mode="rect",
        w_barrier=0.0,
    )
    guidance = GuidanceConfig(enabled=True, scale_alpha=50, start_ratio=1.0, interval=1, grad_clip=10.0, use_jit=True)

    planner = GuidedDiffusionPlannerD3IL(
        agent=agent,
        env=env,
        sim=sim,
        xy_ref=xy_ref,
        device=device,
        weights=weights,
        guidance=guidance,
        fixed_quat=np.array([0, 1, 0, 0], dtype=np.float32),
        settle_steps=10,
        warmstart_steps=3,
        ref_update_mode="nearest",
        ref_nearest_window=160,
    )

    n_ep = 10
    max_steps = 250
    tcp_trajs, sp_trajs, successes, all_results = [], [], [], []
    for ep in range(n_ep):
        print(f"\n=== Episode {ep + 1}/{n_ep} ===")
        result = planner.run_episode(max_steps=max_steps, random_reset=True)
        all_results.append(result)
        tcp_trajs.append(result["traj_xy"])
        sp_trajs.append(result["sp_xy"])
        successes.append(bool(result["success"]))

    succ_rate = float(np.mean(successes))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_obstacles(ax)
    for tcp, succ in zip(tcp_trajs, successes):
        ax.plot(tcp[:, 0], tcp[:, 1], color=("blue" if succ else "red"), alpha=0.20, linewidth=0.8)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.3, 0.5)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"MPC Guided Diffusion (succ={succ_rate:.2f})")
    plt.tight_layout()
    plt.savefig("guided_mpc_tcp_and_setpoint.png", dpi=160)
    plt.show()

    result = all_results[-1]
    if not result["guidance_logs"]:
        return

    plan_log = result["guidance_logs"][0]
    plot_cost_vs_denoising_step(plan_log, save_path="episode0_plan0_cost.png")
    plot_grad_vs_denoising_step(plan_log, save_path="episode0_plan0_grad.png")
    plot_rollout_snapshots(plan_log, save_path="episode0_plan0_snapshots.png")

    summary_last = summarize_episode_guidance_logs(result["guidance_logs"], pick="last")
    plot_episode_cost(summary_last, save_path="episode_all_steps_cost_last.png", title_suffix=" (pick=last)")
    plot_episode_grad(summary_last, save_path="episode_all_steps_grad_last.png", title_suffix=" (pick=last)")

    flat_all = flatten_episode_all_denoising_logs(result["guidance_logs"])
    plot_episode_all_denoising_cost(flat_all, save_path="episode_all_denoising_cost.png", title_suffix=" (all planning x all denoising)")
    plot_episode_all_denoising_grad(flat_all, save_path="episode_all_denoising_grad.png", title_suffix=" (all planning x all denoising)")

    J_tensor, _ = build_flattened_episode_tensor(all_results, value_key="J_total")
    grad_tensor, _ = build_flattened_episode_tensor(all_results, value_key="grad_norm")
    gradc_tensor, _ = build_flattened_episode_tensor(all_results, value_key="grad_norm_clipped")
    alpha_tensor, _ = build_flattened_episode_tensor(all_results, value_key="alpha_grad_norm")
    plot_flattened_mean_std_with_boundaries(
        J_tensor,
        reference_guidance_logs=all_results[0]["guidance_logs"],
        title="Mean ± std J_total over trajectories (flattened all steps × denoise)",
        ylabel="J_total",
        save_path="flattened_mean_std_J_total.png",
    )
    plot_flattened_mean_std_with_boundaries(
        grad_tensor,
        reference_guidance_logs=all_results[0]["guidance_logs"],
        title="Mean ± std grad_norm over trajectories (flattened all steps × denoise)",
        ylabel="grad_norm",
        save_path="flattened_mean_std_grad_norm.png",
    )

    center_tensor, _ = build_flattened_episode_tensor(all_results, value_key="centerx_cost")
    smooth_tensor, _ = build_flattened_episode_tensor(all_results, value_key="smooth_cost")
    track_tensor, _ = build_flattened_episode_tensor(all_results, value_key="track_cost")
    barrier_tensor, _ = build_flattened_episode_tensor(all_results, value_key="barrier_cost")
    plot_episode_all_denoising_cost_mean_std(
        J_tensor,
        center_tensor=center_tensor,
        smooth_tensor=smooth_tensor,
        track_tensor=track_tensor,
        barrier_tensor=barrier_tensor,
        reference_guidance_logs=all_results[0]["guidance_logs"],
        save_path="episode_all_denoising_cost_mean_std.png",
        title_suffix=" (flattened all planning × all denoising)",
    )
    plot_episode_all_denoising_grad_mean_std(
        grad_tensor,
        gradc_tensor=gradc_tensor,
        alpha_tensor=alpha_tensor,
        reference_guidance_logs=all_results[0]["guidance_logs"],
        save_path="episode_all_denoising_grad_mean_std.png",
        title_suffix=" (flattened all planning × all denoising)",
    )


if __name__ == "__main__":
    main()
