from __future__ import annotations

from typing import Dict, Iterable, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .constants import CENTER_X, DEFAULT_OBSTACLES_XY, FINISH_Y


def plot_obstacles(ax, obstacles_xy: np.ndarray = DEFAULT_OBSTACLES_XY, radius: float = 0.03) -> None:
    for i, (cx, cy) in enumerate(obstacles_xy):
        circ = patches.Circle((cx, cy), radius, edgecolor="red", facecolor="none", linewidth=1.5)
        ax.add_patch(circ)
        ax.scatter(cx, cy, s=20, c="red", alpha=0.5, label="obstacle" if i == 0 else None)
    half_w = 0.25
    ax.plot([CENTER_X - half_w, CENTER_X + half_w], [FINISH_Y, FINISH_Y], "g--", linewidth=2, alpha=0.7)


def plot_cost_vs_denoising_step(plan_log: Dict, save_path: Optional[str] = None) -> None:
    logs = plan_log["logs"]
    step = np.array([x["denoise_step_idx"] for x in logs])
    plt.figure(figsize=(7, 4))
    for key, label in [
        ("J_total", "J_total"),
        ("weighted_centerx_cost", "w_centerx * centerx"),
        ("weighted_smooth_cost", "w_smooth * smooth"),
        ("weighted_track_cost", "w_track * track"),
        ("weighted_barrier_cost", "w_barrier * barrier"),
    ]:
        plt.plot(step, np.array([x[key] for x in logs]), marker="o", label=label)
    plt.xlabel("denoising step idx")
    plt.ylabel("cost")
    plt.title("Cost vs denoising step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_grad_vs_denoising_step(plan_log: Dict, save_path: Optional[str] = None) -> None:
    logs = plan_log["logs"]
    step = np.array([x["denoise_step_idx"] for x in logs])
    plt.figure(figsize=(7, 4))
    for key, label in [
        ("grad_norm", "||grad|| raw"),
        ("grad_norm_clipped", "||grad|| clipped"),
        ("alpha_grad_norm", "alpha * ||grad|| clipped"),
    ]:
        plt.plot(step, np.array([x[key] for x in logs]), marker="o", label=label)
    plt.xlabel("denoising step idx")
    plt.ylabel("gradient magnitude")
    plt.title("Gradient magnitude vs denoising step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_rollout_snapshots(plan_log: Dict, save_path: Optional[str] = None, snapshot_indices: Optional[Iterable[int]] = None) -> None:
    logs = plan_log["logs"]
    n = len(logs)
    idxs = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])) if snapshot_indices is None else list(snapshot_indices)
    fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4), squeeze=False)
    axes = axes[0]
    for ax, idx in zip(axes, idxs):
        item = logs[idx]
        tcp = np.asarray(item["rollout_xy_traj"])
        sp = np.asarray(item["sp_xy_traj"])
        plot_obstacles(ax)
        ax.plot(tcp[:, 0], tcp[:, 1], marker="o", linewidth=1.5, label="rollout tcp")
        ax.plot(sp[:, 0], sp[:, 1], marker="x", linewidth=1.2, label="setpoint")
        ax.scatter(tcp[0, 0], tcp[0, 1], s=40, marker="s", label="start")
        ax.scatter(tcp[-1, 0], tcp[-1, 1], s=40, marker="*", label="end")
        ax.set_title(f"step={item['denoise_step_idx']}, t={item['diffusion_t']}")
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(-0.3, 0.5)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def summarize_episode_guidance_logs(guidance_logs, pick: str = "last"):
    episode_step_idx, ref_t = [], []
    J_total, centerx_cost, smooth_cost, track_cost, barrier_cost = [], [], [], [], []
    grad_norm, grad_norm_clipped, alpha_grad_norm = [], [], []
    diffusion_t, denoise_step_idx = [], []

    for plan in guidance_logs:
        logs = plan.get("logs", [])
        if not logs:
            continue
        if pick == "last":
            item = logs[-1]
        elif pick == "first":
            item = logs[0]
        elif pick == "min_cost":
            item = min(logs, key=lambda x: x["J_total"])
        else:
            raise ValueError(f"Unknown pick mode: {pick}")
        episode_step_idx.append(plan.get("episode_step_idx", len(episode_step_idx)))
        ref_t.append(plan.get("ref_t_before_sampling", -1))
        for key, dst in [
            ("J_total", J_total),
            ("centerx_cost", centerx_cost),
            ("smooth_cost", smooth_cost),
            ("track_cost", track_cost),
            ("barrier_cost", barrier_cost),
            ("grad_norm", grad_norm),
            ("grad_norm_clipped", grad_norm_clipped),
            ("alpha_grad_norm", alpha_grad_norm),
            ("diffusion_t", diffusion_t),
            ("denoise_step_idx", denoise_step_idx),
        ]:
            dst.append(item.get(key, np.nan))

    return {k: np.asarray(v) for k, v in {
        "episode_step_idx": episode_step_idx,
        "ref_t": ref_t,
        "J_total": J_total,
        "centerx_cost": centerx_cost,
        "smooth_cost": smooth_cost,
        "track_cost": track_cost,
        "barrier_cost": barrier_cost,
        "grad_norm": grad_norm,
        "grad_norm_clipped": grad_norm_clipped,
        "alpha_grad_norm": alpha_grad_norm,
        "diffusion_t": diffusion_t,
        "denoise_step_idx": denoise_step_idx,
    }.items()}


def flatten_episode_all_denoising_logs(guidance_logs):
    out = {k: [] for k in [
        "x_global", "x_plan_denoise", "planning_idx", "episode_step_idx", "ref_t_before_sampling",
        "denoise_step_idx", "diffusion_t", "J_total", "centerx_cost", "smooth_cost", "track_cost",
        "barrier_cost", "weighted_centerx_cost", "weighted_smooth_cost", "weighted_track_cost",
        "weighted_barrier_cost", "grad_norm", "grad_norm_clipped", "alpha_grad_norm",
    ]}
    k = 0
    for p_idx, plan in enumerate(guidance_logs):
        logs = plan.get("logs", [])
        ep_idx = plan.get("episode_step_idx", p_idx)
        ref_t0 = plan.get("ref_t_before_sampling", -1)
        for item in logs:
            den_idx = item.get("denoise_step_idx", -1)
            out["x_global"].append(k)
            out["x_plan_denoise"].append(ep_idx + den_idx / 100.0)
            out["planning_idx"].append(p_idx)
            out["episode_step_idx"].append(ep_idx)
            out["ref_t_before_sampling"].append(ref_t0)
            for key in [
                "denoise_step_idx", "diffusion_t", "J_total", "centerx_cost", "smooth_cost", "track_cost",
                "barrier_cost", "weighted_centerx_cost", "weighted_smooth_cost", "weighted_track_cost",
                "weighted_barrier_cost", "grad_norm", "grad_norm_clipped", "alpha_grad_norm",
            ]:
                out[key].append(item.get(key, np.nan))
            k += 1
    return {key: np.asarray(val) for key, val in out.items()}


def plot_episode_cost(summary, save_path: Optional[str] = None, title_suffix: str = "") -> None:
    x = summary["episode_step_idx"]
    plt.figure(figsize=(9, 4))
    for key in ["J_total", "centerx_cost", "smooth_cost", "track_cost", "barrier_cost"]:
        plt.plot(x, summary[key], marker="o", label=key)
    plt.xlabel("episode / planning step")
    plt.ylabel("cost")
    plt.title(f"Episode-level cost across all planning steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_episode_grad(summary, save_path: Optional[str] = None, title_suffix: str = "") -> None:
    x = summary["episode_step_idx"]
    plt.figure(figsize=(9, 4))
    for key, label in [("grad_norm", "||grad|| raw"), ("grad_norm_clipped", "||grad|| clipped"), ("alpha_grad_norm", "alpha * ||grad|| clipped")]:
        plt.plot(x, summary[key], marker="o", label=label)
    plt.xlabel("episode / planning step")
    plt.ylabel("gradient magnitude")
    plt.title(f"Episode-level gradient across all planning steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_episode_all_denoising_cost(flat, save_path: Optional[str] = None, title_suffix: str = "") -> None:
    x = flat["x_global"]
    plt.figure(figsize=(14, 4))
    for key in ["J_total", "centerx_cost", "smooth_cost", "track_cost", "barrier_cost"]:
        plt.plot(x, flat[key], linewidth=1.2, label=key)
    planning_idx = flat["planning_idx"]
    change_points = np.where(np.diff(planning_idx) != 0)[0]
    for cp in change_points:
        plt.axvline(x[cp], linewidth=0.8, alpha=0.2)
    plt.xlabel("global denoising index across the whole episode")
    plt.ylabel("cost")
    plt.title(f"Episode-wide cost across all planning x all denoising steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_episode_all_denoising_grad(flat, save_path: Optional[str] = None, title_suffix: str = "") -> None:
    x = flat["x_global"]
    plt.figure(figsize=(12, 4))
    for key, label in [("grad_norm", "||grad|| raw"), ("grad_norm_clipped", "||grad|| clipped"), ("alpha_grad_norm", "alpha * ||grad|| clipped")]:
        plt.plot(x, flat[key], linewidth=1.2, label=label)
    plt.xlabel("global denoising index across the whole episode")
    plt.ylabel("gradient magnitude")
    plt.title(f"Episode-wide gradient across all planning x all denoising steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def flatten_single_episode_to_sequence(guidance_logs, value_key: str = "J_total"):
    seq, planning_idx, denoise_idx, diffusion_t = [], [], [], []
    for p_idx, plan in enumerate(guidance_logs):
        for item in plan.get("logs", []):
            if value_key in item:
                seq.append(item[value_key])
                planning_idx.append(p_idx)
                denoise_idx.append(item.get("denoise_step_idx", -1))
                diffusion_t.append(item.get("diffusion_t", -1))
    return {
        "seq": np.asarray(seq, dtype=np.float32),
        "planning_idx": np.asarray(planning_idx, dtype=np.int32),
        "denoise_idx": np.asarray(denoise_idx, dtype=np.int32),
        "diffusion_t": np.asarray(diffusion_t, dtype=np.int32),
    }


def build_flattened_episode_tensor(all_results, value_key: str = "J_total"):
    flattened = []
    lengths = []
    for result in all_results:
        flat = flatten_single_episode_to_sequence(result["guidance_logs"], value_key=value_key)
        flattened.append(flat)
        lengths.append(len(flat["seq"]))
    N = len(flattened)
    L_max = max(lengths) if lengths else 0
    tensor = np.full((N, L_max), np.nan, dtype=np.float32)
    for i, flat in enumerate(flattened):
        tensor[i, : len(flat["seq"])] = flat["seq"]
    return tensor, {"N": N, "L_max": L_max, "lengths": np.asarray(lengths, dtype=np.int32)}


def plot_flattened_mean_std_with_boundaries(tensor, reference_guidance_logs, title: str, ylabel: str, save_path: Optional[str] = None):
    mean = np.nanmean(tensor, axis=0)
    std = np.nanstd(tensor, axis=0)
    x = np.arange(tensor.shape[1])
    valid = ~np.isnan(mean)
    plt.figure(figsize=(16, 4))
    plt.plot(x[valid], mean[valid], linewidth=1.5, label="mean")
    plt.fill_between(x[valid], mean[valid] - std[valid], mean[valid] + std[valid], alpha=0.25, label="±1 std")
    cursor = 0
    for plan in reference_guidance_logs:
        cursor += len(plan.get("logs", []))
        plt.axvline(cursor, linewidth=0.8, alpha=0.15)
    plt.xlabel("all planning steps × all denoising steps (flattened index)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_episode_all_denoising_cost_mean_std(
    J_tensor,
    center_tensor=None,
    smooth_tensor=None,
    track_tensor=None,
    barrier_tensor=None,
    reference_guidance_logs=None,
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    x = np.arange(J_tensor.shape[1])
    plt.figure(figsize=(16, 4))

    def _plot_mean_std(tensor, label):
        mean = np.nanmean(tensor, axis=0)
        std = np.nanstd(tensor, axis=0)
        valid = ~np.isnan(mean)
        plt.plot(x[valid], mean[valid], linewidth=1.4, label=f"{label} mean")
        plt.fill_between(x[valid], mean[valid] - std[valid], mean[valid] + std[valid], alpha=0.18)

    _plot_mean_std(J_tensor, "J_total")
    for tensor, label in [(center_tensor, "centerx_cost"), (smooth_tensor, "smooth_cost"), (track_tensor, "track_cost"), (barrier_tensor, "barrier_cost")]:
        if tensor is not None:
            _plot_mean_std(tensor, label)
    if reference_guidance_logs is not None:
        cursor = 0
        for plan in reference_guidance_logs:
            cursor += len(plan.get("logs", []))
            plt.axvline(cursor, linewidth=0.8, alpha=0.15)
    plt.xlabel("all planning steps × all denoising steps (flattened index)")
    plt.ylabel("cost")
    plt.title(f"Mean ± std cost over trajectories{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_episode_all_denoising_grad_mean_std(
    grad_tensor,
    gradc_tensor=None,
    alpha_tensor=None,
    reference_guidance_logs=None,
    save_path: Optional[str] = None,
    title_suffix: str = "",
):
    x = np.arange(grad_tensor.shape[1])
    plt.figure(figsize=(16, 4))

    def _plot_mean_std(tensor, label):
        mean = np.nanmean(tensor, axis=0)
        std = np.nanstd(tensor, axis=0)
        valid = ~np.isnan(mean)
        plt.plot(x[valid], mean[valid], linewidth=1.4, label=f"{label} mean")
        plt.fill_between(x[valid], mean[valid] - std[valid], mean[valid] + std[valid], alpha=0.18)

    _plot_mean_std(grad_tensor, "grad_norm")
    for tensor, label in [(gradc_tensor, "grad_norm_clipped"), (alpha_tensor, "alpha_grad_norm")]:
        if tensor is not None:
            _plot_mean_std(tensor, label)
    if reference_guidance_logs is not None:
        cursor = 0
        for plan in reference_guidance_logs:
            cursor += len(plan.get("logs", []))
            plt.axvline(cursor, linewidth=0.8, alpha=0.15)
    plt.xlabel("all planning steps × all denoising steps (flattened index)")
    plt.ylabel("gradient magnitude")
    plt.title(f"Mean ± std gradient over trajectories{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()
