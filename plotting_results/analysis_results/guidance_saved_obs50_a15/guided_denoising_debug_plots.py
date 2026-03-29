import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from guided_diffusion_planner2 import FINISH_Y, CENTER_X


def plot_obstacles(ax):
    obstacles = [
        (0.5,   -0.1, 0.03),
        (0.425,  0.08, 0.03),
        (0.575,  0.08, 0.03),
        (0.35,   0.26, 0.03),
        (0.5,    0.26, 0.03),
        (0.65,   0.26, 0.03),
    ]
    for i, (cx, cy, r) in enumerate(obstacles):
        circ = patches.Circle((cx, cy), r, edgecolor="red", facecolor="none", linewidth=1.5)
        ax.add_patch(circ)
        ax.scatter(cx, cy, s=20, c="red", alpha=0.5)

    half_w = 0.25
    ax.plot([CENTER_X - half_w, CENTER_X + half_w], [FINISH_Y, FINISH_Y], "g--", linewidth=2, alpha=0.7)


def plot_cost_vs_denoising_step(plan_log, save_path=None):
    logs = plan_log["logs"]
    step = np.array([x["denoise_step_idx"] for x in logs])

    J_total = np.array([x["J_total"] for x in logs])
    w_centerx = np.array([x["weighted_centerx_cost"] for x in logs])
    w_smooth = np.array([x["weighted_smooth_cost"] for x in logs])
    w_track = np.array([x["weighted_track_cost"] for x in logs])
    w_barrier = np.array([x["weighted_barrier_cost"] for x in logs])

    plt.figure(figsize=(7, 4))
    plt.plot(step, J_total, marker="o", label="J_total")
    plt.plot(step, w_centerx, marker="o", label="w_centerx * centerx")
    plt.plot(step, w_smooth, marker="o", label="w_smooth * smooth")
    plt.plot(step, w_track, marker="o", label="w_track * track")
    plt.plot(step, w_barrier, marker="o", label="w_barrier * barrier")
    plt.xlabel("denoising step idx")
    plt.ylabel("cost")
    plt.title("Cost vs denoising step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_grad_vs_denoising_step(plan_log, save_path=None):
    logs = plan_log["logs"]
    step = np.array([x["denoise_step_idx"] for x in logs])

    grad_norm = np.array([x["grad_norm"] for x in logs])
    grad_norm_clipped = np.array([x["grad_norm_clipped"] for x in logs])
    alpha_grad_norm = np.array([x["alpha_grad_norm"] for x in logs])

    plt.figure(figsize=(7, 4))
    plt.plot(step, grad_norm, marker="o", label="||grad|| raw")
    plt.plot(step, grad_norm_clipped, marker="o", label="||grad|| clipped")
    plt.plot(step, alpha_grad_norm, marker="o", label="alpha * ||grad|| clipped")
    plt.xlabel("denoising step idx")
    plt.ylabel("gradient magnitude")
    plt.title("Gradient magnitude vs denoising step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_rollout_snapshots(plan_log, save_path=None, snapshot_indices=None):
    logs = plan_log["logs"]
    n = len(logs)

    if snapshot_indices is None:
        idxs = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))
    else:
        idxs = snapshot_indices

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

def summarize_episode_guidance_logs(guidance_logs, pick="last"):
    """
    把整条 episode 的 planning logs 汇总成按 environment/planning step 排列的一维序列。

    Args:
        guidance_logs: result["guidance_logs"]
        pick: "last" | "first" | "min_cost"

    Returns:
        summary: dict of numpy arrays
    """
    episode_step_idx = []
    ref_t = []

    J_total = []
    centerx_cost = []
    smooth_cost = []
    track_cost = []
    barrier_cost = []

    grad_norm = []
    grad_norm_clipped = []
    alpha_grad_norm = []

    diffusion_t = []
    denoise_step_idx = []

    for plan in guidance_logs:
        logs = plan.get("logs", [])
        if len(logs) == 0:
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

        J_total.append(item["J_total"])
        centerx_cost.append(item["centerx_cost"])
        smooth_cost.append(item["smooth_cost"])
        track_cost.append(item["track_cost"])
        barrier_cost.append(item["barrier_cost"])

        grad_norm.append(item["grad_norm"])
        grad_norm_clipped.append(item.get("grad_norm_clipped", np.nan))
        alpha_grad_norm.append(item.get("alpha_grad_norm", np.nan))

        diffusion_t.append(item.get("diffusion_t", -1))
        denoise_step_idx.append(item.get("denoise_step_idx", -1))

    return {
        "episode_step_idx": np.asarray(episode_step_idx),
        "ref_t": np.asarray(ref_t),

        "J_total": np.asarray(J_total),
        "centerx_cost": np.asarray(centerx_cost),
        "smooth_cost": np.asarray(smooth_cost),
        "track_cost": np.asarray(track_cost),
        "barrier_cost": np.asarray(barrier_cost),

        "grad_norm": np.asarray(grad_norm),
        "grad_norm_clipped": np.asarray(grad_norm_clipped),
        "alpha_grad_norm": np.asarray(alpha_grad_norm),

        "diffusion_t": np.asarray(diffusion_t),
        "denoise_step_idx": np.asarray(denoise_step_idx),
    }
def flatten_episode_all_denoising_logs(guidance_logs):
    """
    把整条 episode 的所有 planning 的所有 denoising logs 串成一条长序列。

    Returns:
        flat: dict of numpy arrays
            x_global                全局横轴索引 0,1,2,...,N-1
            planning_idx            这个点属于第几个 planning
            episode_step_idx        对应 episode / planning step
            denoise_step_idx        planning 内的 denoise idx
            diffusion_t             对应 diffusion t
            J_total / centerx_cost / ...
            grad_norm / grad_norm_clipped / alpha_grad_norm
    """
    x_global = []
    x_plan_denoise = []
    planning_idx = []
    episode_step_idx = []
    ref_t_before_sampling = []

    denoise_step_idx = []
    diffusion_t = []

    J_total = []
    centerx_cost = []
    smooth_cost = []
    track_cost = []
    barrier_cost = []

    weighted_centerx_cost = []
    weighted_smooth_cost = []
    weighted_track_cost = []
    weighted_barrier_cost = []

    grad_norm = []
    grad_norm_clipped = []
    alpha_grad_norm = []

    k = 0
    for p_idx, plan in enumerate(guidance_logs):
        logs = plan.get("logs", [])
        ep_idx = plan.get("episode_step_idx", p_idx)
        ref_t0 = plan.get("ref_t_before_sampling", -1)

        for item in logs:
            den_idx = item.get("denoise_step_idx", -1)

            x_global.append(k)
            x_plan_denoise.append(ep_idx + den_idx / 100.0)
            planning_idx.append(p_idx)
            episode_step_idx.append(ep_idx)
            ref_t_before_sampling.append(ref_t0)

            denoise_step_idx.append(item.get("denoise_step_idx", -1))
            diffusion_t.append(item.get("diffusion_t", -1))

            J_total.append(item["J_total"])
            centerx_cost.append(item["centerx_cost"])
            smooth_cost.append(item["smooth_cost"])
            track_cost.append(item["track_cost"])
            barrier_cost.append(item["barrier_cost"])

            weighted_centerx_cost.append(item.get("weighted_centerx_cost", np.nan))
            weighted_smooth_cost.append(item.get("weighted_smooth_cost", np.nan))
            weighted_track_cost.append(item.get("weighted_track_cost", np.nan))
            weighted_barrier_cost.append(item.get("weighted_barrier_cost", np.nan))

            grad_norm.append(item["grad_norm"])
            grad_norm_clipped.append(item.get("grad_norm_clipped", np.nan))
            alpha_grad_norm.append(item.get("alpha_grad_norm", np.nan))

            k += 1

    return {
        "x_global": np.asarray(x_global, dtype=np.int32),
        "planning_idx": np.asarray(planning_idx, dtype=np.int32),
        "episode_step_idx": np.asarray(episode_step_idx, dtype=np.int32),
        "ref_t_before_sampling": np.asarray(ref_t_before_sampling, dtype=np.int32),

        "denoise_step_idx": np.asarray(denoise_step_idx, dtype=np.int32),
        "diffusion_t": np.asarray(diffusion_t, dtype=np.int32),

        "J_total": np.asarray(J_total, dtype=np.float32),
        "centerx_cost": np.asarray(centerx_cost, dtype=np.float32),
        "smooth_cost": np.asarray(smooth_cost, dtype=np.float32),
        "track_cost": np.asarray(track_cost, dtype=np.float32),
        "barrier_cost": np.asarray(barrier_cost, dtype=np.float32),

        "weighted_centerx_cost": np.asarray(weighted_centerx_cost, dtype=np.float32),
        "weighted_smooth_cost": np.asarray(weighted_smooth_cost, dtype=np.float32),
        "weighted_track_cost": np.asarray(weighted_track_cost, dtype=np.float32),
        "weighted_barrier_cost": np.asarray(weighted_barrier_cost, dtype=np.float32),

        "grad_norm": np.asarray(grad_norm, dtype=np.float32),
        "grad_norm_clipped": np.asarray(grad_norm_clipped, dtype=np.float32),
        "alpha_grad_norm": np.asarray(alpha_grad_norm, dtype=np.float32),
        "x_plan_denoise": np.asarray(x_plan_denoise, dtype=np.float32),
    }
    
def plot_episode_cost(summary, save_path=None, title_suffix=""):
    x = summary["episode_step_idx"]

    plt.figure(figsize=(9, 4))
    plt.plot(x, summary["J_total"], marker="o", label="J_total")
    plt.plot(x, summary["centerx_cost"], marker="o", label="centerx_cost")
    plt.plot(x, summary["smooth_cost"], marker="o", label="smooth_cost")
    plt.plot(x, summary["track_cost"], marker="o", label="track_cost")
    plt.plot(x, summary["barrier_cost"], marker="o", label="barrier_cost")

    plt.xlabel("episode / planning step")
    plt.ylabel("cost")
    plt.title(f"Episode-level cost across all planning steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

def plot_episode_grad(summary, save_path=None, title_suffix=""):
    x = summary["episode_step_idx"]

    plt.figure(figsize=(9, 4))
    plt.plot(x, summary["grad_norm"], marker="o", label="||grad|| raw")
    plt.plot(x, summary["grad_norm_clipped"], marker="o", label="||grad|| clipped")
    plt.plot(x, summary["alpha_grad_norm"], marker="o", label="alpha * ||grad|| clipped")

    plt.xlabel("episode / planning step")
    plt.ylabel("gradient magnitude")
    plt.title(f"Episode-level gradient across all planning steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

def plot_episode_all_denoising_cost(flat, save_path=None, title_suffix="", draw_boundaries=True):
    x = flat["x_global"]

    plt.figure(figsize=(14, 4))
    plt.plot(x, flat["J_total"], linewidth=1.5, label="J_total")
    plt.plot(x, flat["centerx_cost"], linewidth=1.0, label="centerx_cost")
    plt.plot(x, flat["smooth_cost"], linewidth=1.0, label="smooth_cost")
    plt.plot(x, flat["track_cost"], linewidth=1.0, label="track_cost")
    plt.plot(x, flat["barrier_cost"], linewidth=1.0, label="barrier_cost")

    if draw_boundaries:
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


def plot_episode_all_denoising_grad(flat, save_path=None, title_suffix=""):
    x = flat["x_global"]

    plt.figure(figsize=(12, 4))
    plt.plot(x, flat["grad_norm"], linewidth=1.5, label="||grad|| raw")
    plt.plot(x, flat["grad_norm_clipped"], linewidth=1.2, label="||grad|| clipped")
    plt.plot(x, flat["alpha_grad_norm"], linewidth=1.2, label="alpha * ||grad|| clipped")

    plt.xlabel("global denoising index across the whole episode")
    plt.ylabel("gradient magnitude")
    plt.title(f"Episode-wide gradient across all planning x all denoising steps{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


#  run 50 trajectory


def flatten_single_episode_to_sequence(guidance_logs, value_key="J_total"):
    """
    把单条 episode 的所有 planning 的所有 denoise 展平成一条长序列。

    Returns:
        seq: shape [L], L = sum over planning of num_denoising_logs
    """
    seq = []
    planning_idx = []
    denoise_idx = []
    diffusion_t = []

    for p_idx, plan in enumerate(guidance_logs):
        logs = plan.get("logs", [])
        for item in logs:
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

def build_flattened_episode_tensor(all_results, value_key="J_total"):
    """
    把多条轨迹 flatten 后，按长序列位置对齐成 [N, L_max].

    Returns:
        tensor: [N, L_max], NaN padding
        meta: dict
    """
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
        L = len(flat["seq"])
        tensor[i, :L] = flat["seq"]

    meta = {
        "N": N,
        "L_max": L_max,
        "lengths": np.asarray(lengths, dtype=np.int32),
    }
    return tensor, meta

def plot_flattened_mean_std_with_boundaries(
    tensor,
    reference_guidance_logs,
    title,
    ylabel,
    save_path=None,
):
    """
    tensor: [N, L]
    reference_guidance_logs: 用第0条轨迹的 guidance_logs 来画 planning 分界线
    """
    mean = np.nanmean(tensor, axis=0)
    std = np.nanstd(tensor, axis=0)
    x = np.arange(tensor.shape[1])

    valid = ~np.isnan(mean)
    x_plot = x[valid]
    mean_plot = mean[valid]
    std_plot = std[valid]

    plt.figure(figsize=(16, 4))
    plt.plot(x_plot, mean_plot, linewidth=1.5, label="mean")
    plt.fill_between(x_plot, mean_plot - std_plot, mean_plot + std_plot, alpha=0.25, label="±1 std")

    # 用第一条轨迹的 planning 边界做参考
    cursor = 0
    for plan in reference_guidance_logs:
        logs = plan.get("logs", [])
        cursor += len(logs)
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
    save_path=None,
    title_suffix="",
):
    x = np.arange(J_tensor.shape[1])

    plt.figure(figsize=(16, 4))

    def _plot_mean_std(tensor, label):
        mean = np.nanmean(tensor, axis=0)
        std = np.nanstd(tensor, axis=0)
        valid = ~np.isnan(mean)
        xv = x[valid]
        mv = mean[valid]
        sv = std[valid]
        plt.plot(xv, mv, linewidth=1.4, label=f"{label} mean")
        plt.fill_between(xv, mv - sv, mv + sv, alpha=0.18)

    _plot_mean_std(J_tensor, "J_total")

    if center_tensor is not None:
        _plot_mean_std(center_tensor, "centerx_cost")
    if smooth_tensor is not None:
        _plot_mean_std(smooth_tensor, "smooth_cost")
    if track_tensor is not None:
        _plot_mean_std(track_tensor, "track_cost")
    if barrier_tensor is not None:
        _plot_mean_std(barrier_tensor, "barrier_cost")

    if reference_guidance_logs is not None:
        cursor = 0
        for plan in reference_guidance_logs:
            logs = plan.get("logs", [])
            cursor += len(logs)
            plt.axvline(cursor, linewidth=0.8, alpha=0.15)

    plt.xlabel("all planning steps × all denoising steps (flattened index)")
    plt.ylabel("cost")
    plt.title(f"Mean ± std cost over 50 trajectories{title_suffix}")
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
    save_path=None,
    title_suffix="",
):
    x = np.arange(grad_tensor.shape[1])

    plt.figure(figsize=(16, 4))

    def _plot_mean_std(tensor, label):
        mean = np.nanmean(tensor, axis=0)
        std = np.nanstd(tensor, axis=0)
        valid = ~np.isnan(mean)
        xv = x[valid]
        mv = mean[valid]
        sv = std[valid]
        plt.plot(xv, mv, linewidth=1.4, label=f"{label} mean")
        plt.fill_between(xv, mv - sv, mv + sv, alpha=0.18)

    _plot_mean_std(grad_tensor, "grad_norm")

    if gradc_tensor is not None:
        _plot_mean_std(gradc_tensor, "grad_norm_clipped")
    if alpha_tensor is not None:
        _plot_mean_std(alpha_tensor, "alpha_grad_norm")

    if reference_guidance_logs is not None:
        cursor = 0
        for plan in reference_guidance_logs:
            logs = plan.get("logs", [])
            cursor += len(logs)
            plt.axvline(cursor, linewidth=0.8, alpha=0.15)

    plt.xlabel("all planning steps × all denoising steps (flattened index)")
    plt.ylabel("gradient magnitude")
    plt.title(f"Mean ± std gradient over 50 trajectories{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

import pickle
import pandas as pd


def build_guidance_log_dataframe(all_results):
    """
    把 all_results 展平成一个 DataFrame。
    每一行对应：
        1 个 episode
        1 个 planning step
        1 个 denoising step
    """
    rows = []

    for episode_idx, result in enumerate(all_results):
        guidance_logs = result.get("guidance_logs", [])

        for plan_idx, plan in enumerate(guidance_logs):
            plan_logs = plan.get("logs", [])

            for item in plan_logs:
                row = {
                    # ---- hierarchy ----
                    "episode_idx": int(episode_idx),
                    "planning_idx": int(plan_idx),
                    "episode_step_idx": int(plan.get("episode_step_idx", plan_idx)),
                    "ref_t_before_sampling": int(plan.get("ref_t_before_sampling", -1)),

                    # ---- denoising identity ----
                    "denoise_step_idx": int(item.get("denoise_step_idx", -1)),
                    "diffusion_t": int(item.get("diffusion_t", -1)),

                    # ---- plan-level meta ----
                    "plan_id": int(plan.get("plan_id", plan_idx)),
                    "guidance_scale": float(plan.get("guidance_scale", np.nan)),
                    "grad_clip": float(plan.get("grad_clip", np.nan)) if plan.get("grad_clip", None) is not None else np.nan,

                    # ---- cost ----
                    "J_total": float(item.get("J_total", np.nan)),
                    "centerx_cost": float(item.get("centerx_cost", np.nan)),
                    "smooth_cost": float(item.get("smooth_cost", np.nan)),
                    "track_cost": float(item.get("track_cost", np.nan)),
                    "barrier_cost": float(item.get("barrier_cost", np.nan)),

                    "weighted_centerx_cost": float(item.get("weighted_centerx_cost", np.nan)),
                    "weighted_smooth_cost": float(item.get("weighted_smooth_cost", np.nan)),
                    "weighted_track_cost": float(item.get("weighted_track_cost", np.nan)),
                    "weighted_barrier_cost": float(item.get("weighted_barrier_cost", np.nan)),

                    # ---- grad ----
                    "grad_norm": float(item.get("grad_norm", np.nan)),
                    "grad_norm_clipped": float(item.get("grad_norm_clipped", np.nan)),
                    "grad_max_abs": float(item.get("grad_max_abs", np.nan)),
                    "grad_max_abs_clipped": float(item.get("grad_max_abs_clipped", np.nan)),
                    "alpha_grad_norm": float(item.get("alpha_grad_norm", np.nan)),

                    # ---- other scalar diagnostics ----
                    "model_mean_norm": float(item.get("model_mean_norm", np.nan)),
                    "dxy_norm": float(item.get("dxy_norm", np.nan)),
                }

                # plan-level optional fields
                if "des_xy_exec" in plan and plan["des_xy_exec"] is not None:
                    row["des_xy_exec_x"] = float(plan["des_xy_exec"][0])
                    row["des_xy_exec_y"] = float(plan["des_xy_exec"][1])

                if "cur_xy" in plan and plan["cur_xy"] is not None:
                    row["cur_xy_x"] = float(plan["cur_xy"][0])
                    row["cur_xy_y"] = float(plan["cur_xy"][1])

                if "target_xy" in plan and plan["target_xy"] is not None:
                    row["target_xy_x"] = float(plan["target_xy"][0])
                    row["target_xy_y"] = float(plan["target_xy"][1])

                if "pred_delta" in plan and plan["pred_delta"] is not None:
                    pred_delta = np.asarray(plan["pred_delta"]).reshape(-1)
                    for k, v in enumerate(pred_delta):
                        row[f"pred_delta_{k}"] = float(v)

                rows.append(row)

    df = pd.DataFrame(rows)

    # 排序，保证后面查询稳定
    sort_cols = ["episode_idx", "planning_idx", "denoise_step_idx"]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    if len(existing_sort_cols) > 0 and len(df) > 0:
        df = df.sort_values(existing_sort_cols).reset_index(drop=True)

    return df


def save_guidance_artifacts(all_results, out_dir="guidance_saved"):
    """
    保存两类文件：
    1) all_results.pkl   原始结构化数据
    2) guidance_logs.csv 扁平表格，方便查询和画图
    3) guidance_logs.parquet 可选，体积更小、读取更快
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. 保存原始结构化数据
    pkl_path = os.path.join(out_dir, "all_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)

    # 2. 保存扁平表
    df = build_guidance_log_dataframe(all_results)

    csv_path = os.path.join(out_dir, "guidance_logs.csv")
    df.to_csv(csv_path, index=False)

    # 3. parquet（如果环境支持）
    parquet_path = os.path.join(out_dir, "guidance_logs.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = None

    return {
        "pkl_path": pkl_path,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "n_rows": len(df),
    }


