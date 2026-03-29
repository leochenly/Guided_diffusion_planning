import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
RUNS = {
    "alpha0": {
        "csv": "guidance_saved_a15/guidance_logs_a0.csv",
        "pkl": "guidance_saved_a15/all_results_a0.pkl",
        "label": "alpha=0",
    },
    "alpha15": {
        "csv": "guidance_saved_a15/guidance_logs_a15.csv",
        "pkl": "guidance_saved_a15/all_results_a15.pkl",
        "label": "alpha=15",
    },
    "alpha50": {
        "csv": "guidance_saved_a15/guidance_logs_a50.csv",
        "pkl": "guidance_saved_a15/all_results_a50.pkl",
        "label": "alpha=50",
    },
}

OUT_DIR = "guidance_saved_a15/alpha_comparison_figs"
BASELINE_T = 15
CENTER_X = 0.5
FIG_DPI = 180

# From your print:
# [time] mujoco model.opt.timestep = 0.001000 s
# [time] sim.n_substeps           = 35
# [time] rollout dt per macro step = 0.035000
ROLLOUT_DT = 0.035


# ============================================================
# Utilities
# ============================================================
def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Saved: {path}")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_csv] {path} -> shape={df.shape}")
    return df


def load_pkl(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[load_pkl] {path} -> type={type(obj)}, len={len(obj) if isinstance(obj, list) else 'n/a'}")
    return obj


def finite_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(df[c].to_numpy())
    return df.loc[mask].copy()


def planning_id_cols(df: pd.DataFrame) -> List[str]:
    req = ["episode_idx", "planning_idx", "diffusion_t"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return ["episode_idx", "planning_idx"]


def summarize_by_t(df: pd.DataFrame, value_key: str) -> pd.DataFrame:
    g = (
        df.groupby("diffusion_t")[value_key]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("diffusion_t")
    )
    return g


def add_normalized_cost(df: pd.DataFrame, cost_key: str = "J_total", baseline_t: int = BASELINE_T, eps: float = 1e-8) -> pd.DataFrame:
    keys = planning_id_cols(df)
    rows = []
    for _, g in df.groupby(keys):
        g = g.sort_values("diffusion_t").copy()
        base_rows = g[g["diffusion_t"] == baseline_t]
        if len(base_rows) == 0:
            continue
        baseline = float(base_rows[cost_key].mean())
        denom = baseline if abs(baseline) > eps else eps
        g["baseline_cost"] = baseline
        g["delta_cost"] = g[cost_key] - baseline
        g["relative_cost"] = g[cost_key] / denom
        rows.append(g)
    if not rows:
        raise ValueError("No valid planning groups with baseline_t found.")
    return pd.concat(rows, ignore_index=True)


def build_next_step_delta(df: pd.DataFrame, cost_key: str = "J_total") -> pd.DataFrame:
    keys = planning_id_cols(df)
    rows = []
    for _, g in df.groupby(keys):
        g = g.sort_values("diffusion_t", ascending=False).copy()  # 15 -> 0
        vals = g[cost_key].to_numpy()
        tvals = g["diffusion_t"].to_numpy()
        delta_next = np.full(len(g), np.nan, dtype=np.float32)
        for j in range(len(g) - 1):
            if tvals[j + 1] == tvals[j] - 1:
                delta_next[j] = vals[j + 1] - vals[j]  # J(t-1) - J(t)
        g["delta_next_cost"] = delta_next
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def bin_by_quantiles(series: pd.Series, q: int = 10) -> pd.Series:
    valid = np.isfinite(series.to_numpy())
    out = pd.Series(np.full(len(series), np.nan), index=series.index)
    if valid.sum() < q:
        out.loc[valid] = 0
        return out
    try:
        out.loc[valid] = pd.qcut(series.loc[valid], q=q, labels=False, duplicates="drop")
    except Exception:
        out.loc[valid] = 0
    return out


def build_executed_centerx_costs(all_results, center_x: float = CENTER_X, rollout_dt: float = ROLLOUT_DT):
    mean_costs = []
    cum_costs = []
    steps = []
    times = []
    success = []

    for result in all_results:
        traj_xy = np.asarray(result["traj_xy"], dtype=np.float32)
        x = traj_xy[:, 0]
        per_step = (x - center_x) ** 2

        step_count = int(result.get("steps", len(traj_xy) - 1))

        mean_costs.append(float(np.mean(per_step)))
        cum_costs.append(float(np.sum(per_step)))
        steps.append(step_count)
        times.append(step_count * rollout_dt)
        success.append(bool(result.get("success", True)))

    return {
        "mean_centerx_cost": np.asarray(mean_costs),
        "cum_centerx_cost": np.asarray(cum_costs),
        "completion_steps": np.asarray(steps),
        "completion_time_sec": np.asarray(times),
        "success": np.asarray(success),
    }


def build_executed_cumulative_tensor(all_results, center_x: float = CENTER_X):
    T_max = max(len(r["traj_xy"]) for r in all_results)
    tensor = np.full((len(all_results), T_max), np.nan, dtype=np.float32)
    for i, result in enumerate(all_results):
        traj_xy = np.asarray(result["traj_xy"], dtype=np.float32)
        x = traj_xy[:, 0]
        per_step = (x - center_x) ** 2
        cum = np.cumsum(per_step)
        tensor[i, : len(cum)] = cum
    return tensor


def run_summary_row(run_name: str, df: pd.DataFrame, result_stats: Dict[str, np.ndarray]) -> Dict[str, float]:
    row = {"run": run_name}
    row["num_csv_rows"] = len(df)
    row["num_episodes"] = len(result_stats["success"])
    row["success_rate"] = float(np.mean(result_stats["success"]))

    row["mean_exec_centerx_mean"] = float(np.mean(result_stats["mean_centerx_cost"]))
    row["std_exec_centerx_mean"] = float(np.std(result_stats["mean_centerx_cost"]))

    row["mean_exec_centerx_cum"] = float(np.mean(result_stats["cum_centerx_cost"]))
    row["std_exec_centerx_cum"] = float(np.std(result_stats["cum_centerx_cost"]))

    row["mean_completion_steps"] = float(np.mean(result_stats["completion_steps"]))
    row["std_completion_steps"] = float(np.std(result_stats["completion_steps"]))

    row["mean_completion_time_sec"] = float(np.mean(result_stats["completion_time_sec"]))
    row["std_completion_time_sec"] = float(np.std(result_stats["completion_time_sec"]))

    # final vs baseline
    dfn = add_normalized_cost(df, cost_key="J_total", baseline_t=BASELINE_T)
    keys = planning_id_cols(dfn)
    last_better = []
    reached_better = []

    for _, g in dfn.groupby(keys):
        g = g.sort_values("diffusion_t")
        jt0 = float(g[g["diffusion_t"] == 0]["J_total"].mean())
        jt15 = float(g[g["diffusion_t"] == BASELINE_T]["J_total"].mean())
        reached_better.append(float(g["J_total"].min() < jt15))
        last_better.append(float(jt0 < jt15))

    row["ratio_final_better_than_t15"] = float(np.mean(last_better))
    row["ratio_any_intermediate_better_than_t15"] = float(np.mean(reached_better))
    return row


# ============================================================
# Plot helpers
# ============================================================
def plot_grouped_lines(
    grouped_dict: Dict[str, pd.DataFrame],
    x_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_name: str,
    reverse_x: bool = False,
    hline: float = None,
):
    plt.figure(figsize=(8, 4.5))
    for run_name, g in grouped_dict.items():
        x = g[x_key].to_numpy()
        mean = g["mean"].to_numpy()
        std = g["std"].fillna(0.0).to_numpy()
        plt.plot(x, mean, marker="o", label=RUNS[run_name]["label"])
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    if reverse_x:
        plt.gca().invert_xaxis()
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.0, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_grouped_drop_ratio(drop_ratio_dict: Dict[str, pd.DataFrame], save_name: str):
    plt.figure(figsize=(8, 4.5))
    for run_name, g in drop_ratio_dict.items():
        plt.plot(g["diffusion_t"], g["drop_ratio"], marker="o", label=RUNS[run_name]["label"])
    plt.axhline(0.5, linestyle="--", linewidth=1.0, alpha=0.8)
    plt.gca().invert_xaxis()
    plt.ylim(0, 1)
    plt.title("Proportion of planning with J(t) < J(15)")
    plt.xlabel("diffusion_t")
    plt.ylabel("drop ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_grouped_argmin_hist(argmin_dict: Dict[str, pd.Series], save_name: str):
    ts = sorted(set().union(*[set(s.index.astype(int)) for s in argmin_dict.values()]))
    x = np.arange(len(ts))
    width = 0.24

    plt.figure(figsize=(9, 4.5))
    offsets = np.linspace(-width, width, len(argmin_dict))

    for off, (run_name, counts) in zip(offsets, argmin_dict.items()):
        y = np.array([counts.get(t, 0) for t in ts], dtype=float)
        plt.bar(x + off, y, width=width, label=RUNS[run_name]["label"])

    plt.xticks(x, ts)
    plt.gca().invert_xaxis()
    plt.title("Where does min J_total occur?")
    plt.xlabel("diffusion_t of argmin")
    plt.ylabel("count of planning")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_binned_delta_vs_grad_compare(
    binned_dict: Dict[str, pd.DataFrame],
    grad_key: str,
    save_name: str,
):
    plt.figure(figsize=(8, 4.5))
    for run_name, g in binned_dict.items():
        x = g["grad_mean"].to_numpy()
        y = g["delta_mean"].to_numpy()
        s = g["delta_std"].fillna(0.0).to_numpy()
        plt.plot(x, y, marker="o", label=RUNS[run_name]["label"])
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.8)
    plt.title(f"Next-step cost change vs {grad_key} (binned)")
    plt.xlabel(f"{grad_key} (bin mean)")
    plt.ylabel("delta_next_cost = J(t-1) - J(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_hist_compare(data_dict: Dict[str, np.ndarray], title: str, xlabel: str, save_name: str, bins: int = 20):
    plt.figure(figsize=(8, 4.5))
    for run_name, arr in data_dict.items():
        plt.hist(arr, bins=bins, alpha=0.45, label=RUNS[run_name]["label"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_box_compare(data_dict: Dict[str, np.ndarray], title: str, ylabel: str, save_name: str):
    labels = [RUNS[k]["label"] for k in data_dict.keys()]
    arrs = [data_dict[k] for k in data_dict.keys()]
    plt.figure(figsize=(7, 4.5))
    plt.boxplot(arrs, tick_labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_bar_with_error(data_dict: Dict[str, np.ndarray], title: str, ylabel: str, save_name: str):
    labels = [RUNS[k]["label"] for k in data_dict.keys()]
    arrs = [data_dict[k] for k in data_dict.keys()]
    means = [np.mean(a) for a in arrs]
    stds = [np.std(a) for a in arrs]

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4.5))
    plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(save_name)
    plt.close()


def plot_exec_cumulative_curve_compare(tensor_dict: Dict[str, np.ndarray], save_name: str):
    plt.figure(figsize=(9, 4.5))
    for run_name, tensor in tensor_dict.items():
        mean = np.nanmean(tensor, axis=0)
        std = np.nanstd(tensor, axis=0)
        x = np.arange(tensor.shape[1])
        valid = ~np.isnan(mean)
        x = x[valid]
        mean = mean[valid]
        std = std[valid]
        plt.plot(x, mean, label=RUNS[run_name]["label"])
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    plt.title("Cumulative executed center-x cost vs episode step")
    plt.xlabel("episode step")
    plt.ylabel("cumulative center-x cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    savefig(save_name)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ensure_out_dir(OUT_DIR)

    dfs: Dict[str, pd.DataFrame] = {}
    results: Dict[str, list] = {}
    result_stats: Dict[str, Dict[str, np.ndarray]] = {}
    exec_cum_tensors: Dict[str, np.ndarray] = {}

    # load
    for run_name, info in RUNS.items():
        df = load_csv(info["csv"])
        res = load_pkl(info["pkl"])

        # keep only finite core subset
        df = finite_df(df, ["episode_idx", "planning_idx", "diffusion_t", "J_total"])
        dfs[run_name] = df
        results[run_name] = res
        result_stats[run_name] = build_executed_centerx_costs(res, center_x=CENTER_X, rollout_dt=ROLLOUT_DT)
        exec_cum_tensors[run_name] = build_executed_cumulative_tensor(res, center_x=CENTER_X)

    # ------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------
    summary_rows = []
    for run_name in RUNS.keys():
        summary_rows.append(run_summary_row(run_name, dfs[run_name], result_stats[run_name]))
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)
    print(summary_df)

    # ------------------------------------------------------------
    # 1) Absolute J_total vs diffusion_t
    # ------------------------------------------------------------
    grouped_abs = {k: summarize_by_t(v, "J_total") for k, v in dfs.items()}
    plot_grouped_lines(
        grouped_abs,
        x_key="diffusion_t",
        title="J_total vs diffusion_t",
        xlabel="diffusion_t",
        ylabel="J_total",
        save_name="01_compare_J_total_vs_t.png",
        reverse_x=True,
    )

    # ------------------------------------------------------------
    # 2) Delta cost vs diffusion_t
    # ------------------------------------------------------------
    dfs_norm = {k: add_normalized_cost(v, cost_key="J_total", baseline_t=BASELINE_T) for k, v in dfs.items()}
    grouped_delta = {k: summarize_by_t(v, "delta_cost") for k, v in dfs_norm.items()}
    plot_grouped_lines(
        grouped_delta,
        x_key="diffusion_t",
        title=f"ΔJ(t) = J(t) - J({BASELINE_T})",
        xlabel="diffusion_t",
        ylabel="delta cost",
        save_name="02_compare_delta_cost_vs_t.png",
        reverse_x=True,
        hline=0.0,
    )

    # ------------------------------------------------------------
    # 3) Relative cost vs diffusion_t
    # ------------------------------------------------------------
    grouped_rel = {k: summarize_by_t(v, "relative_cost") for k, v in dfs_norm.items()}
    plot_grouped_lines(
        grouped_rel,
        x_key="diffusion_t",
        title=f"Relative J(t) = J(t) / J({BASELINE_T})",
        xlabel="diffusion_t",
        ylabel="relative cost",
        save_name="03_compare_relative_cost_vs_t.png",
        reverse_x=True,
        hline=1.0,
    )

    # ------------------------------------------------------------
    # 4) Drop ratio vs diffusion_t
    # ------------------------------------------------------------
    drop_ratio_dict = {}
    for run_name, dfn in dfs_norm.items():
        g = (
            dfn.groupby("diffusion_t")["delta_cost"]
            .apply(lambda s: np.mean(s < 0.0))
            .reset_index(name="drop_ratio")
            .sort_values("diffusion_t")
        )
        drop_ratio_dict[run_name] = g
    plot_grouped_drop_ratio(drop_ratio_dict, "04_compare_drop_ratio_vs_t.png")

    # ------------------------------------------------------------
    # 5) Gradient vs diffusion_t
    # ------------------------------------------------------------
    grad_terms = ["grad_norm", "grad_norm_clipped", "alpha_grad_norm"]
    for grad_key in grad_terms:
        ok = True
        grouped = {}
        for run_name, df in dfs.items():
            if grad_key not in df.columns:
                ok = False
                break
            dff = finite_df(df, ["diffusion_t", grad_key])
            grouped[run_name] = summarize_by_t(dff, grad_key)
        if ok:
            plot_grouped_lines(
                grouped,
                x_key="diffusion_t",
                title=f"{grad_key} vs diffusion_t",
                xlabel="diffusion_t",
                ylabel=grad_key,
                save_name=f"05_compare_{grad_key}_vs_t.png",
                reverse_x=True,
            )

    # ------------------------------------------------------------
    # 6) Cost decomposition vs diffusion_t
    # ------------------------------------------------------------
    cost_terms = ["J_total", "centerx_cost", "smooth_cost", "track_cost", "barrier_cost"]
    for run_name, df in dfs.items():
        available = [k for k in cost_terms if k in df.columns]
        if len(available) == 0:
            continue

        plt.figure(figsize=(8, 4.5))
        dff = finite_df(df, ["diffusion_t"] + available)
        for key in available:
            g = summarize_by_t(dff, key)
            x = g["diffusion_t"].to_numpy()
            mean = g["mean"].to_numpy()
            std = g["std"].fillna(0.0).to_numpy()
            plt.plot(x, mean, marker="o", label=key)
            plt.fill_between(x, mean - std, mean + std, alpha=0.10)
        plt.gca().invert_xaxis()
        plt.title(f"Cost decomposition vs diffusion_t ({RUNS[run_name]['label']})")
        plt.xlabel("diffusion_t")
        plt.ylabel("cost")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        savefig(f"06_cost_decomposition_{run_name}.png")
        plt.close()

    # ------------------------------------------------------------
    # 7) argmin t distribution
    # ------------------------------------------------------------
    argmin_dict = {}
    for run_name, df in dfs.items():
        mins = []
        for _, g in df.groupby(planning_id_cols(df)):
            g = g.sort_values("diffusion_t")
            idx = g["J_total"].idxmin()
            mins.append(int(g.loc[idx, "diffusion_t"]))
        argmin_dict[run_name] = pd.Series(mins).value_counts().sort_index()
    plot_grouped_argmin_hist(argmin_dict, "07_compare_argmin_t_hist.png")

    # ------------------------------------------------------------
    # 8) gradient bin vs next-step delta cost
    # ------------------------------------------------------------
    for grad_key in grad_terms:
        binned_dict = {}
        all_ok = True
        for run_name, df in dfs.items():
            if grad_key not in df.columns:
                all_ok = False
                break
            dfd = build_next_step_delta(df, cost_key="J_total")
            dfd = finite_df(dfd, [grad_key, "delta_next_cost"])
            dfd["grad_bin"] = bin_by_quantiles(dfd[grad_key], q=10)
            gb = (
                dfd.groupby("grad_bin")
                .agg(
                    grad_mean=(grad_key, "mean"),
                    delta_mean=("delta_next_cost", "mean"),
                    delta_std=("delta_next_cost", "std"),
                    count=("delta_next_cost", "count"),
                )
                .reset_index()
                .sort_values("grad_mean")
            )
            binned_dict[run_name] = gb
        if all_ok:
            plot_binned_delta_vs_grad_compare(
                binned_dict, grad_key, f"08_compare_binned_delta_vs_{grad_key}.png"
            )

    # ------------------------------------------------------------
    # 9) Executed center-x costs
    # ------------------------------------------------------------
    plot_hist_compare(
        {k: v["mean_centerx_cost"] for k, v in result_stats.items()},
        title="Mean executed center-x cost",
        xlabel="mean cost",
        save_name="09_compare_exec_mean_centerx_hist.png",
    )
    plot_box_compare(
        {k: v["mean_centerx_cost"] for k, v in result_stats.items()},
        title="Mean executed center-x cost",
        ylabel="mean cost",
        save_name="10_compare_exec_mean_centerx_box.png",
    )
    plot_bar_with_error(
        {k: v["mean_centerx_cost"] for k, v in result_stats.items()},
        title="Mean executed center-x cost comparison",
        ylabel="mean cost",
        save_name="11_compare_exec_mean_centerx_bar.png",
    )

    plot_hist_compare(
        {k: v["cum_centerx_cost"] for k, v in result_stats.items()},
        title="Cumulative executed center-x cost",
        xlabel="cumulative cost",
        save_name="12_compare_exec_cum_centerx_hist.png",
    )
    plot_box_compare(
        {k: v["cum_centerx_cost"] for k, v in result_stats.items()},
        title="Cumulative executed center-x cost",
        ylabel="cumulative cost",
        save_name="13_compare_exec_cum_centerx_box.png",
    )
    plot_bar_with_error(
        {k: v["cum_centerx_cost"] for k, v in result_stats.items()},
        title="Cumulative executed center-x cost comparison",
        ylabel="cumulative cost",
        save_name="14_compare_exec_cum_centerx_bar.png",
    )

    plot_exec_cumulative_curve_compare(exec_cum_tensors, "15_compare_exec_cum_centerx_curve.png")

    # ------------------------------------------------------------
    # 10) Completion steps
    # ------------------------------------------------------------
    plot_hist_compare(
        {k: v["completion_steps"] for k, v in result_stats.items()},
        title="Completion steps distribution",
        xlabel="completion steps",
        save_name="16_compare_completion_steps_hist.png",
    )
    plot_box_compare(
        {k: v["completion_steps"] for k, v in result_stats.items()},
        title="Completion steps",
        ylabel="steps",
        save_name="17_compare_completion_steps_box.png",
    )
    plot_bar_with_error(
        {k: v["completion_steps"] for k, v in result_stats.items()},
        title="Mean completion steps comparison",
        ylabel="steps",
        save_name="18_compare_completion_steps_bar.png",
    )

    # ------------------------------------------------------------
    # 11) Completion time
    # ------------------------------------------------------------
    plot_hist_compare(
        {k: v["completion_time_sec"] for k, v in result_stats.items()},
        title=f"Completion time distribution (dt={ROLLOUT_DT:.3f}s per step)",
        xlabel="completion time (s)",
        save_name="19_compare_completion_time_hist.png",
    )
    plot_box_compare(
        {k: v["completion_time_sec"] for k, v in result_stats.items()},
        title=f"Completion time (dt={ROLLOUT_DT:.3f}s per step)",
        ylabel="time (s)",
        save_name="20_compare_completion_time_box.png",
    )
    plot_bar_with_error(
        {k: v["completion_time_sec"] for k, v in result_stats.items()},
        title="Mean completion time comparison",
        ylabel="time (s)",
        save_name="21_compare_completion_time_bar.png",
    )

    print("\nDone. All comparison figures saved in:")
    print(OUT_DIR)
    print(f"Using ROLLOUT_DT = {ROLLOUT_DT:.6f} s")


if __name__ == "__main__":
    main()