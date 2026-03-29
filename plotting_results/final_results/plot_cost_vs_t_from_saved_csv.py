import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare(csv_path, cost_key="J_total"):
    df = pd.read_csv(csv_path)

    # 只保留必要列
    needed = ["episode_idx", "planning_idx", "diffusion_t", cost_key]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[needed].copy()
    df = df[np.isfinite(df[cost_key])]
    df = df[np.isfinite(df["diffusion_t"])]

    df["diffusion_t"] = df["diffusion_t"].astype(int)
    df["episode_idx"] = df["episode_idx"].astype(int)
    df["planning_idx"] = df["planning_idx"].astype(int)

    return df


def build_normalized_df(df, cost_key="J_total", baseline_t=15, eps=1e-8):
    """
    对每条 planning（episode_idx, planning_idx）：
      baseline = J(t=baseline_t)
      delta_J(t) = J(t) - baseline
      relative_J(t) = J(t) / baseline
    """
    group_cols = ["episode_idx", "planning_idx"]

    rows = []

    for (ep, p), g in df.groupby(group_cols):
        g = g.sort_values("diffusion_t").copy()

        base_rows = g[g["diffusion_t"] == baseline_t]
        if len(base_rows) == 0:
            continue

        # 如果有多个同 t 记录，取均值（正常一般只有一个）
        baseline = base_rows[cost_key].mean()

        # baseline 过小会导致 relative 爆炸，做个保护
        denom = baseline if abs(baseline) > eps else eps

        g["baseline_cost"] = baseline
        g["delta_cost"] = g[cost_key] - baseline
        g["relative_cost"] = g[cost_key] / denom

        rows.append(g)

    if len(rows) == 0:
        raise ValueError("No valid planning groups found with baseline_t present.")

    out = pd.concat(rows, axis=0, ignore_index=True)
    return out


def summarize_by_t(df_norm, value_key):
    grouped = (
        df_norm.groupby("diffusion_t")[value_key]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("diffusion_t")
    )
    return grouped


def plot_mean_std(grouped, value_key, title, ylabel, save_path=None, reverse_x=True, hline=None):
    t = grouped["diffusion_t"].to_numpy()
    mean = grouped["mean"].to_numpy()
    std = grouped["std"].fillna(0.0).to_numpy()
    count = grouped["count"].to_numpy()

    plt.figure(figsize=(7, 4))
    plt.plot(t, mean, marker="o", label=f"{value_key} mean")
    plt.fill_between(t, mean - std, mean + std, alpha=0.25, label="±1 std")

    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.0, alpha=0.8)

    plt.xlabel("diffusion_t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if reverse_x:
        plt.gca().invert_xaxis()

    if save_path:
        plt.savefig(save_path, dpi=160)

    plt.show()

    print(grouped)


def main():
    csv_path = "guidance_saved_a15/guidance_logs.csv"
    cost_key = "J_total"
    baseline_t = 15

    df = load_and_prepare(csv_path, cost_key=cost_key)
    df_norm = build_normalized_df(df, cost_key=cost_key, baseline_t=baseline_t)

    # 1) delta 图
    grouped_delta = summarize_by_t(df_norm, "delta_cost")
    plot_mean_std(
        grouped_delta,
        value_key="delta_cost",
        title=f"ΔJ(t) = J(t) - J({baseline_t})  (all planning, mean ± std)",
        ylabel="delta cost",
        save_path="guidance_saved_a15/delta_cost_vs_diffusion_t_mean_std.png",
        reverse_x=True,
        hline=0.0,
    )

    # 2) relative 图
    grouped_rel = summarize_by_t(df_norm, "relative_cost")
    plot_mean_std(
        grouped_rel,
        value_key="relative_cost",
        title=f"Relative J(t) = J(t) / J({baseline_t})  (all planning, mean ± std)",
        ylabel="relative cost",
        save_path="guidance_saved_a15/relative_cost_vs_diffusion_t_mean_std.png",
        reverse_x=True,
        hline=1.0,
    )


if __name__ == "__main__":
    main()