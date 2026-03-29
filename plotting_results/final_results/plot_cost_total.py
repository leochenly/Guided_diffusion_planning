import pickle
import numpy as np
import matplotlib.pyplot as plt


CENTER_X = 0.5


def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_centerx_exec_costs(all_results, center_x=0.5):
    """
    对每条 episode 计算：
      1) cumulative center-x cost
      2) mean center-x cost
    """
    cum_costs = []
    mean_costs = []

    for result in all_results:
        traj_xy = np.asarray(result["traj_xy"], dtype=np.float32)   # [T,2]
        x = traj_xy[:, 0]
        per_step = (x - center_x) ** 2

        cum_costs.append(np.sum(per_step))
        mean_costs.append(np.mean(per_step))

    return np.asarray(cum_costs), np.asarray(mean_costs)


def print_summary(name, arr):
    print(f"{name}:")
    print(f"  mean   = {np.mean(arr):.6f}")
    print(f"  std    = {np.std(arr):.6f}")
    print(f"  median = {np.median(arr):.6f}")
    print()


def plot_hist_compare(arr_a, arr_b, label_a, label_b, title, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.hist(arr_a, bins=20, alpha=0.5, label=label_a)
    plt.hist(arr_b, bins=20, alpha=0.5, label=label_b)
    plt.title(title)
    plt.xlabel("cost")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def main():
    all_results_a15 = load_results("guidance_saved_a15/all_results.pkl")
    all_results_a0  = load_results("guidance_saved_a15/all_results_a0.pkl")

    cum15, mean15 = compute_centerx_exec_costs(all_results_a15, center_x=CENTER_X)
    cum0,  mean0  = compute_centerx_exec_costs(all_results_a0, center_x=CENTER_X)

    print_summary("alpha=15 cumulative executed centerx cost", cum15)
    print_summary("alpha=0 cumulative executed centerx cost", cum0)

    print_summary("alpha=15 mean executed centerx cost", mean15)
    print_summary("alpha=0 mean executed centerx cost", mean0)

    plot_hist_compare(
        cum15, cum0,
        label_a="alpha=15",
        label_b="alpha=0",
        title="Cumulative executed center-x cost",
        save_path="guidance_saved_a15/hist_cumulative_centerx_cost_alpha15_vs_alpha0.png",
    )

    plot_hist_compare(
        mean15, mean0,
        label_a="alpha=15",
        label_b="alpha=0",
        title="Mean executed center-x cost",
        save_path="guidance_saved_a15/hist_mean_centerx_cost_alpha15_vs_alpha0.png",
    )


if __name__ == "__main__":
    main()