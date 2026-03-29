import pickle
import numpy as np
import matplotlib.pyplot as plt


PKL_PATH = "guidance_saved_a15/all_results_a50.pkl"
SAVE_PATH = "guidance_saved_a15/alpha50_all_trajectories_with_success_rate.png"

OBSTACLES = np.array([
    [0.35, 0.26],
    [0.50, 0.26],
    [0.65, 0.26],
    [0.42, 0.08],
    [0.58, 0.08],
    [0.50, -0.10],
], dtype=np.float32)

OBS_RADIUS = 0.022

# 终点线，仅用于画图
FINISH_Y = 0.35
FINISH_X_MIN = 0.25
FINISH_X_MAX = 0.75


def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def check_collision(traj_xy, obstacles, obs_radius):
    """
    traj_xy: [T, 2]
    返回:
        collided: bool
        collided_obs_idx: 第一个碰撞到的障碍物索引，没有则为 -1
    """
    traj_xy = np.asarray(traj_xy, dtype=np.float32)

    for obs_idx, obs in enumerate(obstacles):
        d2 = np.sum((traj_xy - obs[None, :]) ** 2, axis=1)
        if np.any(d2 <= obs_radius ** 2):
            return True, obs_idx
    return False, -1


def draw_obstacles_and_finish(ax, obstacles, obs_radius):
    for ox, oy in obstacles:
        circ = plt.Circle((ox, oy), obs_radius, fill=False, color="red", linewidth=2)
        ax.add_patch(circ)
        ax.scatter([ox], [oy], s=45, c="red", alpha=0.35)

    ax.plot(
        [FINISH_X_MIN, FINISH_X_MAX],
        [FINISH_Y, FINISH_Y],
        "--",
        linewidth=3,
        alpha=0.8,
        color="green",
    )


def main():
    all_results = load_results(PKL_PATH)

    n_total = len(all_results)
    n_success = 0

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_obstacles_and_finish(ax, OBSTACLES, OBS_RADIUS)

    for result in all_results:
        traj_xy = np.asarray(result["traj_xy"], dtype=np.float32)
        if len(traj_xy) == 0:
            continue

        collided, collided_obs_idx = check_collision(traj_xy, OBSTACLES, OBS_RADIUS)

        # 只要碰撞就判失败；否则成功
        is_success = not collided
        if is_success:
            n_success += 1

        # 成功轨迹画蓝色，失败轨迹画橙红色
        color = "tab:blue" if is_success else "tab:orange"
        alpha = 0.22 if is_success else 0.7
        linewidth = 1.4 if is_success else 2.0

        ax.plot(
            traj_xy[:, 0],
            traj_xy[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        # 起点和终点
        ax.scatter(
            traj_xy[0, 0], traj_xy[0, 1],
            s=12, marker="s", color=color, alpha=min(alpha + 0.15, 1.0)
        )
        ax.scatter(
            traj_xy[-1, 0], traj_xy[-1, 1],
            s=18, marker="*", color=color, alpha=min(alpha + 0.2, 1.0)
        )

    success_rate = n_success / n_total if n_total > 0 else 0.0

    ax.set_title(f"Alpha=15: all trajectories (success rate = {success_rate:.2%})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.3, 0.5)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)

    # 图例
    success_proxy = plt.Line2D([0], [0], color="tab:blue", lw=2, label="success (no collision)")
    fail_proxy = plt.Line2D([0], [0], color="tab:orange", lw=2, label="failure (collision)")
    ax.legend(handles=[success_proxy, fail_proxy], loc="upper right")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=180)
    plt.show()

    print(f"Total trajectories: {n_total}")
    print(f"Success trajectories: {n_success}")
    print(f"Success rate: {success_rate:.4f}")
    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()