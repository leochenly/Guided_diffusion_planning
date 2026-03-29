import matplotlib.pyplot as plt
import matplotlib.image as mpimg

paths = {
    "alpha = 0":   "guided_mpc_new_x0.5_a0_clip10.png",
    #  "alpha = 1 with obs": "guided_mpc_obs30_a1.png",
    # "alpha = 0.1": "guided_mpc_x0.5_100_a0.1.png",
    # "alpha = 1":   "guided_mpc_x0.5_100_a1.png",
    # "alpha = 10":  "guided_mpc_new_x0.5_a10_w10_clip10.png",
    # "alpha = 15":  "guided_mpc_new_x0.5_a15_w10_clip10.png",
    # "alpha = 20":  "guided_mpc_new_x0.5_a20_w10_clip10.png",
    # "alpha = 50":  "guided_mpc_new_x0.5_a50_w10_clip10.png",
    "alpha = 1":  "guided_mpc_obs50_a1.png",
    "alpha = 10":  "guided_mpc_obs50_a10.png",
    

}

fig, axes = plt.subplots(1, 3, figsize=(12, 10))
axes = axes.ravel()

for ax, (title, p) in zip(axes, paths.items()):
    img = mpimg.imread(p)
    ax.imshow(img)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

plt.tight_layout()
# plt.savefig("alpha_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("obs_comparison.png", dpi=300, bbox_inches="tight")
plt.show()