"""
Compare ACWI vs Baseline for ICM and COUNT across 6 environments.
Layout: 6 rows x 2 cols (left=ICM, right=COUNT).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.7,
    "legend.edgecolor": "#cccccc",
})

ENVS = [
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-UnlockPickup-v0",
]
ENV_LABELS = {
    "MiniGrid-DoorKey-8x8-v0":       "DoorKey-8x8",
    "MiniGrid-Empty-16x16-v0":        "Empty-16x16",
    "MiniGrid-KeyCorridorS3R3-v0":    "KeyCorridor-S3R3",
    "MiniGrid-LavaCrossingS9N3-v0":   "LavaCrossing-S9N3",
    "MiniGrid-RedBlueDoors-8x8-v0":   "RedBlueDoors-8x8",
    "MiniGrid-UnlockPickup-v0":       "UnlockPickup",
}

SEEDS = [1, 2, 3, 4, 5]
WINDOW_SMOOTH = 20
WINDOW_VAR = 5

ACWI_ROOT     = "logs"
BASELINE_ROOT = os.path.join("..", "ppo-curiosity", "logs")

METHODS = {
    "ICM":   {"suffix": "_ICM",   "color_acwi": "#1f77b4", "color_base": "#aec7e8"},
    "COUNT": {"suffix": "_COUNT", "color_acwi": "#d62728", "color_base": "#f5b8a8"},
}


def load_seeds(log_dir, env, suffix):
    frames = []
    for seed in SEEDS:
        path = os.path.join(log_dir, env, f"PPO{suffix}_{env}_seed_{seed}.csv")
        try:
            df = pd.read_csv(path)
            frames.append(df.set_index("timestep")["reward"])
        except FileNotFoundError:
            pass
    return frames


def smooth(series, window, min_periods=1):
    return series.rolling(window=window, win_type="triang", min_periods=min_periods).mean()


def plot_pair(ax, env, suffix, color_acwi, color_base):
    configs = [
        (BASELINE_ROOT, "Baseline", color_base,  "-",  0.9),
        (ACWI_ROOT,     "ACWI",     color_acwi, "--",  1.0),
    ]
    for log_root, label, color, ls, alpha in configs:
        frames = load_seeds(log_root, env, suffix)
        if not frames:
            continue
        combined   = pd.concat(frames, axis=1)
        avg        = combined.mean(axis=1)
        std        = combined.std(axis=1)
        avg_smooth = smooth(avg, WINDOW_SMOOTH)
        avg_var    = smooth(avg, WINDOW_VAR)
        std_smooth = smooth(std, WINDOW_SMOOTH)
        ts = avg.index
        ax.plot(ts, avg_smooth, color=color, linewidth=2.0,
                linestyle=ls, alpha=alpha, label=f"{label} (n={len(frames)})")
        ax.fill_between(ts,
                        avg_var - std_smooth,
                        avg_var + std_smooth,
                        color=color, alpha=0.15)


def format_ax(ax, env, method_label):
    ax.set_title(f"{ENV_LABELS[env]}  ·  {method_label}", fontsize=11, pad=6)
    ax.set_xlabel("Timesteps", fontsize=9)
    ax.set_ylabel("Avg Reward", fontsize=9)
    ax.legend(loc="upper left")
    ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.4, alpha=0.3)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x < 1e6 else f"{x/1e6:.1f}M"
    ))


def main():
    n_envs    = len(ENVS)
    n_methods = len(METHODS)
    fig, axes = plt.subplots(n_envs, n_methods,
                             figsize=(13, 3.8 * n_envs),
                             squeeze=False)

    method_list = list(METHODS.items())

    for row, env in enumerate(ENVS):
        for col, (method_label, cfg) in enumerate(method_list):
            ax = axes[row][col]
            plot_pair(ax, env, cfg["suffix"], cfg["color_acwi"], cfg["color_base"])
            format_ax(ax, env, method_label)

    fig.suptitle("ACWI vs Baseline — 6 Environments", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout(h_pad=3.5, w_pad=3.0)

    figs_dir = "figs"
    os.makedirs(figs_dir, exist_ok=True)
    save_path = os.path.join(figs_dir, "compare_acwi_vs_baseline_all_envs.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
