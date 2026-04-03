"""
Compare ACWI vs Baseline PPO training results.

Reads CSV logs from both ppo-curiosity/ (baseline) and ppo-acwi-curiosity/ (ACWI),
then generates comparison figures with mean reward curves and std bands.

Usage:
    python compare_results.py --env MiniGrid-DoorKey-8x8-v0
    python compare_results.py --env MiniGrid-DoorKey-8x8-v0 --methods icm count
    python compare_results.py --env MiniGrid-DoorKey-8x8-v0 --seeds 1:3 --smooth 30
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_LOG_DIR = os.path.join(SCRIPT_DIR, "ppo-curiosity", "logs")
ACWI_LOG_DIR = os.path.join(SCRIPT_DIR, "ppo-acwi-curiosity", "logs")

SUFFIX_MAP = {
    "none": "",
    "icm": "_ICM",
    "count": "_COUNT",
    "ride": "_RIDE",
}

METHOD_LABELS = {
    "none": "PPO (no curiosity)",
    "icm": "PPO + ICM",
    "count": "PPO + Count-Based",
    "ride": "PPO + RIDE",
}

COLORS = {
    "baseline": "#d62728",   # red
    "acwi": "#1f77b4",       # blue
}


def discover_csvs(log_root, env_name, method, seed_start, seed_end):
    """Find all CSV log files for a given method and seed range."""
    suffix = SUFFIX_MAP[method]
    env_dir = os.path.join(log_root, env_name)
    found = []
    for seed in range(seed_start, seed_end + 1):
        fname = f"PPO{suffix}_{env_name}_seed_{seed}.csv"
        fpath = os.path.join(env_dir, fname)
        if os.path.isfile(fpath):
            found.append((seed, fpath))
    return found


def load_and_align(csv_files, smooth_window):
    """
    Load multiple CSV files (each with columns: episode, timestep, reward),
    align them onto a common timestep grid, and compute mean/std.

    Returns (timesteps, mean_reward, std_reward, num_seeds) or None if no data.
    """
    if not csv_files:
        return None

    dataframes = []
    for seed, fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            if df.empty or "timestep" not in df.columns or "reward" not in df.columns:
                continue
            # Smooth the reward (simple rolling mean, no scipy needed)
            df["reward_smooth"] = (
                df["reward"]
                .rolling(window=smooth_window, min_periods=1)
                .mean()
            )
            # Use timestep as index for alignment
            df = df.set_index("timestep")["reward_smooth"]
            df.name = f"seed_{seed}"
            dataframes.append(df)
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")

    if not dataframes:
        return None

    # Align on common timestep index via outer join, then interpolate gaps
    combined = pd.concat(dataframes, axis=1)
    combined = combined.sort_index()
    combined = combined.interpolate(method="linear", limit_direction="forward")

    mean_reward = combined.mean(axis=1)
    std_reward = combined.std(axis=1).fillna(0)
    timesteps = combined.index.values

    return timesteps, mean_reward.values, std_reward.values, len(dataframes)


def plot_method_comparison(ax, env_name, method, seed_start, seed_end, smooth_window):
    """Plot baseline vs ACWI for a single method on the given axes."""

    baseline_csvs = discover_csvs(BASELINE_LOG_DIR, env_name, method, seed_start, seed_end)
    acwi_csvs = discover_csvs(ACWI_LOG_DIR, env_name, method, seed_start, seed_end)

    has_data = False

    # Plot baseline
    result = load_and_align(baseline_csvs, smooth_window)
    if result is not None:
        ts, mean, std, n = result
        label = f"Baseline {METHOD_LABELS[method]} ({n} seeds)"
        ax.plot(ts, mean, color=COLORS["baseline"], linewidth=1.5, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=COLORS["baseline"], alpha=0.15)
        has_data = True

    # Plot ACWI
    result = load_and_align(acwi_csvs, smooth_window)
    if result is not None:
        ts, mean, std, n = result
        label = f"ACWI {METHOD_LABELS[method]} ({n} seeds)"
        ax.plot(ts, mean, color=COLORS["acwi"], linewidth=1.5, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=COLORS["acwi"], alpha=0.15)
        has_data = True

    if not has_data:
        ax.text(
            0.5, 0.5, "No data available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="gray",
        )

    ax.set_title(f"{METHOD_LABELS[method]}", fontsize=12)
    ax.set_xlabel("Timesteps", fontsize=10)
    ax.set_ylabel("Average Reward", fontsize=10)
    if has_data:
        ax.legend(loc="upper left", fontsize=8)
    ax.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.3)


def generate_summary_table(env_name, methods, seed_start, seed_end, smooth_window, out_dir):
    """Generate a CSV table with final reward statistics for each method/variant."""
    rows = []
    for method in methods:
        for variant, log_root in [("Baseline", BASELINE_LOG_DIR), ("ACWI", ACWI_LOG_DIR)]:
            csvs = discover_csvs(log_root, env_name, method, seed_start, seed_end)
            result = load_and_align(csvs, smooth_window)
            if result is not None:
                ts, mean, std, n = result
                # Take the last 10% of timesteps as "final" performance
                cutoff = int(len(mean) * 0.9)
                final_mean = np.mean(mean[cutoff:])
                final_std = np.mean(std[cutoff:])
                rows.append({
                    "Method": METHOD_LABELS[method],
                    "Variant": variant,
                    "Seeds": n,
                    "Final Avg Reward": round(final_mean, 4),
                    "Final Std": round(final_std, 4),
                    "Max Timestep": int(ts[-1]),
                })

    if rows:
        df = pd.DataFrame(rows)
        table_path = os.path.join(out_dir, f"summary_{env_name}.csv")
        df.to_csv(table_path, index=False)
        print(f"\nSummary table saved: {table_path}")
        print(df.to_string(index=False))
    else:
        print("\nNo data found for summary table.")


def main():
    parser = argparse.ArgumentParser(description="Compare ACWI vs Baseline PPO results")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0",
                        help="Environment name")
    parser.add_argument("--methods", nargs="+", default=["none", "icm", "count", "ride"],
                        choices=["none", "icm", "count", "ride"],
                        help="Exploration methods to compare")
    parser.add_argument("--seeds", type=str, default="1:5",
                        help="Seed range as start:end (inclusive)")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Smoothing window size")
    parser.add_argument("--out_dir", type=str, default="comparison_figs",
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"],
                        help="Figure format")
    args = parser.parse_args()

    seed_start, seed_end = map(int, args.seeds.split(":"))
    env_name = args.env
    methods = args.methods

    # Output directory
    out_dir = os.path.join(SCRIPT_DIR, args.out_dir, env_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Environment: {env_name}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seed_start}-{seed_end}")
    print(f"Smooth window: {args.smooth}")
    print(f"Output: {out_dir}")

    # Report what data is available
    print("\n--- Data availability ---")
    for method in methods:
        bl = discover_csvs(BASELINE_LOG_DIR, env_name, method, seed_start, seed_end)
        ac = discover_csvs(ACWI_LOG_DIR, env_name, method, seed_start, seed_end)
        bl_seeds = [s for s, _ in bl]
        ac_seeds = [s for s, _ in ac]
        print(f"  {METHOD_LABELS[method]:25s}  Baseline seeds: {bl_seeds or 'none':>15}  ACWI seeds: {ac_seeds or 'none'}")

    # --- Per-method individual figures ---
    for method in methods:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_method_comparison(ax, env_name, method, seed_start, seed_end, args.smooth)
        fig.suptitle(f"{env_name} — Baseline vs ACWI", fontsize=14, y=1.02)
        fig.tight_layout()
        fpath = os.path.join(out_dir, f"compare_{method}.{args.format}")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {fpath}")

    # --- Combined grid figure ---
    n_methods = len(methods)
    if n_methods <= 2:
        nrows, ncols = 1, n_methods
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, method in enumerate(methods):
        plot_method_comparison(axes[i], env_name, method, seed_start, seed_end, args.smooth)

    # Hide unused subplots
    for j in range(n_methods, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{env_name} — Baseline vs ACWI Comparison", fontsize=14)
    fig.tight_layout()
    grid_path = os.path.join(out_dir, f"compare_all.{args.format}")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid figure: {grid_path}")

    # --- Summary table ---
    generate_summary_table(env_name, methods, seed_start, seed_end, args.smooth, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
