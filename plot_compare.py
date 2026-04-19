"""
Compare baseline (ppo-curiosity) vs ACWI (ppo-acwi-curiosity) for ICM and COUNT.
Run from the repo root:  python plot_compare.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
ENVS = [
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-Empty-16x16-v0',
    'MiniGrid-KeyCorridorS3R3-v0',
    'MiniGrid-LavaCrossingS9N3-v0',
    'MiniGrid-RedBlueDoors-8x8-v0',
    'MiniGrid-UnlockPickup-v0',
]

WINDOW = 20  # smoothing window (triangular)

SERIES = [
    # (sub_dir,            suffix,   display_label,       color,         linestyle)
    ('ppo-curiosity',      '_ICM',   'ICM (baseline)',    'steelblue',   '-'),
    ('ppo-acwi-curiosity', '_ICM',   'ACWI-ICM',          'darkorange',  '--'),
    ('ppo-curiosity',      '_COUNT', 'COUNT (baseline)',  'seagreen',    '-'),
    ('ppo-acwi-curiosity', '_COUNT', 'ACWI-COUNT',        'crimson',     '--'),
]

OUT_DIR = os.path.join('figs', 'compare')
# ─────────────────────────────────────────────────────────────────────────────


def load_seeds(log_dir, env_name, suffix):
    pattern = os.path.join(log_dir, 'logs', env_name, f'PPO{suffix}_{env_name}_seed_*.csv')
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            runs.append(df)
        except Exception as e:
            print(f"  WARNING: {f}: {e}")
    return runs


def average_runs(runs):
    return pd.concat(runs).groupby(level=0).mean().reset_index(drop=True)


def smooth(series, window):
    return series.rolling(window=window, win_type='triang', min_periods=1).mean()


def plot_all_envs():
    os.makedirs(OUT_DIR, exist_ok=True)

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
    fig.suptitle('Baseline vs ACWI (ICM & COUNT) — All Environments',
                 fontsize=15, fontweight='bold', y=1.01)

    for idx, env in enumerate(ENVS):
        ax = axes[idx // ncols][idx % ncols]
        plotted = False

        for sub_dir, suffix, label, color, ls in SERIES:
            runs = load_seeds(sub_dir, env, suffix)
            if not runs:
                print(f"  (no data: {sub_dir} / {env} / {suffix})")
                continue

            data = average_runs(runs)
            data['smooth'] = smooth(data['reward'], WINDOW)
            data['band'] = smooth(data['reward'], 5)

            ax.plot(data['timestep'], data['smooth'],
                    label=f'{label} (n={len(runs)})',
                    color=color, linestyle=ls, linewidth=2)
            ax.fill_between(data['timestep'], data['band'], data['smooth'],
                            color=color, alpha=0.10)
            plotted = True

        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        ax.set_title(env_short, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timesteps', fontsize=9)
        ax.set_ylabel('Avg Reward', fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        if not plotted:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'all_envs_baseline_vs_acwi.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


def plot_per_env():
    """Save individual high-res plots per environment (2-panel: ICM | COUNT)."""
    for env in ENVS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        fig.suptitle(f'{env_short}: Baseline vs ACWI', fontsize=13, fontweight='bold')

        method_groups = [
            ('ICM',   [s for s in SERIES if '_ICM' in s[1]]),
            ('COUNT', [s for s in SERIES if '_COUNT' in s[1]]),
        ]

        for ax, (method_name, series_subset) in zip(axes, method_groups):
            plotted = False
            for sub_dir, suffix, label, color, ls in series_subset:
                runs = load_seeds(sub_dir, env, suffix)
                if not runs:
                    continue
                data = average_runs(runs)
                data['smooth'] = smooth(data['reward'], WINDOW)
                data['band'] = smooth(data['reward'], 5)
                ax.plot(data['timestep'], data['smooth'],
                        label=f'{label} (n={len(runs)})',
                        color=color, linestyle=ls, linewidth=2)
                ax.fill_between(data['timestep'], data['band'], data['smooth'],
                                color=color, alpha=0.12)
                plotted = True

            ax.set_title(method_name, fontsize=11)
            ax.set_xlabel('Timesteps', fontsize=10)
            ax.set_ylabel('Average Reward', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            if not plotted:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f'{env}_baseline_vs_acwi.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


def print_summary():
    """Print final-timestep average reward for each series/env."""
    print("\n=== Summary: mean reward at final timestep ===")
    header = f"{'Env':<35} {'Series':<22} {'Seeds':>5} {'Final Reward':>13}"
    print(header)
    print("-" * len(header))
    for env in ENVS:
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        for sub_dir, suffix, label, _, _ in SERIES:
            runs = load_seeds(sub_dir, env, suffix)
            if not runs:
                continue
            data = average_runs(runs)
            final = data['reward'].iloc[-10:].mean()
            print(f"{env_short:<35} {label:<22} {len(runs):>5} {final:>13.4f}")
        print()


if __name__ == '__main__':
    print("=== Plotting all environments (grid) ===")
    plot_all_envs()
    print("\n=== Saving individual plots per environment ===")
    plot_per_env()
    print_summary()
    print("\nDone.")
