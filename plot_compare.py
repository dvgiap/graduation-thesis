"""
Compare baseline (ppo-curiosity) vs ACWI (ppo-acwi-curiosity) on the same graph.
Run from the repo root:  python plot_compare.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ── CONFIG ────────────────────────────────────────────────────────────────────
ENV_NAME  = 'MiniGrid-DoorKey-8x8-v0'
FIG_NUM   = 0
WINDOW    = 20   # smoothing window (triangular)

BASELINE_DIR = os.path.join('ppo-curiosity',      'logs', ENV_NAME)
ACWI_DIR     = os.path.join('ppo-acwi-curiosity', 'logs', ENV_NAME)

# (log_dir, suffix_in_filename, display_label, color, linestyle)
SERIES = [
    (BASELINE_DIR, '',      'PPO',         'gray',   '-'),
    (BASELINE_DIR, '_ICM',  'ICM',         'blue',   '-'),
    (BASELINE_DIR, '_COUNT','Count',       'green',  '-'),
    (BASELINE_DIR, '_RIDE', 'RIDE',        'orange', '-'),
    (ACWI_DIR,     '_ICM',  'ACWI-ICM',   'blue',   '--'),
    (ACWI_DIR,     '_COUNT','ACWI-Count',  'green',  '--'),
    (ACWI_DIR,     '_RIDE', 'ACWI-RIDE',  'orange', '--'),
]
# ─────────────────────────────────────────────────────────────────────────────


def load_seeds(log_dir, env_name, suffix):
    pattern = os.path.join(log_dir, f'PPO{suffix}_{env_name}_seed_*.csv')
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            runs.append(df)
            print(f"  loaded: {os.path.basename(f)}")
        except Exception as e:
            print(f"  WARNING: {f}: {e}")
    return runs


def average_runs(runs):
    df = pd.concat(runs).groupby(level=0).mean().reset_index(drop=True)
    return df


def smooth(series, window):
    return series.rolling(window=window, win_type='triang', min_periods=1).mean()


def save_graph():
    figs_dir = os.path.join('figs', ENV_NAME)
    os.makedirs(figs_dir, exist_ok=True)
    fig_path = os.path.join(figs_dir, f'compare_{ENV_NAME}_fig_{FIG_NUM}.png')

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False

    for log_dir, suffix, label, color, ls in SERIES:
        runs = load_seeds(log_dir, ENV_NAME, suffix)
        if not runs:
            print(f"  (no data for {label}, skipping)")
            continue

        data = average_runs(runs)
        data['smooth'] = smooth(data['reward'], WINDOW)
        data['var']    = smooth(data['reward'], 5)

        ax.plot(data['timestep'], data['smooth'],
                label=f'{label} (n={len(runs)})',
                color=color, linestyle=ls, linewidth=1.8)
        ax.fill_between(data['timestep'], data['var'], data['smooth'],
                        color=color, alpha=0.08)
        plotted = True

    if not plotted:
        print("No data found. Train models first.")
        return

    # Legend: solid = baseline, dashed = ACWI
    ax.plot([], [], color='black', linestyle='-',  linewidth=1.5, label='── Baseline')
    ax.plot([], [], color='black', linestyle='--', linewidth=1.5, label='-- ACWI')

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(f'{ENV_NAME}: Baseline vs ACWI', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")
    plt.show()


if __name__ == '__main__':
    save_graph()
