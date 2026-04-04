import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ── CONFIG ────────────────────────────────────────────────────────────────────
ENV_NAME   = 'MiniGrid-DoorKey-8x8-v0'
FIG_NUM    = 0          # increment to avoid overwriting
WINDOW     = 20         # smoothing window

# Methods to plot: (suffix_in_filename, display_label, color)
METHODS = [
    ('',      'PPO',   'gray'),
    ('_ICM',  'ICM',   'blue'),
    ('_COUNT','Count', 'green'),
    ('_RIDE', 'RIDE',  'orange'),
]
# ─────────────────────────────────────────────────────────────────────────────


def load_seeds(log_dir, env_name, suffix):
    """Load all seed CSVs for one method; return list of DataFrames."""
    pattern = os.path.join(log_dir, f'PPO{suffix}_{env_name}_seed_*.csv')
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            runs.append(df)
            print(f"  loaded: {os.path.basename(f)}")
        except Exception as e:
            print(f"  WARNING: could not load {f}: {e}")
    return runs


def average_runs(runs):
    """Average reward across seeds, aligned by row index (same timestep spacing)."""
    df = pd.concat(runs).groupby(level=0).mean().reset_index(drop=True)
    return df


def smooth(series, window):
    return series.rolling(window=window, win_type='triang', min_periods=1).mean()


def save_graph():
    log_dir   = os.path.join('logs', ENV_NAME)
    figs_dir  = os.path.join('figs', ENV_NAME)
    os.makedirs(figs_dir, exist_ok=True)
    fig_path  = os.path.join(figs_dir, f'PPO_{ENV_NAME}_fig_{FIG_NUM}.png')

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for suffix, label, color in METHODS:
        runs = load_seeds(log_dir, ENV_NAME, suffix)
        if not runs:
            print(f"  (no data for {label}, skipping)")
            continue

        data = average_runs(runs)
        data['smooth'] = smooth(data['reward'], WINDOW)
        data['var']    = smooth(data['reward'], 5)

        ax.plot(data['timestep'], data['smooth'], label=f'{label} ({len(runs)} seeds)',
                color=color, linewidth=1.5)
        ax.fill_between(data['timestep'], data['var'], data['smooth'],
                        color=color, alpha=0.12)
        plotted = True

    if not plotted:
        print("No data found. Train models first.")
        return

    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(ENV_NAME, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")
    plt.show()


if __name__ == '__main__':
    save_graph()
