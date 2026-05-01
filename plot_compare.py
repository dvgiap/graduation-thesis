"""
Compare PPO baseline, fixed-beta sweep, and CARE for ICM, COUNT, and RIDE.

Each module-level panel shows 6 curves:
  - PPO (no intrinsic)                            — control
  - 4 fixed-beta variants (beta in 0.001..0.1)    — manual sweep
  - CARE (adaptive beta_psi)                      — proposed method

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

# Three intrinsic reward modules evaluated. Each gets its own dedicated panel.
MODULES = [
    # (display_name, suffix_in_filename, care_color)
    ('ICM',   '_ICM',   'darkorange'),
    ('COUNT', '_COUNT', 'crimson'),
    ('RIDE',  '_RIDE',  'goldenrod'),
]

# PPO no-intrinsic control (B_0). Lives under ppo-curiosity/ as historical reference.
PPO_BASELINE = ('ppo-curiosity', '', 'PPO (no intrinsic)', 'gray', ':')

# Fixed-beta sweep: 4 values bracketing the literature-reported optima
# (beta ~ 0.005 for count, beta ~ 0.05 for ICM in sparse MiniGrid).
FIXED_BETA_VALUES = [0.001, 0.005, 0.05, 0.1]
FB_COLORS = ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']  # gradient red

OUT_DIR = os.path.join('figs', 'compare')
# ─────────────────────────────────────────────────────────────────────────────


def build_panel_series(module_suffix, care_color):
    """Return the 6 series shown on one module panel.

    Layout: [PPO baseline] + 4 fixed-beta + [CARE].
    Each entry: (sub_dir, suffix, display_label, color, linestyle).
    """
    module_name = module_suffix.lstrip('_')
    fixed_beta = [
        ('ppo-care-curiosity', f'{module_suffix}_FB{v}',
         f'{module_name} beta={v}', FB_COLORS[i], '-')
        for i, v in enumerate(FIXED_BETA_VALUES)
    ]
    care = ('ppo-care-curiosity', f'{module_suffix}_CARE',
            f'CARE-{module_name}', care_color, '--')
    return [PPO_BASELINE] + fixed_beta + [care]


def load_seeds(log_dir, env_name, suffix):
    """Glob seeded CSV files for a given (sub_dir, suffix, env)."""
    if suffix == '':
        # PPO no-intrinsic control: legacy naming PPO_{env}_seed_N.csv
        pattern = os.path.join(log_dir, 'logs', env_name, f'PPO_{env_name}_seed_*.csv')
    else:
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


def _draw_panel(ax, env, panel_series, title, fontsize_legend=8):
    """Draw one panel of curves on ax. Returns True if any curve was plotted."""
    plotted = False
    for sub_dir, suffix, label, color, ls in panel_series:
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

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Timesteps', fontsize=10)
    ax.set_ylabel('Average Reward', fontsize=10)
    ax.legend(fontsize=fontsize_legend, loc='best')
    ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    if not plotted:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
    return plotted


def plot_per_env():
    """One figure per environment, 3 sub-panels (ICM | COUNT | RIDE).
    Each sub-panel shows: PPO baseline + 4 fixed-beta + CARE = 6 curves.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    for env in ENVS:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        fig.suptitle(f'{env_short}: PPO vs Fixed-beta sweep vs CARE',
                     fontsize=13, fontweight='bold')

        for ax, (module_name, module_suffix, care_color) in zip(axes, MODULES):
            panel = build_panel_series(module_suffix, care_color)
            _draw_panel(ax, env, panel, title=module_name, fontsize_legend=8)

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f'{env}_per_module.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


def plot_per_method():
    """One figure per intrinsic module, 6 sub-panels (1 per env).
    Each sub-panel shows: PPO baseline + 4 fixed-beta + CARE = 6 curves.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ncols = 3
    nrows = 2
    for module_name, module_suffix, care_color in MODULES:
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11))
        fig.suptitle(f'{module_name}: PPO vs Fixed-beta sweep vs CARE — All Environments',
                     fontsize=15, fontweight='bold', y=1.00)

        panel = build_panel_series(module_suffix, care_color)
        for idx, env in enumerate(ENVS):
            ax = axes[idx // ncols][idx % ncols]
            env_short = env.replace('MiniGrid-', '').replace('-v0', '')
            _draw_panel(ax, env, panel, title=env_short, fontsize_legend=8)

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f'all_envs_{module_name.lower()}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


def print_summary():
    """Print final-timestep average reward for every series in every env."""
    print("\n=== Summary: mean reward at final timestep ===")
    header = f"{'Env':<35} {'Series':<26} {'Seeds':>5} {'Final Reward':>13}"
    print(header)
    print("-" * len(header))

    all_series: list = [PPO_BASELINE]
    for _, module_suffix, care_color in MODULES:
        all_series.extend(build_panel_series(module_suffix, care_color)[1:])  # drop dup baseline

    for env in ENVS:
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        for sub_dir, suffix, label, _, _ in all_series:
            runs = load_seeds(sub_dir, env, suffix)
            if not runs:
                continue
            data = average_runs(runs)
            final = data['reward'].iloc[-10:].mean()
            print(f"{env_short:<35} {label:<26} {len(runs):>5} {final:>13.4f}")
        print()


if __name__ == '__main__':
    print("=== Per environment: ICM | COUNT | RIDE ===")
    plot_per_env()
    print("\n=== Per method: 6 environments grid ===")
    plot_per_method()
    print_summary()
    print("\nDone.")
