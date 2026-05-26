"""
Compare CARE vs PPO baseline vs Fixed-beta sweep on MiniGrid.

Renders 5 families of figures into ppo-care-curiosity/figs/compare/:
  1. Reward curves    — per-env (3 modules) and per-method (6 envs).
  2. Aggregate        — sample efficiency + normalized final reward bar charts.
  3. Performance profile (Agarwal et al. NeurIPS 2021).
  4. CARE dynamics    — mean beta(s) over time + extrinsic/intrinsic mix.
  5. Beta histogram   — distribution of beta(s) at end of training.

Run:  cd ppo-care-curiosity && python plot_compare.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Windows console default cp1252 chokes on Greek letters; force UTF-8 if possible.
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGS_ROOT   = os.path.join(SCRIPT_DIR, 'logs')      # reward CSVs
MODELS_ROOT = os.path.join(SCRIPT_DIR, 'models')    # .meta.npz / .samples.npz
OUT_DIR     = os.path.join(SCRIPT_DIR, 'figs', 'compare')

ENVS = [
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-Empty-16x16-v0',
    'MiniGrid-KeyCorridorS3R3-v0',
    'MiniGrid-LavaCrossingS9N3-v0',
    'MiniGrid-RedBlueDoors-8x8-v0',
    'MiniGrid-UnlockPickup-v0',
]

# (display_name, suffix_in_filename, care_color)
MODULES = [
    ('ICM',   '_ICM',   'darkorange'),
    ('COUNT', '_COUNT', 'crimson'),
    ('RIDE',  '_RIDE',  'goldenrod'),
]

WINDOW = 20
FIXED_BETA_VALUES = [0.0005, 0.001, 0.005, 0.01, 0.05]
FB_COLORS = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']  # gradient red (5 shades)
PPO_COLOR = 'gray'
BETA_0    = 1e-4     # CARE cold-start prior — √(β_min·β_max) = √(1e-8·1) = 1e-4

# Hardcoded budget guess for sample-efficiency penalty when threshold is never reached.
DEFAULT_MAX_STEPS = 1_000_000
# ─────────────────────────────────────────────────────────────────────────────


# ── DATA LOADERS ─────────────────────────────────────────────────────────────
def load_seeds(env, suffix):
    """Glob seeded reward CSVs for (env, suffix). suffix='' = PPO no-intrinsic."""
    if suffix == '':
        pat = os.path.join(LOGS_ROOT, env, f'PPO_{env}_seed_*.csv')
    else:
        pat = os.path.join(LOGS_ROOT, env, f'PPO{suffix}_{env}_seed_*.csv')
    return [pd.read_csv(f) for f in sorted(glob.glob(pat))]


def load_meta(env, module_suffix):
    """Load TrainingLogger .meta.npz for all CARE seeds of a module."""
    pat = os.path.join(MODELS_ROOT, env,
                       f'PPO{module_suffix}_CARE_{env}_seed_*.pth.meta.npz')
    return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(pat))]


def load_samples(env, module_suffix):
    """Load .samples.npz (state, beta(s)) snapshots for all CARE seeds."""
    pat = os.path.join(MODELS_ROOT, env,
                       f'PPO{module_suffix}_CARE_{env}_seed_*.pth.samples.npz')
    return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(pat))]


# ── SERIES / CURVE HELPERS ───────────────────────────────────────────────────
def build_panel_series(module_suffix, care_color):
    """Return 6 series for one (module) panel: PPO + 4 FB + CARE.

    Each entry: (suffix, label, color, linestyle).
    """
    module = module_suffix.lstrip('_')
    fixed_beta = [
        (f'{module_suffix}_FB{v}', f'{module} β={v}', FB_COLORS[i], '-')
        for i, v in enumerate(FIXED_BETA_VALUES)
    ]
    care = (f'{module_suffix}_CARE', f'CARE-{module}', care_color, '--')
    ppo  = ('', 'PPO (no intrinsic)', PPO_COLOR, ':')
    return [ppo] + fixed_beta + [care]


def average_runs(runs):
    return pd.concat(runs).groupby(level=0).mean().reset_index(drop=True)


def smooth(series, window):
    return series.rolling(window=window, win_type='triang', min_periods=1).mean()


def _draw_panel(ax, env, panel_series, title, fontsize_legend=8):
    plotted = False
    for suffix, label, color, ls in panel_series:
        runs = load_seeds(env, suffix)
        if not runs:
            continue
        data = average_runs(runs)
        data['smooth'] = smooth(data['reward'], WINDOW)
        data['band']   = smooth(data['reward'], 5)
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


# ── 1. REWARD CURVES ─────────────────────────────────────────────────────────
def plot_per_env():
    """One figure per env: 1×3 panel (ICM | COUNT | RIDE), each with 6 curves."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for env in ENVS:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        fig.suptitle(f'{env_short}: PPO vs Fixed-β sweep vs CARE',
                     fontsize=13, fontweight='bold')
        for ax, (mod, sfx, color) in zip(axes, MODULES):
            _draw_panel(ax, env, build_panel_series(sfx, color), title=mod)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'{env}_per_module.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved {out}")
        plt.close(fig)


def plot_per_method():
    """One figure per module: 2×3 grid envs, each panel with 6 curves."""
    os.makedirs(OUT_DIR, exist_ok=True)
    nrows, ncols = 2, 3
    for mod, sfx, color in MODULES:
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11))
        fig.suptitle(f'{mod}: PPO vs Fixed-β sweep vs CARE — All Environments',
                     fontsize=15, fontweight='bold', y=1.00)
        panel = build_panel_series(sfx, color)
        for idx, env in enumerate(ENVS):
            ax = axes[idx // ncols][idx % ncols]
            env_short = env.replace('MiniGrid-', '').replace('-v0', '')
            _draw_panel(ax, env, panel, title=env_short)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'all_envs_{mod.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved {out}")
        plt.close(fig)


# ── METHOD ENUMERATION (used by aggregate / profile / summary) ───────────────
def all_methods():
    """Return ordered list of every method to compare across plots.

    Each entry: (suffix, label, color).
    """
    methods = [('', 'PPO', PPO_COLOR)]
    for mod, sfx, care_color in MODULES:
        for i, v in enumerate(FIXED_BETA_VALUES):
            methods.append((f'{sfx}_FB{v}', f'{mod}-FB{v}', FB_COLORS[i]))
        methods.append((f'{sfx}_CARE', f'CARE-{mod}', care_color))
    return methods


def per_seed_finals(env, suffix, last_n=10):
    """Return array of per-seed final rewards (mean of last_n timesteps)."""
    runs = load_seeds(env, suffix)
    if not runs:
        return np.array([])
    return np.array([r['reward'].iloc[-last_n:].mean() for r in runs])


def env_max_reward(env):
    """Best per-seed final reward observed across all methods on env (normalizer)."""
    best = 0.0
    for suffix, _, _ in all_methods():
        finals = per_seed_finals(env, suffix)
        if finals.size:
            best = max(best, float(finals.max()))
    return max(best, 1e-6)  # avoid div-by-zero


def first_threshold_step(run_df, threshold, smooth_window=WINDOW, fallback=None):
    """First timestep at which smoothed reward crosses threshold; fallback if never."""
    smoothed = smooth(run_df['reward'], smooth_window).values
    idx = int(np.argmax(smoothed >= threshold))
    if smoothed[idx] >= threshold:
        return int(run_df['timestep'].iloc[idx])
    return int(fallback if fallback is not None else run_df['timestep'].iloc[-1])


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Return (mean, lo, hi) 95% CI by bootstrap. (nan,nan,nan) if empty."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(values.mean()), float(lo), float(hi))


# ── 2. AGGREGATE: SAMPLE EFFICIENCY + FINAL REWARD ───────────────────────────
def plot_aggregate():
    """Two bar charts side by side aggregating across all envs+seeds."""
    os.makedirs(OUT_DIR, exist_ok=True)
    methods = all_methods()

    env_max  = {env: env_max_reward(env) for env in ENVS}
    env_thr  = {env: 0.5 * env_max[env] for env in ENVS}
    # Per-env training budget = max final timestep observed across methods/seeds
    env_budget = {}
    for env in ENVS:
        budgets = []
        for suffix, _, _ in methods:
            for r in load_seeds(env, suffix):
                budgets.append(int(r['timestep'].iloc[-1]))
        env_budget[env] = max(budgets) if budgets else DEFAULT_MAX_STEPS

    eff_stats, fin_stats = [], []
    for suffix, label, color in methods:
        eff_pool, fin_pool = [], []
        for env in ENVS:
            runs = load_seeds(env, suffix)
            if not runs:
                continue
            for r in runs:
                eff_pool.append(first_threshold_step(
                    r, env_thr[env], fallback=env_budget[env]))
                final_raw = r['reward'].iloc[-10:].mean()
                fin_pool.append(final_raw / env_max[env])
        eff_stats.append((label, color, *bootstrap_ci(eff_pool)))
        fin_stats.append((label, color, *bootstrap_ci(fin_pool)))

    fig, (axE, axF) = plt.subplots(1, 2, figsize=(22, 7))

    def _bar(ax, stats, ylabel, title, scale=1.0):
        labels = [s[0] for s in stats]
        colors = [s[1] for s in stats]
        means  = np.array([s[2] for s in stats]) * scale
        los    = np.array([s[3] for s in stats]) * scale
        his    = np.array([s[4] for s in stats]) * scale
        x = np.arange(len(labels))
        yerr = np.vstack([means - los, his - means])
        ax.bar(x, means, color=colors, edgecolor='black', linewidth=0.5,
               yerr=yerr, capsize=3, error_kw={'elinewidth': 1, 'alpha': 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    _bar(axE, eff_stats,
         ylabel='Steps to reach 50% of env-max reward (×1e6)',
         title='Sample efficiency (lower = faster)',
         scale=1e-6)
    _bar(axF, fin_stats,
         ylabel='Final reward (normalized to env-max)',
         title='Final performance (higher = better)')

    fig.suptitle('Aggregate across 6 envs × 5 seeds — 95% bootstrap CI',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'aggregate_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  saved {out}")

    # Per-panel exports for slides (larger when shown alone)
    renderer = fig.canvas.get_renderer()
    for ax, name in [(axE, 'aggregate_efficiency.png'),
                     (axF, 'aggregate_final.png')]:
        extent = ax.get_tightbbox(renderer).transformed(
            fig.dpi_scale_trans.inverted())
        out_panel = os.path.join(OUT_DIR, name)
        fig.savefig(out_panel, dpi=150,
                    bbox_inches=extent.expanded(1.01, 1.15))
        print(f"  saved {out_panel}")
    plt.close(fig)


# ── 3. PERFORMANCE PROFILE (Agarwal et al.) ──────────────────────────────────
def plot_performance_profile():
    """One figure per module: P(τ) = fraction(env,seed) with norm_final ≥ τ."""
    os.makedirs(OUT_DIR, exist_ok=True)
    taus = np.linspace(0, 1, 101)
    env_max = {env: env_max_reward(env) for env in ENVS}

    for mod, mod_sfx, care_color in MODULES:
        series = [('', 'PPO', PPO_COLOR, ':')]
        for i, v in enumerate(FIXED_BETA_VALUES):
            series.append((f'{mod_sfx}_FB{v}', f'{mod}-FB{v}', FB_COLORS[i], '-'))
        series.append((f'{mod_sfx}_CARE', f'CARE-{mod}', care_color, '--'))

        fig, ax = plt.subplots(figsize=(9, 6))
        for suffix, label, color, ls in series:
            normed = []
            for env in ENVS:
                finals = per_seed_finals(env, suffix)
                if finals.size == 0:
                    continue
                normed.extend(list(finals / env_max[env]))
            if not normed:
                continue
            normed = np.asarray(normed)
            profile = np.array([(normed >= t).mean() for t in taus])
            ax.plot(taus, profile, label=f'{label} (n={normed.size})',
                    color=color, linestyle=ls, linewidth=2)

        ax.set_xlabel('Normalized reward threshold τ', fontsize=11)
        ax.set_ylabel('Fraction of (env, seed) runs with final reward ≥ τ',
                      fontsize=11)
        ax.set_title(f'Performance profile — {mod}',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'profile_{mod.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved {out}")
        plt.close(fig)


# ── 4. CARE DYNAMICS: β(s) over time + extrinsic/intrinsic mix ───────────────
def _stack_seed_metric(metas, key):
    """Stack a per-update scalar across seeds, truncated to shortest length."""
    arrs = [m[key] for m in metas if key in m.files]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    return np.stack([a[:n] for a in arrs], axis=0)  # (S, n)


def plot_care_dynamics():
    """One figure per module: 6 envs × 2 panels (β over time | reward mix)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    nrows = len(ENVS)
    for mod, mod_sfx, care_color in MODULES:
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 3.0 * nrows))
        fig.suptitle(f'CARE dynamics — {mod}',
                     fontsize=15, fontweight='bold', y=1.00)
        for r, env in enumerate(ENVS):
            ax_b, ax_r = axes[r, 0], axes[r, 1]
            env_short = env.replace('MiniGrid-', '').replace('-v0', '')
            metas = load_meta(env, mod_sfx)

            # Panel left: mean β across updates
            beta_stack = _stack_seed_metric(metas, 'beta')
            if beta_stack is not None:
                steps = np.arange(beta_stack.shape[1])
                mean = beta_stack.mean(axis=0)
                std  = beta_stack.std(axis=0)
                ax_b.plot(steps, mean, color=care_color, linewidth=2,
                          label=f'mean β (n={beta_stack.shape[0]} seeds)')
                ax_b.fill_between(steps, mean - std, mean + std,
                                  color=care_color, alpha=0.2)
                ax_b.axhline(BETA_0, color='black', linestyle=':',
                             linewidth=1, alpha=0.6, label=f'β₀={BETA_0}')
                ax_b.set_yscale('log')
                ax_b.legend(fontsize=8, loc='best')
            else:
                ax_b.text(0.5, 0.5, 'No CARE meta', ha='center', va='center',
                          transform=ax_b.transAxes, fontsize=10, color='gray')
            ax_b.set_title(f'{env_short} — β(s) over training',
                           fontsize=10, fontweight='bold')
            ax_b.set_xlabel('Update', fontsize=9)
            ax_b.set_ylabel('β (log)', fontsize=9)
            ax_b.grid(linestyle='--', alpha=0.3)

            # Panel right: avg extrinsic vs scaled intrinsic (β·r_int)
            ext  = _stack_seed_metric(metas, 'avg_extrinsic_reward')
            intr = _stack_seed_metric(metas, 'avg_intrinsic_reward')
            beta = beta_stack
            if ext is not None and intr is not None and beta is not None:
                n = min(ext.shape[1], intr.shape[1], beta.shape[1])
                ext_m  = ext[:, :n].mean(axis=0)
                scaled = (beta[:, :n] * intr[:, :n]).mean(axis=0)
                steps = np.arange(n)
                ax_r.fill_between(steps, 0, ext_m, color='steelblue',
                                  alpha=0.7, label='extrinsic')
                ax_r.fill_between(steps, ext_m, ext_m + scaled,
                                  color=care_color, alpha=0.7,
                                  label='β · intrinsic')
                ax_r.legend(fontsize=8, loc='best')
            else:
                ax_r.text(0.5, 0.5, 'No CARE meta', ha='center', va='center',
                          transform=ax_r.transAxes, fontsize=10, color='gray')
            ax_r.set_title(f'{env_short} — reward composition',
                           fontsize=10, fontweight='bold')
            ax_r.set_xlabel('Update', fontsize=9)
            ax_r.set_ylabel('Mean reward / step', fontsize=9)
            ax_r.grid(linestyle='--', alpha=0.3)

        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'care_dynamics_{mod.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved {out}")
        plt.close(fig)


# ── 5. β HISTOGRAM AT END OF TRAINING ────────────────────────────────────────
def plot_beta_histogram():
    """One figure per module: 2×3 envs grid, histogram of β(s) on last snapshot."""
    os.makedirs(OUT_DIR, exist_ok=True)
    nrows, ncols = 2, 3
    for mod, mod_sfx, care_color in MODULES:
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 9))
        fig.suptitle(f'β(s) distribution at end of training — {mod}',
                     fontsize=15, fontweight='bold', y=1.00)
        for idx, env in enumerate(ENVS):
            ax = axes[idx // ncols][idx % ncols]
            env_short = env.replace('MiniGrid-', '').replace('-v0', '')
            samples = load_samples(env, mod_sfx)
            pooled = []
            for s in samples:
                vals = s['values']
                if len(vals) == 0:
                    continue
                last = np.asarray(vals[-1]).flatten()
                pooled.extend(last.tolist())
            if pooled:
                pooled = np.asarray(pooled, dtype=float)
                # log-spaced bins covering the [1e-4, 5e-2] CARE clamp range
                bins = np.logspace(np.log10(1e-4), np.log10(5e-2), 40)
                ax.hist(pooled, bins=bins, color=care_color,
                        edgecolor='black', linewidth=0.3, alpha=0.85)
                ax.set_xscale('log')
                ax.axvline(BETA_0, color='black', linestyle=':',
                           linewidth=1.2, label=f'β₀={BETA_0}')
                for v, c in zip(FIXED_BETA_VALUES, FB_COLORS):
                    ax.axvline(v, color=c, linestyle='--',
                               linewidth=1, alpha=0.7, label=f'FB={v}')
                ax.legend(fontsize=7, loc='best')
            else:
                ax.text(0.5, 0.5, 'No CARE samples', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='gray')
            ax.set_title(env_short, fontsize=10, fontweight='bold')
            ax.set_xlabel('β(s)', fontsize=9)
            ax.set_ylabel('count', fontsize=9)
            ax.grid(linestyle='--', alpha=0.3)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'care_beta_hist_{mod.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  saved {out}")
        plt.close(fig)


# ── 6. SUMMARY TABLE ─────────────────────────────────────────────────────────
def print_summary():
    """Final reward table (mean of last 10 timesteps), with Δ vs PPO."""
    print("\n=== Summary: mean reward at final timestep ===")
    header = (f"{'Env':<28} {'Series':<22} {'Seeds':>5} "
              f"{'Final':>10} {'Δ vs PPO':>10}")
    print(header)
    print('-' * len(header))
    methods = all_methods()
    for env in ENVS:
        env_short = env.replace('MiniGrid-', '').replace('-v0', '')
        ppo_finals = per_seed_finals(env, '')
        ppo_mean = float(ppo_finals.mean()) if ppo_finals.size else None
        for suffix, label, _ in methods:
            finals = per_seed_finals(env, suffix)
            if finals.size == 0:
                continue
            mean = float(finals.mean())
            delta = '' if ppo_mean is None else f"{mean - ppo_mean:+.4f}"
            print(f"{env_short:<28} {label:<22} {finals.size:>5} "
                  f"{mean:>10.4f} {delta:>10}")
        print()


# ── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== 1. Reward curves: per-env ===")
    plot_per_env()
    print("\n=== 2. Reward curves: per-method ===")
    plot_per_method()
    print("\n=== 3. Aggregate sample efficiency + final reward ===")
    plot_aggregate()
    print("\n=== 4. Performance profile (Agarwal et al.) ===")
    plot_performance_profile()
    print("\n=== 5. CARE dynamics: β(s) + reward composition ===")
    plot_care_dynamics()
    print("\n=== 6. β(s) histogram at end of training ===")
    plot_beta_histogram()
    print_summary()
    print("\nDone. Figures saved under:", OUT_DIR)
