"""
Parse reward CSVs in ppo-care-curiosity/logs/<env>/ and emit LaTeX tables
that get \\input{...} into thesis/chapters/c4/c4_chapter.tex Section 4.5.

Output (all under thesis/tables/):
  final_reward_<mod>.tex      — 6 envs × 7 conditions (PPO + 5 FB + CARE)
  sample_efficiency_<mod>.tex — n* (steps to reach 0.7·env_max)
  care_vs_best_fixed.tex      — CARE vs best post-hoc fixed β, all 3 modules

Run:  python thesis/scripts/build_results_tables.py
"""

import os
import glob
import numpy as np
import pandas as pd

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGS_ROOT  = os.path.join(REPO_ROOT, 'ppo-care-curiosity', 'logs')
TABLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'tables')

ENVS = [
    ('MiniGrid-DoorKey-8x8-v0',       'DoorKey-8x8'),
    ('MiniGrid-Empty-16x16-v0',        'Empty-16x16'),
    ('MiniGrid-KeyCorridorS3R3-v0',    'KeyCorridor'),
    ('MiniGrid-LavaCrossingS9N3-v0',   'LavaCrossing'),
    ('MiniGrid-RedBlueDoors-8x8-v0',   'RedBlueDoors'),
    ('MiniGrid-UnlockPickup-v0',       'UnlockPickup'),
]

MODULES = ['COUNT', 'ICM', 'RIDE']
FIXED_BETAS = [0.0005, 0.001, 0.005, 0.01, 0.05]
LAST_N = 10            # episodes to average for final reward
EFFICIENCY_FRAC = 0.7  # threshold for sample efficiency (fraction of env_max)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
def load_seeds(env, suffix):
    """suffix='' for PPO baseline; '_COUNT_FB0.001', '_ICM_CARE', etc. otherwise."""
    if suffix == '':
        pat = os.path.join(LOGS_ROOT, env, f'PPO_{env}_seed_*.csv')
    else:
        pat = os.path.join(LOGS_ROOT, env, f'PPO{suffix}_{env}_seed_*.csv')
    return [pd.read_csv(f) for f in sorted(glob.glob(pat))]


def per_seed_finals(env, suffix):
    runs = load_seeds(env, suffix)
    if not runs:
        return np.array([])
    return np.array([r['reward'].iloc[-LAST_N:].mean() for r in runs])


def env_max(env):
    """Reference value for normalization: max per-seed final across all conditions."""
    best = 0.0
    for suffix in [''] + [f'_{m}_FB{v}' for m in MODULES for v in FIXED_BETAS] \
                       + [f'_{m}_CARE' for m in MODULES]:
        finals = per_seed_finals(env, suffix)
        if finals.size:
            best = max(best, float(finals.mean()))
    return max(best, 1e-6)


def sample_efficiency(env, suffix, threshold):
    """Smallest timestep where mean-across-seeds reward first reaches threshold."""
    runs = load_seeds(env, suffix)
    if not runs:
        return np.inf
    df = pd.concat(runs).groupby('timestep')['reward'].mean().reset_index()
    df = df.sort_values('timestep').reset_index(drop=True)
    df['smooth'] = df['reward'].rolling(window=20, win_type='triang', min_periods=1).mean()
    hit = df[df['smooth'] >= threshold]
    return int(hit['timestep'].iloc[0]) if not hit.empty else np.inf


# ── FORMATTERS ────────────────────────────────────────────────────────────────
def fmt_mean_std(finals):
    if finals.size == 0:
        return '---'
    return f'{finals.mean():.3f}{{\\scriptsize $\\pm$ {finals.std(ddof=1):.3f}}}'


def fmt_steps(n):
    if not np.isfinite(n):
        return '$\\infty$'
    if n >= 1_000_000:
        return f'{n/1e6:.2f}M'
    if n >= 1_000:
        return f'{n/1e3:.0f}k'
    return f'{n}'


def bold_best(rows, lower_is_better=False):
    """Wrap the best numeric value in each row with \\textbf{...}.
    rows: list of (label, [(num, str_value), ...]) — num is the underlying float
    used for ranking, str_value is the formatted cell content.
    """
    out = []
    for label, items in rows:
        nums = np.array([n for n, _ in items], dtype=float)
        values = [v for _, v in items]
        if np.all(~np.isfinite(nums)):
            out.append((label, values))
            continue
        idx = int(np.nanargmin(nums) if lower_is_better else np.nanargmax(nums))
        new_vals = list(values)
        new_vals[idx] = f'\\textbf{{{values[idx]}}}'
        out.append((label, new_vals))
    return out


# ── TABLE BUILDERS ────────────────────────────────────────────────────────────
def build_final_reward_table(module, fout):
    """6 envs × 7 conditions per module."""
    headers = ['PPO'] + [f'$\\beta{{=}}{v}$' for v in FIXED_BETAS] + [f'CARE-{module}']
    suffixes = [''] + [f'_{module}_FB{v}' for v in FIXED_BETAS] + [f'_{module}_CARE']

    rows = []
    for env, env_short in ENVS:
        finals_list = [per_seed_finals(env, s) for s in suffixes]
        items = [
            (float(f.mean()) if f.size else np.nan, fmt_mean_std(f))
            for f in finals_list
        ]
        rows.append((env_short, items))
    rows = bold_best(rows, lower_is_better=False)

    col_spec = '|l|' + 'c|' * len(headers)
    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Final reward (mean of last {LAST_N} logged episodes, '
                 f'averaged over 5 seeds, $\\pm$ standard deviation) for the {module} '
                 f'module. Best entry per row in bold.}}')
    lines.append(f'\\label{{tab:final_reward_{module.lower()}}}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\hline')
    lines.append('\\textbf{Environment} & ' + ' & '.join(f'\\textbf{{{h}}}' for h in headers) + ' \\\\')
    lines.append('\\hline')
    for label, values in rows:
        lines.append(f'{label} & ' + ' & '.join(values) + ' \\\\')
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    fout.write_text('\n'.join(lines), encoding='utf-8')


def build_sample_efficiency_table(module, fout):
    """n* (steps to reach EFFICIENCY_FRAC × env_max) per (env, condition)."""
    headers = ['PPO'] + [f'$\\beta{{=}}{v}$' for v in FIXED_BETAS] + [f'CARE-{module}']
    suffixes = [''] + [f'_{module}_FB{v}' for v in FIXED_BETAS] + [f'_{module}_CARE']

    rows = []
    for env, env_short in ENVS:
        thr = EFFICIENCY_FRAC * env_max(env)
        steps = [sample_efficiency(env, s, thr) for s in suffixes]
        items = [(float(n), fmt_steps(n)) for n in steps]
        rows.append((env_short, items))
    rows = bold_best(rows, lower_is_better=True)

    col_spec = '|l|' + 'c|' * len(headers)
    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append(f'\\caption{{Sample efficiency for the {module} module: '
                 f'first timestep at which the across-seed mean reward reaches '
                 f'$0.7 \\cdot r^\\star_\\text{{env}}$, where $r^\\star_\\text{{env}}$ '
                 f'is the best mean final reward across all conditions on that environment. '
                 f'Lower is better; $\\infty$ means the threshold was never reached '
                 f'within the $10^6$-step budget. Best entry per row in bold.}}')
    lines.append(f'\\label{{tab:sample_efficiency_{module.lower()}}}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\hline')
    lines.append('\\textbf{Environment} & ' + ' & '.join(f'\\textbf{{{h}}}' for h in headers) + ' \\\\')
    lines.append('\\hline')
    for label, values in rows:
        lines.append(f'{label} & ' + ' & '.join(values) + ' \\\\')
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    fout.write_text('\n'.join(lines), encoding='utf-8')


def build_care_vs_best_fixed_table(fout):
    """6 envs × 3 modules: CARE final, best-fixed final, gap, best-β value."""
    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    lines.append('\\caption{CARE versus the best post-hoc fixed-$\\beta$ value, per '
                 '(environment, module). Gap is $r_\\text{CARE} - r_\\text{best-FB}$ in raw reward; '
                 'positive means CARE is better. Best-$\\beta$ is the value of $\\beta$ '
                 'that achieves $r_\\text{best-FB}$ on that environment.}')
    lines.append('\\label{tab:care_vs_best_fixed}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append('\\begin{tabular}{|l|l|c|c|c|c|}')
    lines.append('\\hline')
    lines.append('\\textbf{Environment} & \\textbf{Module} & \\textbf{CARE} & '
                 '\\textbf{Best fixed-$\\beta$} & \\textbf{Best $\\beta$} & '
                 '\\textbf{Gap} \\\\')
    lines.append('\\hline')
    for env, env_short in ENVS:
        for mi, mod in enumerate(MODULES):
            care_finals = per_seed_finals(env, f'_{mod}_CARE')
            r_care = float(care_finals.mean()) if care_finals.size else np.nan
            best_v, best_r = None, -np.inf
            for v in FIXED_BETAS:
                fb = per_seed_finals(env, f'_{mod}_FB{v}')
                if fb.size and fb.mean() > best_r:
                    best_r = float(fb.mean())
                    best_v = v
            gap = r_care - best_r if np.isfinite(r_care) and np.isfinite(best_r) else np.nan
            r_care_s = '---' if not np.isfinite(r_care) else f'{r_care:.3f}'
            best_r_s = '---' if not np.isfinite(best_r) else f'{best_r:.3f}'
            best_v_s = '---' if best_v is None else f'{best_v}'
            if not np.isfinite(gap):
                gap_s = '---'
            else:
                sign = '+' if gap >= 0 else ''
                gap_s = f'\\textbf{{{sign}{gap:.3f}}}' if gap > 0 else f'{sign}{gap:.3f}'
            env_cell = f'\\multirow{{3}}{{*}}{{{env_short}}}' if mi == 0 else ''
            lines.append(f'{env_cell} & {mod} & {r_care_s} & {best_r_s} & {best_v_s} & {gap_s} \\\\')
            if mi < len(MODULES) - 1:
                lines.append('\\cline{2-6}')
            else:
                lines.append('\\hline')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    fout.write_text('\n'.join(lines), encoding='utf-8')


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    from pathlib import Path
    out = Path(TABLES_DIR).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'Writing tables to {out}')

    for mod in MODULES:
        build_final_reward_table(mod, out / f'final_reward_{mod.lower()}.tex')
        print(f'  final_reward_{mod.lower()}.tex')
        build_sample_efficiency_table(mod, out / f'sample_efficiency_{mod.lower()}.tex')
        print(f'  sample_efficiency_{mod.lower()}.tex')

    build_care_vs_best_fixed_table(out / 'care_vs_best_fixed.tex')
    print('  care_vs_best_fixed.tex')


if __name__ == '__main__':
    main()
