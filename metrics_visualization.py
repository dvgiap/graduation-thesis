import os
import argparse
from datetime import datetime
import json
import glob
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt

# -------------------- Helpers --------------------
def find_latest_run(base_results_dir="results", env_name=None):
    """
    If run_dir is not provided, attempt to find the latest run folder.
    Directory structure: results/<env_name>/seed_<s>_YYYYMMDD_HHMMSS
    """
    if not os.path.exists(base_results_dir):
        raise FileNotFoundError(f"{base_results_dir} does not exist")

    candidate_dirs = []
    if env_name:
        env_dir = os.path.join(base_results_dir, env_name)
        if not os.path.exists(env_dir):
            raise FileNotFoundError(f"Env results dir not found: {env_dir}")
        candidate_dirs = [os.path.join(env_dir, d) for d in os.listdir(env_dir)
                          if os.path.isdir(os.path.join(env_dir, d))]
    else:
        for env in os.listdir(base_results_dir):
            env_dir = os.path.join(base_results_dir, env)
            if not os.path.isdir(env_dir):
                continue
            for run in os.listdir(env_dir):
                rd = os.path.join(env_dir, run)
                if os.path.isdir(rd):
                    candidate_dirs.append(rd)

    if not candidate_dirs:
        raise FileNotFoundError("No run directories found under results/")

    candidate_dirs = sorted(candidate_dirs, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidate_dirs[0]


def load_timeseries_npz(run_dir):
    """
    Load timeseries.npz produced by train.py (timeseries.npz).
    Returns dict of arrays or None.
    """
    path = os.path.join(run_dir, "timeseries.npz")
    if not os.path.exists(path):
        return None
    ts = {}
    with np.load(path, allow_pickle=True) as data:
        for k in data.files:
            ts[k] = data[k]
    # Normalize some common names to expected keys for plotting
    mapped = {}
    # prefer explicit keys used in train.py timeseries dict
    expected = ['episode', 'timestep', 'episode_reward', 'avg_recent_return',
                'obs_coverage', 'pos_coverage', 'avg_entropy',
                'reward_discovery_frames', 'total_frames']
    for ek in expected:
        # direct match
        if ek in ts:
            mapped[ek] = ts[ek]
        else:
            # try alternative names
            if ek == 'obs_coverage':
                for alt in ['observation_coverage_count', 'observation_coverage', 'obs_coverage']:
                    if alt in ts:
                        mapped['obs_coverage'] = ts[alt]
                        break
            elif ek == 'pos_coverage':
                for alt in ['position_coverage_fraction', 'pos_coverage', 'position_coverage']:
                    if alt in ts:
                        mapped['pos_coverage'] = ts[alt]
                        break
            elif ek == 'avg_entropy':
                for alt in ['avg_policy_entropy', 'avg_entropy', 'policy_entropy']:
                    if alt in ts:
                        mapped['avg_entropy'] = ts[alt]
                        break
            elif ek == 'reward_discovery_frames':
                for alt in ['reward_discovery_frames', 'reward_discovery']:
                    if alt in ts:
                        mapped['reward_discovery_frames'] = ts[alt]
                        break
            else:
                # fallback if key absent
                mapped[ek] = ts.get(ek, np.array([]))
    # also expose any other keys raw
    for k in ts:
        if k not in mapped:
            mapped[k] = ts[k]
    return mapped


def load_timeseries_csv(run_dir):
    """
    Load a CSV produced by train.py (PPO_*.csv).
    Returns dict of numpy arrays (dtype=object).
    """
    files = glob.glob(os.path.join(run_dir, "PPO_*.csv"))
    if not files:
        files = glob.glob(os.path.join(run_dir, "*.csv"))
    if not files:
        return None
    csv_path = files[0]
    import csv
    ts = { 'episode': [], 'timestep': [], 'episode_reward': [],
           'avg_recent_return': [], 'obs_coverage': [], 'pos_coverage': [],
           'avg_entropy': [], 'reward_discovery_frames': [], 'total_frames': [] }
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # some rows may be empty strings
            def _to_int_or_none(x):
                try:
                    if x is None or x == '':
                        return None
                    return int(x)
                except Exception:
                    return None
            def _to_float_or_nan(x):
                try:
                    if x is None or x == '':
                        return np.nan
                    return float(x)
                except Exception:
                    return np.nan

            ts['episode'].append(_to_int_or_none(row.get('episode','')))
            ts['timestep'].append(_to_int_or_none(row.get('timestep','')))
            ts['episode_reward'].append(_to_float_or_nan(row.get('episode_reward','')))
            ts['avg_recent_return'].append(_to_float_or_nan(row.get('avg_recent_return','')))
            # CSV column may be obs_coverage or observation_coverage_count
            obs_val = row.get('obs_coverage', '') or row.get('observation_coverage_count', '')
            try:
                ts['obs_coverage'].append(int(obs_val) if obs_val != '' else 0)
            except Exception:
                ts['obs_coverage'].append(0)
            ts['pos_coverage'].append(_to_float_or_nan(row.get('pos_coverage','') or row.get('position_coverage_fraction','')))
            ts['avg_entropy'].append(_to_float_or_nan(row.get('avg_entropy','') or row.get('avg_policy_entropy','')))
            ts['reward_discovery_frames'].append(row.get('reward_discovery_frames',''))
            ts['total_frames'].append(_to_int_or_none(row.get('total_frames','')))
    for k in list(ts.keys()):
        ts[k] = np.array(ts[k], dtype=object)
    return ts


def load_metrics_final(run_dir):
    """
    Load final metrics snapshot (metrics_final.npz / metrics_final.json) or recent snapshot.
    Returns a dict (keys may be numpy arrays or python types).
    Normalize likely keys to 'visited_positions' etc.
    """
    npz_path = os.path.join(run_dir, "metrics_final.npz")
    json_path = os.path.join(run_dir, "metrics_final.json")
    data = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as dd:
            for k in dd.files:
                data[k] = dd[k]
        return _normalize_metrics_dict(data)
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            data = json.load(jf)
        return _normalize_metrics_dict(data)
    snapshots = glob.glob(os.path.join(run_dir, "metrics_snapshot_*.npz"))
    if snapshots:
        snapshots = sorted(snapshots, key=lambda p: os.path.getmtime(p), reverse=True)
        with np.load(snapshots[0], allow_pickle=True) as dd:
            for k in dd.files:
                data[k] = dd[k]
        return _normalize_metrics_dict(data)
    jfiles = glob.glob(os.path.join(run_dir, "metrics_summary_*.json"))
    if jfiles:
        jfiles = sorted(jfiles, key=lambda p: os.path.getmtime(p), reverse=True)
        with open(jfiles[0], 'r') as jf:
            data = json.load(jf)
        return _normalize_metrics_dict(data)
    return None


def _normalize_metrics_dict(d):
    """
    Map various possible saved key names into a normalized dict.
    Looks for visited_positions, observation_coverage_count, position_coverage_fraction etc.
    """
    norm = {}
    # if d is numpy-like mapping with array values, handle that
    for k, v in d.items():
        lk = k.lower()
        if lk in ('visited_positions', 'visited_pos', 'vispos'):
            norm['visited_positions'] = v
        elif lk in ('observation_coverage_count', 'observation_coverage', 'obs_coverage'):
            norm['observation_coverage_count'] = v
        elif lk in ('position_coverage_fraction', 'pos_coverage', 'position_coverage'):
            norm['position_coverage_fraction'] = v
        elif lk in ('avg_policy_entropy', 'avg_entropy', 'policy_entropy'):
            norm['avg_policy_entropy'] = v
        elif lk in ('reward_discovery_frames', 'reward_discovery'):
            norm['reward_discovery_frames'] = v
        elif lk in ('total_frames', 'total_frame', 'frames'):
            norm['total_frames'] = v
        else:
            # keep other keys as-is
            norm[k] = v
    return norm


def parse_visited_positions(raw):
    """
    Parse various types for visited_positions into numpy array shape (N,2) of ints,
    or return None if cannot parse.
    Accepts:
      - list of (x,y)
      - numpy object array where each entry is list/tuple
      - string representations like "(1, 2)" or "[1,2]"
    """
    if raw is None:
        return None
    coords = []
    # turn numpy arrays into python list for iteration
    try:
        if isinstance(raw, np.ndarray):
            raw_list = raw.tolist()
        else:
            raw_list = list(raw)
    except Exception:
        raw_list = raw

    for item in raw_list:
        if item is None:
            continue
        # direct tuple/list of numbers
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x = int(item[0]); y = int(item[1])
                coords.append((x, y))
                continue
            except Exception:
                pass
        # numpy array element
        if isinstance(item, np.ndarray) and item.size >= 2:
            try:
                arr = np.asarray(item).astype(int)
                coords.append((int(arr[0]), int(arr[1])))
                continue
            except Exception:
                pass
        # string patterns
        if isinstance(item, str):
            s = item.strip()
            # ignore discovery-frame strings like '12|345'
            if '|' in s and '(' not in s and ',' not in s:
                # not coordinates
                continue
            # extract numbers inside string
            nums = re.findall(r'-?\d+', s)
            if len(nums) >= 2:
                try:
                    coords.append((int(nums[0]), int(nums[1])))
                    continue
                except Exception:
                    pass
            # try literal eval as last resort
            try:
                t = eval(s)
                if isinstance(t, (list, tuple)) and len(t) >= 2:
                    coords.append((int(t[0]), int(t[1])))
                    continue
            except Exception:
                pass
        # fallback: try to coerce via tuple conversion
        try:
            t = tuple(item)
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                coords.append((int(t[0]), int(t[1])))
                continue
        except Exception:
            pass

    if not coords:
        return None
    return np.array(coords, dtype=int)


def build_heatmap_from_positions(positions):
    """
    positions: numpy array shape (N,2) of (x,y)
    returns heatmap array (H, W), width W, height H where heat[row=y, col=x]
    """
    if positions is None or positions.size == 0:
        return None, None, None
    xs = positions[:, 0]
    ys = positions[:, 1]
    max_x = int(xs.max())
    max_y = int(ys.max())
    W = max_x + 1
    H = max_y + 1
    heat = np.zeros((H, W), dtype=int)
    for x, y in positions:
        if 0 <= y < H and 0 <= x < W:
            heat[int(y), int(x)] += 1
    return heat, W, H


# -------------------- Plot helpers --------------------
def _safe_to_float_array(arr, name="y"):
    """
    Convert array-like to numpy float array, coercing non-numeric to np.nan.
    """
    if arr is None:
        return np.array([], dtype=float)
    out = []
    for v in arr:
        try:
            if v is None:
                out.append(np.nan)
            elif isinstance(v, (str, bytes)) and str(v).strip() == '':
                out.append(np.nan)
            else:
                out.append(float(v))
        except Exception:
            try:
                out.append(float(np.asarray(v)))
            except Exception:
                out.append(np.nan)
    outa = np.array(out, dtype=float)
    if outa.size > 0 and np.all(np.isnan(outa)):
        warnings.warn(f"All values for '{name}' are NaN (maybe parsing issue).")
    return outa


def plot_timeseries(ts, run_dir, save=True, show=True):
    """
    ts: dict of arrays with keys like 'timestep', 'episode_reward', 'obs_coverage', 'pos_coverage', 'avg_entropy', 'reward_discovery_frames'
    """
    if ts is None:
        print("[WARN] No timeseries data found.")
        return None

    # x axis
    if 'timestep' in ts and ts['timestep'] is not None and len(ts['timestep']) > 0:
        try:
            x = np.array([float(v) if (v is not None and str(v) != '') else np.nan for v in ts['timestep']], dtype=float)
        except Exception:
            x = np.arange(len(next(iter(ts.values()))), dtype=float)
    else:
        # fallback: length of any timeseries vector
        try:
            sample = next(iter(ts.values()))
            x = np.arange(len(sample), dtype=float)
        except Exception:
            x = np.arange(0, dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # episode reward
    y_reward = _safe_to_float_array(ts.get('episode_reward', []), name='episode_reward')
    axes[0].plot(x[:len(y_reward)], y_reward, marker='o', linewidth=1)
    axes[0].set_ylabel('episode_reward')
    axes[0].grid(True)

    # observation coverage
    y_obs = _safe_to_float_array(ts.get('obs_coverage', []), name='obs_coverage')
    axes[1].plot(x[:len(y_obs)], y_obs, marker='o', linewidth=1)
    axes[1].set_ylabel('obs_coverage (unique count)')
    axes[1].grid(True)

    # position coverage
    y_pos = _safe_to_float_array(ts.get('pos_coverage', []), name='pos_coverage')
    axes[2].plot(x[:len(y_pos)], y_pos, marker='o', linewidth=1)
    axes[2].set_ylabel('pos_coverage (fraction)')
    axes[2].grid(True)

    # avg policy entropy
    y_ent = _safe_to_float_array(ts.get('avg_entropy', []), name='avg_entropy')
    axes[3].plot(x[:len(y_ent)], y_ent, marker='o', linewidth=1)
    axes[3].set_ylabel('avg_policy_entropy')
    axes[3].set_xlabel('timestep')
    axes[3].grid(True)

    # parse reward discovery frames from timeseries (strings like 'f1|f2')
    discovery_frames = []
    rdf = ts.get('reward_discovery_frames', None)
    if rdf is not None:
        for cell in rdf:
            if cell is None:
                continue
            s = str(cell)
            if s == '':
                continue
            parts = s.split('|')
            for p in parts:
                p = p.strip()
                if p == '':
                    continue
                try:
                    discovery_frames.append(int(p))
                except Exception:
                    # could be array-like stored in npz - try to iterate
                    try:
                        for sub in np.asarray(cell).flatten():
                            discovery_frames.append(int(sub))
                    except Exception:
                        pass
    discovery_frames = sorted(set(discovery_frames))[:3]

    # mark vertical lines
    for f in discovery_frames:
        for ax in axes:
            ax.axvline(x=f, linestyle='--', linewidth=0.8, color='gray')
    if discovery_frames:
        txt = f"reward discoveries: {discovery_frames}"
        axes[0].text(0.98, 0.95, txt, transform=axes[0].transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.suptitle('Training time series')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = os.path.join(run_dir, "timeseries_plots.png")
    if save:
        plt.savefig(out_path)
        print("Saved timeseries plot ->", out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_heatmap_for_run(run_dir, save=True, show=True):
    data = load_metrics_final(run_dir)
    if data is None:
        print("[WARN] No metrics_final/metrics_snapshot found for heatmap.")
        return None

    # Try to get visited_positions
    visited = None
    if isinstance(data, dict):
        if 'visited_positions' in data:
            visited = data['visited_positions']
        else:
            # check common alternatives or keys containing 'visited'
            for k in data:
                if 'visited' in k.lower():
                    visited = data[k]
                    break
    else:
        visited = data

    pos_arr = parse_visited_positions(visited)
    if pos_arr is None:
        print("[WARN] No visited_positions (or could not parse) in final snapshot.")
        return None

    heat, W, H = build_heatmap_from_positions(pos_arr)
    if heat is None:
        print("[WARN] Could not build heatmap from positions.")
        return None

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(heat, origin='upper', interpolation='nearest')
    plt.colorbar()
    plt.title("Visited positions heatmap (counts)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "visited_positions_heatmap.png")
    if save:
        plt.savefig(out_path)
        print("Saved heatmap ->", out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# -------------------- Main CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize RL training metrics saved by train.py")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run directory (results/<env>/seed_... )")
    parser.add_argument("--env", type=str, default=None, help="Environment name under results/ to auto-find latest run")
    parser.add_argument("--base_results", type=str, default="results", help="Base results directory")
    parser.add_argument("--no_show", action="store_true", help="Do not call plt.show() (save only)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        try:
            run_dir = find_latest_run(base_results_dir=args.base_results, env_name=args.env)
            print("Auto-selected run_dir:", run_dir)
        except Exception as e:
            print("[ERROR] Couldn't find run_dir automatically:", e)
            return

    show_flag = not args.no_show

    # load timeseries (prefer timeseries.npz)
    ts = load_timeseries_npz(run_dir)
    if ts is None:
        ts = load_timeseries_csv(run_dir)
    if ts is None:
        print("[WARN] No timeseries data found (timeseries.npz or CSV). Skipping timeseries plot.")
    else:
        plot_timeseries(ts, run_dir, save=True, show=show_flag)

    # heatmap
    plot_heatmap_for_run(run_dir, save=True, show=show_flag)

    print("Visualization done. Files saved to:", run_dir)


if __name__ == "__main__":
    main()
