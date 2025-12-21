import os
import glob
import numpy as np
import pandas as pd
import json
import re
from matplotlib import pyplot as plt

# configuration
WINDOW = 10
MIN_WINDOW = 1

def find_seed_dirs(base_results_dir="results", env_name=None):
    """
    find all seed directories inside results.
    structure: results/<env_name>/seed_<s>_YYYYMMDD_HHMMSS
    returns: dict {seed_num: path}
    """
    seed_dirs = {}
    
    if env_name:
        env_dir = os.path.join(base_results_dir, env_name)
        if not os.path.exists(env_dir):
            raise FileNotFoundError(f"env results dir not found: {env_dir}")
        candidates = [os.path.join(env_dir, d) for d in os.listdir(env_dir)
                     if os.path.isdir(os.path.join(env_dir, d))]
    else:
        candidates = []
        for env in os.listdir(base_results_dir):
            env_dir = os.path.join(base_results_dir, env)
            if not os.path.isdir(env_dir):
                continue
            for run in os.listdir(env_dir):
                rd = os.path.join(env_dir, run)
                if os.path.isdir(rd):
                    candidates.append(rd)
    
    # extract seed number from directory name
    for path in candidates:
        dirname = os.path.basename(path)
        match = re.search(r'seed[_-](\d+)', dirname)
        if match:
            seed_num = match.group(1)
            seed_dirs[seed_num] = path
    
    return seed_dirs

def load_timeseries_from_dir(run_dir):
    """
    load timeseries from a seed directory (npz or csv).
    returns: dataframe with columns
    ['timestep', 'episode_reward', 'obs_coverage', 'pos_coverage', 'avg_entropy']
    """
    # try npz first
    npz_path = os.path.join(run_dir, "timeseries.npz")
    if os.path.exists(npz_path):
        data = {}
        with np.load(npz_path, allow_pickle=True) as npz_data:
            for k in npz_data.files:
                data[k] = npz_data[k]
        
        df = pd.DataFrame({
            'timestep': data.get('timestep', data.get('total_frames', [])),
            'episode_reward': data.get('episode_reward', []),
            'obs_coverage': data.get('obs_coverage', data.get('observation_coverage_count', [])),
            'pos_coverage': data.get('pos_coverage', data.get('position_coverage_fraction', [])),
            'avg_entropy': data.get('avg_entropy', data.get('avg_policy_entropy', []))
        })
        return df
    
    # try csv
    csv_files = glob.glob(os.path.join(run_dir, "PPO_*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
    
    if csv_files:
        df = pd.read_csv(csv_files[0])
        
        # normalize column names
        col_mapping = {
            'observation_coverage_count': 'obs_coverage',
            'position_coverage_fraction': 'pos_coverage',
            'avg_policy_entropy': 'avg_entropy'
        }
        df = df.rename(columns=col_mapping)
        
        # select relevant columns
        available_cols = ['timestep', 'episode_reward', 'obs_coverage', 'pos_coverage', 'avg_entropy']
        df = df[[c for c in available_cols if c in df.columns]]
        return df
    
    raise FileNotFoundError(f"no timeseries data found in {run_dir}")

def aggregate_metrics_multi_seed(
    seed_dirs,
    metrics=['episode_reward', 'obs_coverage', 'pos_coverage', 'avg_entropy'],
    window=WINDOW,
    min_periods=MIN_WINDOW
):
    """
    aggregate metrics from multiple seeds with interpolation and smoothing.
    
    args:
        seed_dirs: dict {seed_num: path}
        metrics: list of metric names to aggregate
        window: smoothing window size
        min_periods: minimum periods for rolling window
    
    returns:
        dict {metric_name: dataframe with columns ['timestep', 'mean', 'std']}
    """
    # load all seed data
    seed_data = {}
    for seed_num, path in seed_dirs.items():
        try:
            df = load_timeseries_from_dir(path)
            seed_data[seed_num] = df
            print(f"loaded seed {seed_num}: {len(df)} timesteps")
        except Exception as e:
            print(f"warning: could not load seed {seed_num}: {e}")
    
    if not seed_data:
        raise RuntimeError("no valid seed data loaded")
    
    results = {}
    
    for metric in metrics:
        print(f"\nprocessing metric: {metric}")
        
        # merge all seeds on timestep
        df_merged = None
        for seed_num, df in seed_data.items():
            if metric not in df.columns:
                print(f"  skipping seed {seed_num}: metric not found")
                continue
            
            tmp = df[['timestep', metric]].copy()
            tmp = tmp.rename(columns={metric: f'{metric}_seed_{seed_num}'})
            
            # convert to numeric
            tmp['timestep'] = pd.to_numeric(tmp['timestep'], errors='coerce')
            tmp[f'{metric}_seed_{seed_num}'] = pd.to_numeric(
                tmp[f'{metric}_seed_{seed_num}'], errors='coerce'
            )
            tmp = tmp.dropna(subset=['timestep'])
            
            if df_merged is None:
                df_merged = tmp
            else:
                df_merged = pd.merge(df_merged, tmp, on='timestep', how='outer')
        
        if df_merged is None or len(df_merged) == 0:
            print(f"  no data for metric {metric}")
            continue
        
        df_merged = df_merged.sort_values('timestep').reset_index(drop=True)
        
        # get all seed columns for this metric
        seed_cols = [c for c in df_merged.columns if c.startswith(f'{metric}_seed_')]
        if not seed_cols:
            continue
        
        # interpolate per seed
        for col in seed_cols:
            df_merged[col] = df_merged[col].interpolate(
                method='linear', limit_direction='both'
            )
            df_merged[col] = df_merged[col].ffill().bfill()
        
        # smooth per seed
        smooth_cols = []
        for col in seed_cols:
            col_smooth = col + '_smooth'
            try:
                df_merged[col_smooth] = df_merged[col].rolling(
                    window=window,
                    win_type='triang',
                    min_periods=min_periods
                ).mean()
            except:
                df_merged[col_smooth] = df_merged[col].rolling(
                    window=window,
                    min_periods=min_periods
                ).mean()
            smooth_cols.append(col_smooth)
        
        # calculate statistics across seeds
        if smooth_cols:
            df_merged[f'{metric}_mean'] = df_merged[smooth_cols].mean(axis=1)
            df_merged[f'{metric}_std'] = df_merged[smooth_cols].std(axis=1)
        
        # store results
        result_df = df_merged[['timestep', f'{metric}_mean', f'{metric}_std']].copy()
        result_df.columns = ['timestep', 'mean', 'std']
        results[metric] = result_df
        
        print(f"  aggregated {len(seed_cols)} seeds, {len(result_df)} timesteps")
    
    return results

def save_aggregated_results(results, output_dir="aggregated_results"):
    """
    save aggregated results to csv files and create summary plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for metric, df in results.items():
        # save csv
        csv_path = os.path.join(output_dir, f"{metric}_aggregated.csv")
        df.to_csv(csv_path, index=False)
        print(f"saved {metric} -> {csv_path}")
        
        # create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        timesteps = df['timestep'].values
        mean = df['mean'].values
        std = df['std'].values
        
        # plot mean with std shading
        ax.plot(timesteps, mean, label='mean', linewidth=2)
        ax.fill_between(
            timesteps,
            mean - std,
            mean + std,
            alpha=0.3,
            label='±1 std dev'
        )
        
        ax.set_xlabel('timestep')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} across seeds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"{metric}_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"saved plot -> {plot_path}")
    
    # save summary statistics
    summary = {}
    for metric, df in results.items():
        summary[metric] = {
            'final_mean': float(df['mean'].iloc[-1]) if len(df) > 0 else None,
            'final_std': float(df['std'].iloc[-1]) if len(df) > 0 else None,
            'max_mean': float(df['mean'].max()),
            'timesteps': int(len(df))
        }
    
    summary_path = os.path.join(output_dir, "summary_statistics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary -> {summary_path}")

def main():
    """
    main function to aggregate metrics from multiple seeds.
    """
    # configuration - modify these
    base_results_dir = "results"
    env_name = ""  # or none to search all envs
    output_dir = "aggregated_results"
    
    print("=" * 60)
    print("multi-seed metrics aggregation with variance")
    print("=" * 60)
    
    # find all seed directories
    print(f"\nsearching for seed directories in: {base_results_dir}")
    if env_name:
        print(f"environment: {env_name}")
    
    seed_dirs = find_seed_dirs(base_results_dir, env_name)
    
    if not seed_dirs:
        raise RuntimeError(f"no seed directories found in {base_results_dir}")
    
    print(f"\nfound {len(seed_dirs)} seeds:")
    for seed_num, path in sorted(seed_dirs.items()):
        print(f"  seed {seed_num}: {path}")
    
    # aggregate metrics
    print("\n" + "=" * 60)
    print("aggregating metrics across seeds...")
    print("=" * 60)
    
    metrics = ['episode_reward', 'obs_coverage', 'pos_coverage', 'avg_entropy']
    results = aggregate_metrics_multi_seed(seed_dirs, metrics=metrics)
    
    # save results
    print("\n" + "=" * 60)
    print("saving results...")
    print("=" * 60)
    
    save_aggregated_results(results, output_dir)
    
    print("\n" + "=" * 60)
    print("done!")
    print("=" * 60)
    print(f"results saved to: {output_dir}/")
    print("  - csv files: <metric>_aggregated.csv")
    print("  - plots: <metric>_plot.png")
    print("  - summary: summary_statistics.json")

if __name__ == "__main__":
    main()
