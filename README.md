# PPO with Correlation-Aware Reward Exploration (CARE) on MiniGrid

This repository implements and benchmarks PPO agents with curiosity-driven exploration in sparse-reward [MiniGrid](https://minigrid.farama.org/) environments. The main contribution is **CARE (Correlation-Aware Reward Exploration)** — a learned state-dependent β(s) scaling function that replaces fixed intrinsic-reward coefficients. CARE is evaluated on top of three curiosity baselines: **ICM**, **count-based** exploration, and **RIDE**.

A full derivation of the method and the experimental results lives in the LaTeX thesis under [thesis/](thesis/).

---

## Repository layout

```
graduation-thesis/
├── ppo-care-curiosity/         # all training + plotting code
│   ├── train.py                # unified entry point (method/env/seed/beta flags)
│   ├── ppo.py                  # baseline PPO (no intrinsic reward)
│   ├── ppo_icm.py              # PPO + ICM (optionally + CARE)
│   ├── ppo_count.py            # PPO + count-based bonus (optionally + CARE)
│   ├── ppo_ride.py             # PPO + RIDE (optionally + CARE)
│   ├── curiosity/
│   │   ├── icm.py
│   │   ├── count_based.py
│   │   ├── ride.py
│   │   └── care_module.py      # BetaNetwork + correlation-with-advantage update
│   ├── training_logger.py      # CARE diagnostics (.meta.npz / .samples.npz)
│   ├── plot_graph.py           # single-env reward curves
│   ├── plot_compare.py         # full thesis figure set
│   └── requirements.txt
└── thesis/                     # LaTeX thesis source
```

---

## Quick start — Google Colab

A ready-to-run Colab notebook is available here:

> **[Open in Colab](https://colab.research.google.com/drive/1oBcsHoY81DZ_x9poIW-CsWmBY5-Q-HmA)**

If the shared notebook is unavailable, paste this into any fresh Colab runtime:

```python
!git clone https://github.com/dvgiap/graduation-thesis.git
%cd graduation-thesis/ppo-care-curiosity
!pip install -r requirements.txt

!python train.py --method count --env MiniGrid-DoorKey-8x8-v0
```

Swap `--method count` for `icm`, `ride`, or `none` to switch the exploration module; see the full flag list under [Training](#training) below.

A free Colab CPU runtime is sufficient — MiniGrid states are flat vectors, so a GPU gives only a modest speedup.

---

## Training

All training is launched through `train.py`. The CLI surface is:

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--method` | `none` / `icm` / `count` / `ride` | `none` | Exploration module |
| `--env` | any Gymnasium MiniGrid id | `MiniGrid-DoorKey-8x8-v0` | Environment |
| `--seed_start` | int | `1` | First seed (inclusive) |
| `--seed_end` | int | `5` | Last seed (inclusive) |
| `--max_steps` | int | `1000000` | Per-seed timestep budget |
| `--fixed-beta` | float or unset | unset | If set, disable state-dependent β(s) and use this scalar instead (ablation) |

### Examples

```bash
# CARE + RIDE on DoorKey (5 seeds, default 1M steps each)
python train.py --method ride --env MiniGrid-DoorKey-8x8-v0 --seed_start 1 --seed_end 5

# Fixed-β ablation: ICM with constant β = 0.01
python train.py --method icm --fixed-beta 0.01 --seed_start 1 --seed_end 5

# Pure PPO baseline (no intrinsic reward)
python train.py --method none --seed_start 1 --seed_end 5
```

### Benchmark environments

The thesis reports results on six MiniGrid environments — pass any of these via `--env`:

- `MiniGrid-DoorKey-8x8-v0`
- `MiniGrid-Empty-16x16-v0`
- `MiniGrid-KeyCorridorS3R3-v0`
- `MiniGrid-LavaCrossingS9N3-v0`
- `MiniGrid-RedBlueDoors-8x8-v0`
- `MiniGrid-UnlockPickup-v0`

Any other Gymnasium MiniGrid env id also works.

---

## Visualizing results

```bash
# Single-environment reward curves
python plot_graph.py
```

`plot_graph.py` has the env name hardcoded near the top of the file (`env_name = 'MiniGrid-DoorKey-8x8-v0'`); edit that line to plot a different environment. Output goes to `figs/{env}/`.

```bash
# Full thesis figure set (per-env curves, aggregate, performance profile,
# β dynamics, β histogram)
python plot_compare.py
```

`plot_compare.py` reads all CSVs under `logs/` and CARE diagnostics under `models/`, then writes figures to `figs/compare/`.

---

## Output artifacts

Training writes the following files (paths are relative to `ppo-care-curiosity/`):

| Path | Contents |
|------|----------|
| `logs/{env}/PPO{suffix}{cond}_{env}_seed_{seed}.csv` | Reward curve (episode, timestep, reward) |
| `models/{env}/PPO{suffix}{cond}_{env}_seed_{seed}.pth` | Policy checkpoint |
| `models/{env}/*.meta.npz` | CARE diagnostics — β over time, meta-loss split, advantage stats |
| `models/{env}/*.samples.npz` | CARE diagnostics — (state, β(s)) snapshots for histogram analysis |

Filename suffix convention:

- `_ICM`, `_COUNT`, `_RIDE` — picked from `--method`
- `_CARE` — state-dependent β(s) is active (the default when an exploration method is selected)
- `_FB{value}` — `--fixed-beta {value}` was passed (state-dependent β disabled)

So for example `PPO_RIDE_CARE_MiniGrid-DoorKey-8x8-v0_seed_3.pth` is the RIDE + CARE checkpoint for seed 3 on DoorKey.

---

## Method in one paragraph

CARE learns a network `β_ψ(s)` that scales the standardized intrinsic signal per state: `R_combined = R_extrinsic + β(s) · R_intrinsic⁺`. `β_ψ` is trained by maximizing the Pearson correlation between the weighted intrinsic reward `β(s) · R_intrinsic⁺` and the extrinsic GAE advantage `Â^E`, with a log-space regularizer toward a cold-start prior `β₀ = √(β_min · β_max)`. See the thesis for the full equations, hyperparameters, and ablations.
