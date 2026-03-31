# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graduation thesis research project implementing and comparing PPO-based reinforcement learning agents with various curiosity-driven exploration methods in sparse-reward MiniGrid environments. The novel contribution is **ACWI (Adaptive Correlation-Weighted Intrinsic rewards)** — a learned state-dependent β(s) scaling function that replaces fixed intrinsic reward coefficients.

## Running Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Train an agent (from either ppo-curiosity/ or ppo-acwi-curiosity/)
python train.py --method [none|icm|count|ride] \
                --env MiniGrid-DoorKey-8x8-v0 \
                --seed_start 1 \
                --seed_end 5 \
                --max_steps 1000000

# Visualize results
python plot_graph.py

# Generate episode GIFs
python make_gif.py
```

**Available environments:** `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-Empty-16x16-v0`, `MiniGrid-RedBlueDoors-8x8-v0`, `MiniGrid-UnlockPickup-v0`

## Repository Structure

```
graduation-thesis/
├── ppo-curiosity/          # Baseline reference implementation
├── ppo-acwi-curiosity/     # ACWI-enhanced implementation (main contribution)
└── thesis/                 # LaTeX thesis document
```

Both `ppo-curiosity/` and `ppo-acwi-curiosity/` share the same layout — differences are in how intrinsic rewards are weighted and the addition of `training_logger.py` in the ACWI variant.

## Architecture

### Core PPO Components (shared across both implementations)

**`ActorCritic` (nn.Module)** — two-headed network with shared input:
- Actor: `Linear(64) → Tanh → Linear(64) → Tanh → Linear(action_dim) → Softmax`
- Critic: `Linear(64) → Tanh → Linear(64) → Tanh → Linear(1)`

**`RolloutBuffer`** — stores `(state, action, logprob, reward, is_terminal, next_state)` tuples; `next_state` is needed by ICM/RIDE.

**`PPO` agent** — wraps ActorCritic, handles `select_action()`, `update()` (K=80 gradient epochs with GAE advantage estimation), and checkpoint save/load.

### Exploration Modules (`curiosity/`)

| Module | File | Intrinsic Reward Signal |
|---|---|---|
| ICM | `curiosity/icm.py` | Forward model prediction error in embedding space |
| Count-based | `curiosity/count_based.py` | `1/sqrt(visit_count)` via random projection hash |
| RIDE | `curiosity/ride.py` | k-NN distance in embedding space (episodic + global memory) |

Each module exposes `compute_intrinsic_reward(state, action, next_state)` and is updated alongside the PPO policy.

### ACWI Enhancement (`ppo-acwi-curiosity/` only)

A **Beta Network** learns β(s): state → scalar. The combined reward is:

```
R_combined = R_extrinsic + α * β(s) * R_intrinsic
```

β is trained via a **correlation loss** that aligns `β(s) * R_intrinsic` with future extrinsic returns, making exploration adaptive per-state without manual tuning. `training_logger.py` logs β values, meta-loss, and reward component ratios for analysis.

### Training Loop (`train.py`)

- Outer loop: seeds (`seed_start` → `seed_end`)
- Inner loop: timesteps up to `max_steps`
- PPO update every `update_timestep = max_ep_len * 4` steps (default: 4000)
- Checkpoints + trajectory heatmaps saved every 100k steps

### Key Hyperparameters (defined at top of `train.py`)

```python
K_epochs = 80, eps_clip = 0.2, gamma = 0.99, gae_lambda = 0.95
lr_actor = 0.0003, lr_critic = 0.001
icm_intr_strength = 0.001, count_intr_strength = 0.001, ride_intr_strength = 0.001
```

### Output Artifacts

- `logs/{env}/PPO{suffix}_{env}_seed_{seed}.csv` — reward curves
- `models/{env}/PPO{suffix}_{env}_seed_{seed}.pth` — model checkpoints
- `*.trajectory.csv` / `*.trajectory.heatmap.png` — agent position data
