# Calibrating Conservatism in Offline Reinforcement Learning

Unified comparison of seven offline RL algorithms (TD3-BC, CQL, Q-ensemble, BEAR, UWAC, MOPO, APTQ-CQL) with coverage-tier CER/URC diagnostics, testing whether fixed-α pessimism is spatially miscalibrated across D4RL maze2d navigation and hopper locomotion.

| Domain | Dataset | Results dir |
|---|---|---|
| Navigation (sparse) | `maze2d-medium-sparse-v1` (default) | `maze_2d_results/` |
| Navigation (sparse) | `maze2d-large-sparse-v1` | `maze2d_large_results/` |
| Locomotion (dense) | `hopper-medium-v2` | `hopper_results/` |

Dataset is selected via the `MAZE2D_DATASET` env var (registry in `experiments/common.py`). Hopper rewards are scaled by 1e-2 to share the maze Q-value range.

## Setup
```
pip install -r requirements.txt
```

## Run
```bash
cd experiments

# Train and evaluate offline RL baselines
python offline_rl.py --dataset maze2d-medium-sparse-v1
python offline_rl.py --dataset maze2d-large-sparse-v1
python offline_rl.py --dataset hopper-medium-v2

# Compute coverage-tier diagnostics (requires trained checkpoints)
python compute_diagnostics.py --dataset maze2d-medium-sparse-v1
python compute_diagnostics.py --dataset maze2d-large-sparse-v1
python compute_diagnostics.py --dataset hopper-medium-v2

# Generate figures for the paper and Overleaf
python plot_figures.py --dataset maze2d-medium-sparse-v1
python plot_figures.py --dataset maze2d-large-sparse-v1
python plot_figures.py --dataset hopper-medium-v2
```
Each results dir gets `results_summary.json/csv`, `all_runs.json`, `training_curves.json`, `checkpoints/`, `config.json`.

**Protocol:** matched 256–256 MLPs, Adam 3e-4, target-critic Bellman backups (γ=0.99, τ=0.005), target clip ±10, 10×200 grad steps/seed. Score = mean Q(s, π(s)) on 5,000 dataset states - a within-dataset proxy, not a D4RL normalised return.

## Key findings
- Stable ranking on all three datasets: CQL > APTQ-CQL ≫ {UWAC, Ensemble, TD3-BC} > MOPO/BEAR.
- CER < 0 in every tier/dataset - CQL-family critics are optimistic vs. a Monte Carlo baseline; APTQ-CQL reduces the excess, most in well-covered tiers.
- URC flips sign by domain: positive on sparse mazes (uncertainty doesn't track return), negative on hopper (uncertainty anticorrelates with return) - ensemble disagreement is only a reliable calibration signal where coverage is dense.
