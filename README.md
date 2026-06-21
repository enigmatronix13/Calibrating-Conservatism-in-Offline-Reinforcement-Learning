# Calibrating Conservatism in Offline Reinforcement Learning

Empirical study of pessimism calibration across three D4RL datasets spanning
two domains:

| Domain | Dataset | `MAZE2D_DATASET` value | Results dir |
|--------|---------|------------------------|-------------|
| Navigation (sparse) | maze2d-medium-sparse-v1 | `maze2d-medium-sparse-v1` (default) | `maze_2d_results/` |
| Navigation (sparse) | maze2d-large-sparse-v1  | `maze2d-large-sparse-v1` | `maze2d_large_results/` |
| Locomotion (dense)  | hopper-medium-v2        | `hopper-medium-v2` | `hopper_results/` |

The active dataset is selected with the `MAZE2D_DATASET` environment variable
(see the registry in `experiments/common.py`). Dense-reward hopper is scaled
by `1e-2` so its Q-values share the maze value range.

## Requirements

```bash
pip install -r requirements.txt
```

## Reproduce (training + proxy scores)

```bash
cd experiments
# default = maze2d-medium-sparse-v1
python maze2d_offline_rl.py
# other datasets (bash):
MAZE2D_DATASET=maze2d-large-sparse-v1 python maze2d_offline_rl.py
MAZE2D_DATASET=hopper-medium-v2       python maze2d_offline_rl.py
```

PowerShell:

```powershell
$env:MAZE2D_DATASET="hopper-medium-v2"; python maze2d_offline_rl.py
```

Each run writes to its dataset's results dir:
- `results_summary.json` / `.csv` — proxy scores (3 seeds)
- `all_runs.json`, `training_curves.json` — per-seed curves
- `checkpoints/` — per-algorithm actor/critic weights
- `config.json` — full protocol metadata

**Protocol:** matched 256–256 MLPs, Adam `3e-4`, Bellman backups with target
critics (`γ=0.99`, `τ=0.005`), Bellman target clipped to ±10, `10×200`
gradient steps per seed. Score = mean `Q(s, π(s))` on 5,000 dataset states
(a within-dataset proxy, **not** a D4RL rollout / normalised return).

## Coverage-tier diagnostics (CER / URC)

```bash
cd experiments
MAZE2D_DATASET=<dataset> python compute_diagnostics.py
```

Requires checkpoints from the training step. Uses a `5e4`-transition kNN
subsample (`k=10`). Writes `<results_dir>/diagnostics.json`.

## Figures

```bash
cd experiments
MAZE2D_DATASET=<dataset> python plot_figures.py
```

Writes to `paper/figures/` and `overleaf/figures/`. Filenames are prefixed per
dataset (`large_`, `hopper_`; none for medium).

## Paper (Overleaf)

**Upload folder:** [`overleaf/`](overleaf/) — `main.tex` + `figures/` + `data/`
(ready for Overleaf zip upload). Targets a 6-page IEEE conference layout.

Source copy: [`paper/calibrating_conservatism.tex`](paper/calibrating_conservatism.tex)

Compile locally (needs a TeX distribution):

```bash
pdflatex paper/calibrating_conservatism.tex
pdflatex paper/calibrating_conservatism.tex
```

## Key findings

- **Stable ranking** on all three datasets: CQL > APTQ-CQL ≫ {UWAC, Ensemble, TD3-BC} > MOPO/BEAR.
- **CER < 0** in every tier/dataset: CQL-family critics are optimistic vs. a Monte Carlo value baseline; APTQ-CQL reduces the excess, most in well-covered tiers.
- **URC sign flips by domain**: positive on the sparse mazes (uncertainty fails to track return), negative on hopper (uncertainty anticorrelates with return) → ensemble disagreement is a usable calibration signal only where coverage is dense.

## Submission checklist

- [x] Unified training protocol across 7 algorithms
- [x] Bellman + target networks documented
- [x] Multi-domain evaluation (2 maze + 1 locomotion dataset)
- [x] Coverage-tier CER/URC diagnostics on all datasets
- [x] Honest limitation of proxy vs D4RL normalised return
- [ ] D4RL MuJoCo rollouts (optional; requires `d4rl` + MuJoCo install)
