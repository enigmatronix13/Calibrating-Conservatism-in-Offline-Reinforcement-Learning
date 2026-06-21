# Calibrating Conservatism in Offline Reinforcement Learning

Rigorous empirical study of pessimism calibration on `maze2d-medium-sparse-v1`.

## Requirements

```bash
pip install -r requirements.txt
```

## Reproduce (training + proxy scores)

```bash
cd experiments
python maze2d_offline_rl.py
```

Writes to `maze_2d_results/`:
- `results_summary.json` — Table I proxy scores (3 seeds)
- `checkpoints/` — per-algorithm actor/critic weights
- `config.json` — full protocol metadata

**Protocol:** matched 256–256 MLPs, Adam $3\times10^{-4}$, Bellman backups with target critics ($\gamma{=}0.99$, $\tau{=}0.005$), $10\times200$ gradient steps per seed. Score = mean $Q(s,\pi(s))$ on 5,000 dataset states (not D4RL rollout).

## Coverage-tier diagnostics (CER / URC)

```bash
cd experiments
python compute_diagnostics.py
```

Requires checkpoints from the training step. Writes `maze_2d_results/diagnostics.json`.

## Paper (Overleaf)

**Upload folder:** [`overleaf/`](overleaf/) — `main.tex` + `figures/` (ready for Overleaf zip upload).

Source copy: [`paper/calibrating_conservatism.tex`](paper/calibrating_conservatism.tex)

Regenerate figures:

```bash
python experiments/plot_figures.py
```

Compile locally:

```bash
pdflatex paper/calibrating_conservatism.tex
pdflatex paper/calibrating_conservatism.tex
```

## Notebooks

`maze2d-dm.ipynb` — legacy exploratory notebook. Use `experiments/` scripts for submission-aligned results.

## Submission checklist

- [x] Unified training protocol across 7 algorithms
- [x] Bellman + target networks documented
- [x] Coverage-tier CER/URC diagnostics
- [x] Honest limitation of proxy vs D4RL normalised return
- [ ] D4RL MuJoCo rollouts (optional; requires `d4rl` + MuJoCo install)
