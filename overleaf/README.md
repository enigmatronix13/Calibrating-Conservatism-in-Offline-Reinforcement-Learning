# Overleaf upload bundle

Upload **the contents of this folder** to Overleaf (or zip and upload the zip).
The paper documents three D4RL datasets — `maze2d-medium-sparse-v1`,
`maze2d-large-sparse-v1`, and `hopper-medium-v2` — and targets a 6-page IEEE
conference layout.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full paper source (IEEE conference format) |
| `figures/*.pdf` | Training curves, CER/URC, and coverage maps for all three datasets |
| `data/*.{csv,json}` | Per-dataset proxy scores (`*_results_summary`) and CER/URC diagnostics (`*_diagnostics`) |

Figure naming: no prefix = `maze2d-medium-sparse`, `large_` = `maze2d-large-sparse`,
`hopper_` = `hopper-medium-v2`.

## Compile on Overleaf

1. New Project → Upload Project → select a zip of this folder.
2. Menu → Compiler: **pdfLaTeX**.
3. Main document: **main.tex** (auto-detected).
4. Click **Recompile** (run twice so cross-references resolve).

Overleaf bundles the `IEEEtran` class by default; if not, pick the **IEEE
Conference Template** and replace its `main.tex` with this one, keeping the
`figures/` folder.

## Regenerate figures from code

From the repository root, for each dataset:

```bash
pip install -r requirements.txt
cd experiments
MAZE2D_DATASET=maze2d-medium-sparse-v1 python plot_figures.py
MAZE2D_DATASET=maze2d-large-sparse-v1  python plot_figures.py
MAZE2D_DATASET=hopper-medium-v2        python plot_figures.py
```

This refreshes both `paper/figures/` and `overleaf/figures/`.

## Repository

https://github.com/enigmatronix13/Calibrating-Conservatism-in-Offline-Reinforcement-Learning
