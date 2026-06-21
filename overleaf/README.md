# Overleaf upload bundle

Upload **the contents of this folder** to Overleaf (or zip and upload the zip).

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Full paper source (IEEE conference format) |
| `figures/*.pdf` | Training curves, scores, CER/URC, coverage map |

## Compile on Overleaf

1. New Project → Upload Project → select this folder (or a zip of it).
2. Menu → Compiler: **pdfLaTeX**.
3. Main document: **main.tex** (should be auto-detected).
4. Click **Recompile** (run twice if references look wrong).

Overleaf includes the `IEEEtran` class by default for conference templates; if needed, choose the **IEEE Conference Template** and replace its `main.tex` with this one, keeping the `figures/` folder.

## Regenerate figures from code

From the repository root:

```bash
pip install matplotlib numpy h5py
python experiments/plot_figures.py
```

This refreshes both `paper/figures/` and `overleaf/figures/`.

## Repository

https://github.com/enigmatronix13/Calibrating-Conservatism-in-Offline-Reinforcement-Learning
