"""Generate paper figures from maze_2d_results JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "maze_2d_results"
FIG_DIRS = [
    ROOT / "paper" / "figures",
    ROOT / "overleaf" / "figures",
]

ALGO_ORDER = [
    "CQL",
    "APTQ-CQL",
    "UWAC",
    "Ensemble",
    "TD3-BC",
    "MOPO",
    "BEAR",
]
COLORS = {
    "CQL": "#1f77b4",
    "APTQ-CQL": "#ff7f0e",
    "UWAC": "#2ca02c",
    "Ensemble": "#9467bd",
    "TD3-BC": "#8c564b",
    "MOPO": "#e377c2",
    "BEAR": "#7f7f7f",
}
TIERS = ("High", "Med", "Low")


def save_all(fig: plt.Figure, name: str) -> None:
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves() -> None:
    runs = json.load(open(RESULTS / "all_runs.json"))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    checkpoints = np.arange(1, 11)
    for algo in ALGO_ORDER:
        if algo not in runs:
            continue
        arr = np.array(runs[algo], dtype=np.float64)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(checkpoints, mean, label=algo, color=COLORS.get(algo), linewidth=2)
        ax.fill_between(checkpoints, mean - std, mean + std, color=COLORS.get(algo), alpha=0.15)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel(r"Mean $Q(s,\pi(s))$ (proxy)")
    ax.set_title("Unified offline policy-value proxy during training")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    save_all(fig, "training_curves")


def plot_final_scores() -> None:
    summary = json.load(open(RESULTS / "results_summary.json"))
    means = [summary[a][0] for a in ALGO_ORDER if a in summary]
    stds = [summary[a][1] for a in ALGO_ORDER if a in summary]
    labels = [a for a in ALGO_ORDER if a in summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS.get(a, "#333333") for a in labels],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"Final proxy score")
    ax.set_title(r"Final score (mean of last 3 checkpoints, 3 seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    save_all(fig, "final_scores")


def plot_cer_urc() -> None:
    diag = json.load(open(RESULTS / "diagnostics.json"))
    algos = ("CQL", "APTQ-CQL")
    x = np.arange(len(TIERS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    for i, algo in enumerate(algos):
        cer = [diag["algorithms"][algo][t]["CER"] for t in TIERS]
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, cer, width, label=algo, color=COLORS.get(algo))

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TIERS)
    axes[0].set_ylabel("CER")
    axes[0].set_title("Conservative Excess Ratio")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    for i, algo in enumerate(algos):
        urc = [diag["algorithms"][algo][t]["URC"] for t in TIERS]
        offset = (i - 0.5) * width
        axes[1].bar(x + offset, urc, width, label=algo, color=COLORS.get(algo))

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TIERS)
    axes[1].set_ylabel("URC")
    axes[1].set_title("Uncertainty--Return Correlation")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_all(fig, "cer_urc")


def plot_coverage_map() -> None:
    import h5py

    h5_path = RESULTS / "maze2d-medium-sparse-v1.hdf5"
    if not h5_path.exists():
        print("Skipping coverage map: HDF5 not found")
        return

    with h5py.File(h5_path, "r") as f:
        obs = f["observations"][:].astype(np.float32)

    rng = np.random.default_rng(0)
    n = len(obs)
    idx = rng.choice(n, size=min(20_000, n), replace=False)
    xy = obs[idx, :2]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.25, c="#1f77b4", rasterized=True)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Position $y$")
    ax.set_title("Dataset state coverage (subsample)")
    ax.set_aspect("equal", adjustable="box")
    save_all(fig, "coverage_map")


def main():
    plt.rcParams.update({"font.size": 9, "figure.dpi": 120})
    plot_training_curves()
    plot_final_scores()
    plot_cer_urc()
    plot_coverage_map()
    print("Figures written to paper/figures/ and overleaf/figures/")


if __name__ == "__main__":
    main()
