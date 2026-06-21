"""
Coverage-tier diagnostics: CER and URC on maze2d-medium-sparse-v1.

Requires checkpoints from experiments/maze2d_offline_rl.py.
Uses a subsample for kNN coverage density (2M transitions).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

from common import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    ENSEMBLE_K,
    GAMMA,
    INNER_STEPS,
    KNN_K,
    LR,
    OUTER_STEPS,
    OUTPUT_DIR,
    Actor,
    Critic,
    device,
    download_dataset,
    load_checkpoint,
    load_dataset_cpu,
    sample_batch,
    set_seed,
)

DIAG_SAMPLE = 10_000
TIER_LABELS = ("High", "Med", "Low")


def coverage_tiers(obs: np.ndarray, k: int = KNN_K) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign subsampled transitions a coverage density rho and tier label."""
    n = len(obs)
    idx = np.arange(n)
    if n > DIAG_SAMPLE:
        idx = np.sort(np.random.choice(n, DIAG_SAMPLE, replace=False))
    sub_obs = obs[idx]

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=1)
    nn.fit(sub_obs)
    dist, _ = nn.kneighbors(sub_obs)
    d_k = dist[:, k]
    rho = 1.0 / (d_k + 1e-6)

    tertiles = np.quantile(rho, [1 / 3, 2 / 3])
    tiers = np.empty(len(idx), dtype=object)
    tiers[rho >= tertiles[1]] = "High"
    tiers[(rho >= tertiles[0]) & (rho < tertiles[1])] = "Med"
    tiers[rho < tertiles[0]] = "Low"
    return idx, rho.astype(np.float32), tiers


def train_mc_critic(data, steps: int = 500) -> Critic:
    """Behavioural Monte Carlo value baseline Q_BC(s,a) approximating return."""
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    critic = Critic(sd, ad).to(device)
    opt = optim.Adam(critic.parameters(), lr=LR)
    target = torch.tensor(data.mc_returns, dtype=torch.float32).unsqueeze(1)
    n_batches = max(steps // 5, 1)
    for _ in range(n_batches):
        idx = np.random.randint(0, len(data.obs_t), BATCH_SIZE)
        obs = data.obs_t[idx].to(device)
        act = data.act_t[idx].to(device)
        q = critic(obs, act)
        loss = ((q - target[idx].to(device)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return critic


def train_uncertainty_ensemble(data) -> list[Critic]:
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    ens = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    opts = [optim.Adam(q.parameters(), lr=LR) for q in ens]
    target = torch.tensor(data.mc_returns, dtype=torch.float32).unsqueeze(1)
    for _ in range(5):
        for _ in range(50):
            idx = np.random.randint(0, len(data.obs_t), BATCH_SIZE)
            obs = data.obs_t[idx].to(device)
            act = data.act_t[idx].to(device)
            for q, opt in zip(ens, opts):
                pred = q(obs, act)
                loss = ((pred - target[idx].to(device)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    return ens


@torch.no_grad()
def batch_q(critic: Critic, obs_t, act_t, batch_size: int = 8192) -> np.ndarray:
    out = []
    for i in range(0, len(obs_t), batch_size):
        out.append(critic(obs_t[i : i + batch_size], act_t[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out).reshape(-1)


@torch.no_grad()
def batch_uncertainty(ensemble: list[Critic], obs_t, act_t, batch_size: int = 8192) -> np.ndarray:
    qs = [batch_q(q, obs_t, act_t, batch_size) for q in ensemble]
    return np.std(np.stack(qs), axis=0)


def load_algo_critic(name: str, seed: int = 0) -> Critic:
    ckpt = load_checkpoint(CHECKPOINT_DIR / f"{name}_seed{seed}.pt")
    sd, ad = ckpt["state_dim"], ckpt["action_dim"]
    critic = Critic(sd, ad).to(device)
    if "critics" in ckpt:
        critic.load_state_dict(ckpt["critics"][0])
    else:
        critic.load_state_dict(ckpt["critic"])
    critic.eval()
    return critic


def compute_tier_stats(
    tiers: np.ndarray,
    q_bc: np.ndarray,
    q_algo: np.ndarray,
    uncertainty: np.ndarray,
    mc_returns: np.ndarray,
) -> dict:
    u_mean = float(uncertainty.mean())
    out = {}
    for tier in TIER_LABELS:
        mask = tiers == tier
        if mask.sum() < 100:
            continue
        excess = q_bc[mask] - q_algo[mask]
        cer = float(excess.mean() / (u_mean + 1e-8))
        if np.std(uncertainty[mask]) < 1e-8 or np.std(mc_returns[mask]) < 1e-8:
            urc = 0.0
        else:
            urc = float(np.corrcoef(uncertainty[mask], mc_returns[mask])[0, 1])
        out[tier] = {
            "n": int(mask.sum()),
            "CER": cer,
            "URC": urc,
            "mean_Q_BC": float(q_bc[mask].mean()),
            "mean_Q_algo": float(q_algo[mask].mean()),
            "mean_uncertainty": float(uncertainty[mask].mean()),
        }
    return out


def main():
    set_seed(0)
    path = download_dataset(OUTPUT_DIR)
    data = load_dataset_cpu(path)
    obs_norm = data.obs_t.cpu().numpy()

    print("Computing coverage tiers (subsampled kNN) ...")
    idx, rho, tiers = coverage_tiers(obs_norm)

    print("Training Q_BC and uncertainty ensemble ...")
    q_bc_net = train_mc_critic(data)
    ensemble = train_uncertainty_ensemble(data)

    obs_s = data.obs_t[idx].to(device)
    act_s = data.act_t[idx].to(device)
    mc = data.mc_returns[idx]

    print(f"Scoring {len(idx)} transitions ...")
    q_bc = batch_q(q_bc_net, obs_s, act_s)
    uncertainty = batch_uncertainty(ensemble, obs_s, act_s)

    results = {
        "dataset": "maze2d-medium-sparse-v1",
        "knn_k": KNN_K,
        "diag_sample": DIAG_SAMPLE,
        "global_mean_uncertainty": float(uncertainty.mean()),
        "algorithms": {},
    }

    for algo in ("CQL", "APTQ-CQL"):
        ckpt_path = CHECKPOINT_DIR / f"{algo}_seed0.pt"
        if not ckpt_path.exists():
            print(f"Skipping {algo}: checkpoint missing ({ckpt_path})")
            continue
        critic = load_algo_critic(algo, seed=0)
        q_algo = batch_q(critic, obs_s, act_s)
        results["algorithms"][algo] = compute_tier_stats(tiers, q_bc, q_algo, uncertainty, mc)

    out_path = OUTPUT_DIR / "diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved diagnostics to {out_path}")
    for algo, tiers_stats in results["algorithms"].items():
        print(f"\n{algo}:")
        for tier, stats in tiers_stats.items():
            print(
                f"  {tier:4s}  CER={stats['CER']:+.3f}  URC={stats['URC']:+.3f}  "
                f"n={stats['n']}"
            )


if __name__ == "__main__":
    main()
