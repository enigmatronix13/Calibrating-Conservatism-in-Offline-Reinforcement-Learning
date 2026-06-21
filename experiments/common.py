"""Shared utilities for maze2d offline RL experiments."""

from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATASET_NAME = "maze2d-medium-sparse-v1"
DATA_URL = (
    "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/"
    "maze2d-medium-sparse-v1.hdf5"
)
HDF5_FILE = "maze2d-medium-sparse-v1.hdf5"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "maze_2d_results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

SEEDS = [0, 1, 2]
OUTER_STEPS = 10
INNER_STEPS = 200
BATCH_SIZE = 256
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
GRAD_CLIP = 1.0
Q_CLIP = 10.0
ENSEMBLE_K = 5
CQL_ALPHA = 1.0
BC_WEIGHT = 0.1
BEAR_MMD_WEIGHT = 10.0
MOPO_PENALTY = 0.5
EVAL_STATES = 5000
KNN_K = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DatasetBundle:
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    mc_returns: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    obs_t: torch.Tensor
    act_t: torch.Tensor
    rew_t: torch.Tensor
    next_obs_t: torch.Tensor
    done_t: torch.Tensor


@dataclass
class Batch:
    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.mean = nn.Linear(256, latent_dim)
        self.log_std = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def encode(self, s, a):
        x = self.encoder(torch.cat([s, a], dim=1))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-4, 4)
        return mean, torch.exp(log_std)

    def decode(self, s, z=None):
        if z is None:
            z = torch.randn((s.shape[0], 16), device=s.device)
        return self.decoder(torch.cat([s, z], dim=1))

    def forward(self, s, a):
        mean, std = self.encode(s, a)
        z = mean + std * torch.randn_like(std)
        return self.decode(s, z), mean, std


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


def download_dataset(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / HDF5_FILE
    if not path.exists():
        print(f"Downloading {DATASET_NAME} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def compute_mc_returns(rew: np.ndarray, episode_end: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Backward discounted return for each transition index."""
    mc = np.zeros_like(rew, dtype=np.float32)
    i = len(rew) - 1
    g = 0.0
    while i >= 0:
        g = rew[i] + gamma * g
        mc[i] = g
        if episode_end[i] and i > 0:
            g = 0.0
        i -= 1
    return mc


def load_dataset_cpu(path: Path) -> DatasetBundle:
    """Load dataset with tensors on CPU (for diagnostics on large HDF5 files)."""
    with h5py.File(path, "r") as f:
        obs = f["observations"][:].astype(np.float32)
        act = f["actions"][:].astype(np.float32)
        rew = f["rewards"][:].astype(np.float32).reshape(-1)
        terminals = f["terminals"][:].astype(np.bool_).reshape(-1)
        timeouts = f["timeouts"][:].astype(np.bool_).reshape(-1)

    episode_end = terminals | timeouts
    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    for i in range(len(obs) - 1):
        if episode_end[i]:
            next_obs[i] = obs[i]

    obs_mean = obs.mean(0)
    obs_std = obs.std(0) + 1e-3
    obs_norm = (obs - obs_mean) / obs_std
    mc_returns = compute_mc_returns(rew, episode_end)

    return DatasetBundle(
        obs=obs,
        act=act,
        rew=rew,
        next_obs=next_obs,
        done=episode_end.astype(np.float32),
        mc_returns=mc_returns,
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_t=torch.tensor(obs_norm),
        act_t=torch.tensor(act),
        rew_t=torch.tensor(rew),
        next_obs_t=torch.tensor((next_obs - obs_mean) / obs_std),
        done_t=torch.tensor(episode_end.astype(np.float32)),
    )


def load_dataset(path: Path) -> DatasetBundle:
    bundle = load_dataset_cpu(path)
    return DatasetBundle(
        obs=bundle.obs,
        act=bundle.act,
        rew=bundle.rew,
        next_obs=bundle.next_obs,
        done=bundle.done,
        mc_returns=bundle.mc_returns,
        obs_mean=bundle.obs_mean,
        obs_std=bundle.obs_std,
        obs_t=bundle.obs_t.to(device),
        act_t=bundle.act_t.to(device),
        rew_t=bundle.rew_t.to(device),
        next_obs_t=bundle.next_obs_t.to(device),
        done_t=bundle.done_t.to(device),
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_batch(data: DatasetBundle) -> Batch:
    idx = np.random.randint(0, len(data.obs_t), BATCH_SIZE)
    return Batch(
        data.obs_t[idx],
        data.act_t[idx],
        data.rew_t[idx],
        data.next_obs_t[idx],
        data.done_t[idx],
    )


def mmd_loss(a1, a2, sigma=10.0):
    diff = a1.unsqueeze(1) - a2.unsqueeze(0)
    dist = (diff ** 2).mean(-1)
    return torch.exp(-dist / (2 * sigma)).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float = TAU):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def bellman_target(
    critic_t: Critic,
    next_obs: torch.Tensor,
    rew: torch.Tensor,
    done: torch.Tensor,
    actor: Actor,
    gamma: float = GAMMA,
):
    with torch.no_grad():
        next_a = actor(next_obs)
        next_q = critic_t(next_obs, next_a).clamp(-Q_CLIP, Q_CLIP)
        target = rew.unsqueeze(1) + gamma * (1.0 - done.unsqueeze(1)) * next_q
        return target.clamp(-Q_CLIP, Q_CLIP)


def critic_bellman_loss(critic, critic_t, actor, batch: Batch) -> torch.Tensor:
    q = critic(batch.obs, batch.act)
    target = bellman_target(critic_t, batch.next_obs, batch.rew, batch.done, actor)
    return ((q - target) ** 2).mean()


def update_critic(critic, critic_t, actor, opt, batch: Batch) -> None:
    loss = critic_bellman_loss(critic, critic_t, actor, batch)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    opt.step()
    soft_update(critic_t, critic)


def actor_bc_q_loss(actor, critic, batch: Batch, critics=None) -> torch.Tensor:
    pred = actor(batch.obs)
    if critics is None:
        q_val = critic(batch.obs, pred)
    else:
        q_val = torch.stack([c(batch.obs, pred) for c in critics]).mean(0)
    bc = ((pred - batch.act) ** 2).mean()
    return -q_val.mean() + BC_WEIGHT * bc


def eval_score(actor: Actor, critics: list[Critic] | Critic, obs_t: torch.Tensor) -> float:
    eval_obs = obs_t[: min(EVAL_STATES, len(obs_t))]
    with torch.no_grad():
        actions = actor(eval_obs)
        if isinstance(critics, list):
            q = torch.stack([c(eval_obs, actions) for c in critics]).mean(0)
        else:
            q = critics(eval_obs, actions)
        return q.mean().item()


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location=device, weights_only=False)
