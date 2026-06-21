"""
Unified offline RL baseline comparison on maze2d-medium-sparse-v1.

Protocol:
  - Shared MLP architecture, Adam lr=3e-4, batch 256, gamma=0.99
  - Target critics with polyak averaging (tau=0.005)
  - Bellman backup for all actor-critic methods
  - Evaluation: mean Q(s, pi(s)) on first 5000 dataset states
  - Reported score: mean of final 3 checkpoints, mean +/- std over 3 seeds

Educational reimplementations aligned in training budget; not official author code.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from common import (
    CHECKPOINT_DIR,
    CQL_ALPHA,
    DATASET_NAME,
    ENSEMBLE_K,
    INNER_STEPS,
    LR,
    MOPO_PENALTY,
    OUTER_STEPS,
    OUTPUT_DIR,
    SEEDS,
    Actor,
    BEAR_MMD_WEIGHT,
    Critic,
    DatasetBundle,
    DynamicsModel,
    VAE,
    actor_bc_q_loss,
    critic_bellman_loss,
    device,
    download_dataset,
    eval_score,
    load_dataset,
    mmd_loss,
    sample_batch,
    save_checkpoint,
    set_seed,
    soft_update,
    update_critic,
)

# imported below after train fns defined - fix circular import by defining trains here


def train_td3bc(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    actor = Actor(sd, ad).to(device)
    critic = Critic(sd, ad).to(device)
    critic_t = Critic(sd, ad).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    opt_c = optim.Adam(critic.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            update_critic(critic, critic_t, actor, opt_c, b)
            loss = actor_bc_q_loss(actor, critic, b)
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
        hist.append(eval_score(actor, critic, data.obs_t))
    return hist, actor, critic


def train_cql(data: DatasetBundle, alpha: float = CQL_ALPHA):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    actor = Actor(sd, ad).to(device)
    critic = Critic(sd, ad).to(device)
    critic_t = Critic(sd, ad).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    opt_c = optim.Adam(critic.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            q_data = critic(b.obs, b.act)
            rand_a = torch.randn_like(b.act).clamp(-1, 1)
            q_rand = critic(b.obs, rand_a)
            bellman = critic_bellman_loss(critic, critic_t, actor, b)
            cql = alpha * (q_rand.mean() - q_data.mean())
            loss = bellman + cql
            opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_c.step()
            soft_update(critic_t, critic)
            loss_a = actor_bc_q_loss(actor, critic, b)
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
        hist.append(eval_score(actor, critic, data.obs_t))
    return hist, actor, critic


def train_ensemble(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    actor = Actor(sd, ad).to(device)
    ens = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    ens_t = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    for q, qt in zip(ens, ens_t):
        qt.load_state_dict(q.state_dict())
    opts = [optim.Adam(q.parameters(), lr=LR) for q in ens]
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            for q, qt, opt in zip(ens, ens_t, opts):
                update_critic(q, qt, actor, opt, b)
            loss = actor_bc_q_loss(actor, ens[0], b, critics=ens)
            opt_a.zero_grad()
            loss.backward()
            opt_a.step()
        hist.append(eval_score(actor, ens, data.obs_t))
    return hist, actor, ens


def train_bear(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    vae = VAE(sd, ad).to(device)
    actor = Actor(sd, ad).to(device)
    critic = Critic(sd, ad).to(device)
    critic_t = Critic(sd, ad).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_v = optim.Adam(vae.parameters(), lr=1e-3)
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    opt_c = optim.Adam(critic.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            recon, mean, std = vae(b.obs, b.act)
            vae_loss = ((recon - b.act) ** 2).mean() - 0.5 * (
                1 + torch.log(std ** 2) - mean ** 2 - std ** 2
            ).mean()
            opt_v.zero_grad()
            vae_loss.backward()
            opt_v.step()
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            update_critic(critic, critic_t, actor, opt_c, b)
            pred = actor(b.obs)
            with torch.no_grad():
                sampled = vae.decode(b.obs)
            actor_loss = -critic(b.obs, pred).mean() + BEAR_MMD_WEIGHT * mmd_loss(pred, sampled)
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()
        hist.append(eval_score(actor, critic, data.obs_t))
    return hist, actor, critic


def train_uwac(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    actor = Actor(sd, ad).to(device)
    ens = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    ens_t = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    for q, qt in zip(ens, ens_t):
        qt.load_state_dict(q.state_dict())
    opts = [optim.Adam(q.parameters(), lr=LR) for q in ens]
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            for q, qt, opt in zip(ens, ens_t, opts):
                b = sample_batch(data)
                update_critic(q, qt, actor, opt, b)
            b = sample_batch(data)
            pred = actor(b.obs)
            q_preds = torch.stack([q(b.obs, pred) for q in ens])
            q_mean = q_preds.mean(0)
            q_std = q_preds.std(0)
            actor_loss = -(q_mean - q_std).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()
        hist.append(eval_score(actor, ens, data.obs_t))
    return hist, actor, ens


def train_mopo(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    model = DynamicsModel(sd, ad).to(device)
    actor = Actor(sd, ad).to(device)
    critic = Critic(sd, ad).to(device)
    critic_t = Critic(sd, ad).to(device)
    critic_t.load_state_dict(critic.state_dict())
    opt_m = optim.Adam(model.parameters(), lr=LR)
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    opt_c = optim.Adam(critic.parameters(), lr=LR)
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            pred_next = model(b.obs, b.act)
            model_loss = ((pred_next - b.next_obs) ** 2).mean()
            opt_m.zero_grad()
            model_loss.backward()
            opt_m.step()
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            pred_next = model(b.obs, b.act)
            penalty = (pred_next - b.next_obs).pow(2).mean(1, keepdim=True)
            q = critic(b.obs, b.act)
            target = b.rew.unsqueeze(1) - MOPO_PENALTY * penalty
            critic_loss = ((q - target.detach()) ** 2).mean()
            opt_c.zero_grad()
            critic_loss.backward()
            opt_c.step()
            soft_update(critic_t, critic)
            loss_a = actor_bc_q_loss(actor, critic, b)
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
        hist.append(eval_score(actor, critic, data.obs_t))
    return hist, actor, critic


def train_aptq_cql(data: DatasetBundle):
    sd, ad = data.obs_t.shape[1], data.act_t.shape[1]
    actor = Actor(sd, ad).to(device)
    critic = Critic(sd, ad).to(device)
    critic_t = Critic(sd, ad).to(device)
    critic_t.load_state_dict(critic.state_dict())
    ens = [Critic(sd, ad).to(device) for _ in range(ENSEMBLE_K)]
    ens_opts = [optim.Adam(q.parameters(), lr=LR) for q in ens]
    opt_c = optim.Adam(critic.parameters(), lr=LR)
    opt_a = optim.Adam(actor.parameters(), lr=LR)
    alpha, beta = 1.0, 0.1
    hist = []
    for _ in range(OUTER_STEPS):
        for _ in range(INNER_STEPS):
            b = sample_batch(data)
            q_data = critic(b.obs, b.act)
            rand_a = torch.randn_like(b.act).clamp(-1, 1)
            q_rand = critic(b.obs, rand_a)
            with torch.no_grad():
                qs = torch.stack([q(b.obs, b.act) for q in ens])
                uncertainty = qs.std(0).mean().item()
            phi = (q_rand.mean() - q_data.mean()).item()
            alpha = float(np.clip((1 - beta) * alpha + beta * (phi + uncertainty), 0.5, 5.0))
            bellman = critic_bellman_loss(critic, critic_t, actor, b)
            cql = alpha * (q_rand.mean() - q_data.mean())
            loss = bellman + cql
            opt_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_c.step()
            soft_update(critic_t, critic)
            for q, e_opt in zip(ens, ens_opts):
                pred = q(b.obs, b.act)
                e_loss = ((pred - b.rew.unsqueeze(1)) ** 2).mean()
                e_opt.zero_grad()
                e_loss.backward()
                e_opt.step()
            loss_a = actor_bc_q_loss(actor, critic, b)
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
        hist.append(eval_score(actor, critic, data.obs_t))
    return hist, actor, critic


TRAINERS = {
    "TD3-BC": train_td3bc,
    "CQL": train_cql,
    "Ensemble": train_ensemble,
    "BEAR": train_bear,
    "UWAC": train_uwac,
    "MOPO": train_mopo,
    "APTQ-CQL": train_aptq_cql,
}


def summarize_runs(all_runs: dict[str, np.ndarray]) -> dict[str, list[float]]:
    summary = {}
    for name, runs in all_runs.items():
        finals = np.array([np.mean(run[-3:]) for run in runs])
        summary[name] = [float(finals.mean()), float(finals.std(ddof=0))]
    return summary


def save_results(all_runs, summary, config):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = {k: np.mean(v, axis=0).tolist() for k, v in all_runs.items()}
    with open(OUTPUT_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    with open(OUTPUT_DIR / "all_runs.json", "w") as f:
        json.dump({k: v.tolist() for k, v in all_runs.items()}, f, indent=2)
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["Algorithm,Mean,Std\n"]
    for algo, (mean, std) in summary.items():
        lines.append(f"{algo},{mean},{std}\n")
    (OUTPUT_DIR / "results_summary.csv").write_text("".join(lines))
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    readme = f"""# Offline RL Experiments — {DATASET_NAME}

## Protocol

Educational reimplementations with matched architecture and training budget.
Bellman backups use target critics (gamma={config['gamma']}, tau={config['tau']}).

- **Dataset:** `{DATASET_NAME}`
- **Seeds:** {SEEDS}
- **Score:** mean Q(s, pi(s)) over first {config['eval_states']} dataset states;
  table = mean +/- std of final 3 checkpoints across seeds.
- **Note:** Not a D4RL environment rollout normalised return.

## Results

| Algorithm | Mean ± Std |
|-----------|------------|
"""
    for algo, (mean, std) in sorted(summary.items(), key=lambda x: -x[1][0]):
        readme += f"| {algo} | {mean:.3f} ± {std:.3f} |\n"
    readme += f"\nGenerated: {datetime.now().isoformat()}\n"
    (OUTPUT_DIR / "README.md").write_text(readme)
    print(f"\nSaved results to {OUTPUT_DIR}")


def main():
    data_path = download_dataset(OUTPUT_DIR)
    data = load_dataset(data_path)
    config = {
        "dataset": DATASET_NAME,
        "date": datetime.now().isoformat(),
        "device": str(device),
        "seeds": SEEDS,
        "outer_steps": OUTER_STEPS,
        "inner_steps": INNER_STEPS,
        "gamma": 0.99,
        "tau": 0.005,
        "eval_states": 5000,
        "implementation": "educational reimplementation with unified Bellman backup",
        "evaluation": "mean_Q(s, pi(s)); not D4RL rollout",
    }
    all_runs = {}
    for name, train_fn in TRAINERS.items():
        print(f"Running {name} ...")
        runs = []
        for seed in SEEDS:
            set_seed(seed)
            hist, actor, critic = train_fn(data)
            runs.append(hist)
            ckpt = CHECKPOINT_DIR / f"{name.replace(' ', '_')}_seed{seed}.pt"
            payload = {
                "algorithm": name,
                "seed": seed,
                "actor": actor.state_dict(),
                "state_dim": data.obs_t.shape[1],
                "action_dim": data.act_t.shape[1],
            }
            if isinstance(critic, list):
                payload["critics"] = [c.state_dict() for c in critic]
            else:
                payload["critic"] = critic.state_dict()
            save_checkpoint(ckpt, **payload)
            print(f"  seed {seed}: final={np.mean(hist[-3:]):.4f}")
        all_runs[name] = np.array(runs)
    summary = summarize_runs(all_runs)
    save_results(all_runs, summary, config)
    print("\n=== Summary (mean ± std) ===")
    for algo, (mean, std) in sorted(summary.items(), key=lambda x: -x[1][0]):
        print(f"  {algo:10s}  {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
