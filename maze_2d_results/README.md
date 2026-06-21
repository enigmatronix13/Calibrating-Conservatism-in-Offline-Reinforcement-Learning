# Offline RL Experiments — maze2d-medium-sparse-v1

## Protocol

Educational reimplementations with matched architecture and training budget.
Bellman backups use target critics (gamma=0.99, tau=0.005).

- **Dataset:** `maze2d-medium-sparse-v1`
- **Seeds:** [0, 1, 2]
- **Score:** mean Q(s, pi(s)) over first 5000 dataset states;
  table = mean +/- std of final 3 checkpoints across seeds.
- **Note:** Not a D4RL environment rollout normalised return.

## Results

| Algorithm | Mean ± Std |
|-----------|------------|
| CQL | 2.293 ± 0.141 |
| APTQ-CQL | 1.333 ± 0.143 |
| UWAC | 0.144 ± 0.016 |
| Ensemble | 0.118 ± 0.010 |
| TD3-BC | 0.112 ± 0.023 |
| MOPO | 0.014 ± 0.001 |
| BEAR | -0.001 ± 0.054 |

Generated: 2026-06-21T10:18:03.065695
