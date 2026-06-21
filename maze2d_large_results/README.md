# Offline RL Experiments — maze2d-large-sparse-v1

## Protocol

Educational reimplementations with matched architecture and training budget.
Bellman backups use target critics (gamma=0.99, tau=0.005).

- **Dataset:** `maze2d-large-sparse-v1`
- **Seeds:** [0, 1, 2]
- **Score:** mean Q(s, pi(s)) over first 5000 dataset states;
  table = mean +/- std of final 3 checkpoints across seeds.
- **Note:** Not a D4RL environment rollout normalised return.

## Results

| Algorithm | Mean ± Std |
|-----------|------------|
| CQL | 2.899 ± 0.061 |
| APTQ-CQL | 1.632 ± 0.050 |
| UWAC | 0.268 ± 0.004 |
| Ensemble | 0.222 ± 0.010 |
| TD3-BC | 0.202 ± 0.030 |
| BEAR | 0.105 ± 0.015 |
| MOPO | 0.024 ± 0.004 |

Generated: 2026-06-21T15:24:05.386888
