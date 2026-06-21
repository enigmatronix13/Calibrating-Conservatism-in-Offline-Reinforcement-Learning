# Offline RL Experiments — hopper-medium-v2

## Protocol

Educational reimplementations with matched architecture and training budget.
Bellman backups use target critics (gamma=0.99, tau=0.005).

- **Dataset:** `hopper-medium-v2`
- **Seeds:** [0, 1, 2]
- **Score:** mean Q(s, pi(s)) over first 5000 dataset states;
  table = mean +/- std of final 3 checkpoints across seeds.
- **Note:** Not a D4RL environment rollout normalised return.

## Results

| Algorithm | Mean ± Std |
|-----------|------------|
| CQL | 11.881 ± 0.121 |
| APTQ-CQL | 8.919 ± 0.163 |
| UWAC | 0.530 ± 0.116 |
| TD3-BC | 0.372 ± 0.122 |
| BEAR | 0.328 ± 1.434 |
| Ensemble | 0.307 ± 0.044 |
| MOPO | 0.029 ± 0.000 |

Generated: 2026-06-21T16:32:42.175414
