# Offline RL Experiments - Maze2D

## Results Summary

Training curves and final results for offline reinforcement learning algorithms on Maze2D.

### Saved Files
- `training_curves.json` / `training_curves.pkl` - Training curves
- `results_summary.csv` - Final results (mean ± std)
- `results_summary.json` - Results in JSON format  
- `all_runs.pkl` - Multiple runs (if applicable)
- `config.json` - Experiment config
- `index.json` - File index

### Algorithms
- TD3-BC
- CQL
- Ensemble
- BEAR
- UWAC
- MOPO
- APTQ-CQL

### How to Load Results in Python

```python
import json
import pickle

# Load training curves
with open('training_curves.pkl', 'rb') as f:
    curves = pickle.load(f)

# Load summary
with open('results_summary.json', 'r') as f:
    results = json.load(f)
    
# Plot curves
import matplotlib.pyplot as plt
for algo, curve in curves.items():
    plt.plot(curve, label=algo)
plt.legend()
plt.show()
```

Generated: 2026-04-22T13:53:31.895347
