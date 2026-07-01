# Committed policy checkpoints

These are the best SB3 checkpoints used for evaluation, demos, and the control vs. treatment comparison. Each is ~300 KB.

| Directory | Policy | Training | Best eval | Source run |
|---|---|---|---:|---|
| `flat/` | Ant-v5 baseline (flat ground) | 2M steps PPO | 2421 | `results/ppo_ant_v5_1782846694` |
| `terrain/` | TerrainAnt-v0 (boost fine-tune) | 3M steps PPO | 1013 @ diff 0.35 | `results/ppo_terrainant_v0_1782867891` |

Load in code:

```python
from stable_baselines3 import PPO
model = PPO.load("checkpoints/terrain/best_model/best_model")
```

Or use the scripts with defaults (they prefer `checkpoints/` over local `results/`):

```bash
python compare_policies.py
python evaluate.py --config terrain
python evaluate.py --config ant
```

Re-train locally with `python train.py --config ant` / `terrain_boost`; new runs land in gitignored `results/`.
