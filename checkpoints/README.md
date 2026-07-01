# Committed policy checkpoints

These are the best SB3 checkpoints used for evaluation, demos, and the control vs. treatment comparison. Each is ~300 KB.

| Directory | Policy | Training | Best eval | Source run |
|---|---|---|---:|---|
| `flat/` | Ant-v5 baseline (flat ground) | 2M steps PPO | 2421 | `results/ppo_ant_v5_1782846694` |
| `terrain/` | TerrainAnt-v0 (speed fine-tune) | 2M steps PPO | 1152 @ diff 0.38 | `results/ppo_terrainant_v0_1782945141` |

The terrain checkpoint is a **speed fine-tune** (`configs/ppo_terrain_speed.py`) of the boost checkpoint with `forward_reward_weight=2.5`.

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

Re-train locally:

```bash
python train.py --config terrain_boost   # stable baseline
python train.py --config terrain_speed   # speed fine-tune (current checkpoint recipe)
```

New runs land in gitignored `results/`.
