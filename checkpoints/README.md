# Committed policy checkpoints

These are the best SB3 checkpoints used for evaluation, demos, and the control vs. treatment comparison. Each is ~300 KB.

| Directory | Policy | Training | Best eval | Source run |
|---|---|---|---:|---|
| `flat/` | Ant-v5 baseline (flat ground) | 1M fine-tune | 3385 | `results/ppo_ant_v5_1782953305` |
| `terrain/` | TerrainAnt-v0 (balanced fine-tune) | 3M steps PPO | 994 @ diff 0.4 | `results/ppo_terrainant_v0_1782950206` |

The terrain checkpoint is a **balanced fine-tune** (`configs/ppo_terrain_balanced.py`) of the boost checkpoint with `forward_reward_weight=2.0`, trained and evaluated at difficulty 0.4.

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
python train.py --config ant_finetune      # flat baseline polish
python train.py --config terrain_balanced  # terrain checkpoint recipe
```

New runs land in gitignored `results/`.
