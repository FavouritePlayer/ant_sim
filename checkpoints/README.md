# Committed policy checkpoints

| Directory | Policy | Best eval | Recipe |
|---|---|---:|---|
| `flat/` | Ant-v5 baseline | 3385 | `ant_finetune` |
| `terrain/` | TerrainAnt-v0 | 994 @ diff 0.4 | `terrain_balanced` |
| `damage/` | DamageAnt-v0, leg 1 amputated | 3383 | `damage_upright` → `damage_speed` → `damage_gait` |

```bash
python compare_policies.py
python compare_damage.py
python evaluate.py --config terrain
```

Re-train damage checkpoint:

```bash
python train.py --config damage_upright
python train.py --config damage_speed
python train.py --config damage_gait
```
