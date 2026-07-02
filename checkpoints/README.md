# Committed policy checkpoints

| Directory | Policy | Best eval | Recipe |
|---|---|---:|---|
| `flat/` | Ant-v5 baseline | 3385 | `ant_finetune` |
| `terrain/` | TerrainAnt-v0 | 994 @ diff 0.4 | `terrain_balanced` |
| `damage/` | DamageAnt-v0, leg 1 amputated | 5477 | `damage_upright` → `damage_final` |

```bash
python compare_policies.py
python compare_damage.py
python evaluate.py --config terrain
```

Re-train damage checkpoint:

```bash
python train.py --config damage_upright
python train.py --config damage_final
```
