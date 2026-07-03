# Canonical Checkpoints

These are the **canonical, committed** checkpoints used by the README metrics and demo scripts:

- `flat/` — shared flat-ground baseline
- `terrain/` — canonical terrain-adapted policy used in `compare_policies.py`
- `damage/` — canonical fixed-amputation damage policy used in `compare_damage.py`

## Important provenance note

Some exploratory configs under `configs/` still reference historical parents inside `results/`.
Those recipes are useful as experiment notes, but they are **not** the clean reproducible
path for the shipped portfolio artifacts.

If you want the repo's headline results from a fresh clone, use:

- committed checkpoints in this directory
- `docs/assets/terrain/` and `docs/assets/damage/`
- `compare_policies.py` and `compare_damage.py`

## Canonical recipes

- `flat/` corresponds to the flat-ground baseline used across both experiments
- `terrain/` corresponds to the `terrain_balanced` result described in the README
- `damage/` corresponds to the staged `damage_upright -> damage_speed -> damage_gait` result

Until the exploratory configs are fully cleaned up, treat this directory as the source of truth
for benchmarked models.
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
