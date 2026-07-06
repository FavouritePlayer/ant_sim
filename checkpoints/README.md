# Canonical Checkpoints

These are the **canonical, committed** checkpoints used by the README metrics and demo scripts:

- `flat/` — shared flat-ground baseline
- `terrain/` — canonical terrain-adapted policy used in `compare_policies.py`
- `damage/` — canonical fixed-amputation damage policy used in `compare_damage.py`
- `replications/` — tracked handoff checkpoints copied out of `results/` for cross-machine resume

## Important provenance note

Exploratory configs under `configs/` now anchor to committed checkpoints where possible.
The main exception is `terrain_velocity_v2`, which has no committed `TerrainAnt-v1`
parent because the velocity-command variant changes the observation space; resume that
kind of run with an explicit `--pretrained` override when needed.

If you want the repo's headline results from a fresh clone, use:

- committed checkpoints in this directory
- `docs/assets/terrain/` and `docs/assets/damage/`
- `compare_policies.py` and `compare_damage.py`

If you want to resume interrupted multi-seed training on another machine, check:

- `current_instructions.md`
- `checkpoints/replications/`

## Canonical recipes

- `flat/` corresponds to the flat-ground baseline used across both experiments
- `terrain/` corresponds to the `terrain_balanced` result described in the README
- `damage/` corresponds to the staged `damage_upright -> damage_speed -> damage_gait` result

Treat this directory as the source of truth for benchmarked models and the default parent
anchors for reproducible fine-tunes.
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
