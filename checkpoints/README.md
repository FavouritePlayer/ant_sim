# Canonical Checkpoints

Committed policy checkpoints used by the README metrics and demo scripts.

| Directory | Policy | Best eval | Recipe |
|---|---|---:|---|
| `flat/` | Ant-v5 baseline | 3385 | `ant_finetune` |
| `terrain/` | TerrainAnt-v0 | 994 @ diff 0.4 | `terrain_balanced` |
| `damage/` | DamageAnt-v0, leg 1 amputated | 3383 | `damage_upright` → `damage_speed` → `damage_gait` |
| `replications/` | Multi-seed handoffs | — | See `current_instructions.md` |

## Usage

From a fresh clone, run benchmarks without retraining:

```bash
python compare_policies.py
python compare_damage.py
python evaluate.py --config terrain
```

Re-train the damage checkpoint:

```bash
python train.py --config damage_upright
python train.py --config damage_speed
python train.py --config damage_gait
```

## Provenance

Exploratory configs under `configs/` now anchor to committed checkpoints where possible. The main exception is `terrain_velocity_v2`, which has no committed `TerrainAnt-v1` parent because the velocity-command variant changes the observation space; resume that kind of run with an explicit `--pretrained` override when needed.

For headline results from a fresh clone, use:

- committed checkpoints in this directory
- `docs/assets/terrain/` and `docs/assets/damage/`
- `compare_policies.py` and `compare_damage.py`

To resume interrupted multi-seed training on another machine, see `checkpoints/replications/`.

Treat this directory as the source of truth for benchmarked models and the default parent anchors for reproducible fine-tunes.
