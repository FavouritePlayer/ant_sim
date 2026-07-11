# Project Scope

Internal scope document for the ant locomotion project. For setup, results, and reproduction commands, see [README.md](README.md).

## Delivered

- Flat, terrain-adapted, and damage-specialist PPO policies with committed checkpoints
- Custom `TerrainAnt-v0` and `DamageAnt-v0` environments
- Reproducible 10-seed comparison benchmarks and multi-seed training replication
- Extended eval sweeps (terrain difficulty, all-leg damage, sudden amputation)
- Cross-leg training experiment and compound policy router
- Comparison videos, JSON artifacts, and regression tests

## Status

**Portfolio complete.** Optional future work: per-leg specialists for legs 2–3, or a unified damage policy matching compound macro performance without routing.

## Canonical training recipes

| Checkpoint | Config chain |
|---|---|
| `checkpoints/flat/` | `ant_finetune` |
| `checkpoints/terrain/` | `terrain_balanced` |
| `checkpoints/damage/` | `damage_upright` → `damage_speed` → `damage_gait` |

Multi-seed replication: `python replicate_training.py --profiles terrain_canonical damage_canonical --seeds 0 1 2`

## Design notes

- **TerrainAnt-v0** — procedural heightfield, spawn safety, boundary termination. Fixed difficulty during training (curriculum caused catastrophic forgetting).
- **DamageAnt-v0** — leg amputation via removed collision geometry and zeroed actuators; gated forward reward and tip-over termination.
- **Baseline** — flat-trained `checkpoints/flat/` is shared control for both experiments.
- **Compound router** — switches between specialist and cross-leg checkpoints per amputated leg; best macro damage strategy in the repo.
- **Evaluation** — `n_envs=4` during training; best model checkpointed on eval reward every 50k steps.

## Related docs

- [README.md](README.md) — public-facing overview and quick start
- [LEARNING.md](LEARNING.md) — RL concepts and code walkthrough
- [checkpoints/README.md](checkpoints/README.md) — checkpoint lineage and usage
