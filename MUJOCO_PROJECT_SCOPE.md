# MuJoCo RL Project — Scope

## Status (as of 2026-07-02)

| Item | Status |
|---|---|
| Phase 1 — baseline (flat Ant, PPO, render pipeline) | **DONE** |
| Phase 2 — terrain adaptation | **DONE** |
| Phase 2 — leg-damage robustness | **DONE** |
| Phase 3 — repo, README, LEARNING.md, comparison artifacts | **DONE** |

## Experiments (from `docs/assets/*/comparison_results.json`)

### A — Terrain adaptation

| Metric | Flat-trained | Terrain-adapted |
|---|---:|---:|
| Mean reward | 424 ± 106 | 893 ± 130 |
| Fall rate | 50% | 30% |
| Forward velocity | 0.22 m/s | 0.10 m/s |

### B — Leg amputation (leg 1 / front-right)

| Metric | Flat-trained | Damage-robust |
|---|---:|---:|
| Mean reward | 44 ± 37 | 2148 ± 749 |
| Fall rate | 100% | 20% |
| Mean episode length | 21 steps | 809 steps |
| Forward velocity | -0.19 m/s | 0.31 m/s |

## Resume bullet

> Trained terrain-adapted and damage-robust quadruped policies (PPO, MuJoCo Ant-v5). On unseen heightfield terrain: **893 ± 130** vs **424 ± 106** reward, **30% vs 50%** fall rate. Under front-right leg amputation: **2148 ± 749** vs **44 ± 37** reward, **20% vs 100%** fall rate, **0.31 m/s** tripod locomotion. [github.com/FavouritePlayer/ant_sim](https://github.com/FavouritePlayer/ant_sim)

## Canonical training recipes

| Checkpoint | Config chain |
|---|---|
| `checkpoints/terrain/` | `terrain_balanced` |
| `checkpoints/damage/` | `damage_upright` → `damage_speed` → `damage_gait` |
| `checkpoints/flat/` | `ant_finetune` |

## Remaining Gaps (priority order)

- [x] Set up replicated canonical terrain/damage training pipeline with multiple training seeds (`replicate_training.py`)
- [ ] Run and summarize the replicated training sweeps themselves
- [ ] Expand evaluation beyond one terrain difficulty and one amputated leg
- [ ] Promote sudden-amputation recovery from demo-only artifact to a first-class reported benchmark
- [ ] Add automated regression tests for `envs/terrain_ant.py`, `envs/damage_ant.py`, and comparison-script metrics
- [ ] Clean up config provenance so canonical recipes do not depend on historical `results/` directories
