# Replicated Training Sweep Summary

Generated: `2026-07-06T01:46:32Z`

## Per-seed final eval rewards

| Profile | Seed | Final eval reward | Timesteps | Run dir |
|---|---:|---:|---:|---|
| `damage_canonical` | 0 | 2394.16 ± 1506.69 | 1500000 | `results/ppo_damageant_v0_1783161704_seed0_replicate_damage_canonical_damage_gait` |
| `damage_canonical` | 1 | 3300.41 ± 1528.77 | 1500000 | `results/ppo_damageant_v0_1783163648_seed1_replicate_damage_canonical_damage_gait` |
| `damage_canonical` | 2 | 2478.82 ± 1690.99 | 1150000 | `results/ppo_damageant_v0_1783165386_seed2_replicate_damage_canonical_damage_gait` |
| `damage_crossleg` | 0 | 981.51 ± 1387.89 | 1050000 | `results/ppo_damageant_v0_1783295334_seed0_replicate_damage_crossleg_crossleg_gait` |
| `damage_crossleg` | 1 | 648.95 ± 1070.18 | 1500000 | `results/ppo_damageant_v0_1783296345_seed1_replicate_damage_crossleg_crossleg_gait` |
| `damage_crossleg` | 2 | 1256.94 ± 1144.24 | 1000000 | `results/ppo_damageant_v0_1783298003_seed2_replicate_damage_crossleg_crossleg_gait` |
| `terrain_canonical` | 0 | 501.99 ± 244.34 | 3000000 | `results/ppo_terrainant_v0_1783236040_seed0_replicate_terrain_canonical_terrain_balanced` |
| `terrain_canonical` | 1 | 397.01 ± 15.92 | 3000000 | `results/ppo_terrainant_v0_1783145145_seed1_resume_from_tracked_seed1_balanced` |
| `terrain_canonical` | 2 | 384.97 ± 30.04 | 3000000 | `results/ppo_terrainant_v0_1783157709_seed2_replicate_terrain_canonical_terrain_balanced` |

## Aggregate

- Terrain seeds present: **3** (mean final eval `427.99 ± 52.55`)
- Damage seeds present: **3** (mean final eval `2724.46 ± 408.72`)

## Notes

- Terrain seed 1 uses the resumed tracked handoff run.
- Entries are discovered from replication manifests with KNOWN_FINAL_RUNS fallbacks.
- Terrain canonical seeds present locally: [0, 1, 2].
- Cross-leg damage replications are included when manifests exist under results/replications/.
