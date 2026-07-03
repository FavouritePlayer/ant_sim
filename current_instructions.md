# Current Instructions

## Current state

The long replication run was **intentionally stopped after a clean checkpoint boundary** so work can continue later on this machine or another computer.

The last clean completed stage is:

- profile: `terrain_canonical`
- seed: `0`
- completed stage: `terrain_boost`

Clean resume checkpoint:

- `results/ppo_terrainant_v0_1783110751_seed0_replicate_terrain_canonical_terrain_boost/best_model/best_model.zip`

Replication manifest from the interrupted run:

- `results/replications/replication_1783104843_full_seed_sweeps.json`

Note: that manifest still shows `terrain_balanced` for seed `0` as `"running"` because the Python process was stopped manually after the clean `terrain_boost` save. Treat that partial `terrain_balanced` work as discarded. Resume from the `terrain_boost` checkpoint above.

## Important GitHub note

The checkpoint above is **not on GitHub** because it lives under `results/`, which is gitignored.

If resuming on another computer, you must manually transfer:

- the whole repo, or at minimum
- `results/ppo_terrainant_v0_1783110751_seed0_replicate_terrain_canonical_terrain_boost/`
- `results/replications/replication_1783104843_full_seed_sweeps.json`

## Resume options

### Option A — continue manually from the clean checkpoint

Run this from the repo root:

```bash
source .venv/bin/activate

python train.py \
  --config terrain_balanced \
  --seed 0 \
  --pretrained "results/ppo_terrainant_v0_1783110751_seed0_replicate_terrain_canonical_terrain_boost/best_model/best_model" \
  --run-tag replicate_terrain_canonical_terrain_balanced
```

When that finishes, continue the remaining staged replications manually:

1. `terrain_canonical`, seed `1`
2. `terrain_canonical`, seed `2`
3. `damage_canonical`, seed `0`
4. `damage_canonical`, seed `1`
5. `damage_canonical`, seed `2`

### Option B — use the existing replication runner as a checklist

`replicate_training.py` is ready, but it does **not** yet resume partially completed manifests automatically.

So for now:

1. use the checkpoint above to finish `terrain_balanced` for seed `0`
2. then either:
   - launch new one-off `train.py` commands for the remaining stages/seeds, or
   - extend `replicate_training.py` later with manifest-resume support before restarting a large batch

## Useful context

- Last fully completed seed-stage chain:
  - `terrain_finetune` seed `0` completed
  - `terrain_boost` seed `0` completed
- Not started yet:
  - `terrain_canonical` seeds `1`, `2`
  - all `damage_canonical` seeds `0`, `1`, `2`

## Recommended next step

On the destination machine:

1. copy the repo and the required `results/` checkpoint folder
2. recreate/activate the virtualenv
3. run the `terrain_balanced` resume command above
4. only after the replicated training is complete, run the broadened evaluation sweeps
