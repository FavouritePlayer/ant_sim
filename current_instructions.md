# Current Instructions

## Current state

The replication run was stopped intentionally. The newest **clean completed** checkpoint is now:

- profile: `terrain_canonical`
- seed: `1`
- completed stage: `terrain_balanced`

Tracked resume checkpoint in the repo:

- `checkpoints/replications/terrain_seed1_balanced/best_model/best_model.zip`

Source run for that tracked checkpoint:

- `results/ppo_terrainant_v0_1783136920_seed1_replicate_terrain_canonical_terrain_balanced/`

Relevant manifests:

- earlier interrupted batch: `results/replications/replication_1783104843_full_seed_sweeps.json`
- later terrain-only batch: `results/replications/replication_1783127241_remaining_terrain_seeds.json`

Important note about the final stopped work:

- `terrain_canonical`, seed `2`, `terrain_finetune` had started but was stopped manually mid-stage
- treat that partial seed-2 work as discarded
- resume from the tracked seed-1 checkpoint above, not from the interrupted seed-2 run

## Important GitHub note

The recommended resume checkpoint is now copied into a **tracked** path under `checkpoints/replications/`, so after pulling this commit on another machine you can resume directly from GitHub without manually transferring `results/`.

The historical `results/` directories and manifests are still useful for provenance, but they are not required for the basic resume path below.

## Resume options

### Option A — continue manually from the tracked checkpoint

Run this from the repo root:

```bash
source .venv/bin/activate

python train.py \
  --config terrain_balanced \
  --seed 1 \
  --pretrained "checkpoints/replications/terrain_seed1_balanced/best_model/best_model" \
  --run-tag resume_from_tracked_seed1_balanced
```

That command resumes from the latest clean replicated terrain checkpoint that is now stored in the repo.

If your goal is to continue the broader replication campaign, the next unfinished work is:

1. rerun `terrain_canonical`, seed `2` from the start of that seed chain
2. then run `damage_canonical`, seeds `0`, `1`, `2`

### Option B — use the existing replication runner as a checklist

`replicate_training.py` is ready, but it does **not** yet resume partially completed manifests automatically.

So for now:

1. use the tracked checkpoint above as the clean resume base
2. then either:
   - launch new one-off `train.py` commands for the remaining stages/seeds, or
   - extend `replicate_training.py` later with manifest-resume support before restarting a large batch
3. if you want a simple pull-and-run workflow on the Mac mini, prefer the tracked checkpoint path under `checkpoints/replications/`

## Useful context

- Fully completed terrain seeds:
  - `terrain_canonical` seed `0`
  - `terrain_canonical` seed `1`
- Interrupted and should be treated as incomplete:
  - `terrain_canonical` seed `2` during `terrain_finetune`
- Not started yet:
  - all `damage_canonical` seeds `0`, `1`, `2`

## Recommended next step

On the destination machine:

1. pull the repo on the Mac mini
2. recreate/activate the virtualenv
3. run the `terrain_balanced` resume command above using the tracked checkpoint in `checkpoints/replications/`
4. only after the replicated training is complete, run the broadened evaluation sweeps
