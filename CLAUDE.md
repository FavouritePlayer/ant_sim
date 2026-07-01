# Quadruped Locomotion RL — Ant-v5

Reinforcement learning project training a 4-legged robot to walk on flat and rough terrain using PPO (Proximal Policy Optimization). Built with MuJoCo + Stable-Baselines3.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

**Train baseline (Ant-v5, flat ground, 2M steps):**
```bash
python train.py --config ant
```

**Train terrain agent (TerrainAnt-v0, 6M steps):**
```bash
python train.py --config terrain
```

**Fine-tune terrain agent (recommended — boost from stable checkpoint, 3M steps):**
```bash
python train.py --config terrain_boost
```

**Compare flat-trained vs terrain-adapted on matched terrain (control experiment):**
```bash
python compare_policies.py --difficulty 0.4 --seeds 0 1 2 3 4 5 6 7 8 9
```

Uses `checkpoints/flat/` and `checkpoints/terrain/` by default.

**Evaluate and record demo video (uses latest run):**
```bash
python evaluate.py --episodes 3
python evaluate.py --config terrain
python evaluate.py --config ant
```

**Plot reward curve (uses latest run):**
```bash
python plot_results.py
```

**Check environment is working:**
```bash
python check_env.py
```

Results land in `results/ppo_<env>_<timestamp>/`. TensorBoard logs go in `results/tb_logs/`.

## Repo structure

```
ant_sim/
├── train.py               # main training loop (PPO via SB3)
├── compare_policies.py    # control vs treatment evaluation on terrain
├── evaluate.py            # loads best model, records demo.mp4
├── plot_results.py        # plots eval reward curve from evaluations.npz
├── check_env.py           # quick sanity check for the environment
├── collect_artifacts.py   # copy run outputs to docs/assets/
├── configs/
│   ├── ppo_ant.py         # baseline config (Ant-v5, 2M steps)
│   ├── ppo_terrain.py     # terrain config (TerrainAnt-v0, 6M steps)
│   └── ppo_terrain_boost.py  # fine-tune from stable terrain checkpoint
├── envs/
│   ├── __init__.py        # registers TerrainAnt-v0 with gymnasium
│   ├── terrain_ant.py     # custom env + CurriculumCallback
│   └── assets/
│       └── ant_terrain.xml  # MuJoCo XML with hfield terrain mesh
└── results/               # gitignored — training outputs live here
```

## Custom environment: TerrainAnt-v0

`TerrainAntEnv` subclasses `AntEnv` from gymnasium and replaces the flat floor with a procedurally generated heightfield. Each episode randomizes the terrain using a sum of 6 low-frequency sinusoids with sharpened peaks and troughs, scaled by a `difficulty` parameter in [0, 1]. Difficulty 0 = flat, 1 = full relief (~3 m elevation range above base).

The spawn point is zeroed (terrain height set to 0 at center) and the ant spawns at local terrain height + 0.55 m clearance.

**CurriculumCallback** linearly ramps difficulty from `start_difficulty` to `max_difficulty` over training. Note: curriculum was found to cause catastrophic forgetting — current configs use fixed difficulty instead.

## Key checkpoints

Scripts prefer committed checkpoints over gitignored `results/`:

| Policy | Path | Best eval |
|---|---|---:|
| Flat baseline | `checkpoints/flat/` | 2421 |
| Terrain (boost fine-tune) | `checkpoints/terrain/` | 1013 @ diff 0.35 |

Comparison artifacts (10 seeds @ diff 0.4): `docs/assets/terrain/comparison_*`

**Leg-damage robustness is not in this repo.** Scope pivoted to terrain adaptation only. See LEARNING.md §3.

## Key design notes

- `n_envs=4` parallel environments for data collection efficiency
- `eval_freq=50_000 // n_envs` steps between evaluations; best model is checkpointed automatically
- `results/` is gitignored — models and logs are not versioned
- The terrain XML uses a 256×256 heightfield; `hfield_data` is written each reset

## Stack

- `gymnasium[mujoco]` — environment (Ant-v5, MuJoCo physics)
- `stable-baselines3` — PPO implementation
- `torch` — underlying ML framework
- Python 3.10+
