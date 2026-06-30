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

**Evaluate and record demo video (uses latest run):**
```bash
python evaluate.py --episodes 3
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
├── evaluate.py            # loads best model, records demo.mp4
├── plot_results.py        # plots eval reward curve from evaluations.npz
├── check_env.py           # quick sanity check for the environment
├── configs/
│   ├── ppo_ant.py         # baseline config (Ant-v5, 2M steps)
│   └── ppo_terrain.py     # terrain config (TerrainAnt-v0, 6M steps)
├── envs/
│   ├── __init__.py        # registers TerrainAnt-v0 with gymnasium
│   ├── terrain_ant.py     # custom env + CurriculumCallback
│   └── assets/
│       └── ant_terrain.xml  # MuJoCo XML with hfield terrain mesh
└── results/               # gitignored — training outputs live here
```

## Custom environment: TerrainAnt-v0

`TerrainAntEnv` subclasses `AntEnv` from gymnasium and replaces the flat floor with a procedurally generated heightfield. Each episode randomizes the terrain using a sum of 8 random sinusoids, scaled by a `difficulty` parameter in [0, 1]. Difficulty 0 = flat, 1 = ~0.6m peak bumps.

The spawn point is always zeroed (terrain height set to 0 at center) so the ant never spawns embedded in the ground.

**CurriculumCallback** linearly ramps difficulty from `start_difficulty` to `max_difficulty` over training. Note: curriculum was found to cause catastrophic forgetting — the current `ppo_terrain.py` config uses fixed difficulty instead.

## Key design notes

- `n_envs=4` parallel environments for data collection efficiency
- `eval_freq=50_000 // n_envs` steps between evaluations; best model is checkpointed automatically
- `results/` is gitignored — models and logs are not versioned
- The terrain XML uses a 128×128 heightfield; `hfield_data` is written each reset

## Stack

- `gymnasium[mujoco]` — environment (Ant-v5, MuJoCo physics)
- `stable-baselines3` — PPO implementation
- `torch` — underlying ML framework
- Python 3.10+
