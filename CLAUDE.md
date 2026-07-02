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

**Fine-tune terrain agent (speed — current checkpoint recipe, 2M steps):**
```bash
python train.py --config terrain_speed
```

**Compare flat-trained vs terrain-adapted on matched terrain (control experiment):**
```bash
python compare_policies.py --difficulty 0.4 --seeds 0 1 2 3 4 5 6 7 8 9
```

**Train damage-robust agent (DamageAnt-v0, fine-tune from flat):**
```bash
python train.py --config damage
```

**Compare flat-trained vs damage-robust under leg failure:**
```bash
python compare_damage.py --disabled-legs 1 --seeds 0 1 2 3 4 5 6 7 8 9
```

Uses `checkpoints/flat/` and `checkpoints/terrain/` by default for terrain comparison; `checkpoints/damage/` for leg-damage comparison.

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
├── compare_policies.py    # terrain control vs treatment
├── compare_damage.py      # leg damage control vs treatment
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
│   ├── terrain_ant.py     # TerrainAnt-v0 + CurriculumCallback
│   ├── damage_ant.py      # DamageAnt-v0 (leg actuator failure)
│   └── assets/
│       └── ant_terrain.xml  # MuJoCo XML with hfield terrain mesh
└── results/               # gitignored — training outputs live here
```

## Custom environment: TerrainAnt-v0

`TerrainAntEnv` subclasses `AntEnv` from gymnasium and replaces the flat floor with a procedurally generated heightfield. Each episode randomizes the terrain using a sum of 6 low-frequency sinusoids with sharpened peaks and troughs, scaled by a `difficulty` parameter in [0, 1]. Difficulty 0 = flat, 1 = full relief (~3 m elevation range above base).

The spawn point is zeroed (terrain height set to 0 at center) and the ant spawns at local terrain height + 0.55 m clearance.

**CurriculumCallback** linearly ramps difficulty from `start_difficulty` to `max_difficulty` over training. Note: curriculum was found to cause catastrophic forgetting — current configs use fixed difficulty instead.

## Custom environment: DamageAnt-v0

`DamageAntEnv` subclasses `AntEnv` on flat ground. Leg failure is modelled as zero actuator torques on selected legs (2 actuators per leg). Training randomizes 0..`max_disabled_legs` legs per episode; evaluation uses `fixed_disabled_legs` (e.g. `[1]`).

## Key checkpoints

Scripts prefer committed checkpoints over gitignored `results/`:

| Policy | Path | Best eval |
|---|---|---:|
| Flat baseline | `checkpoints/flat/` | 3385 |
| Terrain (balanced fine-tune) | `checkpoints/terrain/` | 994 @ diff 0.4 |
| Damage (leg-failure fine-tune) | `checkpoints/damage/` | 3068 @ leg 1 out |

Comparison artifacts: `docs/assets/terrain/comparison_*`, `docs/assets/damage/comparison_*`

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
