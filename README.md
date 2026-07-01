# Quadruped Locomotion with PPO

Train a 4-legged MuJoCo ant to walk on flat ground and procedurally generated rough terrain using Proximal Policy Optimization (PPO).

**Stack:** Python · PyTorch · Gymnasium · MuJoCo · Stable-Baselines3

## Demo

| Flat ground (Ant-v5) | Rough terrain (TerrainAnt-v0) |
|---|---|
| ![Baseline reward curve](docs/assets/ant/reward_curve.png) | ![Terrain reward curve](docs/assets/terrain/reward_curve.png) |

Videos: [baseline demo](docs/assets/ant/demo.mp4) · [terrain demo](docs/assets/terrain/demo.mp4)

## Results

| Environment | Timesteps | Best eval reward | Notes |
|---|---:|---:|---|
| Ant-v5 (flat) | 2M | **2421** | Baseline locomotion |
| TerrainAnt-v0 (difficulty 0.4) | 3M boost | **1013** | Fine-tuned from stable terrain agent |

The terrain agent reaches ~43% of flat-ground reward — expected, since uneven footing makes locomotion harder and episodes are shorter (avg ~321 steps vs ~811 on flat).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify environments
python check_env.py

# Train
python train.py --config ant       # flat ground, 2M steps
python train.py --config terrain   # rough terrain, 6M steps

# Evaluate latest run
python plot_results.py --config ant --no-show
python evaluate.py --config ant

# Copy artifacts for README
python collect_artifacts.py --config ant
```

TensorBoard: `tensorboard --logdir results/tb_logs`

## How it works

### Reinforcement learning loop

1. **Agent** (PPO policy network) observes the ant's state (joint angles, velocities, contact forces — 105-dim vector).
2. **Action** — 8 continuous torques applied to leg joints, clipped to [-1, 1].
3. **Environment** (MuJoCo physics) simulates one timestep and returns reward + next state.
4. **PPO** collects rollouts from 4 parallel envs, then updates the policy to increase expected reward while staying close to the old policy (clipped surrogate objective).

### Custom terrain environment

`TerrainAnt-v0` subclasses Gymnasium's `AntEnv` and swaps the flat floor for a MuJoCo heightfield. Each episode:

1. Generates terrain as a sum of 6 low-frequency sinusoids with sharpened peaks and troughs.
2. Maps to MuJoCo heightfield data in [0, 1] (up to ~3 m elevation above a 0.1 m base).
3. Sets spawn height to local terrain + 0.55 m clearance so the ant never clips into slopes.

### Curriculum experiment

A `CurriculumCallback` can linearly ramp terrain difficulty during training. In practice this caused **catastrophic forgetting** — the policy peaked on easy terrain, then degraded as difficulty increased faster than it could adapt. The terrain config uses **fixed difficulty 0.3** instead.

## Project structure

```
train.py              # PPO training loop
evaluate.py           # Record demo.mp4 from best checkpoint
plot_results.py       # Plot eval reward curve
check_env.py          # Sanity-check both environments
collect_artifacts.py  # Copy results into docs/assets/
configs/              # Hyperparameters for ant vs terrain
envs/terrain_ant.py   # Custom env + curriculum callback
docs/assets/          # Committed plots and demo videos
```

## Key hyperparameters

| Parameter | Ant-v5 | TerrainAnt-v0 |
|---|---|---|
| Total timesteps | 2M | 6M |
| Parallel envs | 4 | 4 |
| Learning rate | 3e-4 | 2e-4 |
| Rollout length (`n_steps`) | 2048 | 2048 |
| Eval frequency | 12,500 steps | 12,500 steps |

## Further reading

See [LEARNING.md](LEARNING.md) for a walkthrough of the RL and code concepts behind this project.

## License

MIT — see [LICENSE](LICENSE).
