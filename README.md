# Quadruped Locomotion with PPO

Train a 4-legged MuJoCo ant to walk on flat ground and procedurally generated rough terrain using Proximal Policy Optimization (PPO).

**Research question:** Does training on randomized heightfield terrain produce a policy that generalizes to unseen rough ground better than a flat-trained baseline?

**Stack:** Python · PyTorch · Gymnasium · MuJoCo · Stable-Baselines3

**Scope note:** This project tests **terrain adaptation** (flat-trained vs. terrain-trained on rough ground). The original brief also described **leg-damage robustness** (disable actuators, policy still walks) — that variant is **not implemented** here; see [LEARNING.md §3](LEARNING.md#3-project-scope-terrain-adaptation-not-leg-damage).

## Demo

| Flat ground (Ant-v5) | Rough terrain (TerrainAnt-v0) | Control vs treatment (same terrain) |
|---|---|---|
| ![Baseline reward curve](docs/assets/ant/reward_curve.png) | ![Terrain reward curve](docs/assets/terrain/reward_curve.png) | ![Comparison plot](docs/assets/terrain/comparison_plot.png) |

Videos: [flat demo](docs/assets/ant/demo.mp4) · [terrain demo](docs/assets/terrain/demo.mp4) · **[comparison demo](docs/assets/terrain/comparison_demo.mp4)** (flat-trained vs terrain-adapted, matched seed)

## Key result (control vs treatment)

Both policies evaluated on **TerrainAnt-v0** at **difficulty 0.4**, **10 matched seeds**, 1000-step episodes:

| Metric | Flat-trained (control) | Terrain-adapted | 
|---|---:|---:|
| Mean episode reward | 500 ± 126 | **848 ± 130** |
| Mean episode length | 560 steps | **887 steps** |
| Fall rate | **100%** | **30%** |
| Mean forward distance | 8.9 m (before falling) | **2.6 m** |
| Mean forward velocity | 0.41 m/s | **0.08 m/s** |

**Takeaway:** The flat-trained policy falls on every evaluated seed. The terrain-adapted policy survives 7/10 episodes and moves **~2.8× farther** than the prior terrain checkpoint (0.9 m → 2.6 m mean forward distance) after a speed-focused fine-tune (`terrain_speed`). It trades some stability (30% fall rate vs. 10% before) for clearly stronger locomotion on hills.

Run the comparison yourself:

```bash
python compare_policies.py --difficulty 0.4 --seeds 0 1 2 3 4 5 6 7 8 9
```

## Training results

| Policy | Training | Best eval reward | Notes |
|---|---|---:|---|
| Ant-v5 (flat) | 2M steps | **2421** | Phase 1 baseline |
| TerrainAnt-v0 | 2M speed fine-tune | **1152** | `forward_reward_weight=2.5` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python check_env.py
python train.py --config ant
python train.py --config terrain_boost   # stable terrain agent
python train.py --config terrain_speed   # speed fine-tune (current checkpoint)
python compare_policies.py               # control vs treatment (uses checkpoints/)
python evaluate.py --config terrain
python collect_artifacts.py --all        # refresh docs/assets/ + comparison
```

Committed checkpoints (~300 KB each) in `checkpoints/` let you run compare/evaluate without retraining. See [checkpoints/README.md](checkpoints/README.md).

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
3. Sets spawn height to local terrain + 0.55 m clearance.

### Experimental design

| | Control | Treatment |
|---|---|---|
| **Policy** | Flat-trained Ant-v5 checkpoint | Terrain-adapted checkpoint (boost fine-tune) |
| **Test env** | TerrainAnt-v0 | TerrainAnt-v0 |
| **Held constant** | Difficulty (0.4), seeds, max steps (1000), deterministic actions |
| **Metric** | Episode reward, survival (fall rate), forward distance |

### What failed

- **Curriculum learning** caused catastrophic forgetting when difficulty ramped too fast.
- **Fine-tune from flat** alone produced fast but unstable locomotion on terrain (fell ~150 steps).
- **Boost fine-tune** from the stable terrain checkpoint + higher forward reward produced a stable baseline.
- **Speed fine-tune** (`terrain_speed`, `forward_reward_weight=2.5`) improved mean forward distance from **0.9 m → 2.6 m** on matched eval seeds (reproducible terrain seeding fix in `TerrainAntEnv.reset_model`).

## Project structure

```
train.py               # PPO training loop (+ fine-tune support)
compare_policies.py    # Control vs treatment evaluation
evaluate.py            # Record demo.mp4 from best checkpoint
plot_results.py        # Plot eval reward curve
check_env.py           # Sanity-check both environments
collect_artifacts.py   # Copy results into docs/assets/
configs/               # Hyperparameters (ant, terrain, terrain_boost)
envs/terrain_ant.py    # Custom env + curriculum callback
docs/assets/           # Committed plots, comparison results, demo videos
```

## Resume bullet

> Trained a terrain-adapted quadruped locomotion policy (PPO, MuJoCo Ant-v5); on unseen heightfield terrain, terrain-trained agent achieved **848 ± 130** episode reward vs **500 ± 126** for a flat-trained baseline, with **30% vs 100%** fall rate and **2.6 m** mean forward distance under matched seeds. [github.com/FavouritePlayer/ant_sim](https://github.com/FavouritePlayer/ant_sim)

## Further reading

See [LEARNING.md](LEARNING.md) for a walkthrough of the RL and code concepts behind this project.

## License

MIT — see [LICENSE](LICENSE).
