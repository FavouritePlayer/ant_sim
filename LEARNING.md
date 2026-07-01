# Learning Guide — RL Concepts in This Project

This document explains the reinforcement learning and code foundations behind the ant locomotion project.

## 1. What is reinforcement learning?

An **agent** learns by trial and error in an **environment**:

```
observe state  →  choose action  →  get reward + new state  →  repeat
```

Unlike supervised learning (labeled examples), the agent only gets a scalar **reward** signal telling it how well it did. The goal is to learn a **policy** π(a|s) that maximizes cumulative reward over time.

## 2. The Ant environment

**Gymnasium Ant-v5** simulates a quadruped robot in MuJoCo:

| Component | Shape | Meaning |
|---|---|---|
| Observation | (105,) | Joint positions/velocities, torso orientation, contact forces |
| Action | (8,) | Torques on 8 leg joints, range [-1, 1] |
| Reward | scalar | Forward velocity + alive bonus − control cost |

Each episode runs up to 1000 steps. The ant "dies" if it flips over (torso too low/high).

## 3. Project scope: terrain adaptation, not leg damage

The original project brief (`MUJOCO_PROJECT_SCOPE.md`) described two possible contributions:

| Option | Idea | Status in this repo |
|---|---|---|
| **Leg-damage robustness** | Randomly disable leg actuators during training; compare damaged baseline vs. robust policy | **Not implemented** — would need a failure-injection wrapper that zeros torques on selected joints and retraining under that distribution |
| **Terrain adaptation** | Train on randomized heightfield terrain; compare flat-trained vs. terrain-adapted policy on unseen rough ground | **Implemented** — this is the actual contribution |

We picked **one** contribution (terrain) and did not straddle. The resume bullet and comparison experiment are about **generalization to rough terrain**, not walking with missing limbs.

If you wanted leg-damage next, the env change would look roughly like: at each reset, sample k ∈ {0,1,2} leg indices and multiply those action dimensions by 0 before `step()`. The flat-trained baseline would collapse under damage; a policy trained with randomized damage would be the treatment. That code path does not exist in this repository.

## 4. PPO — Proximal Policy Optimization

PPO is an **on-policy** actor-critic algorithm. It maintains:

- **Actor** (policy): maps observations → actions
- **Critic** (value function): estimates expected future reward from a state

### Training cycle (one iteration)

1. **Collect rollouts**: run the current policy in 4 parallel envs for `n_steps=2048` steps each → 8,192 transitions per iteration.
2. **Compute advantages**: using GAE (Generalized Advantage Estimation) with `gamma=0.99`, `gae_lambda=0.95`.
3. **Update policy**: for `n_epochs=10`, shuffle data into `batch_size=256` minibatches and apply the clipped surrogate loss:

   ```
   L = min(r(θ)·A,  clip(r(θ), 1-ε, 1+ε)·A)
   ```

   where r(θ) is the probability ratio between new and old policy, and ε = 0.2 (`clip_range`).

The clip prevents destructively large policy updates — this is what makes PPO stable.

### Where this lives in code

```python
# train.py — create PPO agent
model = PPO(policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=2048, ...)

# train.py — learn
model.learn(total_timesteps=2_000_000, callback=eval_callback)
```

`MlpPolicy` = two hidden-layer MLPs (64 units each by default) for actor and critic.

## 5. Parallel environments

```python
env = make_vec_env("Ant-v5", n_envs=4, seed=42)
```

`make_vec_env` runs 4 independent Ant instances in parallel (vectorized). This collects experience 4× faster — critical because RL is sample-inefficient.

## 6. Evaluation and checkpointing

```python
EvalCallback(eval_env, best_model_save_path=..., eval_freq=12500, n_eval_episodes=5)
```

Every 12,500 training steps, the agent runs 5 deterministic episodes in a held-out env. If mean reward beats the previous best, the model is saved to `best_model/`. Committed copies for reproducibility live in `checkpoints/flat/` and `checkpoints/terrain/` (~300 KB each). Scripts default to those paths so a fresh clone can run demos without retraining.

## 7. Custom terrain — TerrainAnt-v0

### Why subclass AntEnv?

Gymnasium's `AntEnv` already handles observations, rewards, and termination. We only need to change the **physics scene** — swap flat ground for a heightfield.

### Procedural terrain generation

Each episode builds a **256×256** heightfield over a **16×16 m** footprint:

```python
# Sum of 6 random 2D sinusoids, then sharpen peaks/troughs
terrain += amp * sin(fx * X + px) * cos(fy * Y + py)
centered = sign(centered) * (abs(centered * 2) ** 0.75) * 0.5
terrain = 0.5 + (terrain - 0.5) * difficulty   # difficulty in [0, 1]
```

At `difficulty=1`, elevation spans roughly **0.1–3.0 m** above the MuJoCo base plane. At `difficulty=0.4` (comparison/eval setting), relief is scaled to 40% of that range — visible hills and valleys, not flat ground.

### Spawn safety

The height at the spawn cell is normalized so the center is mid-range, then:

```python
qpos[2] = terrain_z + SPAWN_CLEARANCE  # 0.55 m above local ground
```

A contact-based nudge in `step()` corrects rare floor penetrations reported by MuJoCo.

### Writing to MuJoCo

```python
self.model.hfield_data[adr : adr + NROW * NCOL] = terrain.flatten()
mujoco.mj_forward(self.model, self.data)
```

We write `hfield_data` on every `reset()`. Do **not** call `spec.recompile()` after writing — that was a bug that zeroed collision geometry while visual hills remained.

## 8. Curriculum learning (and why we disabled it)

**Idea**: start on flat ground, gradually increase terrain difficulty so the agent learns incrementally.

**What happened**: the policy learned to walk on easy terrain, but when difficulty ramped up, performance collapsed — the policy forgot earlier skills faster than it acquired new ones (**catastrophic forgetting**).

**Fix**: train at fixed difficulty (~0.35) so the reward landscape stays stable. The winning terrain agent came from a **boost fine-tune** of an earlier stable checkpoint with higher `forward_reward_weight`.

The `CurriculumCallback` code remains in `envs/terrain_ant.py` if you want to experiment:

```python
# configs/ppo_terrain.py — add to enable curriculum
"curriculum": {"start": 0.05, "max": 0.8, "interval": 100_000},
```

## 9. Control vs. treatment experiment

This is the resume-grade result — implemented in `compare_policies.py`.

| | Control | Treatment |
|---|---|---|
| **Policy** | Flat-trained Ant-v5 (`checkpoints/flat/`) | Terrain-adapted (`checkpoints/terrain/`) |
| **Test env** | TerrainAnt-v0 | TerrainAnt-v0 |
| **Held constant** | Difficulty 0.4, matched seeds, 1000 max steps, deterministic actions |
| **Metrics** | Episode reward, steps survived, fall rate, forward distance |

**Results (10 seeds, difficulty 0.4, reproducible terrain seeding):**

| Metric | Flat-trained | Terrain-adapted (speed) |
|---|---:|---:|
| Mean reward | 500 ± 126 | 848 ± 130 |
| Mean steps | 560 | 887 |
| Fall rate | 100% | 30% |
| Mean forward distance | 8.9 m | 2.6 m |
| Mean forward velocity | 0.41 m/s | 0.08 m/s |

**Interpretation:** The flat policy always falls but often covers more ground in x before doing so. The speed fine-tune (`configs/ppo_terrain_speed.py`, `forward_reward_weight=2.5`) roughly **tripled** mean forward distance vs. the prior boost checkpoint (0.9 m → 2.6 m) at the cost of higher fall rate (10% → 30%). Still far better than flat (100% fall).

**Bug fixed:** terrain was generated before Gymnasium applied the episode seed, making comparisons non-reproducible. `_randomise_terrain()` now runs inside `reset_model()` after the RNG is seeded.

```bash
python compare_policies.py --difficulty 0.4 --seeds 0 1 2 3 4 5 6 7 8 9
python collect_artifacts.py --all   # refresh docs/assets/ including comparison
```

## 10. What we tried (and what failed)

1. **Curriculum ramp** — catastrophic forgetting when difficulty increased too fast.
2. **Fine-tune flat → terrain** — fast initial motion but unstable (~150–220 steps before falling).
3. **Boost fine-tune** from stable terrain checkpoint + `forward_reward_weight=1.5` — stable but cautious locomotion.
4. **Speed fine-tune** (`terrain_speed`, `forward_reward_weight=2.5`) — current checkpoint; 2.6 m mean forward distance, 30% fall rate.

## 11. Reading training output

```
| rollout/           |          |
|    ep_len_mean     | 76.1     |   ← avg episode length (higher = survives longer)
|    ep_rew_mean     | -82.7    |   ← avg reward during rollouts (should increase)
| time/              |          |
|    fps             | 4656     |   ← simulation steps per second
|    total_timesteps | 8192     |
```

Early training: negative rewards, short episodes (ant falls immediately).
Late training: positive rewards, episodes near 1000 steps (ant runs far).

## 12. Glossary

| Term | Definition |
|---|---|
| **Policy** | Function mapping states → actions |
| **Value function** | Predicted total future reward from a state |
| **Advantage** | How much better an action was vs. the average (A = Q - V) |
| **On-policy** | Train only on data collected by the current policy |
| **Heightfield** | 2D grid of terrain heights used by MuJoCo |
| **GAE** | Generalized Advantage Estimation — reduces variance in advantage estimates |
