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

## 3. PPO — Proximal Policy Optimization

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

## 4. Parallel environments

```python
env = make_vec_env("Ant-v5", n_envs=4, seed=42)
```

`make_vec_env` runs 4 independent Ant instances in parallel (vectorized). This collects experience 4× faster — critical because RL is sample-inefficient.

## 5. Evaluation and checkpointing

```python
EvalCallback(eval_env, best_model_save_path=..., eval_freq=12500, n_eval_episodes=5)
```

Every 12,500 training steps, the agent runs 5 deterministic episodes in a held-out env. If mean reward beats the previous best, the model is saved to `best_model/`. This is what `evaluate.py` loads for demo videos.

## 6. Custom terrain — TerrainAnt-v0

### Why subclass AntEnv?

Gymnasium's `AntEnv` already handles observations, rewards, and termination. We only need to change the **physics scene** — swap flat ground for a heightfield.

### Procedural terrain generation

```python
# Sum of 8 random 2D sinusoids
terrain += sin(fx * X + px) * cos(fy * Y + py)
```

Each episode gets a unique landscape. `difficulty` scales amplitude: 0.3 means bumps up to ~18 cm.

### Spawn safety

```python
terrain -= terrain[cx, cy]  # zero height at center
```

Without this, the ant could spawn inside a hill.

### Writing to MuJoCo

```python
self.model.hfield_data[:] = (terrain * self.difficulty).flatten()
```

MuJoCo reads `hfield_data` each physics step. We overwrite it on every `reset()`.

## 7. Curriculum learning (and why we disabled it)

**Idea**: start on flat ground, gradually increase terrain difficulty so the agent learns incrementally.

**What happened**: the policy learned to walk on easy terrain, but when difficulty ramped up, performance collapsed — the policy forgot earlier skills faster than it acquired new ones (**catastrophic forgetting**).

**Fix**: train at fixed `difficulty=0.3` so the reward landscape stays stable.

The `CurriculumCallback` code remains in `envs/terrain_ant.py` if you want to experiment:

```python
# configs/ppo_terrain.py — add to enable curriculum
"curriculum": {"start": 0.05, "max": 0.8, "interval": 100_000},
```

## 8. Reading training output

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

## 9. Suggested experiments

1. **Transfer learning**: train on flat (`ant`), then fine-tune on terrain with `--timesteps 500000`.
2. **Difficulty ablation**: try `difficulty` = 0.0, 0.3, 0.6, 1.0 and compare eval rewards.
3. **Re-enable curriculum** with slower ramp (`interval=500_000`) and compare to fixed difficulty.
4. **Entropy bonus**: set `ent_coef=0.01` in terrain config to encourage exploration.

## 10. Glossary

| Term | Definition |
|---|---|
| **Policy** | Function mapping states → actions |
| **Value function** | Predicted total future reward from a state |
| **Advantage** | How much better an action was vs. the average (A = Q - V) |
| **On-policy** | Train only on data collected by the current policy |
| **Heightfield** | 2D grid of terrain heights used by MuJoCo |
| **GAE** | Generalized Advantage Estimation — reduces variance in advantage estimates |
