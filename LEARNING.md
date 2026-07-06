# Learning Guide — RL Concepts in This Project

This document explains the reinforcement learning foundations, custom environments, and experiments in the ant locomotion project. For headline numbers, see [README.md](README.md).

---

## 1. Reinforcement learning basics

An **agent** interacts with an **environment** in a loop:

```
observe state → choose action → receive reward + next state → repeat
```

The goal is to learn a **policy** π(a|s) that maximizes cumulative reward. Unlike supervised learning, labels are not provided — only scalar rewards.

**Gymnasium Ant-v5** (MuJoCo):

| Component | Shape | Meaning |
|---|---|---|
| Observation | (105,) | Joint state, torso pose, contacts |
| Action | (8,) | Leg joint torques in [−1, 1] |
| Reward | scalar | Forward velocity + survival − control cost |

Episodes last up to 1000 steps. Termination occurs if the torso tips beyond healthy bounds.

---

## 2. Algorithm: PPO

**Proximal Policy Optimization** is an on-policy actor-critic method:

1. Collect rollouts from parallel envs (`n_steps=2048`, `n_envs=4–8`)
2. Estimate advantages with GAE (`gamma=0.99`, `gae_lambda=0.95`)
3. Update with clipped surrogate loss (`clip_range` ≈ 0.05–0.2)

```python
# train.py
model = PPO("MlpPolicy", env, learning_rate=..., n_steps=2048, ...)
model.learn(total_timesteps=..., callback=EvalCallback(...))
```

`EvalCallback` saves `best_model/` when held-out eval reward improves. Committed copies live in `checkpoints/`.

---

## 3. Custom environments

### TerrainAnt-v0 (`envs/terrain_ant.py`)

- Procedural **256×256 heightfield** over 16×16 m, built from summed sinusoids scaled by `difficulty ∈ [0, 1]`
- Spawn height set above local terrain; boundary termination if the ant leaves the field
- Training uses randomized difficulty; comparison eval uses **difficulty 0.4**

### DamageAnt-v0 (`envs/damage_ant.py`)

- Amputated legs: invisible geoms, no contact, zero actuators, damped joints
- Reward shaping encourages upright tripod locomotion (upright bonus, tilt penalty, optional gait terms)
- **Specialist** training: `fixed_disabled_legs=[1]` every episode
- **Cross-leg** training: `min_disabled_legs=1`, `max_disabled_legs=1` (random leg each episode)

---

## 4. Experiments and methodology

All comparisons hold the **test environment**, **seeds**, and **episode length** constant; only the policy changes.

### A — Terrain adaptation (`compare_policies.py`)

| | Control | Treatment |
|---|---|---|
| Policy | `checkpoints/flat/` | `checkpoints/terrain/` |
| Test | TerrainAnt-v0 @ diff 0.4 | Same |

**Result:** 893 ± 130 vs 424 ± 106 reward; 30% vs 50% fall rate (10 seeds).

**Interpretation:** Heightfield training improves robustness; locomotion is slower on hills.

Terrain holds from difficulty 0.2–0.8 (see `docs/assets/terrain/difficulty_sweep.json`).

### B — Leg damage specialist (`compare_damage.py`)

| | Control | Treatment |
|---|---|---|
| Policy | Flat | `checkpoints/damage/` (upright → speed → gait) |
| Test | DamageAnt-v0, leg 1 out at reset | Same |

**Result:** 2148 ± 749 vs 44 ± 37 reward; 20% vs 100% fall; 809 vs 21 mean steps.

**Interpretation:** Strong on trained leg; **100% fall** on legs 0, 2, 3 in all-leg sweep.

### C — Sudden amputation

Both policies start on 4 legs; leg 1 removed at **step 120** (~4 s).

**Result:** Specialist 1386 ± 758 reward, 80% fall vs flat 520 ± 21, 100% fall.

**Interpretation:** Harder than fixed amputation; specialist was not trained for mid-episode recovery.

### D — Cross-leg training

Train with random single-leg amputation; test each leg separately.

**Result:** Cross-leg wins leg 0 (60% fall); specialist wins leg 1 (50% fall); both fail on legs 2–3.

### E — Compound router (`compound_damage.py`)

Not a learned policy — a **runtime switch** by amputated leg id:

| Leg | Route to |
|---:|---|
| 0 | Cross-leg checkpoint |
| 1 | Specialist checkpoint |
| 2, 3 | Cross-leg checkpoint |

**Macro result (40 episodes):** 836 ± 909 reward, 78% fall — best overall damage approach.

**Sudden-amputation demo:** flat policy on 4 legs → switch to specialist at step 120; compared against cross-leg with no switch.

---

## 5. Training recipes

| Checkpoint | Config chain |
|---|---|
| `checkpoints/flat/` | `ant_finetune` |
| `checkpoints/terrain/` | `terrain_finetune` → `terrain_boost` → `terrain_balanced` |
| `checkpoints/damage/` | `damage_upright` → `damage_speed` → `damage_gait` |
| Cross-leg (results/) | `damage_crossleg_upright` → `damage_crossleg_speed` → `damage_crossleg_gait` |

Multi-seed replication: `python replicate_training.py --profiles terrain_canonical damage_canonical --seeds 0 1 2`

---

## 6. What we tried and learned

| Approach | Outcome |
|---|---|
| Curriculum difficulty ramp | Catastrophic forgetting when difficulty increased |
| Fine-tune flat → terrain | Fast but unstable early training |
| Fixed-difficulty terrain + staged fine-tunes | Stable terrain policy (shipped checkpoint) |
| Leg-1-only damage training | Excellent on leg 1, no cross-leg transfer |
| Random-leg damage training | Helps leg 0, hurts leg 1 vs specialist |
| Compound routing | Best macro damage numbers without new training |

---

## 7. Reproducing results

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Primary benchmarks
python compare_policies.py --difficulty 0.4 --seeds 0 1 2 3 4 5 6 7 8 9
python compare_damage.py --disabled-legs 1 --seeds 0 1 2 3 4 5 6 7 8 9
python compound_damage.py

# Regenerate comparison videos
python generate_policy_videos.py

# Tests
python -m unittest discover -s tests -v
```

Artifacts land in `docs/assets/`. JSON files are the source of truth for tables in the README.

---

## 8. Glossary

| Term | Definition |
|---|---|
| **Policy** | Mapping from observations to actions |
| **PPO** | Proximal Policy Optimization — stable on-policy RL |
| **Heightfield** | 2D grid of terrain heights in MuJoCo |
| **GAE** | Generalized Advantage Estimation |
| **Compound router** | Rule-based checkpoint selection by amputated leg |
| **Fall rate** | Fraction of episodes ending in tip-over termination |
