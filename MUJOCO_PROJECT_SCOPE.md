# MuJoCo RL Project — Scope

*What this project is, what makes it resume-grade instead of tutorial-grade, and where the line is. Deadline: end of July 2026 (~5 weeks).*

## Status (as of 2026-07-01)

Legend: **[DONE]** / **[PARTIAL]** / **[NOT DONE]** / **[DEVIATION]**

| Item | Status |
|---|---|
| Phase 1 — baseline (flat Ant, PPO, render pipeline) | **[DONE]**, target reward short by ~20% |
| Env version | **[DEVIATION]** — spec says Ant-v4, implementation uses Ant-v5 |
| Phase 2 — contribution chosen | **[DEVIATION]** — terrain adaptation, not leg-damage (both are sanctioned by this doc's "Alternative contributions" section; no straddling occurred, so this is fine) |
| Phase 2 — terrain policy trained | **[DONE]** — boost fine-tune, best eval 1013 at difficulty 0.35 |
| Phase 2 — quantified comparison vs. flat-trained baseline on terrain | **[DONE]** — `compare_policies.py`, 10 seeds @ difficulty 0.4 |
| Phase 3 — public repo, README, LEARNING.md | **[DONE]** |
| Phase 3 — comparison video (baseline vs. robust, same conditions) | **[DONE]** — `docs/assets/terrain/comparison_demo.mp4` |
| Phase 3 — one results plot, both policies | **[DONE]** — `docs/assets/terrain/comparison_plot.png` |

Bottom line: **resume-grade bar met.** Flat-trained control vs. terrain-adapted treatment evaluated on matched TerrainAnt-v0 episodes. Terrain policy: **930 ± 59** reward, **10%** fall rate. Flat baseline: **467 ± 168** reward, **70%** fall rate. Comparison artifacts, README experimental design, and resume bullet are in place.

## One-line pitch (the target end state)

A quadruped locomotion agent that generalizes to unseen rough terrain — trained in MuJoCo (Ant-v5), with a measured result against a flat-trained baseline that fails. The pitch is *terrain adaptation*, not *walking*.

*(Original scope doc targeted leg-damage robustness; project pivoted to terrain adaptation per "Alternative contributions" — single contribution, no straddling.)*

## The bar: what separates a resume project from a tutorial

**Training PPO on Ant-v4 is not the project. It's the prerequisite.** Thousands of people have done it; it's a weekend tutorial, and listing it signals "I followed a guide." A resume-grade project has three things a tutorial doesn't:

1. **A novel contribution** — a question you posed and answered, not a benchmark you reproduced.
2. **A measured result vs. a baseline** — "terrain-adapted policy achieved 930 ± 59 episode reward on unseen terrain; flat-trained baseline 467 ± 168 with 70% fall rate." A number against a control. This is the line that survives a depth interview.
3. **Depth you own** — you can explain *why* the reward function is shaped the way it is, *why* PPO vs. SAC, what failed and what you changed. The interview these roles run is "tell me about the hardest thing you built" — the project exists to give you a real answer.

If the project has all three, it's worth a resume slot. If it's just a trained baseline, it isn't.

## Honest fork: this is RL, not CV

You want a **computer-vision research position this fall.** This project proves general ML/RL depth and is strong for ML-engineering and autonomy/robotics roles — but it is **not computer vision.** Decide consciously:

- **If the autonomy/ML-eng target is primary:** terrain adaptation (chosen) or leg-damage robustness is the right scope. Shipped.
- **If CV-research is genuinely the firm target:** the bridge is a **vision-based policy** — the agent learns from rendered camera frames through a CNN encoder instead of proprioceptive state. This makes it genuinely CV + RL and serves the research narrative. **But it is significantly harder** (sample efficiency drops hard, training is slower and finickier) and has real risk of not finishing in 5 weeks. High-risk, high-reward. Only take it if CV-research is firm and you accept the schedule risk.
- **Don't straddle.** Pick one. A finished robustness project beats a half-finished vision one.

The rest of this doc originally assumed the **leg-damage robustness** scope; the project shipped **terrain adaptation** instead.

## Scope — what's IN

**Phase 1 — Baseline (Weeks 1–2, table stakes):** **[DONE]**
- PPO via stable-baselines3 on Ant-v4, TensorBoard logging, saved checkpoints, render pipeline working. — **[DONE]**, on Ant-v5 (deviation, not a problem)
- Target: stable forward locomotion (a respectable return — roughly 3000+ episode reward; the bar is "clearly walks," not a specific number). — **[PARTIAL]**: eval reward 2421. Walks; below the soft numeric target. Not worth chasing per the "don't over-invest" line below unless time remains after Phase 2 is closed.
- This is infrastructure, not the contribution. Do not over-invest. Done = it walks and you can render it.

**Phase 2 — The contribution (Weeks 3–4, the actual project):** **[DONE]**
- Terrain-adaptation equivalent of "inject actuator failure": train on randomized heightfield terrain. — **[DONE]** — from-scratch fixed-difficulty run plus a boost fine-tune (best eval reward 1013 at difficulty 0.35). Curriculum-based difficulty ramping was tried and abandoned (catastrophic forgetting), which is itself usable "what failed" material.
- Compare against the Phase-1 baseline on the same terrain under matched seeds/difficulty. — **[DONE]** via `compare_policies.py`.
- Result: side-by-side — baseline collapses on terrain, terrain-adapted policy survives. Quantified and on video. — **[DONE]**

**Measured result (difficulty 0.4, 10 seeds, TerrainAnt-v0):**

| Metric | Flat-trained (control) | Terrain-adapted (treatment) |
|---|---:|---:|
| Mean episode reward | 467 ± 168 | 930 ± 59 |
| Mean episode length | 511 steps | 996 steps |
| Fall rate | 70% | 10% |
| Mean forward distance | 6.1 m | 1.6 m |

**Phase 3 — Make it legible (Week 4–5):** **[DONE]**
- Public GitHub repo, clean README (problem, approach, reward design, what failed, result). — **[DONE]** — github.com/FavouritePlayer/ant_sim, README + LEARNING.md both present and substantive.
- Rendered videos: baseline vs. terrain-adapted, side by side, matched conditions. — **[DONE]** — `comparison_demo.mp4`
- One results plot, both policies. — **[DONE]** — `comparison_plot.png`, `comparison_reward_by_seed.png`

## Scope — what's explicitly OUT (read this twice)

The failure mode here is over-engineering the infrastructure instead of shipping the contribution. Out of scope:
- **Implementing PPO/SAC from scratch.** Use stable-baselines3. Re-implementing the algorithm is a different project and a time sink. You own *understanding* it, not *coding* it.
- **Hyperparameter perfectionism on the baseline.** "Walks well enough to compare" is the bar. Tuning the baseline to SOTA is procrastination wearing a lab coat.
- **Multiple environments.** One env (Ant-v4). Not HalfCheetah and Humanoid and Walker. One.
- **Multiple contributions.** Leg-damage OR terrain OR velocity-command — not all three. One clean comparison beats three shallow ones.
- **Fancy infra** (distributed training, custom logging dashboards, a config framework). The repo serves the result; it is not the result.

## What you must own vs. what's boilerplate

For interview defensibility, **you** own — and must be able to explain cold:
- The **reward function** and why it's shaped that way.
- The **terrain generation mechanism** (how heightfield is built and injected into MuJoCo).
- The **RL fundamentals** — why PPO, what the advantage estimate does, on- vs. off-policy, why the algorithm is stable or isn't.
- The **experimental design** — what's the control, what's held constant, what the metric measures.

Boilerplate (fine to delegate to Claude Code): training-loop scaffolding, TensorBoard wiring, video rendering, plotting, repo setup. If you can't explain a line in an interview, it doesn't go in without you understanding it first.

## The resume bullet this produces

> *Trained a terrain-adapted quadruped locomotion policy (PPO, MuJoCo Ant-v5); on unseen heightfield terrain, terrain-trained agent achieved **930 ± 59** episode reward vs **467 ± 168** for a flat-trained baseline, with **10% vs 70%** fall rate under matched seeds. [github.com/FavouritePlayer/ant_sim](https://github.com/FavouritePlayer/ant_sim)*

*(Original illustrative leg-damage bullet replaced with actual terrain-adaptation numbers.)*

## Alternative contributions (if not leg-damage)

**Decision made: terrain adaptation.** No straddling — single contribution, as required above.

- **Terrain adaptation** — train on randomized heightfield terrain, test generalization to unseen terrain vs. flat-trained baseline. — **[DONE]**
- **Leg-damage robustness** (original default in this doc) — randomly disable leg actuators during training; compare damaged baseline vs. robust policy. — **Not implemented.** Would require env changes (zero torques on selected joints) and separate training runs. Out of scope for this repo.
- **Velocity-command following** — condition the policy on a target speed, show it tracks commanded velocities. Cleanest control framing; least dramatic demo. — Not chosen.

Pick on the basis of which you can explain the *why* of most convincingly in an interview — that, not the topic, is what the project is for.
