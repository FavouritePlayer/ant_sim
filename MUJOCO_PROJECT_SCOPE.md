# MuJoCo RL Project — Scope

*What this project is, what makes it resume-grade instead of tutorial-grade, and where the line is. Deadline: end of July 2026 (~5 weeks).*

## One-line pitch (the target end state)

A quadruped locomotion agent that stays robust under actuator failure — trained in MuJoCo (Ant-v4), with a measured result against a baseline that fails. The pitch is *robustness*, not *walking*.

## The bar: what separates a resume project from a tutorial

**Training PPO on Ant-v4 is not the project. It's the prerequisite.** Thousands of people have done it; it's a weekend tutorial, and listing it signals "I followed a guide." A resume-grade project has three things a tutorial doesn't:

1. **A novel contribution** — a question you posed and answered, not a benchmark you reproduced.
2. **A measured result vs. a baseline** — "robust policy retained 70% of forward velocity under single-leg failure; baseline collapsed to 12%." A number against a control. This is the line that survives a depth interview.
3. **Depth you own** — you can explain *why* the reward function is shaped the way it is, *why* PPO vs. SAC, what failed and what you changed. The interview these roles run is "tell me about the hardest thing you built" — the project exists to give you a real answer.

If the project has all three, it's worth a resume slot. If it's just a trained baseline, it isn't.

## Honest fork: this is RL, not CV

You want a **computer-vision research position this fall.** This project proves general ML/RL depth and is strong for ML-engineering and autonomy/robotics roles — but it is **not computer vision.** Decide consciously:

- **If the autonomy/ML-eng target is primary:** leg-damage robustness (below) is the right scope. Ship it.
- **If CV-research is genuinely the firm target:** the bridge is a **vision-based policy** — the agent learns from rendered camera frames through a CNN encoder instead of proprioceptive state. This makes it genuinely CV + RL and serves the research narrative. **But it is significantly harder** (sample efficiency drops hard, training is slower and finickier) and has real risk of not finishing in 5 weeks. High-risk, high-reward. Only take it if CV-research is firm and you accept the schedule risk.
- **Don't straddle.** Pick one. A finished robustness project beats a half-finished vision one.

The rest of this doc assumes the **leg-damage robustness** scope (the recommended default) and notes the vision variant where relevant.

## Scope — what's IN

**Phase 1 — Baseline (Weeks 1–2, table stakes):**
- PPO via stable-baselines3 on Ant-v4, TensorBoard logging, saved checkpoints, render pipeline working.
- Target: stable forward locomotion (a respectable return — roughly 3000+ episode reward; the bar is "clearly walks," not a specific number).
- This is infrastructure, not the contribution. Do not over-invest. Done = it walks and you can render it.

**Phase 2 — The contribution (Weeks 3–4, the actual project):**
- Inject actuator failure: during training, randomly disable a leg (zero its actuator torque) with some probability — domain randomization over failure.
- Retrain so the policy learns to compensate. Compare against the Phase-1 baseline under the same failures.
- Result: a side-by-side — baseline collapses when damaged, your robust policy adapts. Quantified (velocity/return retained under k disabled actuators) and on video.

**Phase 3 — Make it legible (Week 4–5):**
- Public GitHub repo, clean README (problem, approach, reward design, what failed, result).
- Rendered videos: baseline-damaged vs. robust-damaged, side by side.
- One results plot (return vs. number of disabled actuators, both policies).

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
- The **failure-injection mechanism** (how damage is modeled in the env).
- The **RL fundamentals** — why PPO, what the advantage estimate does, on- vs. off-policy, why the algorithm is stable or isn't.
- The **experimental design** — what's the control, what's held constant, what the metric measures.

Boilerplate (fine to delegate to Claude Code): training-loop scaffolding, TensorBoard wiring, video rendering, plotting, repo setup. If you can't explain a line in an interview, it doesn't go in without you understanding it first.

## The resume bullet this produces (target)

> *Trained a damage-robust quadruped locomotion policy (PPO, MuJoCo Ant-v4); via randomized actuator-failure training, retained ~70% forward velocity under single-leg failure vs. ~12% for a standard baseline. [repo link]*

Numbers are illustrative — fill with real results. One specific, measured bullet like this outweighs five tool-listing bullets.

## Alternative contributions (if not leg-damage)

- **Terrain adaptation** — train on randomized heightfield terrain, test generalization to unseen terrain vs. flat-trained baseline. Strong narrative; more env-engineering overhead (heightfield generation).
- **Velocity-command following** — condition the policy on a target speed, show it tracks commanded velocities. Cleanest control framing; least dramatic demo.

Pick on the basis of which you can explain the *why* of most convincingly in an interview — that, not the topic, is what the project is for.
