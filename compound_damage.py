"""Compound damage policy: route to specialist or cross-leg checkpoint by amputated leg."""

from __future__ import annotations

import argparse
import json
import os
import time

import imageio
import numpy as np
from stable_baselines3 import PPO

from compare_damage import (
    DEFAULT_FLAT_RUN,
    DEFAULT_AMPUTATION_STEP,
    EpisodeMetrics,
    LEG_LABELS,
    _damage_caption,
    _label_frame,
    _load_model,
    _render_demo,
    _score_demo_seed,
    make_damage_env,
    make_sudden_amputation_env,
    rollout,
    summarize,
)
from render_utils import clear_tracking_state
from results_utils import default_checkpoint

DEFAULT_SPECIALIST_RUN = default_checkpoint("damage")
DEFAULT_CROSSLEG_RUN = (
    "results/ppo_damageant_v0_1783295334_seed0_replicate_damage_crossleg_crossleg_gait"
)

# Per-leg routing from fixed-amputation sweep (10 seeds): pick best policy per leg.
COMPOUND_ROUTING: dict[int, str] = {
    0: "crossleg",  # 1491 reward / 60% fall vs specialist 50 / 100%
    1: "specialist",  # 1449 / 50% vs crossleg 94 / 100%
    2: "crossleg",  # 305 / 100% vs specialist 54 / 100%
    3: "crossleg",  # 99 / 100% vs specialist 70 / 100%
}

DAMAGE_LEGS = [0, 1, 2, 3]
EVAL_SEEDS = list(range(10))


class CompoundDamageRouter:
    """Select PPO checkpoint from which leg is amputated (single-leg amputation only)."""

    def __init__(
        self,
        specialist_run: str,
        crossleg_run: str,
        routing: dict[int, str] | None = None,
    ):
        self.routing = dict(routing or COMPOUND_ROUTING)
        self._runs = {
            "specialist": specialist_run,
            "crossleg": crossleg_run,
        }
        self._models: dict[str, PPO] = {}
        for policy_key in set(self.routing.values()):
            self._models[policy_key] = _load_model(self._runs[policy_key], policy_key)

    def policy_for_legs(self, disabled_legs: list[int]) -> tuple[str, PPO]:
        if len(disabled_legs) != 1:
            raise ValueError(
                f"Compound router expects exactly one amputated leg, got {disabled_legs}"
            )
        leg = int(disabled_legs[0])
        policy_key = self.routing[leg]
        return policy_key, self._models[policy_key]


def compound_rollout(
    router: CompoundDamageRouter,
    env,
    seed: int,
    max_steps: int = 1000,
) -> tuple[EpisodeMetrics, str]:
    obs, _ = env.reset(seed=seed)
    disabled = list(env.unwrapped._disabled_legs)
    policy_key, model = router.policy_for_legs(disabled)

    x0 = env.unwrapped.data.qpos[0]
    reward_sum = 0.0
    fell = False
    steps = 0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        disabled = info.get("disabled_legs", disabled)
        reward_sum += reward
        steps += 1
        if terminated:
            fell = True
            break
        if truncated:
            break

    dt = steps * env.unwrapped.dt
    forward = float(env.unwrapped.data.qpos[0] - x0)
    mean_vel = forward / dt if dt > 0 else 0.0

    return (
        EpisodeMetrics(
            seed=seed,
            reward=float(reward_sum),
            steps=steps,
            forward_distance=forward,
            mean_forward_velocity=mean_vel,
            fell=fell,
            disabled_legs=disabled,
        ),
        policy_key,
    )


def _evaluate_legs(
    *,
    legs: list[int],
    seeds: list[int],
    max_steps: int,
    label: str,
    flat_run: str | None = None,
    policy_run: str | None = None,
    router: CompoundDamageRouter | None = None,
) -> dict:
    by_leg = {}
    all_episodes: list[EpisodeMetrics] = []

    for leg in legs:
        episodes: list[EpisodeMetrics] = []
        for seed in seeds:
            env = make_damage_env([leg], render=False)
            if router is not None:
                episode, _ = compound_rollout(router, env, seed, max_steps)
            elif policy_run is not None:
                model = _load_model(policy_run, label)
                episode = rollout(model, env, seed, max_steps)
            else:
                model = _load_model(flat_run, "flat")
                episode = rollout(model, env, seed, max_steps)
            env.close()
            episodes.append(episode)
            all_episodes.append(episode)

        by_leg[str(leg)] = {
            "leg_id": leg,
            "leg_label": LEG_LABELS[leg],
            "summary": summarize(episodes),
        }

    return {
        "label": label,
        "legs": legs,
        "seeds": seeds,
        "max_steps": max_steps,
        "by_leg": by_leg,
        "macro_summary": summarize(all_episodes),
    }


def compare_compound(
    *,
    flat_run: str = DEFAULT_FLAT_RUN,
    specialist_run: str | None = DEFAULT_SPECIALIST_RUN,
    crossleg_run: str = DEFAULT_CROSSLEG_RUN,
    legs: list[int] | None = None,
    seeds: list[int] | None = None,
    max_steps: int = 1000,
    routing: dict[int, str] | None = None,
) -> dict:
    legs = legs or DAMAGE_LEGS
    seeds = seeds or EVAL_SEEDS
    router = CompoundDamageRouter(specialist_run, crossleg_run, routing)

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "flat_run": flat_run,
        "specialist_run": specialist_run,
        "crossleg_run": crossleg_run,
        "routing": router.routing,
        "flat": _evaluate_legs(
            legs=legs,
            seeds=seeds,
            max_steps=max_steps,
            label="flat",
            flat_run=flat_run,
        ),
        "specialist": _evaluate_legs(
            legs=legs,
            seeds=seeds,
            max_steps=max_steps,
            label="specialist",
            policy_run=specialist_run,
        ),
        "crossleg": _evaluate_legs(
            legs=legs,
            seeds=seeds,
            max_steps=max_steps,
            label="crossleg",
            policy_run=crossleg_run,
        ),
        "compound": _evaluate_legs(
            legs=legs,
            seeds=seeds,
            max_steps=max_steps,
            label="compound",
            router=router,
        ),
    }


def record_compound_side_by_side(
    flat_run: str,
    router: CompoundDamageRouter,
    disabled_legs: list[int],
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    flat_model = _load_model(flat_run, "flat")
    policy_key, compound_model = router.policy_for_legs(disabled_legs)

    env_flat = make_damage_env(
        disabled_legs,
        render=True,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_compound = make_damage_env(disabled_legs, render=True)

    obs_f, _ = env_flat.reset(seed=seed)
    obs_c, _ = env_compound.reset(seed=seed)

    caption = _damage_caption(disabled_legs)
    route_caption = f"compound → {policy_key}"
    frames = []
    clear_tracking_state()

    for _ in range(max_steps):
        a_f, _ = flat_model.predict(obs_f, deterministic=True)
        obs_f, _, _, _, _ = env_flat.step(a_f)
        frame_f = _render_demo(env_flat)

        a_c, _ = compound_model.predict(obs_c, deterministic=True)
        obs_c, _, term_c, trunc_c, _ = env_compound.step(a_c)
        frame_c = _render_demo(env_compound)

        left = _label_frame(frame_f, "Flat-trained (control)", caption)
        right = _label_frame(frame_c, f"Compound ({route_caption})", caption)
        frames.append(np.hstack([left, right]))

        if term_c or trunc_c:
            break

    env_flat.close()
    env_compound.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved compound video: {out_path}  ({len(frames)} frames)")


def record_crossleg_vs_compound(
    crossleg_run: str,
    router: CompoundDamageRouter,
    disabled_legs: list[int],
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    """Show why compound exists: cross-leg fails, router swaps in the right policy."""
    crossleg_model = _load_model(crossleg_run, "crossleg")
    policy_key, compound_model = router.policy_for_legs(disabled_legs)

    env_cross = make_damage_env(
        disabled_legs,
        render=True,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_compound = make_damage_env(disabled_legs, render=True)

    obs_x, _ = env_cross.reset(seed=seed)
    obs_c, _ = env_compound.reset(seed=seed)

    caption = _damage_caption(disabled_legs)
    frames = []
    clear_tracking_state()

    for _ in range(max_steps):
        a_x, _ = crossleg_model.predict(obs_x, deterministic=True)
        obs_x, _, term_x, trunc_x, _ = env_cross.step(a_x)
        frame_x = _render_demo(env_cross)

        a_c, _ = compound_model.predict(obs_c, deterministic=True)
        obs_c, _, term_c, trunc_c, _ = env_compound.step(a_c)
        frame_c = _render_demo(env_compound)

        left = _label_frame(
            frame_x,
            "Cross-leg policy (single model)",
            caption,
        )
        right = _label_frame(
            frame_c,
            f"Compound router → {policy_key}",
            caption,
        )
        frames.append(np.hstack([left, right]))

        if (term_x or trunc_x) and (term_c or trunc_c):
            break

    env_cross.close()
    env_compound.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved crossleg vs compound video: {out_path}  ({len(frames)} frames)")


def pick_sudden_compound_demo_seed(
    flat_run: str,
    specialist_run: str,
    amputation_legs: list[int],
    amputation_step: int,
    candidates: list[int] | None = None,
) -> int:
    """Pick seed where flat walks pre-amputation and specialist survives post-switch."""
    flat_model = _load_model(flat_run, "flat")
    specialist_model = _load_model(specialist_run, "specialist")
    seeds = candidates if candidates is not None else list(range(32))
    best_seed = seeds[0]
    best_score = -1.0

    for seed in seeds:
        env = make_sudden_amputation_env(render=False)
        obs, _ = env.reset(seed=seed)
        fell = False
        steps = 0
        forward = 0.0
        reward_sum = 0.0
        amputated = False

        for step_i in range(1000):
            if step_i == amputation_step and not amputated:
                env.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
                amputated = True

            model = specialist_model if amputated else flat_model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            steps += 1
            if terminated:
                fell = True
                break
            if truncated:
                break

        if amputated:
            forward = float(env.unwrapped.data.qpos[0])
        env.close()

        score = steps + (500.0 if not fell else 0.0) + 300.0 * max(0.0, forward)
        if score > best_score:
            best_score = score
            best_seed = seed

    print(f"Sudden compound demo seed: {best_seed} (score {best_score:.0f})")
    return best_seed


def record_sudden_crossleg_vs_compound(
    flat_run: str,
    crossleg_run: str,
    specialist_run: str,
    amputation_legs: list[int],
    amputation_step: int,
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    """
    Sudden amputation demo: both ants start on 4 legs.

    Left: cross-leg policy throughout (fails after leg-1 removal).
    Right: flat 4-leg policy, then compound switches to specialist at amputation.
    """
    crossleg_model = _load_model(crossleg_run, "crossleg")
    flat_model = _load_model(flat_run, "flat")
    specialist_model = _load_model(specialist_run, "specialist")

    env_cross = make_sudden_amputation_env(
        render=True,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_compound = make_sudden_amputation_env(render=True)

    obs_x, _ = env_cross.reset(seed=seed)
    obs_c, _ = env_compound.reset(seed=seed)

    pre_caption = "4 legs — walking normally"
    post_caption = _damage_caption(amputation_legs)
    frames = []
    clear_tracking_state()
    amputated = False
    cross_done = False
    last_cross = None

    for step_i in range(max_steps):
        if step_i == amputation_step and not amputated:
            env_cross.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
            env_compound.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
            amputated = True

        if not cross_done:
            a_x, _ = crossleg_model.predict(obs_x, deterministic=True)
            obs_x, _, term_x, trunc_x, _ = env_cross.step(a_x)
            last_cross = _render_demo(env_cross)
            if amputated and (term_x or trunc_x):
                cross_done = True

        compound_model = specialist_model if amputated else flat_model
        compound_label = (
            f"Compound: flat → specialist"
            if amputated
            else "Compound: flat (4 legs)"
        )
        a_c, _ = compound_model.predict(obs_c, deterministic=True)
        obs_c, _, term_c, trunc_c, _ = env_compound.step(a_c)
        frame_c = _render_demo(env_compound)

        caption = post_caption if amputated else pre_caption
        left = _label_frame(
            last_cross if last_cross is not None else frame_c,
            "Cross-leg policy (no switch)",
            caption,
        )
        right = _label_frame(frame_c, compound_label, caption)
        frames.append(np.hstack([left, right]))

        if cross_done and (term_c or trunc_c):
            break

    env_cross.close()
    env_compound.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved sudden crossleg vs compound video: {out_path}  ({len(frames)} frames)")


def record_single_policy(
    run_dir: str,
    disabled_legs: list[int],
    seed: int,
    max_steps: int,
    out_path: str,
    title: str,
    fps: int = 30,
):
    model = _load_model(run_dir, "policy")
    env = make_damage_env(disabled_legs, render=True)
    obs, _ = env.reset(seed=seed)
    caption = _damage_caption(disabled_legs)
    frames = []
    clear_tracking_state()

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        frame = _render_demo(env)
        frames.append(_label_frame(frame, title, caption))
        if term or trunc:
            break

    env.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved policy video: {out_path}  ({len(frames)} frames)")


def pick_demo_seed_for_run(run_dir: str, disabled_legs: list[int]) -> int:
    model = _load_model(run_dir, "policy")
    seeds = list(range(32))
    best_seed = seeds[0]
    best_score = -1.0
    for seed in seeds:
        score = _score_demo_seed(model, disabled_legs, seed)
        if score > best_score:
            best_score = score
            best_seed = seed
    print(f"Demo seed for {run_dir} legs {disabled_legs}: {best_seed}")
    return best_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate compound leg-routed damage policy")
    parser.add_argument("--flat-run", default=DEFAULT_FLAT_RUN)
    parser.add_argument("--specialist-run", default=DEFAULT_SPECIALIST_RUN)
    parser.add_argument("--crossleg-run", default=DEFAULT_CROSSLEG_RUN)
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/assets/damage/compound")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.specialist_run is None:
        parser.error("No specialist checkpoint found. Pass --specialist-run.")

    results = compare_compound(
        flat_run=args.flat_run,
        specialist_run=args.specialist_run,
        crossleg_run=args.crossleg_run,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "compound_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    for label in ("flat", "specialist", "crossleg", "compound"):
        macro = results[label]["macro_summary"]
        print(
            f"{label:12} | reward {macro['mean_reward']:.0f} ± {macro['std_reward']:.0f} | "
            f"fall {macro['fall_rate']*100:.0f}%"
        )

    if not args.no_video:
        amputation_legs = [1]
        seed = pick_sudden_compound_demo_seed(
            args.flat_run,
            args.specialist_run,
            amputation_legs,
            DEFAULT_AMPUTATION_STEP,
        )
        record_sudden_crossleg_vs_compound(
            args.flat_run,
            args.crossleg_run,
            args.specialist_run,
            amputation_legs,
            DEFAULT_AMPUTATION_STEP,
            seed,
            args.max_steps,
            "docs/assets/damage/compound_comparison_demo.mp4",
        )
