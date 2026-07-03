"""Compare flat-trained vs damage-robust policies under leg amputation."""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from results_utils import default_checkpoint

from envs.damage_ant import LEG_LABELS

DEFAULT_FLAT_RUN = default_checkpoint("flat") or "checkpoints/flat"
DEFAULT_DAMAGE_RUN = default_checkpoint("damage")
DEFAULT_DISABLED_LEGS = [1]
DEFAULT_VIDEO_SEED = 7
DEFAULT_AMPUTATION_STEP = 120
DEMO_CAMERA_NAME = "demo"
_LOOKAT: dict[int, np.ndarray] = {}
_LOOKAT_BLEND = 0.55
_DEMO_DISTANCE = 6.5
_DEMO_AZIMUTH = 90.0
_DEMO_ELEVATION = -15.0


@dataclass
class EpisodeMetrics:
    seed: int
    reward: float
    steps: int
    forward_distance: float
    mean_forward_velocity: float
    fell: bool
    disabled_legs: list[int]


def make_damage_env(disabled_legs: list[int], render: bool = False, **kwargs):
    from envs import register

    register()
    env_kwargs = {"fixed_disabled_legs": list(disabled_legs), **kwargs}
    if render:
        env_kwargs.update(
            render_mode="rgb_array",
            width=640,
            height=480,
        )
    env = gym.make("DamageAnt-v0", **env_kwargs)
    if render:
        env.unwrapped.model.vis.global_.fovy = 50.0
    return env


def make_sudden_amputation_env(render: bool = False, **kwargs):
    """Start with all 4 legs intact; amputate later via set_damage()."""
    return make_damage_env(
        [],
        render=render,
        fixed_disabled_legs=[],
        min_disabled_legs=0,
        max_disabled_legs=0,
        **kwargs,
    )


def _checkpoint_prefix(run_dir: str | None, label: str) -> str:
    if not run_dir:
        raise FileNotFoundError(
            f"No {label} checkpoint configured. Pass --{label}-run explicitly."
        )
    prefix = os.path.join(run_dir, "best_model", "best_model")
    if not os.path.isfile(prefix + ".zip"):
        raise FileNotFoundError(f"No checkpoint found at {prefix}.zip")
    return prefix


def _load_model(run_dir: str | None, label: str) -> PPO:
    return PPO.load(_checkpoint_prefix(run_dir, label))


def _render_demo(env) -> np.ndarray:
    """Free camera with smoothed torso lookat — body trackcom drifts off-frame when walking far."""
    renderer = env.unwrapped.mujoco_renderer
    viewer = renderer._get_viewer(render_mode="rgb_array")
    cam = viewer.cam
    torso_id = env.unwrapped.model.body("torso").id
    target = env.unwrapped.data.xpos[torso_id].copy()

    key = id(env)
    prev = _LOOKAT.get(key)
    if prev is None:
        _LOOKAT[key] = target.copy()
    else:
        _LOOKAT[key] = _LOOKAT_BLEND * target + (1.0 - _LOOKAT_BLEND) * prev

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = _LOOKAT[key]
    cam.distance = _DEMO_DISTANCE
    cam.azimuth = _DEMO_AZIMUTH
    cam.elevation = _DEMO_ELEVATION
    return viewer.render(render_mode="rgb_array", camera_id=-1)


def _damage_caption(disabled_legs: list[int]) -> str:
    names = ", ".join(f"leg {i} ({LEG_LABELS[i]}) amputated" for i in disabled_legs)
    return names


def rollout(model: PPO, env, seed: int, max_steps: int = 1000) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    x0 = env.unwrapped.data.qpos[0]
    reward_sum = 0.0
    fell = False
    steps = 0
    disabled = []

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
    return EpisodeMetrics(
        seed=seed,
        reward=float(reward_sum),
        steps=steps,
        forward_distance=forward,
        mean_forward_velocity=mean_vel,
        fell=fell,
        disabled_legs=disabled,
    )


def summarize(episodes: list[EpisodeMetrics]) -> dict:
    rewards = [e.reward for e in episodes]
    steps = [e.steps for e in episodes]
    dists = [e.forward_distance for e in episodes]
    vels = [e.mean_forward_velocity for e in episodes]
    fall_rate = sum(e.fell for e in episodes) / len(episodes)
    return {
        "n_episodes": len(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_forward_distance": float(np.mean(dists)),
        "std_forward_distance": float(np.std(dists)),
        "mean_forward_velocity": float(np.mean(vels)),
        "std_forward_velocity": float(np.std(vels)),
        "fall_rate": float(fall_rate),
    }


def _comparison_metrics(flat_s: dict, damage_s: dict) -> dict:
    flat_vel = float(flat_s["mean_forward_velocity"])
    damage_vel = float(damage_s["mean_forward_velocity"])
    velocity_retention_pct = None
    if flat_vel > 1e-6:
        velocity_retention_pct = float(100.0 * damage_vel / flat_vel)

    return {
        "velocity_retention_pct": velocity_retention_pct,
        "velocity_gain_mps": float(damage_vel - flat_vel),
        "reward_retention_pct": float(
            100.0 * damage_s["mean_reward"] / max(flat_s["mean_reward"], 1e-6)
        ),
        "reward_gain": float(damage_s["mean_reward"] - flat_s["mean_reward"]),
        "fall_rate_reduction_pct": float(
            100.0 * (flat_s["fall_rate"] - damage_s["fall_rate"])
        ),
        "episode_length_gain_steps": float(
            damage_s["mean_steps"] - flat_s["mean_steps"]
        ),
    }


def compare(
    flat_run: str,
    damage_run: str,
    disabled_legs: list[int],
    seeds: list[int],
    max_steps: int = 1000,
) -> dict:
    flat_model = _load_model(flat_run, "flat")
    damage_model = _load_model(damage_run, "damage")

    flat_eps: list[EpisodeMetrics] = []
    damage_eps: list[EpisodeMetrics] = []

    for seed in seeds:
        env = make_damage_env(disabled_legs, render=False)
        flat_eps.append(rollout(flat_model, env, seed, max_steps))
        env.close()

        env = make_damage_env(disabled_legs, render=False)
        damage_eps.append(rollout(damage_model, env, seed, max_steps))
        env.close()

        print(
            f"seed {seed:3d} | flat: R={flat_eps[-1].reward:7.0f} "
            f"vel={flat_eps[-1].mean_forward_velocity:.2f} fell={flat_eps[-1].fell} | "
            f"damage: R={damage_eps[-1].reward:7.0f} "
            f"vel={damage_eps[-1].mean_forward_velocity:.2f} fell={damage_eps[-1].fell}"
        )

    flat_s = summarize(flat_eps)
    damage_s = summarize(damage_eps)
    results = {
        "disabled_legs": disabled_legs,
        "max_steps": max_steps,
        "seeds": seeds,
        "flat_run": flat_run,
        "damage_run": damage_run,
        "flat": {"episodes": [asdict(e) for e in flat_eps], "summary": flat_s},
        "damage": {"episodes": [asdict(e) for e in damage_eps], "summary": damage_s},
    }
    results.update(_comparison_metrics(flat_s, damage_s))
    return results


def rollout_sudden_amputation(
    model: PPO,
    env,
    amputation_legs: list[int],
    amputation_step: int,
    seed: int,
    max_steps: int = 1000,
) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    x0 = env.unwrapped.data.qpos[0]
    reward_sum = 0.0
    fell = False
    steps = 0
    disabled: list[int] = []

    for step_i in range(max_steps):
        if step_i == amputation_step:
            env.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
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
    return EpisodeMetrics(
        seed=seed,
        reward=float(reward_sum),
        steps=steps,
        forward_distance=forward,
        mean_forward_velocity=mean_vel,
        fell=fell,
        disabled_legs=disabled,
    )


def compare_sudden_amputation(
    flat_run: str,
    damage_run: str,
    amputation_legs: list[int],
    amputation_step: int,
    seeds: list[int],
    max_steps: int = 1000,
) -> dict:
    flat_model = _load_model(flat_run, "flat")
    damage_model = _load_model(damage_run, "damage")

    flat_eps: list[EpisodeMetrics] = []
    damage_eps: list[EpisodeMetrics] = []

    for seed in seeds:
        env = make_sudden_amputation_env(render=False)
        flat_eps.append(
            rollout_sudden_amputation(
                flat_model, env, amputation_legs, amputation_step, seed, max_steps
            )
        )
        env.close()

        env = make_sudden_amputation_env(render=False)
        damage_eps.append(
            rollout_sudden_amputation(
                damage_model, env, amputation_legs, amputation_step, seed, max_steps
            )
        )
        env.close()

    flat_s = summarize(flat_eps)
    damage_s = summarize(damage_eps)
    results = {
        "disabled_legs": amputation_legs,
        "amputation_step": amputation_step,
        "max_steps": max_steps,
        "seeds": seeds,
        "flat_run": flat_run,
        "damage_run": damage_run,
        "flat": {"episodes": [asdict(e) for e in flat_eps], "summary": flat_s},
        "damage": {"episodes": [asdict(e) for e in damage_eps], "summary": damage_s},
    }
    results.update(_comparison_metrics(flat_s, damage_s))
    return results


def plot_comparison(results: dict, out_path: str):
    labels = ["Flat-trained\n(control)", "Damage-robust\n(treatment)"]
    flat_s = results["flat"]["summary"]
    damage_s = results["damage"]["summary"]
    metrics = [
        ("Mean reward", flat_s["mean_reward"], damage_s["mean_reward"]),
        ("Forward velocity (m/s)", flat_s["mean_forward_velocity"], damage_s["mean_forward_velocity"]),
        ("Forward distance (m)", flat_s["mean_forward_distance"], damage_s["mean_forward_distance"]),
        ("Fall rate (%)", flat_s["fall_rate"] * 100, damage_s["fall_rate"] * 100),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (title, flat_v, damage_v) in zip(axes, metrics):
        bars = ax.bar(labels, [flat_v, damage_v], color=["#dc2626", "#16a34a"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, [flat_v, damage_v]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    leg = results["disabled_legs"]
    fig.suptitle(f"Leg damage comparison — disabled legs {leg}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _label_frame(frame: np.ndarray, title: str, subtitle: str = "", banner_h: int = 48) -> np.ndarray:
    img = Image.fromarray(frame)
    canvas = Image.new("RGB", (img.width, img.height + banner_h), (30, 30, 30))
    canvas.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(255, 255, 255))
    if subtitle:
        draw.text((12, 26), subtitle, fill=(200, 200, 200))
    return np.array(canvas)


def _score_demo_seed(
    model: PPO,
    disabled_legs: list[int],
    seed: int,
    *,
    max_steps: int = 1000,
    amputation_step: int | None = None,
) -> float:
    if amputation_step is None:
        env = make_damage_env(disabled_legs, render=False)
    else:
        env = make_sudden_amputation_env(render=False)

    obs, _ = env.reset(seed=seed)
    x0 = env.unwrapped.data.qpos[0]
    steps = 0
    fell = False
    min_up = 1.0

    for step_i in range(max_steps):
        if amputation_step is not None and step_i == amputation_step:
            env.unwrapped.set_damage(disabled_legs, reset_tip_grace=True)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        min_up = min(min_up, info.get("torso_uprightness", 0.0))
        if terminated:
            fell = True
            break
        if truncated:
            break

    forward = float(env.unwrapped.data.qpos[0] - x0)
    env.close()
    return steps + (500.0 if not fell else 0.0) + 300.0 * forward + 150.0 * max(0.0, min_up)


def pick_demo_seed(
    damage_run: str,
    disabled_legs: list[int],
    candidates: list[int] | None = None,
) -> int:
    """Pick the best seed for the fixed-amputation comparison demo."""
    model = _load_model(damage_run, "damage")
    seeds = candidates if candidates is not None else list(range(32))
    best_seed = seeds[0]
    best_score = -1.0

    for seed in seeds:
        score = _score_demo_seed(model, disabled_legs, seed)
        if score > best_score:
            best_score = score
            best_seed = seed
    print(f"Fixed-amputation demo seed: {best_seed} (score {best_score:.0f})")
    return best_seed


def pick_sudden_amputation_seed(
    damage_run: str,
    amputation_legs: list[int],
    amputation_step: int,
    candidates: list[int] | None = None,
) -> int:
    """Pick the best seed for the sudden-amputation demo."""
    model = _load_model(damage_run, "damage")
    seeds = candidates if candidates is not None else list(range(32))
    best_seed = seeds[0]
    best_score = -1.0

    for seed in seeds:
        score = _score_demo_seed(
            model,
            amputation_legs,
            seed,
            amputation_step=amputation_step,
        )
        if score > best_score:
            best_score = score
            best_seed = seed
    print(f"Sudden-amputation demo seed: {best_seed} (score {best_score:.0f})")
    return best_seed


def record_side_by_side(
    flat_run: str,
    damage_run: str,
    disabled_legs: list[int],
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    flat_model = _load_model(flat_run, "flat")
    damage_model = _load_model(damage_run, "damage")

    # Flat control: keep stepping after collapse so legs keep flailing (no tip/health kill).
    env_flat = make_damage_env(
        disabled_legs,
        render=True,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_damage = make_damage_env(disabled_legs, render=True)

    obs_f, _ = env_flat.reset(seed=seed)
    obs_d, _ = env_damage.reset(seed=seed)

    caption = _damage_caption(disabled_legs)
    frames = []
    _LOOKAT.clear()

    for _ in range(max_steps):
        a_f, _ = flat_model.predict(obs_f, deterministic=True)
        obs_f, _, _, _, _ = env_flat.step(a_f)
        frame_f = _render_demo(env_flat)

        a_d, _ = damage_model.predict(obs_d, deterministic=True)
        obs_d, _, term_d, trunc_d, _ = env_damage.step(a_d)
        frame_d = _render_demo(env_damage)

        left = _label_frame(frame_f, "Flat-trained (control)", caption)
        right = _label_frame(frame_d, "Damage-robust (treatment)", caption)
        frames.append(np.hstack([left, right]))

        if term_d or trunc_d:
            break

    env_flat.close()
    env_damage.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved video: {out_path}  ({len(frames)} frames)")


def record_sudden_amputation(
    flat_run: str,
    damage_run: str,
    amputation_legs: list[int],
    amputation_step: int,
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    """Record side-by-side demo: both ants walk on 4 legs, then leg(s) amputated mid-episode."""
    flat_model = _load_model(flat_run, "flat")
    damage_model = _load_model(damage_run, "damage")

    env_flat = make_sudden_amputation_env(
        render=True,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_damage = make_sudden_amputation_env(render=True)

    obs_f, _ = env_flat.reset(seed=seed)
    obs_d, _ = env_damage.reset(seed=seed)

    pre_caption = "4 legs — walking normally"
    post_caption = _damage_caption(amputation_legs)
    frames = []
    _LOOKAT.clear()
    flat_done = False
    last_flat = None
    amputated = False

    for step_i in range(max_steps):
        if step_i == amputation_step and not amputated:
            env_flat.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
            env_damage.unwrapped.set_damage(amputation_legs, reset_tip_grace=True)
            amputated = True

        if not flat_done:
            a_f, _ = flat_model.predict(obs_f, deterministic=True)
            obs_f, _, term_f, trunc_f, _ = env_flat.step(a_f)
            last_flat = _render_demo(env_flat)
            if amputated and (term_f or trunc_f):
                flat_done = True

        a_d, _ = damage_model.predict(obs_d, deterministic=True)
        obs_d, _, term_d, trunc_d, _ = env_damage.step(a_d)
        frame_d = _render_demo(env_damage)

        caption = post_caption if amputated else pre_caption
        left = _label_frame(
            last_flat if last_flat is not None else frame_d,
            "Flat-trained (control)",
            caption,
        )
        right = _label_frame(frame_d, "Damage-robust (treatment)", caption)
        frames.append(np.hstack([left, right]))

        if term_d or trunc_d:
            break

    env_flat.close()
    env_damage.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved sudden-amputation video: {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare flat vs damage-robust under leg failure")
    parser.add_argument("--flat-run", default=DEFAULT_FLAT_RUN)
    parser.add_argument("--damage-run", default=DEFAULT_DAMAGE_RUN)
    parser.add_argument("--disabled-legs", type=int, nargs="+", default=DEFAULT_DISABLED_LEGS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/assets/damage")
    parser.add_argument("--video-seed", type=int, default=None, help="Auto-pick best if omitted")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--amputation-step",
        type=int,
        default=DEFAULT_AMPUTATION_STEP,
        help="Simulation step at which leg(s) are amputated in sudden-amputation demo",
    )
    args = parser.parse_args()

    if args.damage_run is None:
        parser.error(
            "No committed damage checkpoint found. Pass --damage-run explicitly."
        )

    results = compare(
        args.flat_run,
        args.damage_run,
        args.disabled_legs,
        args.seeds,
        args.max_steps,
    )
    sudden_results = compare_sudden_amputation(
        args.flat_run,
        args.damage_run,
        args.disabled_legs,
        args.amputation_step,
        args.seeds,
        args.max_steps,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        payload = dict(results)
        payload["sudden_amputation"] = sudden_results
        json.dump(payload, f, indent=2)
    print(f"Saved metrics: {json_path}")

    plot_comparison(results, os.path.join(args.out_dir, "comparison_plot.png"))

    flat_s = results["flat"]["summary"]
    damage_s = results["damage"]["summary"]
    print("\n=== Summary ===")
    print(
        f"Flat-trained     | reward {flat_s['mean_reward']:.0f} | "
        f"vel {flat_s['mean_forward_velocity']:.2f} m/s | fall {flat_s['fall_rate']*100:.0f}%"
    )
    print(
        f"Damage-robust    | reward {damage_s['mean_reward']:.0f} | "
        f"vel {damage_s['mean_forward_velocity']:.2f} m/s | fall {damage_s['fall_rate']*100:.0f}%"
    )
    if results["velocity_retention_pct"] is None:
        print(
            f"Velocity gain: {results['velocity_gain_mps']:+.2f} m/s "
            "(retention undefined because flat baseline velocity <= 0)"
        )
    else:
        print(f"Velocity retention: {results['velocity_retention_pct']:.0f}%")

    sudden_flat = sudden_results["flat"]["summary"]
    sudden_damage = sudden_results["damage"]["summary"]
    print("\n=== Sudden Amputation Summary ===")
    print(
        f"Flat-trained     | reward {sudden_flat['mean_reward']:.0f} | "
        f"vel {sudden_flat['mean_forward_velocity']:.2f} m/s | fall {sudden_flat['fall_rate']*100:.0f}%"
    )
    print(
        f"Damage-robust    | reward {sudden_damage['mean_reward']:.0f} | "
        f"vel {sudden_damage['mean_forward_velocity']:.2f} m/s | fall {sudden_damage['fall_rate']*100:.0f}%"
    )

    if not args.no_video:
        side_by_side_seed = args.video_seed
        sudden_seed = args.video_seed
        if side_by_side_seed is None:
            side_by_side_seed = pick_demo_seed(args.damage_run, args.disabled_legs)
            sudden_seed = pick_sudden_amputation_seed(
                args.damage_run,
                args.disabled_legs,
                args.amputation_step,
            )
        record_side_by_side(
            args.flat_run,
            args.damage_run,
            args.disabled_legs,
            side_by_side_seed,
            args.max_steps,
            os.path.join(args.out_dir, "comparison_demo.mp4"),
        )
        record_sudden_amputation(
            args.flat_run,
            args.damage_run,
            args.disabled_legs,
            args.amputation_step,
            sudden_seed,
            args.max_steps,
            os.path.join(args.out_dir, "sudden_amputation_demo.mp4"),
        )
