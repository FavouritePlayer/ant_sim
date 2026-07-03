"""Compare flat-trained vs damage-robust policies under leg actuator failure."""

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
DEFAULT_DAMAGE_RUN = default_checkpoint("damage") or "checkpoints/flat"
DEFAULT_DISABLED_LEGS = [1]
DEFAULT_VIDEO_SEED = 7
DEFAULT_AMPUTATION_STEP = 120
DEMO_CAMERA_NAME = "demo"


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
    return gym.make("DamageAnt-v0", **env_kwargs)


def _render_demo(env) -> np.ndarray:
    """Use the XML trackcom camera — stable framing without manual lookat jitter."""
    renderer = env.unwrapped.mujoco_renderer
    viewer = renderer._get_viewer(render_mode="rgb_array")
    cam_id = mujoco.mj_name2id(
        env.unwrapped.model, mujoco.mjtObj.mjOBJ_CAMERA, DEMO_CAMERA_NAME
    )
    return viewer.render(render_mode="rgb_array", camera_id=cam_id)


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


def compare(
    flat_run: str,
    damage_run: str,
    disabled_legs: list[int],
    seeds: list[int],
    max_steps: int = 1000,
) -> dict:
    flat_model = PPO.load(os.path.join(flat_run, "best_model", "best_model"))
    damage_model = PPO.load(os.path.join(damage_run, "best_model", "best_model"))

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
    vel_ret = (
        100.0 * damage_s["mean_forward_velocity"] / flat_s["mean_forward_velocity"]
        if flat_s["mean_forward_velocity"] > 1e-6
        else 0.0
    )
    return {
        "disabled_legs": disabled_legs,
        "max_steps": max_steps,
        "seeds": seeds,
        "flat_run": flat_run,
        "damage_run": damage_run,
        "flat": {"episodes": [asdict(e) for e in flat_eps], "summary": flat_s},
        "damage": {"episodes": [asdict(e) for e in damage_eps], "summary": damage_s},
        "velocity_retention_pct": float(vel_ret),
        "reward_retention_pct": float(
            100.0 * damage_s["mean_reward"] / max(flat_s["mean_reward"], 1e-6)
        ),
    }


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


def pick_demo_seed(
    damage_run: str,
    disabled_legs: list[int],
    candidates: list[int] | None = None,
) -> int:
    """Pick seed with longest survival and best forward distance for the damage policy."""
    from stable_baselines3 import PPO

    model = PPO.load(os.path.join(damage_run, "best_model", "best_model"))
    seeds = candidates if candidates is not None else list(range(32))
    best_seed = seeds[0]
    best_score = -1.0

    for seed in seeds:
        env = make_damage_env(disabled_legs, render=False)
        obs, _ = env.reset(seed=seed)
        x0 = env.unwrapped.data.qpos[0]
        steps = 0
        fell = False
        min_up = 1.0
        for _ in range(1000):
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
        score = steps + (500.0 if not fell else 0.0) + 300.0 * forward + 150.0 * max(0.0, min_up)
        if score > best_score:
            best_score = score
            best_seed = seed
    print(f"Demo seed: {best_seed} (score {best_score:.0f})")
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
    flat_model = PPO.load(os.path.join(flat_run, "best_model", "best_model"))
    damage_model = PPO.load(os.path.join(damage_run, "best_model", "best_model"))

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
    flat_model = PPO.load(os.path.join(flat_run, "best_model", "best_model"))
    damage_model = PPO.load(os.path.join(damage_run, "best_model", "best_model"))

    env_flat = make_damage_env(
        [],
        render=True,
        fixed_disabled_legs=[],
        min_disabled_legs=0,
        max_disabled_legs=0,
        terminate_when_tipped=False,
        terminate_when_unhealthy=False,
    )
    env_damage = make_damage_env(
        [],
        render=True,
        fixed_disabled_legs=[],
        min_disabled_legs=0,
        max_disabled_legs=0,
    )

    obs_f, _ = env_flat.reset(seed=seed)
    obs_d, _ = env_damage.reset(seed=seed)

    pre_caption = "4 legs — walking normally"
    post_caption = _damage_caption(amputation_legs)
    frames = []
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

    results = compare(
        args.flat_run,
        args.damage_run,
        args.disabled_legs,
        args.seeds,
        args.max_steps,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
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
    print(f"Velocity retention: {results['velocity_retention_pct']:.0f}%")

    if not args.no_video:
        video_seed = args.video_seed
        if video_seed is None:
            video_seed = pick_demo_seed(args.damage_run, args.disabled_legs)
        record_side_by_side(
            args.flat_run,
            args.damage_run,
            args.disabled_legs,
            video_seed,
            args.max_steps,
            os.path.join(args.out_dir, "comparison_demo.mp4"),
        )
        record_sudden_amputation(
            args.flat_run,
            args.damage_run,
            args.disabled_legs,
            args.amputation_step,
            video_seed,
            args.max_steps,
            os.path.join(args.out_dir, "sudden_amputation_demo.mp4"),
        )
