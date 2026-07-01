"""Compare flat-trained vs terrain-adapted policies on TerrainAnt-v0.

Runs matched seeds/difficulty, writes metrics JSON, comparison plots,
and a side-by-side demo video (flat policy freezes on fall).
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from results_utils import default_checkpoint

DEFAULT_FLAT_RUN = default_checkpoint("flat") or "checkpoints/flat"
DEFAULT_TERRAIN_RUN = default_checkpoint("terrain") or "checkpoints/terrain"


@dataclass
class EpisodeMetrics:
    seed: int
    reward: float
    steps: int
    forward_distance: float
    mean_forward_velocity: float
    fell: bool


def make_terrain_env(difficulty: float, render: bool = False):
    from envs import register

    register()
    kwargs = {"difficulty": difficulty}
    if render:
        kwargs.update(
            render_mode="rgb_array",
            camera_name="track",
            width=640,
            height=480,
        )
    return gym.make("TerrainAnt-v0", **kwargs)


def rollout(model: PPO, env, seed: int, max_steps: int = 1000) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    x0 = env.unwrapped.data.qpos[0]
    reward_sum = 0.0
    fell = False
    steps = 0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
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
    terrain_run: str,
    difficulty: float,
    seeds: list[int],
    max_steps: int = 1000,
) -> dict:
    flat_model = PPO.load(os.path.join(flat_run, "best_model", "best_model"))
    terrain_model = PPO.load(os.path.join(terrain_run, "best_model", "best_model"))

    flat_eps: list[EpisodeMetrics] = []
    terrain_eps: list[EpisodeMetrics] = []

    for seed in seeds:
        env = make_terrain_env(difficulty, render=False)
        flat_eps.append(rollout(flat_model, env, seed, max_steps))
        env.close()

        env = make_terrain_env(difficulty, render=False)
        terrain_eps.append(rollout(terrain_model, env, seed, max_steps))
        env.close()

        print(
            f"seed {seed:3d} | flat: R={flat_eps[-1].reward:7.0f} "
            f"steps={flat_eps[-1].steps:4d} vel={flat_eps[-1].mean_forward_velocity:.2f} fell={flat_eps[-1].fell} | "
            f"terrain: R={terrain_eps[-1].reward:7.0f} "
            f"steps={terrain_eps[-1].steps:4d} vel={terrain_eps[-1].mean_forward_velocity:.2f} fell={terrain_eps[-1].fell}"
        )

    flat_summary = summarize(flat_eps)
    terrain_summary = summarize(terrain_eps)

    reward_retention = (
        terrain_summary["mean_reward"] / flat_summary["mean_reward"] * 100
        if flat_summary["mean_reward"] > 0
        else float("nan")
    )
    velocity_retention = (
        terrain_summary["mean_forward_velocity"] / flat_summary["mean_forward_velocity"] * 100
        if flat_summary["mean_forward_velocity"] > 0
        else float("nan")
    )

    return {
        "difficulty": difficulty,
        "max_steps": max_steps,
        "seeds": seeds,
        "flat_run": flat_run,
        "terrain_run": terrain_run,
        "flat": {"episodes": [asdict(e) for e in flat_eps], "summary": flat_summary},
        "terrain": {"episodes": [asdict(e) for e in terrain_eps], "summary": terrain_summary},
        "terrain_reward_retention_pct": reward_retention,
        "terrain_velocity_retention_pct": velocity_retention,
    }


def plot_comparison_summary(results: dict, out_path: str):
    flat = results["flat"]["summary"]
    terrain = results["terrain"]["summary"]
    labels = ["Episode reward", "Episode length", "Forward distance (m)", "Mean fwd velocity (m/s)"]
    flat_vals = [
        flat["mean_reward"],
        flat["mean_steps"],
        flat["mean_forward_distance"],
        flat["mean_forward_velocity"],
    ]
    terrain_vals = [
        terrain["mean_reward"],
        terrain["mean_steps"],
        terrain["mean_forward_distance"],
        terrain["mean_forward_velocity"],
    ]
    flat_err = [
        flat["std_reward"],
        flat["std_steps"],
        flat["std_forward_distance"],
        flat["std_forward_velocity"],
    ]
    terrain_err = [
        terrain["std_reward"],
        terrain["std_steps"],
        terrain["std_forward_distance"],
        terrain["std_forward_velocity"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, flat_vals, width, yerr=flat_err, capsize=4,
           label="Flat-trained (Ant-v5)", color="#94a3b8")
    ax.bar(x + width / 2, terrain_vals, width, yerr=terrain_err, capsize=4,
           label="Terrain-adapted", color="#2563eb")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_title(
        f"Control vs treatment on TerrainAnt-v0  (difficulty={results['difficulty']}, "
        f"n={results['flat']['summary']['n_episodes']} seeds)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_reward_by_seed(results: dict, out_path: str):
    seeds = results["seeds"]
    flat_rewards = [e["reward"] for e in results["flat"]["episodes"]]
    terrain_rewards = [e["reward"] for e in results["terrain"]["episodes"]]

    x = np.arange(len(seeds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, flat_rewards, width, label="Flat-trained", color="#94a3b8")
    ax.bar(x + width / 2, terrain_rewards, width, label="Terrain-adapted", color="#2563eb")
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("Episode seed (matched terrain layout)")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Return on unseen terrain  (difficulty={results['difficulty']})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def _label_frame(frame: np.ndarray, title: str) -> np.ndarray:
    img = Image.fromarray(frame)
    banner_h = 36
    canvas = Image.new("RGB", (img.width, img.height + banner_h), (30, 30, 30))
    canvas.paste(img, (0, banner_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(255, 255, 255))
    return np.array(canvas)


def record_side_by_side(
    flat_run: str,
    terrain_run: str,
    difficulty: float,
    seed: int,
    max_steps: int,
    out_path: str,
    fps: int = 30,
):
    flat_model = PPO.load(os.path.join(flat_run, "best_model", "best_model"))
    terrain_model = PPO.load(os.path.join(terrain_run, "best_model", "best_model"))

    env_flat = make_terrain_env(difficulty, render=True)
    env_terrain = make_terrain_env(difficulty, render=True)

    obs_f, _ = env_flat.reset(seed=seed)
    obs_t, _ = env_terrain.reset(seed=seed)

    flat_done = False
    last_flat_frame = None
    frames = []

    for _ in range(max_steps):
        if not flat_done:
            action_f, _ = flat_model.predict(obs_f, deterministic=True)
            obs_f, _, term_f, trunc_f, _ = env_flat.step(action_f)
            last_flat_frame = env_flat.render()
            if term_f or trunc_f:
                flat_done = True

        action_t, _ = terrain_model.predict(obs_t, deterministic=True)
        obs_t, _, term_t, trunc_t, _ = env_terrain.step(action_t)
        frame_t = env_terrain.render()

        frame_f = last_flat_frame if last_flat_frame is not None else frame_t
        left = _label_frame(frame_f, "Flat-trained (control)")
        right = _label_frame(frame_t, "Terrain-adapted")
        frames.append(np.hstack([left, right]))

        if flat_done and (term_t or trunc_t):
            break

    env_flat.close()
    env_terrain.close()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Saved video: {out_path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare flat vs terrain policies on terrain")
    parser.add_argument("--flat-run", default=DEFAULT_FLAT_RUN)
    parser.add_argument("--terrain-run", default=DEFAULT_TERRAIN_RUN)
    parser.add_argument("--difficulty", type=float, default=0.4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="docs/assets/terrain")
    parser.add_argument("--video-seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    results = compare(
        args.flat_run,
        args.terrain_run,
        args.difficulty,
        args.seeds,
        args.max_steps,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {json_path}")

    plot_comparison_summary(results, os.path.join(args.out_dir, "comparison_plot.png"))
    plot_reward_by_seed(results, os.path.join(args.out_dir, "comparison_reward_by_seed.png"))

    flat_s = results["flat"]["summary"]
    terrain_s = results["terrain"]["summary"]
    print("\n=== Summary ===")
    print(f"Flat-trained     | reward {flat_s['mean_reward']:.0f} ± {flat_s['std_reward']:.0f} | "
          f"steps {flat_s['mean_steps']:.0f} | fall rate {flat_s['fall_rate']*100:.0f}% | "
          f"vel {flat_s['mean_forward_velocity']:.2f} m/s")
    print(f"Terrain-adapted  | reward {terrain_s['mean_reward']:.0f} ± {terrain_s['std_reward']:.0f} | "
          f"steps {terrain_s['mean_steps']:.0f} | fall rate {terrain_s['fall_rate']*100:.0f}% | "
          f"vel {terrain_s['mean_forward_velocity']:.2f} m/s")
    print(f"Reward retention:   {results['terrain_reward_retention_pct']:.0f}%")
    print(f"Velocity retention: {results['terrain_velocity_retention_pct']:.0f}%")

    if not args.no_video:
        record_side_by_side(
            args.flat_run,
            args.terrain_run,
            args.difficulty,
            args.video_seed,
            args.max_steps,
            os.path.join(args.out_dir, "comparison_demo.mp4"),
        )
