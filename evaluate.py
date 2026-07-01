import argparse
import os

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

from results_utils import find_latest_run, load_run_config

# Demo defaults — more dramatic than training difficulty (0.3) for portfolio video
DEMO_DIFFICULTY = 0.4
DEMO_SEED = 42
DEMO_CAMERA = "track"
TARGET_FRAMES = 2000  # ~67 s at 30 fps (matches ant demo length)


def make_eval_env(
    cfg: dict,
    difficulty: float | None = None,
    camera: str = DEMO_CAMERA,
    width: int = 640,
    height: int = 480,
):
    env_id = cfg["env_id"]
    if env_id == "TerrainAnt-v0":
        from envs import register
        register()
        diff = difficulty if difficulty is not None else cfg.get("difficulty", DEMO_DIFFICULTY)
        return gym.make(
            env_id,
            difficulty=diff,
            render_mode="rgb_array",
            camera_name=camera,
            width=width,
            height=height,
            terminate_when_unhealthy=True,
        )
    return gym.make(
        env_id,
        render_mode="rgb_array",
        camera_name=camera,
        width=width,
        height=height,
    )


def record(
    run_dir: str,
    n_episodes: int = 1,
    fps: int = 30,
    difficulty: float | None = None,
    seed: int | None = DEMO_SEED,
    camera: str = DEMO_CAMERA,
    target_frames: int | None = TARGET_FRAMES,
):
    cfg = load_run_config(run_dir)
    model_path = os.path.join(run_dir, "best_model", "best_model")
    model = PPO.load(model_path)
    diff = difficulty if difficulty is not None else (
        DEMO_DIFFICULTY if cfg.get("env_id") == "TerrainAnt-v0" else None
    )
    env = make_eval_env(cfg, difficulty=diff, camera=camera)
    frames = []

    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        ep_reward = 0
        ep_steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            frames.append(env.render())
            ep_steps += 1

            if target_frames is not None and len(frames) >= target_frames:
                break

            if terminated or truncated:
                break

        print(f"Episode {ep + 1}: reward = {ep_reward:.1f}  ({ep_steps} steps)")

        if target_frames is not None and len(frames) >= target_frames:
            break

    # If we still need frames (episode ended early), keep resetting and recording
    continuation = 0
    while target_frames is not None and len(frames) < target_frames:
        continuation += 1
        cont_seed = (seed + 100 + continuation) if seed is not None else None
        obs, _ = env.reset(seed=cont_seed)
        ep_reward = 0
        ep_steps = 0
        while len(frames) < target_frames:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            frames.append(env.render())
            ep_steps += 1
            if terminated or truncated:
                break
        print(f"Continuation {continuation}: reward = {ep_reward:.1f}  ({ep_steps} steps)")

    env.close()

    out_path = os.path.join(run_dir, "demo.mp4")
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Saved: {out_path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["ant", "terrain"], default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--difficulty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    parser.add_argument("--camera", default=DEMO_CAMERA)
    parser.add_argument("--target-frames", type=int, default=TARGET_FRAMES,
                        help="Total frames to record (auto-continues across episode resets)")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(config=args.config)
    print(f"Loading from: {run_dir}")
    record(
        run_dir,
        n_episodes=args.episodes,
        fps=args.fps,
        difficulty=args.difficulty,
        seed=args.seed,
        camera=args.camera,
        target_frames=args.target_frames,
    )
