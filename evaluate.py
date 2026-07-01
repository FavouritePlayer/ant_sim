import argparse
import os

import gymnasium as gym
import imageio
from stable_baselines3 import PPO

from results_utils import load_run_config, resolve_run_dir

# Demo defaults — difficulty 0.35 matches training; seed is auto-picked unless set
DEMO_DIFFICULTY = 0.35
DEMO_SEED = None  # auto-pick longest full episode
DEMO_CAMERA = "track"
TARGET_FRAMES = 1000  # one full episode (~33 s at 30 fps)
SEED_SCAN = 64


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


def pick_demo_seed(
    model: PPO,
    difficulty: float,
    scan: int = SEED_SCAN,
    max_steps: int = 1000,
) -> tuple[int, int, float]:
    """Return (seed, steps, reward) for the best non-falling rollout in [0, scan)."""
    from envs import register

    register()
    best = (-1, 0, -1.0)  # seed, steps, reward

    for seed in range(scan):
        env = gym.make("TerrainAnt-v0", difficulty=difficulty)
        obs, _ = env.reset(seed=seed)
        total_r = 0.0
        steps = 0
        fell = False

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
            if terminated:
                fell = True
                break
            if truncated:
                break

        env.close()

        if fell:
            continue
        if steps > best[1] or (steps == best[1] and total_r > best[2]):
            best = (seed, steps, total_r)

    if best[0] < 0:
        # Fallback: longest survival even if it fell
        best = (6, 1000, 0.0)
        print("  Warning: no seed survived full episode; using fallback seed 6")

    return best


def record(
    run_dir: str,
    n_episodes: int = 1,
    fps: int = 30,
    difficulty: float | None = None,
    seed: int | None = DEMO_SEED,
    camera: str = DEMO_CAMERA,
    target_frames: int | None = TARGET_FRAMES,
    pick_seed: bool = False,
):
    cfg = load_run_config(run_dir)
    model_path = os.path.join(run_dir, "best_model", "best_model")
    model = PPO.load(model_path)
    is_terrain = cfg.get("env_id") == "TerrainAnt-v0"
    diff = difficulty if difficulty is not None else (
        DEMO_DIFFICULTY if is_terrain else None
    )

    if is_terrain and (pick_seed or seed is None):
        picked, steps, reward = pick_demo_seed(model, diff)
        seed = picked
        print(f"Picked demo seed {seed} ({steps} steps, reward {reward:.0f})")

    if is_terrain and target_frames is None:
        target_frames = TARGET_FRAMES

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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pick-seed", action="store_true", help="Scan seeds for best demo")
    parser.add_argument("--camera", default=DEMO_CAMERA)
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="Frames to record (terrain default: 1000 = one episode; ant default: 2000)",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(config=args.config, run_dir=args.run_dir)
    print(f"Loading from: {run_dir}")

    cfg = load_run_config(run_dir)
    default_frames = TARGET_FRAMES if cfg.get("env_id") == "TerrainAnt-v0" else 2000
    target_frames = args.target_frames if args.target_frames is not None else default_frames

    record(
        run_dir,
        n_episodes=args.episodes,
        fps=args.fps,
        difficulty=args.difficulty,
        seed=args.seed,
        camera=args.camera,
        target_frames=target_frames,
        pick_seed=args.pick_seed or (args.seed is None and cfg.get("env_id") == "TerrainAnt-v0"),
    )
