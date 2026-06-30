import os
import glob
import argparse
import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import PPO


def find_latest_run(results_dir="results"):
    runs = sorted(glob.glob(os.path.join(results_dir, "ppo_ant_*")))
    if not runs:
        raise FileNotFoundError("No training runs found in results/")
    return runs[-1]


def record(run_dir: str, n_episodes: int = 3, fps: int = 30):
    model_path = os.path.join(run_dir, "best_model", "best_model")
    model = PPO.load(model_path)

    env = gym.make("Ant-v5", render_mode="rgb_array")
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            frames.append(env.render())
            done = terminated or truncated
        print(f"Episode {ep + 1}: reward = {ep_reward:.1f}")

    env.close()

    out_path = os.path.join(run_dir, "demo.mp4")
    imageio.mimwrite(out_path, frames, fps=fps)
    print(f"Saved: {out_path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    run_dir = find_latest_run()
    print(f"Loading from: {run_dir}")
    record(run_dir, n_episodes=args.episodes, fps=args.fps)
