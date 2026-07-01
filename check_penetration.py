"""Check for ant-ground penetration during a rollout."""

import argparse
import os

from stable_baselines3 import PPO

from evaluate import make_eval_env
from results_utils import find_latest_run, load_run_config


def check_penetration(run_dir: str, difficulty: float = 0.65, seed: int = 7, steps: int = 1000):
    cfg = load_run_config(run_dir)
    model = PPO.load(os.path.join(run_dir, "best_model", "best_model"))
    env = make_eval_env(cfg, difficulty=difficulty)
    unwrapped = env.unwrapped

    obs, _ = env.reset(seed=seed)
    worst_gap = float("inf")
    worst_step = 0

    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)

        x, y, z = unwrapped.data.qpos[0], unwrapped.data.qpos[1], unwrapped.data.qpos[2]
        ground = unwrapped.sample_terrain_z(float(x), float(y))
        gap = z - ground
        if gap < worst_gap:
            worst_gap = gap
            worst_step = t

        if term or trunc:
            break

    env.close()
    print(f"Worst torso-ground gap: {worst_gap:.3f} m at step {worst_step}")
    print("PASS" if worst_gap > -0.05 else "FAIL — likely penetration")
    return worst_gap


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["terrain"], default="terrain")
    parser.add_argument("--difficulty", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    run_dir = find_latest_run(config=args.config)
    check_penetration(run_dir, args.difficulty, args.seed, args.steps)
