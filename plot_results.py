import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from results_utils import find_latest_run, run_label


def plot(run_dir: str, show: bool = True):
    npz_path = os.path.join(run_dir, "eval", "evaluations.npz")
    data = np.load(npz_path)

    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards = data["results"].std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, mean_rewards, color="#2563eb", linewidth=2, label="Mean eval reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        color="#2563eb",
        label="±1 std",
    )
    ax.axhline(mean_rewards.max(), color="#16a34a", linewidth=1, linestyle="--",
               label=f"Best: {mean_rewards.max():.0f}")

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title(run_label(run_dir), fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(run_dir, "reward_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "best_reward": float(mean_rewards.max()),
        "final_reward": float(mean_rewards[-1]),
        "timesteps": int(timesteps[-1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["ant", "terrain", "damage"], default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(config=args.config)
    print(f"Plotting run: {run_dir}")
    metrics = plot(run_dir, show=not args.no_show)
    print(f"Best eval reward: {metrics['best_reward']:.0f}")
