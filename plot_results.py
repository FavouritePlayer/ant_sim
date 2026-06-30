import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def find_latest_run(results_dir="results"):
    runs = sorted(glob.glob(os.path.join(results_dir, "ppo_ant_*")))
    if not runs:
        raise FileNotFoundError("No training runs found in results/")
    return runs[-1]


def plot(run_dir: str):
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
    ax.set_title("PPO baseline — Ant-v5 (flat ground)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(run_dir, "reward_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    run_dir = find_latest_run()
    print(f"Plotting run: {run_dir}")
    plot(run_dir)
