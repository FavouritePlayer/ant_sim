import argparse

import gymnasium as gym


def check_env(env_id: str, steps: int = 200, **make_kwargs):
    if env_id.startswith(("TerrainAnt", "DamageAnt")):
        from envs import register
        register()

    env = gym.make(env_id, **make_kwargs) if make_kwargs else gym.make(env_id)

    obs, info = env.reset()
    print(f"\n=== {env_id} ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Obs shape:         {obs.shape}")

    total_reward = 0
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"{steps} random steps completed. Cumulative reward: {total_reward:.2f}")
    print("Environment OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["ant", "terrain", "both"], default="both")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.env in ("ant", "both"):
        check_env("Ant-v5", steps=args.steps)
    if args.env in ("terrain", "both"):
        check_env("TerrainAnt-v0", steps=args.steps, difficulty=0.3)
        check_env(
            "DamageAnt-v0",
            steps=args.steps,
            fixed_disabled_legs=[1],
        )
