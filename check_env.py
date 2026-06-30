import gymnasium as gym

env = gym.make("Ant-v5")
obs, info = env.reset()

print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print(f"Obs shape:         {obs.shape}")

total_reward = 0
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()

print(f"200 random steps completed. Cumulative reward: {total_reward:.2f}")
env.close()
print("Environment OK.")
