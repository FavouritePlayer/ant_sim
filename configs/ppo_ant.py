config = {
    "env_id": "Ant-v5",
    "n_envs": 4,              # parallel envs for data collection
    "total_timesteps": 2_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,          # steps per env per rollout
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
}
