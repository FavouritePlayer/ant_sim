config = {
    "env_id": "TerrainAnt-v0",
    "n_envs": 4,
    "total_timesteps": 6_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 2e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.45,
    # No curriculum — fixed difficulty throughout so the policy has a stable target.
    # Curriculum caused catastrophic forgetting: policy peaked early then degraded as
    # terrain difficulty outpaced what the policy could handle.
}
