# Boost the stable terrain agent: fine-tune from the committed terrain checkpoint on the
# current heightfield with a stronger forward reward to walk farther without falling.

config = {
    "env_id": "TerrainAnt-v0",
    "n_envs": 4,
    "total_timesteps": 3_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 5e-5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.12,
    "ent_coef": 0.002,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.35,
    "forward_reward_weight": 1.5,
    "pretrained_path": "checkpoints/terrain/best_model/best_model",
}
