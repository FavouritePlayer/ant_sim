# Refine speed model: start from the committed terrain checkpoint and train at eval
# difficulty to cut fall rate while keeping forward motion.

config = {
    "env_id": "TerrainAnt-v0",
    "n_envs": 4,
    "total_timesteps": 750_000,
    "policy": "MlpPolicy",
    "learning_rate": 2e-5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.08,
    "ent_coef": 0.002,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.4,
    "forward_reward_weight": 2.2,
    "ctrl_cost_weight": 0.42,
    "pretrained_path": "checkpoints/terrain/best_model/best_model",
}
