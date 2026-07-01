# Speed boost: fine-tune the terrain checkpoint with stronger forward reward
# while keeping control costs moderate so the agent walks farther on hills.

config = {
    "env_id": "TerrainAnt-v0",
    "n_envs": 4,
    "total_timesteps": 2_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 4e-5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.004,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.38,
    "forward_reward_weight": 2.5,
    "ctrl_cost_weight": 0.4,
    "pretrained_path": "checkpoints/terrain/best_model/best_model",
}
