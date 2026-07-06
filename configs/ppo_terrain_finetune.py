# Second-stage fine-tune: flat-ground expert -> current heightfield at moderate difficulty.
# Use the committed flat checkpoint as the reproducible parent instead of a historical run dir.

config = {
    "env_id": "TerrainAnt-v0",
    "n_envs": 4,
    "total_timesteps": 6_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.35,
    "pretrained_path": "checkpoints/flat/best_model/best_model",
}
