# Extra flat-ground training from the baseline checkpoint.

config = {
    "env_id": "Ant-v5",
    "n_envs": 4,
    "total_timesteps": 1_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "pretrained_path": "checkpoints/flat/best_model/best_model",
}
