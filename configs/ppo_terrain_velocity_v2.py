# Follow-up velocity-command terrain recipe: less tracking penalty, more forward reward.
# `TerrainAnt-v1` adds a commanded-speed observation, so there is no committed v1 parent
# checkpoint in the repo; pass `--pretrained` explicitly when continuing a prior v1 run.

config = {
    "env_id": "TerrainAnt-v1",
    "n_envs": 6,
    "total_timesteps": 1_500_000,
    "policy": "MlpPolicy",
    "learning_rate": 1.5e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 8,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.12,
    "ent_coef": 0.003,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "difficulty": 0.4,
    "eval_difficulty": 0.4,
    "n_eval_episodes": 10,
    "forward_reward_weight": 2.2,
    "ctrl_cost_weight": 0.45,
    "velocity_tracking_weight": 0.6,
    "target_speed_range": [0.2, 0.45],
    "eval_target_speed_range": [0.35, 0.35],
    "early_stop_patience": 10,
}
