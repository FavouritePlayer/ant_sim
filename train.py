import json
import os
import time
import argparse
import importlib

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnNoModelImprovement

CONFIG_MODULES = {
    "ant": "configs.ppo_ant",
    "ant_finetune": "configs.ppo_ant_finetune",
    "terrain": "configs.ppo_terrain",
    "terrain_finetune": "configs.ppo_terrain_finetune",
    "terrain_boost": "configs.ppo_terrain_boost",
    "terrain_speed": "configs.ppo_terrain_speed",
    "terrain_speed_refine": "configs.ppo_terrain_speed_refine",
    "terrain_balanced": "configs.ppo_terrain_balanced",
    "terrain_diverse": "configs.ppo_terrain_diverse",
    "terrain_refined": "configs.ppo_terrain_refined",
    "terrain_polish": "configs.ppo_terrain_polish",
    "terrain_velocity": "configs.ppo_terrain_velocity",
    "terrain_velocity_v2": "configs.ppo_terrain_velocity_v2",
    "damage": "configs.ppo_damage",
    "damage_boost": "configs.ppo_damage_boost",
    "damage_retrain": "configs.ppo_damage_retrain",
    "damage_upright": "configs.ppo_damage_upright",
    "damage_polish": "configs.ppo_damage_polish",
    "damage_final": "configs.ppo_damage_final",
    "damage_speed": "configs.ppo_damage_speed",
    "damage_gait": "configs.ppo_damage_gait",
    "damage_holistic": "configs.ppo_damage_holistic",
    "damage_holistic_v2": "configs.ppo_damage_holistic_v2",
    "damage_holistic_v3": "configs.ppo_damage_holistic_v3",
    "damage_upright_polish": "configs.ppo_damage_upright_polish",
    "damage_crossleg_upright": "configs.ppo_damage_crossleg_upright",
    "damage_crossleg_speed": "configs.ppo_damage_crossleg_speed",
    "damage_crossleg_gait": "configs.ppo_damage_crossleg_gait",
}


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def load_config(config_name: str) -> dict:
    module = importlib.import_module(CONFIG_MODULES[config_name])
    return module.config.copy()


def train(cfg: dict, *, config_name: str | None = None, run_tag: str | None = None) -> str:
    from envs import register
    register()

    cfg = cfg.copy()
    if config_name:
        cfg["config_name"] = config_name
    timestamp = int(time.time())
    run_name = f"ppo_{cfg['env_id'].lower().replace('-', '_')}_{timestamp}"
    if "seed" in cfg:
        run_name += f"_seed{cfg['seed']}"
    if run_tag:
        cfg["run_tag"] = run_tag
        run_name += f"_{_slug(run_tag)}"
    log_dir = os.path.join("results", run_name)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    env_kwargs = {}
    if "difficulty" in cfg:
        env_kwargs["difficulty"] = cfg["difficulty"]
    for key in (
        "forward_reward_weight",
        "ctrl_cost_weight",
        "contact_cost_weight",
        "progress_reward_weight",
        "velocity_tracking_weight",
        "target_speed_range",
        "upright_reward_weight",
        "height_reward_weight",
        "tilt_penalty_weight",
        "progress_reward_weight",
        "forward_gate_uprightness",
        "forward_gate_height",
        "velocity_tracking_weight",
        "target_speed",
        "backward_penalty_weight",
        "leg_balance_weight",
        "foot_gait_weight",
        "shuffle_penalty_weight",
        "lateral_penalty_weight",
        "min_leg_activity_weight",
        "reset_noise_scale",
        "min_uprightness",
    ):
        if key in cfg:
            env_kwargs[key] = tuple(cfg[key]) if key == "target_speed_range" else cfg[key]
    if "terminate_when_tipped" in cfg:
        env_kwargs["terminate_when_tipped"] = cfg["terminate_when_tipped"]
    if "tip_grace_steps" in cfg:
        env_kwargs["tip_grace_steps"] = cfg["tip_grace_steps"]
    if "min_uprightness" in cfg:
        env_kwargs["min_uprightness"] = cfg["min_uprightness"]

    if "difficulty_range" in cfg:
        env_kwargs["difficulty_range"] = tuple(cfg["difficulty_range"])
    if "max_disabled_legs" in cfg:
        env_kwargs["max_disabled_legs"] = cfg["max_disabled_legs"]
    if "min_disabled_legs" in cfg:
        env_kwargs["min_disabled_legs"] = cfg["min_disabled_legs"]
    if "fixed_disabled_legs" in cfg:
        env_kwargs["fixed_disabled_legs"] = list(cfg["fixed_disabled_legs"])

    env = make_vec_env(cfg["env_id"], n_envs=cfg["n_envs"], seed=cfg["seed"], env_kwargs=env_kwargs or None)

    eval_kwargs = dict(env_kwargs)
    if "eval_difficulty" in cfg:
        eval_kwargs["difficulty"] = cfg["eval_difficulty"]
    if "eval_target_speed_range" in cfg:
        eval_kwargs["target_speed_range"] = tuple(cfg["eval_target_speed_range"])
    if "eval_fixed_disabled_legs" in cfg:
        eval_kwargs["fixed_disabled_legs"] = list(cfg["eval_fixed_disabled_legs"])
    eval_kwargs.pop("difficulty_range", None)
    eval_kwargs.pop("max_disabled_legs", None)
    eval_kwargs.pop("min_disabled_legs", None)
    eval_env = make_vec_env(
        cfg["env_id"], n_envs=1, seed=cfg["seed"] + 100, env_kwargs=eval_kwargs or None
    )

    pretrained = cfg.get("pretrained_path")
    if pretrained:
        print(f"Fine-tuning from: {pretrained}")
        model = PPO.load(
            pretrained,
            env=env,
            learning_rate=cfg["learning_rate"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            tensorboard_log=os.path.join("results", "tb_logs"),
        )
        model.set_random_seed(cfg["seed"])
    else:
        model = PPO(
            policy=cfg["policy"],
            env=env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            tensorboard_log=os.path.join("results", "tb_logs"),
            seed=cfg["seed"],
            verbose=1,
        )

    callbacks = []

    stop_callback = None
    patience = cfg.get("early_stop_patience")
    if patience:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=patience,
            min_evals=5,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(50_000 // cfg["n_envs"], 1),
        n_eval_episodes=cfg.get("n_eval_episodes", 5),
        deterministic=True,
        callback_after_eval=stop_callback,
    )
    callbacks.append(eval_callback)

    if cfg.get("curriculum"):
        from envs.terrain_ant import CurriculumCallback
        callbacks.append(CurriculumCallback(
            start_difficulty=cfg["curriculum"]["start"],
            max_difficulty=cfg["curriculum"]["max"],
            update_interval=cfg["curriculum"]["interval"],
        ))

    print(f"Training run: {run_name}")
    print(f"Total timesteps: {cfg['total_timesteps']:,}  |  Envs: {cfg['n_envs']}")

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        tb_log_name=run_name,
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "final_model"))
    print(f"Saved to {log_dir}")

    env.close()
    eval_env.close()
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=sorted(CONFIG_MODULES.keys()),
        default="ant",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to a .zip model to fine-tune (overrides config pretrained_path)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional suffix to label this run (useful for replication groups)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.pretrained:
        cfg["pretrained_path"] = args.pretrained
    if args.seed is not None:
        cfg["seed"] = args.seed

    train(cfg, config_name=args.config, run_tag=args.run_tag)
