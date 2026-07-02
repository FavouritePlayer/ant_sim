import json
import os
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnNoModelImprovement


def train(cfg: dict):
    from envs import register
    register()

    run_name = f"ppo_{cfg['env_id'].lower().replace('-', '_')}_{int(time.time())}"
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
    ):
        if key in cfg:
            env_kwargs[key] = cfg[key]

    if "difficulty_range" in cfg:
        env_kwargs["difficulty_range"] = tuple(cfg["difficulty_range"])

    env = make_vec_env(cfg["env_id"], n_envs=cfg["n_envs"], seed=cfg["seed"], env_kwargs=env_kwargs or None)

    eval_kwargs = dict(env_kwargs)
    if "eval_difficulty" in cfg:
        eval_kwargs["difficulty"] = cfg["eval_difficulty"]
    eval_kwargs.pop("difficulty_range", None)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=[
            "ant",
            "ant_finetune",
            "terrain",
            "terrain_finetune",
            "terrain_boost",
            "terrain_speed",
            "terrain_speed_refine",
            "terrain_balanced",
            "terrain_diverse",
            "terrain_refined",
            "terrain_polish",
        ],
        default="ant",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to a .zip model to fine-tune (overrides config pretrained_path)",
    )
    args = parser.parse_args()

    if args.config == "ant_finetune":
        from configs.ppo_ant_finetune import config
    elif args.config == "terrain":
        from configs.ppo_terrain import config
    elif args.config == "terrain_finetune":
        from configs.ppo_terrain_finetune import config
    elif args.config == "terrain_boost":
        from configs.ppo_terrain_boost import config
    elif args.config == "terrain_speed":
        from configs.ppo_terrain_speed import config
    elif args.config == "terrain_speed_refine":
        from configs.ppo_terrain_speed_refine import config
    elif args.config == "terrain_balanced":
        from configs.ppo_terrain_balanced import config
    elif args.config == "terrain_diverse":
        from configs.ppo_terrain_diverse import config
    elif args.config == "terrain_refined":
        from configs.ppo_terrain_refined import config
    elif args.config == "terrain_polish":
        from configs.ppo_terrain_polish import config
    else:
        from configs.ppo_ant import config

    cfg = config.copy()
    if args.timesteps:
        cfg["total_timesteps"] = args.timesteps
    if args.pretrained:
        cfg["pretrained_path"] = args.pretrained

    train(cfg)
