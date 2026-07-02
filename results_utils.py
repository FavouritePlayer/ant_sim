import glob
import json
import os

CHECKPOINT_DIRS = {
    "ant": "checkpoints/flat",
    "flat": "checkpoints/flat",
    "terrain": "checkpoints/terrain",
    "damage": "checkpoints/damage",
}


def _has_checkpoint(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "best_model", "best_model.zip"))


def find_latest_run(results_dir: str = "results", config: str | None = None) -> str:
    """Return the most recent training run directory.

    config: 'ant', 'terrain', or None (any run).
    """
    if config == "ant":
        pattern = "ppo_ant_*"
    elif config == "terrain":
        pattern = "ppo_terrainant_*"
    elif config == "damage":
        pattern = "ppo_damageant_*"
    else:
        pattern = "ppo_*"

    runs = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not runs:
        label = config or "any"
        raise FileNotFoundError(f"No {label} training runs found in {results_dir}/")
    return runs[-1]


def default_checkpoint(config: str | None) -> str | None:
    if not config:
        return None
    path = CHECKPOINT_DIRS.get(config)
    if path and _has_checkpoint(path):
        return path
    return None


def resolve_run_dir(config: str | None = None, run_dir: str | None = None) -> str:
    """Prefer explicit run_dir, then committed checkpoints, then latest results/."""
    if run_dir:
        if not _has_checkpoint(run_dir):
            raise FileNotFoundError(
                f"No checkpoint at {run_dir}/best_model/best_model.zip"
            )
        return run_dir

    checkpoint = default_checkpoint(config)
    if checkpoint:
        return checkpoint

    if config:
        return find_latest_run(config=config)

    raise ValueError("Provide --run-dir or --config to locate a checkpoint")


def load_run_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)
    # Infer from directory name for older runs
    name = os.path.basename(run_dir.rstrip("/"))
    if name in ("flat", "terrain", "damage") or "terrainant" in name or "damageant" in name:
        if "damage" in name:
            return {"env_id": "DamageAnt-v0", "fixed_disabled_legs": [1]}
        return {"env_id": "TerrainAnt-v0", "difficulty": 0.35}
    return {"env_id": "Ant-v5"}


def run_label(run_dir: str) -> str:
    cfg = load_run_config(run_dir)
    env_id = cfg.get("env_id", "Ant-v5")
    if env_id == "TerrainAnt-v0":
        diff = cfg.get("difficulty", 0.3)
        return f"PPO — TerrainAnt-v0 (difficulty {diff})"
    if env_id == "DamageAnt-v0":
        legs = cfg.get("eval_fixed_disabled_legs", [1])
        return f"PPO — DamageAnt-v0 (eval leg {legs} disabled)"
    return "PPO — Ant-v5 (flat ground)"
