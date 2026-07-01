import glob
import json
import os


def find_latest_run(results_dir: str = "results", config: str | None = None) -> str:
    """Return the most recent training run directory.

    config: 'ant', 'terrain', or None (any run).
    """
    if config == "ant":
        pattern = "ppo_ant_*"
    elif config == "terrain":
        pattern = "ppo_terrainant_*"
    else:
        pattern = "ppo_*"

    runs = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not runs:
        label = config or "any"
        raise FileNotFoundError(f"No {label} training runs found in {results_dir}/")
    return runs[-1]


def load_run_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)
    # Infer from directory name for older runs
    name = os.path.basename(run_dir)
    if "terrainant" in name:
        return {"env_id": "TerrainAnt-v0", "difficulty": 0.3}
    return {"env_id": "Ant-v5"}


def run_label(run_dir: str) -> str:
    cfg = load_run_config(run_dir)
    env_id = cfg.get("env_id", "Ant-v5")
    if env_id == "TerrainAnt-v0":
        diff = cfg.get("difficulty", 0.3)
        return f"PPO — TerrainAnt-v0 (difficulty {diff})"
    return "PPO — Ant-v5 (flat ground)"
