"""Launch replicated multi-seed training for canonical experiment chains."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

from train import load_config, train


@dataclass(frozen=True)
class StageSpec:
    name: str
    config_name: str
    pretrained_path: str | None = None


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    stages: tuple[StageSpec, ...]


REPLICATION_PROFILES: dict[str, ProfileSpec] = {
    "terrain_canonical": ProfileSpec(
        name="terrain_canonical",
        description=(
            "Flat baseline -> terrain fine-tune -> terrain boost -> terrain balanced "
            "using committed checkpoints plus seed-specific intermediate outputs."
        ),
        stages=(
            StageSpec(
                name="terrain_finetune",
                config_name="terrain_finetune",
                pretrained_path="checkpoints/flat/best_model/best_model",
            ),
            StageSpec(name="terrain_boost", config_name="terrain_boost"),
            StageSpec(name="terrain_balanced", config_name="terrain_balanced"),
        ),
    ),
    "damage_canonical": ProfileSpec(
        name="damage_canonical",
        description=(
            "Flat baseline -> damage upright -> damage speed -> damage gait using "
            "committed checkpoints plus seed-specific intermediate outputs."
        ),
        stages=(
            StageSpec(
                name="damage_upright",
                config_name="damage_upright",
                pretrained_path="checkpoints/flat/best_model/best_model",
            ),
            StageSpec(name="damage_speed", config_name="damage_speed"),
            StageSpec(name="damage_gait", config_name="damage_gait"),
        ),
    ),
}


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _best_or_final_model_prefix(run_dir: str) -> str:
    best_prefix = os.path.join(run_dir, "best_model", "best_model")
    if os.path.isfile(best_prefix + ".zip"):
        return best_prefix
    final_prefix = os.path.join(run_dir, "final_model")
    if os.path.isfile(final_prefix + ".zip"):
        return final_prefix
    raise FileNotFoundError(
        f"No chained checkpoint found in {run_dir} "
        "(expected best_model/best_model.zip or final_model.zip)"
    )


def _manifest_dir() -> str:
    path = os.path.join("results", "replications")
    os.makedirs(path, exist_ok=True)
    return path


def _manifest_path(tag: str | None) -> str:
    timestamp = int(time.time())
    suffix = f"_{_slug(tag)}" if tag else ""
    return os.path.join(_manifest_dir(), f"replication_{timestamp}{suffix}.json")


def save_manifest(path: str, manifest: dict):
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def build_manifest(
    profiles: list[str],
    seeds: list[int],
    *,
    timesteps: int | None = None,
    tag: str | None = None,
) -> dict:
    requested_profiles = []
    for profile_name in profiles:
        spec = REPLICATION_PROFILES[profile_name]
        requested_profiles.append(
            {
                "name": spec.name,
                "description": spec.description,
                "stages": [asdict(stage) for stage in spec.stages],
            }
        )

    replications = []
    for profile_name in profiles:
        for seed in seeds:
            replications.append(
                {
                    "profile": profile_name,
                    "seed": seed,
                    "status": "pending",
                    "stages": [],
                }
            )

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": tag,
        "timesteps_override": timesteps,
        "requested_profiles": requested_profiles,
        "replications": replications,
    }


def run_replications(
    profiles: list[str],
    seeds: list[int],
    *,
    timesteps: int | None = None,
    tag: str | None = None,
    dry_run: bool = False,
) -> str:
    manifest = build_manifest(profiles, seeds, timesteps=timesteps, tag=tag)
    manifest_path = _manifest_path(tag)
    save_manifest(manifest_path, manifest)

    if dry_run:
        return manifest_path

    for replication in manifest["replications"]:
        profile = REPLICATION_PROFILES[replication["profile"]]
        seed = replication["seed"]
        replication["status"] = "running"
        chained_pretrained = None

        for stage_index, stage in enumerate(profile.stages):
            cfg = load_config(stage.config_name)
            cfg["seed"] = seed
            if timesteps is not None:
                cfg["total_timesteps"] = timesteps

            pretrained_path = stage.pretrained_path or chained_pretrained
            if pretrained_path:
                cfg["pretrained_path"] = pretrained_path
            elif "pretrained_path" in cfg:
                cfg.pop("pretrained_path")

            cfg["replication"] = {
                "profile": profile.name,
                "description": profile.description,
                "seed": seed,
                "stage_index": stage_index,
                "stage_name": stage.name,
            }

            run_tag = f"replicate_{profile.name}_{stage.name}"
            stage_record = {
                "name": stage.name,
                "config_name": stage.config_name,
                "seed": seed,
                "pretrained_path": pretrained_path,
                "status": "running",
            }
            replication["stages"].append(stage_record)
            save_manifest(manifest_path, manifest)

            run_dir = train(cfg, config_name=stage.config_name, run_tag=run_tag)
            chained_pretrained = _best_or_final_model_prefix(run_dir)

            stage_record["run_dir"] = run_dir
            stage_record["chained_model_prefix"] = chained_pretrained
            stage_record["status"] = "completed"
            replication["final_run_dir"] = run_dir
            replication["final_model_prefix"] = chained_pretrained
            save_manifest(manifest_path, manifest)

        replication["status"] = "completed"
        save_manifest(manifest_path, manifest)

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replicate canonical terrain/damage training across multiple seeds"
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(REPLICATION_PROFILES.keys()),
        default=["terrain_canonical", "damage_canonical"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps for every stage (useful for smoke tests)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for the replication manifest filename",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest without launching training",
    )
    args = parser.parse_args()

    manifest_path = run_replications(
        args.profiles,
        args.seeds,
        timesteps=args.timesteps,
        tag=args.tag,
        dry_run=args.dry_run,
    )
    print(f"Replication manifest: {manifest_path}")
