"""Run every remaining project-backlog item unattended.

Covers:
1. Summarize replicated terrain/damage training sweeps
2. Broaden evaluation across terrain difficulties and amputated legs
3. Promote sudden-amputation recovery to a first-class reported benchmark
4. Update docs / overnight tracking files to mark the backlog complete
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from compare_damage import (
    DEFAULT_AMPUTATION_STEP,
    _comparison_metrics,
    compare as compare_damage,
    compare_sudden_amputation,
    plot_comparison as plot_damage_comparison,
)
from compare_policies import (
    compare as compare_terrain,
    plot_comparison_summary as plot_terrain_summary,
)
from envs.damage_ant import LEG_LABELS
from results_utils import default_checkpoint

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
REPLICATIONS_DIR = RESULTS_DIR / "replications"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
REPLICATION_ASSETS = ASSETS_DIR / "replications"
TERRAIN_ASSETS = ASSETS_DIR / "terrain"
DAMAGE_ASSETS = ASSETS_DIR / "damage"

TERRAIN_DIFFICULTIES = [0.2, 0.4, 0.6, 0.8]
DAMAGE_LEGS = [0, 1, 2, 3]
EVAL_SEEDS = list(range(10))

KNOWN_FINAL_RUNS = {
    ("terrain_canonical", 1): (
        "results/ppo_terrainant_v0_1783145145_seed1_resume_from_tracked_seed1_balanced"
    ),
    ("terrain_canonical", 2): (
        "results/ppo_terrainant_v0_1783157709_seed2_replicate_terrain_canonical_terrain_balanced"
    ),
    ("terrain_canonical", 0): (
        "results/ppo_terrainant_v0_1783236040_seed0_replicate_terrain_canonical_terrain_balanced"
    ),
    ("damage_canonical", 0): (
        "results/ppo_damageant_v0_1783161704_seed0_replicate_damage_canonical_damage_gait"
    ),
    ("damage_canonical", 1): (
        "results/ppo_damageant_v0_1783163648_seed1_replicate_damage_canonical_damage_gait"
    ),
    ("damage_canonical", 2): (
        "results/ppo_damageant_v0_1783165386_seed2_replicate_damage_canonical_damage_gait"
    ),
}


def _discover_replication_entries() -> dict[tuple[str, int], str]:
    """Return the best-known final run dir for each (profile, seed) pair."""
    discovered: dict[tuple[str, int], tuple[str, str]] = {}

    for (profile, seed), run_dir in KNOWN_FINAL_RUNS.items():
        discovered[(profile, seed)] = (run_dir, "known_final_run")

    for manifest in _load_manifests():
        for replication in manifest.get("replications", []):
            if replication.get("status") != "completed":
                continue
            profile = replication.get("profile")
            seed = replication.get("seed")
            final_run_dir = replication.get("final_run_dir")
            if profile is None or seed is None or not final_run_dir:
                continue
            key = (profile, seed)
            manifest_path = manifest["_manifest_path"]
            discovered[key] = (final_run_dir, manifest_path)

    return {key: run_dir for key, (run_dir, _) in discovered.items()}


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _append_overnight_log(section: str, bullets: list[str]) -> None:
    path = REPO_ROOT / "overnight_log.md"
    with path.open("a") as f:
        f.write(f"\n## {section}\n\n")
        for bullet in bullets:
            f.write(f"- {bullet}\n")


def _read_final_eval(run_dir: str | Path) -> dict | None:
    eval_path = Path(run_dir) / "eval" / "evaluations.npz"
    if not eval_path.is_file():
        return None
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    results = data["results"]
    last = results[-1]
    return {
        "timesteps": int(timesteps[-1]),
        "mean_reward": float(last.mean()),
        "std_reward": float(last.std()),
        "n_eval_episodes": int(last.shape[0]),
    }


def _load_manifests() -> list[dict]:
    manifests = []
    if not REPLICATIONS_DIR.is_dir():
        return manifests
    for path in sorted(REPLICATIONS_DIR.glob("replication_*.json")):
        with path.open() as f:
            payload = json.load(f)
        payload["_manifest_path"] = str(path.relative_to(REPO_ROOT))
        manifests.append(payload)
    return manifests


def summarize_replications() -> dict:
    _log("Step 1/4: summarizing replicated training sweeps")
    REPLICATION_ASSETS.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_replication_entries()
    entries = []
    for (profile, seed), run_dir in sorted(run_dirs.items()):
        eval_metrics = _read_final_eval(REPO_ROOT / run_dir)
        entries.append(
            {
                "profile": profile,
                "seed": seed,
                "final_run_dir": run_dir,
                "status": "completed" if eval_metrics else "missing_eval",
                "final_eval": eval_metrics,
            }
        )

    # Attach manifest provenance when available.
    manifests = _load_manifests()
    manifest_index = {}
    for manifest in manifests:
        for replication in manifest.get("replications", []):
            key = (replication.get("profile"), replication.get("seed"))
            manifest_index[key] = {
                "manifest_path": manifest["_manifest_path"],
                "status": replication.get("status"),
                "final_run_dir": replication.get("final_run_dir"),
                "stages": replication.get("stages", []),
            }

    for entry in entries:
        key = (entry["profile"], entry["seed"])
        if key in manifest_index:
            entry["manifest"] = manifest_index[key]

    terrain_rewards = [
        e["final_eval"]["mean_reward"]
        for e in entries
        if e["profile"] == "terrain_canonical" and e["final_eval"]
    ]
    damage_rewards = [
        e["final_eval"]["mean_reward"]
        for e in entries
        if e["profile"] == "damage_canonical" and e["final_eval"]
    ]
    crossleg_rewards = [
        e["final_eval"]["mean_reward"]
        for e in entries
        if e["profile"] == "damage_crossleg" and e["final_eval"]
    ]

    notes = [
        "Terrain seed 1 uses the resumed tracked handoff run.",
        "Entries are discovered from replication manifests with KNOWN_FINAL_RUNS fallbacks.",
    ]
    terrain_seeds = sorted(
        e["seed"] for e in entries if e["profile"] == "terrain_canonical" and e["final_eval"]
    )
    if terrain_seeds:
        notes.append(f"Terrain canonical seeds present locally: {terrain_seeds}.")
    if crossleg_rewards:
        notes.append(
            "Cross-leg damage replications are included when manifests exist under results/replications/."
        )

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries,
        "aggregate": {
            "terrain_canonical": {
                "n_seeds": len(terrain_rewards),
                "mean_final_eval_reward": float(np.mean(terrain_rewards))
                if terrain_rewards
                else None,
                "std_final_eval_reward": float(np.std(terrain_rewards))
                if terrain_rewards
                else None,
            },
            "damage_canonical": {
                "n_seeds": len(damage_rewards),
                "mean_final_eval_reward": float(np.mean(damage_rewards))
                if damage_rewards
                else None,
                "std_final_eval_reward": float(np.std(damage_rewards))
                if damage_rewards
                else None,
            },
            "damage_crossleg": {
                "n_seeds": len(crossleg_rewards),
                "mean_final_eval_reward": float(np.mean(crossleg_rewards))
                if crossleg_rewards
                else None,
                "std_final_eval_reward": float(np.std(crossleg_rewards))
                if crossleg_rewards
                else None,
            },
        },
        "notes": notes,
    }

    json_path = REPLICATION_ASSETS / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    md_path = REPLICATION_ASSETS / "SUMMARY.md"
    lines = [
        "# Replicated Training Sweep Summary",
        "",
        f"Generated: `{summary['created_at']}`",
        "",
        "## Per-seed final eval rewards",
        "",
        "| Profile | Seed | Final eval reward | Timesteps | Run dir |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in entries:
        eval_metrics = entry["final_eval"] or {}
        reward = (
            f"{eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}"
            if eval_metrics
            else "n/a"
        )
        steps = eval_metrics.get("timesteps", "n/a")
        lines.append(
            f"| `{entry['profile']}` | {entry['seed']} | {reward} | {steps} | `{entry['final_run_dir']}` |"
        )

    terrain_agg = summary["aggregate"]["terrain_canonical"]
    damage_agg = summary["aggregate"]["damage_canonical"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Terrain seeds present: **{terrain_agg['n_seeds']}** "
            f"(mean final eval `{terrain_agg['mean_final_eval_reward']:.2f} ± {terrain_agg['std_final_eval_reward']:.2f}`)"
            if terrain_agg["mean_final_eval_reward"] is not None
            else "- Terrain seeds present: **0**",
            f"- Damage seeds present: **{damage_agg['n_seeds']}** "
            f"(mean final eval `{damage_agg['mean_final_eval_reward']:.2f} ± {damage_agg['std_final_eval_reward']:.2f}`)"
            if damage_agg["mean_final_eval_reward"] is not None
            else "- Damage seeds present: **0**",
            "",
            "## Notes",
            "",
        ]
    )
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    md_path.write_text("\n".join(lines))

    _log(f"Wrote {json_path.relative_to(REPO_ROOT)} and {md_path.relative_to(REPO_ROOT)}")
    return summary


def run_terrain_difficulty_sweep(
    *,
    difficulties: list[float],
    seeds: list[int],
    max_steps: int,
) -> dict:
    _log("Step 2/4: terrain difficulty evaluation sweep")
    TERRAIN_ASSETS.mkdir(parents=True, exist_ok=True)

    flat_run = default_checkpoint("flat") or "checkpoints/flat"
    terrain_run = default_checkpoint("terrain") or "checkpoints/terrain"

    by_difficulty = {}
    for difficulty in difficulties:
        _log(f"  terrain difficulty={difficulty}")
        results = compare_terrain(
            flat_run,
            terrain_run,
            difficulty,
            seeds,
            max_steps=max_steps,
        )
        out_dir = TERRAIN_ASSETS / f"difficulty_{difficulty:.1f}".replace(".", "p")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "comparison_results.json"
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)
        plot_terrain_summary(results, str(out_dir / "comparison_plot.png"))
        by_difficulty[str(difficulty)] = {
            "out_dir": str(out_dir.relative_to(REPO_ROOT)),
            "flat_summary": results["flat"]["summary"],
            "terrain_summary": results["terrain"]["summary"],
            "terrain_reward_retention_pct": results["terrain_reward_retention_pct"],
            "terrain_velocity_retention_pct": results["terrain_velocity_retention_pct"],
        }

    sweep = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "flat_run": flat_run,
        "terrain_run": terrain_run,
        "seeds": seeds,
        "max_steps": max_steps,
        "difficulties": difficulties,
        "by_difficulty": by_difficulty,
    }
    sweep_path = TERRAIN_ASSETS / "difficulty_sweep.json"
    with sweep_path.open("w") as f:
        json.dump(sweep, f, indent=2)
    _log(f"Wrote {sweep_path.relative_to(REPO_ROOT)}")
    return sweep


def run_damage_leg_sweep(
    *,
    legs: list[int],
    seeds: list[int],
    max_steps: int,
) -> dict:
    _log("Step 3/4: damage all-leg fixed-amputation evaluation sweep")
    DAMAGE_ASSETS.mkdir(parents=True, exist_ok=True)

    flat_run = default_checkpoint("flat") or "checkpoints/flat"
    damage_run = default_checkpoint("damage")
    if damage_run is None:
        raise FileNotFoundError("No committed damage checkpoint found")

    by_leg = {}
    for leg in legs:
        label = LEG_LABELS[leg]
        _log(f"  damage fixed amputation leg={leg} ({label})")
        results = compare_damage(
            flat_run,
            damage_run,
            [leg],
            seeds,
            max_steps=max_steps,
        )
        out_dir = DAMAGE_ASSETS / f"leg_{leg}"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "comparison_results.json"
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)
        plot_damage_comparison(results, str(out_dir / "comparison_plot.png"))
        by_leg[str(leg)] = {
            "leg_id": leg,
            "leg_label": label,
            "out_dir": str(out_dir.relative_to(REPO_ROOT)),
            "flat_summary": results["flat"]["summary"],
            "damage_summary": results["damage"]["summary"],
            "metrics": {
                key: results[key]
                for key in (
                    "velocity_retention_pct",
                    "velocity_gain_mps",
                    "reward_retention_pct",
                    "reward_gain",
                    "fall_rate_reduction_pct",
                    "episode_length_gain_steps",
                )
            },
        }

    sweep = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "flat_run": flat_run,
        "damage_run": damage_run,
        "seeds": seeds,
        "max_steps": max_steps,
        "legs": legs,
        "by_leg": by_leg,
    }
    sweep_path = DAMAGE_ASSETS / "leg_sweep.json"
    with sweep_path.open("w") as f:
        json.dump(sweep, f, indent=2)
    _log(f"Wrote {sweep_path.relative_to(REPO_ROOT)}")
    return sweep


def promote_sudden_amputation(
    *,
    seeds: list[int],
    max_steps: int,
    amputation_step: int,
) -> dict:
    _log("Step 4/4: promoting sudden-amputation to first-class benchmark")
    DAMAGE_ASSETS.mkdir(parents=True, exist_ok=True)

    flat_run = default_checkpoint("flat") or "checkpoints/flat"
    damage_run = default_checkpoint("damage")
    if damage_run is None:
        raise FileNotFoundError("No committed damage checkpoint found")

    results = compare_sudden_amputation(
        flat_run,
        damage_run,
        [1],
        amputation_step,
        seeds,
        max_steps=max_steps,
    )
    results["benchmark"] = "sudden_amputation"
    results["leg_label"] = LEG_LABELS[1]
    results["comparison_metrics"] = _comparison_metrics(
        results["flat"]["summary"],
        results["damage"]["summary"],
    )

    json_path = DAMAGE_ASSETS / "sudden_amputation_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Keep the main damage comparison payload in sync with a dedicated sudden block.
    main_path = DAMAGE_ASSETS / "comparison_results.json"
    if main_path.is_file():
        with main_path.open() as f:
            payload = json.load(f)
    else:
        payload = {}
    payload["sudden_amputation"] = results
    with main_path.open("w") as f:
        json.dump(payload, f, indent=2)

    _log(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    return results


def _fmt_reward(summary: dict) -> str:
    return f"{summary['mean_reward']:.0f} ± {summary['std_reward']:.0f}"


def _fmt_fall(summary: dict) -> str:
    return f"{summary['fall_rate'] * 100:.0f}%"


def _fmt_vel(summary: dict) -> str:
    return f"{summary['mean_forward_velocity']:.2f} m/s"


def update_docs(
    replication_summary: dict,
    terrain_sweep: dict,
    damage_sweep: dict,
    sudden_results: dict,
) -> None:
    _log("Updating project docs and overnight tracking files")

    terrain_rows = []
    for difficulty in terrain_sweep["difficulties"]:
        entry = terrain_sweep["by_difficulty"][str(difficulty)]
        terrain_rows.append(
            (
                difficulty,
                entry["flat_summary"],
                entry["terrain_summary"],
            )
        )

    damage_rows = []
    for leg in damage_sweep["legs"]:
        entry = damage_sweep["by_leg"][str(leg)]
        damage_rows.append(
            (
                leg,
                entry["leg_label"],
                entry["flat_summary"],
                entry["damage_summary"],
            )
        )

    sudden_flat = sudden_results["flat"]["summary"]
    sudden_damage = sudden_results["damage"]["summary"]

    terrain_agg = replication_summary["aggregate"]["terrain_canonical"]
    damage_agg = replication_summary["aggregate"]["damage_canonical"]

    # README remaining-work section and new result sections.
    readme_path = REPO_ROOT / "README.md"
    readme = readme_path.read_text()
    remaining_block = """## Remaining Work

- [x] Baseline flat, terrain adaptation, and fixed-amputation damage experiments
- [x] Comparison videos and sudden-amputation demo
- [x] Committed checkpoints and artifact collection scripts
- [x] Multi-seed training replication pipeline via `replicate_training.py`
- [x] Regression coverage for terrain/damage env behavior and comparison metrics
- [x] Cleaner provenance for exploratory terrain configs
- [x] Run and summarize replicated terrain/damage training across multiple seeds
- [x] Broader evaluation sweeps: terrain difficulty curve and all-leg damage cases
- [x] Sudden-amputation benchmark promoted from demo to first-class reported result
"""
    if "## Remaining Work" in readme:
        before, _, after = readme.partition("## Remaining Work")
        # Keep everything after the remaining-work bullets (Further reading onward).
        if "## Further reading" in after:
            _, _, rest = after.partition("## Further reading")
            readme = before + remaining_block + "\n## Further reading" + rest
        else:
            readme = before + remaining_block

    # Insert or replace extended results section before Remaining Work.
    extended = [
        "## Extended evaluation results",
        "",
        "Generated by `python run_overnight_backlog.py`.",
        "",
        "### Replicated training finals",
        "",
        f"- Terrain seeds present: **{terrain_agg['n_seeds']}** "
        f"(mean final eval `{terrain_agg['mean_final_eval_reward']:.1f} ± {terrain_agg['std_final_eval_reward']:.1f}`)",
        f"- Damage seeds present: **{damage_agg['n_seeds']}** "
        f"(mean final eval `{damage_agg['mean_final_eval_reward']:.1f} ± {damage_agg['std_final_eval_reward']:.1f}`)",
        "",
        "Details: [`docs/assets/replications/SUMMARY.md`](docs/assets/replications/SUMMARY.md)",
        "",
        "### Terrain difficulty curve",
        "",
        "Flat-trained vs terrain-adapted on `TerrainAnt-v0`, 10 matched seeds:",
        "",
        "| Difficulty | Flat reward | Terrain reward | Flat fall | Terrain fall | Flat vel | Terrain vel |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for difficulty, flat_s, terrain_s in terrain_rows:
        extended.append(
            f"| {difficulty:.1f} | {_fmt_reward(flat_s)} | {_fmt_reward(terrain_s)} | "
            f"{_fmt_fall(flat_s)} | {_fmt_fall(terrain_s)} | {_fmt_vel(flat_s)} | {_fmt_vel(terrain_s)} |"
        )
    extended.extend(
        [
            "",
            "Artifacts: `docs/assets/terrain/difficulty_sweep.json`",
            "",
            "### All-leg fixed amputation",
            "",
            "Flat-trained vs damage-robust on `DamageAnt-v0`, 10 matched seeds:",
            "",
            "| Leg | Label | Flat reward | Damage reward | Flat fall | Damage fall | Flat vel | Damage vel |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for leg, label, flat_s, damage_s in damage_rows:
        extended.append(
            f"| {leg} | {label} | {_fmt_reward(flat_s)} | {_fmt_reward(damage_s)} | "
            f"{_fmt_fall(flat_s)} | {_fmt_fall(damage_s)} | {_fmt_vel(flat_s)} | {_fmt_vel(damage_s)} |"
        )
    extended.extend(
        [
            "",
            "Artifacts: `docs/assets/damage/leg_sweep.json`",
            "",
            "### Sudden amputation (first-class benchmark)",
            "",
            f"Both policies start on 4 legs; leg 1 (`{LEG_LABELS[1]}`) is amputated at step "
            f"**{sudden_results['amputation_step']}**. 10 matched seeds:",
            "",
            "| Metric | Flat-trained | Damage-robust |",
            "|---|---:|---:|",
            f"| Mean reward | {_fmt_reward(sudden_flat)} | {_fmt_reward(sudden_damage)} |",
            f"| Fall rate | {_fmt_fall(sudden_flat)} | {_fmt_fall(sudden_damage)} |",
            f"| Mean episode length | {sudden_flat['mean_steps']:.0f} | {sudden_damage['mean_steps']:.0f} |",
            f"| Mean forward velocity | {_fmt_vel(sudden_flat)} | {_fmt_vel(sudden_damage)} |",
            "",
            "Artifacts: `docs/assets/damage/sudden_amputation_results.json`",
            "",
        ]
    )
    extended_text = "\n".join(extended)
    if "## Extended evaluation results" in readme:
        before, _, after = readme.partition("## Extended evaluation results")
        if "## Remaining Work" in after:
            _, _, rest = after.partition("## Remaining Work")
            readme = before + extended_text + "## Remaining Work" + rest
        else:
            readme = before + extended_text
    elif "## Remaining Work" in readme:
        readme = readme.replace("## Remaining Work", extended_text + "## Remaining Work")
    else:
        readme = readme.rstrip() + "\n\n" + extended_text

    readme_path.write_text(readme)

    # Scope doc.
    scope_path = REPO_ROOT / "MUJOCO_PROJECT_SCOPE.md"
    scope = scope_path.read_text()
    scope_gaps = """## Remaining Gaps (priority order)

- [x] Set up replicated canonical terrain/damage training pipeline with multiple training seeds (`replicate_training.py`)
- [x] Clean up exploratory config provenance so reproducible recipes no longer depend on historical `results/` directories
- [x] Add automated regression coverage for terrain env and comparison metrics
- [x] Run and summarize the replicated training sweeps themselves
- [x] Expand evaluation beyond one terrain difficulty and one amputated leg
- [x] Promote sudden-amputation recovery from demo-only artifact to a first-class reported benchmark
"""
    if "## Remaining Gaps" in scope:
        before, _, _ = scope.partition("## Remaining Gaps")
        scope = before + scope_gaps
    else:
        scope = scope.rstrip() + "\n\n" + scope_gaps
    # Refresh status date.
    scope = scope.replace("## Status (as of 2026-07-02)", "## Status (as of 2026-07-05)")
    scope_path.write_text(scope)

    # Overnight todo: mark remaining items complete.
    todo_path = REPO_ROOT / "overnight_todo.md"
    todo = todo_path.read_text()
    replacements = {
        "- [ ] Summarize the replicated terrain / damage training sweeps now that all seeds are finished.": (
            "- [x] Summarize the replicated terrain / damage training sweeps now that all seeds are finished.\n"
            "  - Completed via `python run_overnight_backlog.py`.\n"
            "  - Artifacts: `docs/assets/replications/summary.json`, `docs/assets/replications/SUMMARY.md`."
        ),
        "- [ ] Broaden evaluation sweeps across more terrain difficulties and amputated-leg cases.": (
            "- [x] Broaden evaluation sweeps across more terrain difficulties and amputated-leg cases.\n"
            "  - Terrain difficulties: 0.2, 0.4, 0.6, 0.8 → `docs/assets/terrain/difficulty_sweep.json`.\n"
            "  - Damage legs: 0, 1, 2, 3 → `docs/assets/damage/leg_sweep.json`."
        ),
        "- [ ] Promote sudden-amputation recovery into a first-class reported benchmark instead of a demo-only artifact.": (
            "- [x] Promote sudden-amputation recovery into a first-class reported benchmark instead of a demo-only artifact.\n"
            "  - Artifact: `docs/assets/damage/sudden_amputation_results.json`.\n"
            "  - Reported in `README.md` and `MUJOCO_PROJECT_SCOPE.md`."
        ),
    }
    for old, new in replacements.items():
        todo = todo.replace(old, new)
    todo_path.write_text(todo)

    # Current instructions: backlog complete.
    instructions_path = REPO_ROOT / "current_instructions.md"
    instructions = instructions_path.read_text()
    instructions = instructions.replace(
        """## Remaining project gaps

Based on `MUJOCO_PROJECT_SCOPE.md` and `README.md`, the highest-value unfinished work is:

1. run and summarize the replicated training sweeps
2. expand evaluation beyond one terrain difficulty and one amputated leg
3. promote sudden-amputation recovery from demo-only artifact to first-class benchmark
""",
        """## Remaining project gaps

All previously listed project-backlog gaps are complete as of the overnight backlog run:

1. [x] run and summarize the replicated training sweeps
2. [x] expand evaluation beyond one terrain difficulty and one amputated leg
3. [x] promote sudden-amputation recovery from demo-only artifact to first-class benchmark

Primary artifacts:

- `docs/assets/replications/SUMMARY.md`
- `docs/assets/terrain/difficulty_sweep.json`
- `docs/assets/damage/leg_sweep.json`
- `docs/assets/damage/sudden_amputation_results.json`
""",
    )
    instructions = instructions.replace(
        """## Canonical next experiment queue

When code/test tasks are in a good state, the next experimental work is:

1. summarize replication results after the terrain and damage sweeps
2. only then broaden evaluation sweeps
""",
        """## Canonical next experiment queue

The original project backlog is complete. If continuing autonomously, prefer:

1. optional follow-ups discovered from the new evaluation artifacts
2. packaging / publishing updates only when explicitly requested
""",
    )
    instructions_path.write_text(instructions)

    _append_overnight_log(
        "2026-07-05 overnight backlog completion",
        [
            "Task completed: finished every remaining project-backlog item via `run_overnight_backlog.py`",
            "Artifacts: `docs/assets/replications/SUMMARY.md`, `docs/assets/terrain/difficulty_sweep.json`, `docs/assets/damage/leg_sweep.json`, `docs/assets/damage/sudden_amputation_results.json`",
            "Docs updated: `README.md`, `MUJOCO_PROJECT_SCOPE.md`, `overnight_todo.md`, `current_instructions.md`",
            "Result: project backlog is fully closed",
        ],
    )


def run_all(
    *,
    difficulties: list[float],
    legs: list[int],
    seeds: list[int],
    max_steps: int,
    amputation_step: int,
    skip_docs: bool = False,
) -> dict:
    started = time.time()
    _append_overnight_log(
        "2026-07-05 planned overnight backlog run",
        [
            "Command: `python run_overnight_backlog.py`",
            f"Terrain difficulties: {difficulties}",
            f"Damage legs: {legs}",
            f"Eval seeds: {seeds}",
            f"Max steps: {max_steps}",
            f"Sudden amputation step: {amputation_step}",
        ],
    )

    replication_summary = summarize_replications()
    terrain_sweep = run_terrain_difficulty_sweep(
        difficulties=difficulties,
        seeds=seeds,
        max_steps=max_steps,
    )
    damage_sweep = run_damage_leg_sweep(
        legs=legs,
        seeds=seeds,
        max_steps=max_steps,
    )
    sudden_results = promote_sudden_amputation(
        seeds=seeds,
        max_steps=max_steps,
        amputation_step=amputation_step,
    )
    if not skip_docs:
        update_docs(
            replication_summary,
            terrain_sweep,
            damage_sweep,
            sudden_results,
        )

    elapsed = time.time() - started
    _log(f"Overnight backlog complete in {elapsed / 60:.1f} minutes")
    return {
        "replication_summary": replication_summary,
        "terrain_sweep": terrain_sweep,
        "damage_sweep": damage_sweep,
        "sudden_results": sudden_results,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run every remaining project-backlog item unattended"
    )
    parser.add_argument(
        "--difficulties",
        type=float,
        nargs="+",
        default=TERRAIN_DIFFICULTIES,
        help="Terrain difficulties for the broadened evaluation sweep",
    )
    parser.add_argument(
        "--legs",
        type=int,
        nargs="+",
        default=DAMAGE_LEGS,
        help="Leg ids for the all-leg fixed-amputation sweep",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=EVAL_SEEDS,
        help="Matched evaluation seeds",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--amputation-step",
        type=int,
        default=DEFAULT_AMPUTATION_STEP,
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip README / scope / overnight tracking updates",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write the replication summary artifacts",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if args.summary_only:
        summarize_replications()
    else:
        run_all(
            difficulties=args.difficulties,
            legs=args.legs,
            seeds=args.seeds,
            max_steps=args.max_steps,
            amputation_step=args.amputation_step,
            skip_docs=args.skip_docs,
        )
