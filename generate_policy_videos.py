"""Generate essential comparison demo videos (not per-leg solo clips)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from compound_damage import (
    DEFAULT_CROSSLEG_RUN,
    DEFAULT_SPECIALIST_RUN,
    compare_compound,
    pick_sudden_compound_demo_seed,
    record_sudden_crossleg_vs_compound,
)
from compare_damage import DEFAULT_AMPUTATION_STEP
from results_utils import default_checkpoint

DEFAULT_FLAT_RUN = default_checkpoint("flat") or "checkpoints/flat"
DEFAULT_DAMAGE_RUN = default_checkpoint("damage")

# Essential portfolio videos (comparisons only).
ESSENTIAL_VIDEOS = [
    "docs/assets/terrain/comparison_demo.mp4",
    "docs/assets/damage/comparison_demo.mp4",
    "docs/assets/damage/sudden_amputation_demo.mp4",
    "docs/assets/damage/compound_comparison_demo.mp4",
]

# Sudden leg-1 amputation at this step (~4 s at 30 fps) while still upright.
COMPOUND_DEMO_LEG = 1


def generate_terrain_comparison():
    subprocess.run(
        [
            sys.executable,
            "compare_policies.py",
            "--difficulty",
            "0.4",
            "--seeds",
            *[str(s) for s in range(10)],
            "--out-dir",
            "docs/assets/terrain",
        ],
        check=True,
    )


def generate_damage_comparisons(*, specialist_run: str, max_steps: int = 1000):
    """Fixed leg-1 amputation + sudden mid-episode amputation (both vs flat)."""
    os.makedirs("docs/assets/damage", exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "compare_damage.py",
            "--flat-run",
            DEFAULT_FLAT_RUN,
            "--damage-run",
            specialist_run,
            "--disabled-legs",
            "1",
            "--seeds",
            *[str(s) for s in range(10)],
            "--max-steps",
            str(max_steps),
            "--out-dir",
            "docs/assets/damage",
        ],
        check=True,
    )


def generate_compound_comparison(
    *,
    flat_run: str,
    specialist_run: str,
    crossleg_run: str,
    max_steps: int = 1000,
    amputation_step: int = DEFAULT_AMPUTATION_STEP,
):
    """Sudden amputation: cross-leg vs flat→specialist compound switch."""
    os.makedirs("docs/assets/damage", exist_ok=True)
    compound_dir = "docs/assets/damage/compound"
    os.makedirs(compound_dir, exist_ok=True)

    results = compare_compound(
        flat_run=flat_run,
        specialist_run=specialist_run,
        crossleg_run=crossleg_run,
        max_steps=max_steps,
    )
    with open(os.path.join(compound_dir, "compound_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    amputation_legs = [1]
    seed = pick_sudden_compound_demo_seed(
        flat_run,
        specialist_run,
        amputation_legs,
        amputation_step,
    )
    record_sudden_crossleg_vs_compound(
        flat_run,
        crossleg_run,
        specialist_run,
        amputation_legs,
        amputation_step,
        seed,
        max_steps,
        "docs/assets/damage/compound_comparison_demo.mp4",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate essential comparison demo videos"
    )
    parser.add_argument("--specialist-run", default=DEFAULT_DAMAGE_RUN)
    parser.add_argument("--crossleg-run", default=DEFAULT_CROSSLEG_RUN)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--skip-terrain", action="store_true")
    parser.add_argument("--skip-damage", action="store_true")
    parser.add_argument("--skip-compound", action="store_true")
    args = parser.parse_args()

    if args.specialist_run is None:
        parser.error("No damage specialist checkpoint. Pass --specialist-run.")

    if not args.skip_terrain:
        print("=== Terrain: flat vs adapted ===")
        generate_terrain_comparison()

    if not args.skip_damage:
        print("=== Damage: flat vs specialist (fixed + sudden) ===")
        generate_damage_comparisons(
            specialist_run=args.specialist_run,
            max_steps=args.max_steps,
        )

    if not args.skip_compound:
        print(f"=== Compound: sudden cross-leg vs flat→specialist (leg {COMPOUND_DEMO_LEG}) ===")
        generate_compound_comparison(
            flat_run=DEFAULT_FLAT_RUN,
            specialist_run=args.specialist_run,
            crossleg_run=args.crossleg_run,
            max_steps=args.max_steps,
        )

    print("Essential comparison videos:")
    for path in ESSENTIAL_VIDEOS:
        status = "ok" if os.path.isfile(path) else "MISSING"
        print(f"  [{status}] {path}")
    print("Done.")


if __name__ == "__main__":
    main()
