"""Copy training artifacts into docs/assets/ for README and portfolio."""

import argparse
import os
import shutil
import subprocess
import sys

from results_utils import find_latest_run, resolve_run_dir

COMPARISON_FILES = (
    "comparison_results.json",
    "comparison_plot.png",
    "comparison_reward_by_seed.png",
    "comparison_demo.mp4",
    "sudden_amputation_demo.mp4",
)


def collect(config: str, dest_subdir: str, run_dir: str | None = None):
    run_dir = resolve_run_dir(config=config, run_dir=run_dir)
    dest = os.path.join("docs", "assets", dest_subdir)
    os.makedirs(dest, exist_ok=True)

    for name in ("reward_curve.png", "demo.mp4", "config.json"):
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, name))

    eval_npz = os.path.join(run_dir, "eval", "evaluations.npz")
    if os.path.isfile(eval_npz):
        shutil.copy2(eval_npz, os.path.join(dest, "evaluations.npz"))

    print(f"Collected artifacts from {run_dir} → {dest}/")


def collect_comparison(
    out_dir: str = "docs/assets/terrain",
    difficulty: float = 0.4,
    seeds: list[int] | None = None,
    regenerate: bool = False,
):
    """Refresh comparison JSON/plots/video in docs/assets/terrain/."""
    os.makedirs(out_dir, exist_ok=True)
    missing = [name for name in COMPARISON_FILES if not os.path.isfile(os.path.join(out_dir, name))]

    if regenerate or missing:
        seed_args = seeds if seeds is not None else list(range(10))
        cmd = [
            sys.executable,
            "compare_policies.py",
            "--difficulty",
            str(difficulty),
            "--seeds",
            *[str(s) for s in seed_args],
            "--out-dir",
            out_dir,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    print(f"Comparison artifacts already present in {out_dir}/ (use --regenerate-comparison to rebuild)")


def collect_comparison_damage(
    out_dir: str = "docs/assets/damage",
    disabled_legs: list[int] | None = None,
    seeds: list[int] | None = None,
    regenerate: bool = False,
):
    """Refresh leg-damage comparison artifacts in docs/assets/damage/."""
    os.makedirs(out_dir, exist_ok=True)
    missing = [name for name in COMPARISON_FILES if not os.path.isfile(os.path.join(out_dir, name))]

    if regenerate or missing:
        seed_args = seeds if seeds is not None else list(range(10))
        leg_args = disabled_legs if disabled_legs is not None else [1]
        cmd = [
            sys.executable,
            "compare_damage.py",
            "--disabled-legs",
            *[str(l) for l in leg_args],
            "--seeds",
            *[str(s) for s in seed_args],
            "--out-dir",
            out_dir,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    print(f"Damage comparison artifacts already present in {out_dir}/ (use --regenerate-comparison to rebuild)")


def collect_all(
    regenerate_comparison: bool = False,
    comparison_seeds: list[int] | None = None,
):
    """Collect ant + terrain + damage run artifacts and comparison outputs."""
    for config, dest in (("ant", "ant"), ("terrain", "terrain"), ("damage", "damage")):
        try:
            collect(config, dest)
        except FileNotFoundError as exc:
            print(f"Skipping {config}: {exc}")

    collect_comparison(
        regenerate=regenerate_comparison,
        seeds=comparison_seeds,
    )
    collect_comparison_damage(
        regenerate=regenerate_comparison,
        seeds=comparison_seeds,
    )
    print("Done — docs/assets/ updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["ant", "terrain", "damage"], default=None)
    parser.add_argument("--dest", default=None, help="Subfolder under docs/assets/")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Collect ant + terrain + damage + comparison artifacts",
    )
    parser.add_argument(
        "--with-comparison",
        action="store_true",
        help="Also refresh comparison artifacts (after single-config collect)",
    )
    parser.add_argument(
        "--regenerate-comparison",
        action="store_true",
        help="Force re-run compare_policies.py",
    )
    parser.add_argument("--comparison-seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.all:
        collect_all(
            regenerate_comparison=args.regenerate_comparison,
            comparison_seeds=args.comparison_seeds,
        )
    elif args.config:
        dest = args.dest or args.config
        collect(args.config, dest, run_dir=args.run_dir)
        if args.with_comparison:
            collect_comparison(
                regenerate=args.regenerate_comparison,
                seeds=args.comparison_seeds,
            )
    else:
        parser.error("Provide --config or --all")
