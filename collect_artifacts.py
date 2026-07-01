"""Copy training artifacts into docs/assets/ for README and portfolio."""

import argparse
import os
import shutil

from results_utils import find_latest_run


def collect(config: str, dest_subdir: str):
    run_dir = find_latest_run(config=config)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["ant", "terrain"], required=True)
    parser.add_argument("--dest", default=None, help="Subfolder under docs/assets/")
    args = parser.parse_args()
    dest = args.dest or args.config
    collect(args.config, dest)
