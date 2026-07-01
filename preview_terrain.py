"""Render a terrain snapshot and report relief stats for quick iteration."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from envs import register
from envs.terrain_ant import TerrainAntEnv, terrain_height_from_data


def preview(difficulty: float = 0.65, seed: int = 7, out: str = "terrain_preview.png"):
    register()
    env = TerrainAntEnv(difficulty=difficulty, render_mode="rgb_array")
    env.reset(seed=seed)

    grid = env._terrain_grid
    heights = terrain_height_from_data(grid)
    relief = heights.max() - heights.min()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(heights, cmap="terrain", origin="lower", extent=[-8, 8, -8, 8])
    axes[0].set_title(f"Terrain height map (relief {relief:.2f} m)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    plt.colorbar(im, ax=axes[0], label="z (m)")

    frame = env.render()
    axes[1].imshow(frame)
    axes[1].set_title(f"Ant view  |  difficulty={difficulty}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Relief: {relief:.2f} m  |  z range [{heights.min():.2f}, {heights.max():.2f}]")
    print(f"Saved: {out}")
    env.close()
    return relief


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="terrain_preview.png")
    args = parser.parse_args()
    preview(args.difficulty, args.seed, args.out)
