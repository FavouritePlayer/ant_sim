import os
import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from stable_baselines3.common.callbacks import BaseCallback

_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "ant_terrain.xml")

NROW, NCOL = 128, 128


class TerrainAntEnv(AntEnv):
    """
    Ant-v5 on procedurally generated bumpy terrain.

    The terrain is a sum of random sinusoids, regenerated every episode.
    difficulty in [0, 1] controls amplitude: 0 = flat, 1 = 0.6m peak bumps.
    The spawn point is always zeroed so the ant doesn't start embedded in ground.
    """

    def __init__(self, difficulty: float = 0.3, **kwargs):
        self.difficulty = difficulty
        kwargs.setdefault("healthy_z_range", (0.2, 2.0))
        super().__init__(xml_file=_XML_PATH, **kwargs)

    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def reset(self, **kwargs):
        self._randomise_terrain()
        return super().reset(**kwargs)

    def _randomise_terrain(self):
        x = np.linspace(0, 2 * np.pi, NCOL)
        y = np.linspace(0, 2 * np.pi, NROW)
        X, Y = np.meshgrid(x, y)

        terrain = np.zeros((NROW, NCOL))
        rng = self.np_random if hasattr(self, "np_random") else np.random
        for _ in range(8):
            fx = rng.uniform(0.5, 3.0)
            fy = rng.uniform(0.5, 3.0)
            px = rng.uniform(0, 2 * np.pi)
            py = rng.uniform(0, 2 * np.pi)
            # multiply independent X and Y waves → 2D hills, not diagonal ridges
            terrain += np.sin(fx * X + px) * np.cos(fy * Y + py)

        # Normalise to [0, 1] then scale by difficulty
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())

        # Zero the spawn point so the ant always starts on solid ground
        cx, cy = NROW // 2, NCOL // 2
        terrain -= terrain[cx, cy]
        terrain = np.clip(terrain, 0, None)
        terrain /= terrain.max() + 1e-8  # re-normalise after shift

        self.model.hfield_data[:] = (terrain * self.difficulty).flatten()


class CurriculumCallback(BaseCallback):
    """
    Linearly ramps terrain difficulty from start_difficulty to max_difficulty
    over the course of training, updating every update_interval steps.
    """

    def __init__(
        self,
        start_difficulty: float = 0.05,
        max_difficulty: float = 0.8,
        update_interval: int = 100_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_difficulty = start_difficulty
        self.max_difficulty = max_difficulty
        self.update_interval = update_interval
        self._last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update >= self.update_interval:
            progress = min(self.num_timesteps / self.model._total_timesteps, 1.0)
            difficulty = self.start_difficulty + progress * (self.max_difficulty - self.start_difficulty)
            self.training_env.env_method("set_difficulty", difficulty)
            if self.verbose:
                print(f"  [curriculum] timestep {self.num_timesteps:,} → difficulty {difficulty:.2f}")
            self._last_update = self.num_timesteps
        return True
