import os

import mujoco
import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from stable_baselines3.common.callbacks import BaseCallback

_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "ant_terrain.xml")

NROW, NCOL = 256, 256
HFIELD_X_HALF = 8.0
HFIELD_Y_HALF = 8.0
Z_TOP = 3.0
Z_BASE = 0.1
SPAWN_CLEARANCE = 0.55
MIN_GROUND_GAP = 0.02  # only correct actual penetrations reported by MuJoCo contacts


def _bilinear_sample(grid: np.ndarray, x: float, y: float) -> float:
    nrow, ncol = grid.shape
    fi = (y + HFIELD_Y_HALF) / (2 * HFIELD_Y_HALF) * (nrow - 1)
    fj = (x + HFIELD_X_HALF) / (2 * HFIELD_X_HALF) * (ncol - 1)
    fi = np.clip(fi, 0, nrow - 1)
    fj = np.clip(fj, 0, ncol - 1)
    i0, j0 = int(np.floor(fi)), int(np.floor(fj))
    i1, j1 = min(i0 + 1, nrow - 1), min(j0 + 1, ncol - 1)
    di, dj = fi - i0, fj - j0
    return float(
        (1 - di) * (1 - dj) * grid[i0, j0]
        + (1 - di) * dj * grid[i0, j1]
        + di * (1 - dj) * grid[i1, j0]
        + di * dj * grid[i1, j1]
    )


def terrain_height_from_data(data: float) -> float:
    return Z_BASE + data * (Z_TOP - Z_BASE)


def _smooth_grid(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
    kernel /= kernel.sum()
    out = grid.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        smoothed = np.zeros_like(out)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                smoothed[i, j] = (padded[i : i + 3, j : j + 3] * kernel).sum()
        out = smoothed
    return out


class TerrainAntEnv(AntEnv):
    """
    Ant-v5 on procedurally generated bumpy terrain.

    difficulty in [0, 1] scales relief: 0 = flat, 1 = full ~2.9 m peak-to-trough.
    """

    def __init__(self, difficulty: float = 0.3, **kwargs):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self._terrain_grid = np.full((NROW, NCOL), 0.5, dtype=np.float64)
        kwargs.setdefault("healthy_z_range", (0.08, 4.0))
        super().__init__(xml_file=_XML_PATH, **kwargs)
        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._refresh_hfield_adr()

    def _refresh_hfield_adr(self):
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self._hfield_adr = self.model.hfield_adr[hid]

    def set_difficulty(self, difficulty: float):
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def sample_terrain_z(self, x: float, y: float) -> float:
        data = _bilinear_sample(self._terrain_grid, x, y)
        return terrain_height_from_data(data)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def reset_model(self):
        self._randomise_terrain()

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        terrain_z = self.sample_terrain_z(float(qpos[0]), float(qpos[1]))
        qpos[2] = terrain_z + SPAWN_CLEARANCE

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _enforce_ground_clearance(self):
        """Nudge upward only when MuJoCo reports floor penetration."""
        mujoco.mj_forward(self.model, self.data)
        lift = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= 0:
                continue
            if c.geom1 != self._floor_gid and c.geom2 != self._floor_gid:
                continue
            lift = max(lift, -c.dist + MIN_GROUND_GAP)
        if lift > 0:
            self.data.qpos[2] += lift
            mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        self._enforce_ground_clearance()
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _write_terrain(self, terrain: np.ndarray):
        self._terrain_grid = terrain
        self.model.hfield_data[self._hfield_adr : self._hfield_adr + NROW * NCOL] = (
            terrain.flatten()
        )
        mujoco.mj_forward(self.model, self.data)

    def _randomise_terrain(self):
        x = np.linspace(0, 2 * np.pi, NCOL)
        y = np.linspace(0, 2 * np.pi, NROW)
        X, Y = np.meshgrid(x, y)

        terrain = np.zeros((NROW, NCOL))
        rng = self.np_random if hasattr(self, "np_random") else np.random

        for _ in range(6):
            fx = rng.uniform(0.25, 1.0)
            fy = rng.uniform(0.25, 1.0)
            px = rng.uniform(0, 2 * np.pi)
            py = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.6, 1.0)
            terrain += amp * np.sin(fx * X + px) * np.cos(fy * Y + py)

        terrain -= terrain.min()
        terrain /= terrain.max() + 1e-8

        centered = terrain - 0.5
        centered = np.sign(centered) * (np.abs(centered * 2) ** 0.75) * 0.5
        terrain = np.clip(0.5 + centered, 0.0, 1.0)
        terrain = 0.5 + (terrain - 0.5) * self.difficulty
        terrain = np.clip(terrain, 0.0, 1.0)
        terrain = _smooth_grid(terrain, passes=2)

        self._write_terrain(terrain)


class CurriculumCallback(BaseCallback):
    """Linearly ramps terrain difficulty over training."""

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
            difficulty = self.start_difficulty + progress * (
                self.max_difficulty - self.start_difficulty
            )
            self.training_env.env_method("set_difficulty", difficulty)
            if self.verbose:
                print(
                    f"  [curriculum] timestep {self.num_timesteps:,} → difficulty {difficulty:.2f}"
                )
            self._last_update = self.num_timesteps
        return True
