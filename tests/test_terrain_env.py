import unittest

import gymnasium as gym
import numpy as np

from envs import register
from envs.terrain_ant import BOUNDARY_MARGIN, HFIELD_X_HALF, SPAWN_CLEARANCE


class TerrainEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register()

    def test_zero_difficulty_produces_flat_heightfield(self):
        env = gym.make("TerrainAnt-v0", difficulty=0.0)
        try:
            env.reset(seed=0)
            ant = env.unwrapped
            self.assertTrue(np.allclose(ant._terrain_grid, 0.5))
        finally:
            env.close()

    def test_reset_respects_difficulty_range_and_spawn_clearance(self):
        env = gym.make("TerrainAnt-v0", difficulty=0.9, difficulty_range=(0.2, 0.3))
        try:
            obs, _ = env.reset(seed=1)
            ant = env.unwrapped

            self.assertGreaterEqual(ant.difficulty, 0.2)
            self.assertLessEqual(ant.difficulty, 0.3)
            self.assertEqual(obs.shape, env.observation_space.shape)

            terrain_z = ant._local_terrain_z(float(ant.data.qpos[0]), float(ant.data.qpos[1]))
            self.assertGreaterEqual(float(ant.data.qpos[2]), terrain_z + SPAWN_CLEARANCE)
            self.assertGreater(float(np.std(ant._terrain_grid)), 0.0)
        finally:
            env.close()

    def test_step_terminates_when_ant_leaves_heightfield_bounds(self):
        env = gym.make("TerrainAnt-v0", difficulty=0.4)
        try:
            env.reset(seed=2)
            ant = env.unwrapped

            qpos = ant.data.qpos.copy()
            qvel = ant.data.qvel.copy()
            qpos[0] = HFIELD_X_HALF - BOUNDARY_MARGIN + 0.1
            ant.set_state(qpos, qvel)

            _, _, terminated, truncated, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
        finally:
            env.close()

    def test_velocity_env_appends_target_speed_and_reports_tracking_reward(self):
        env = gym.make("TerrainAnt-v1", target_speed_range=(0.35, 0.35), difficulty=0.4)
        try:
            obs, _ = env.reset(seed=3)
            ant = env.unwrapped

            self.assertEqual(obs.shape, env.observation_space.shape)
            self.assertAlmostEqual(float(obs[-1]), 0.35, places=6)

            next_obs, _, _, _, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            self.assertEqual(next_obs.shape, env.observation_space.shape)
            self.assertIn("target_speed", info)
            self.assertIn("velocity_tracking_reward", info)
            self.assertAlmostEqual(info["target_speed"], 0.35, places=6)
            self.assertAlmostEqual(float(next_obs[-1]), 0.35, places=6)
            self.assertEqual(ant.observation_space.shape, env.observation_space.shape)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
