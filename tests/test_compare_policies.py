import unittest
from unittest.mock import MagicMock, patch

from compare_policies import EpisodeMetrics, _comparison_metrics, compare


class ComparePoliciesMetricsTest(unittest.TestCase):
    def test_velocity_retention_is_none_for_nonpositive_flat_velocity(self):
        flat = {
            "mean_forward_velocity": -0.10,
            "mean_reward": 100.0,
        }
        terrain = {
            "mean_forward_velocity": 0.20,
            "mean_reward": 250.0,
        }

        metrics = _comparison_metrics(flat, terrain)

        self.assertIsNone(metrics["terrain_velocity_retention_pct"])
        self.assertAlmostEqual(metrics["terrain_reward_retention_pct"], 250.0, places=6)

    def test_compare_aggregates_episode_metrics_into_expected_schema(self):
        dummy_env = MagicMock()
        episodes = [
            EpisodeMetrics(
                seed=0,
                reward=100.0,
                steps=400,
                forward_distance=40.0,
                mean_forward_velocity=0.20,
                fell=True,
            ),
            EpisodeMetrics(
                seed=0,
                reward=300.0,
                steps=800,
                forward_distance=96.0,
                mean_forward_velocity=0.30,
                fell=False,
            ),
            EpisodeMetrics(
                seed=1,
                reward=140.0,
                steps=500,
                forward_distance=60.0,
                mean_forward_velocity=0.24,
                fell=False,
            ),
            EpisodeMetrics(
                seed=1,
                reward=280.0,
                steps=900,
                forward_distance=126.0,
                mean_forward_velocity=0.35,
                fell=False,
            ),
        ]

        with (
            patch("compare_policies.PPO.load", side_effect=[object(), object()]),
            patch("compare_policies.make_terrain_env", return_value=dummy_env),
            patch("compare_policies.rollout", side_effect=episodes),
        ):
            results = compare(
                flat_run="checkpoints/flat",
                terrain_run="checkpoints/terrain",
                difficulty=0.4,
                seeds=[0, 1],
                max_steps=1000,
            )

        self.assertEqual(results["difficulty"], 0.4)
        self.assertEqual(results["seeds"], [0, 1])
        self.assertEqual(results["flat_run"], "checkpoints/flat")
        self.assertEqual(results["terrain_run"], "checkpoints/terrain")
        self.assertEqual(len(results["flat"]["episodes"]), 2)
        self.assertEqual(len(results["terrain"]["episodes"]), 2)

        flat_summary = results["flat"]["summary"]
        terrain_summary = results["terrain"]["summary"]
        self.assertEqual(flat_summary["n_episodes"], 2)
        self.assertEqual(terrain_summary["n_episodes"], 2)
        self.assertAlmostEqual(flat_summary["mean_reward"], 120.0, places=6)
        self.assertAlmostEqual(terrain_summary["mean_reward"], 290.0, places=6)
        self.assertAlmostEqual(flat_summary["fall_rate"], 0.5, places=6)
        self.assertAlmostEqual(terrain_summary["fall_rate"], 0.0, places=6)
        self.assertAlmostEqual(results["terrain_reward_retention_pct"], 241.6666666667, places=6)
        self.assertAlmostEqual(results["terrain_velocity_retention_pct"], 147.7272727273, places=6)


if __name__ == "__main__":
    unittest.main()
