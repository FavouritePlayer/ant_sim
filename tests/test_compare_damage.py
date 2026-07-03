import unittest

from compare_damage import _comparison_metrics


class CompareDamageMetricsTest(unittest.TestCase):
    def test_velocity_retention_is_none_for_nonpositive_flat_velocity(self):
        flat = {
            "mean_forward_velocity": -0.19,
            "mean_reward": 44.0,
            "fall_rate": 1.0,
            "mean_steps": 20.6,
        }
        damage = {
            "mean_forward_velocity": 0.31,
            "mean_reward": 2148.0,
            "fall_rate": 0.2,
            "mean_steps": 809.0,
        }

        metrics = _comparison_metrics(flat, damage)

        self.assertIsNone(metrics["velocity_retention_pct"])
        self.assertAlmostEqual(metrics["velocity_gain_mps"], 0.50, places=6)
        self.assertAlmostEqual(metrics["fall_rate_reduction_pct"], 80.0, places=6)
        self.assertAlmostEqual(metrics["episode_length_gain_steps"], 788.4, places=6)

    def test_velocity_retention_is_computed_for_positive_flat_velocity(self):
        flat = {
            "mean_forward_velocity": 0.20,
            "mean_reward": 100.0,
            "fall_rate": 0.6,
            "mean_steps": 100.0,
        }
        damage = {
            "mean_forward_velocity": 0.30,
            "mean_reward": 250.0,
            "fall_rate": 0.1,
            "mean_steps": 300.0,
        }

        metrics = _comparison_metrics(flat, damage)

        self.assertAlmostEqual(metrics["velocity_retention_pct"], 150.0, places=6)
        self.assertAlmostEqual(metrics["reward_retention_pct"], 250.0, places=6)
        self.assertAlmostEqual(metrics["reward_gain"], 150.0, places=6)


if __name__ == "__main__":
    unittest.main()
