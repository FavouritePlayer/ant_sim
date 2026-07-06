import importlib
import unittest


EXPLORATORY_TERRAIN_CONFIGS = {
    "configs.ppo_terrain_finetune": "checkpoints/flat/best_model/best_model",
    "configs.ppo_terrain_boost": "checkpoints/terrain/best_model/best_model",
    "configs.ppo_terrain_balanced": "checkpoints/terrain/best_model/best_model",
    "configs.ppo_terrain_speed_refine": "checkpoints/terrain/best_model/best_model",
}


class ConfigProvenanceTest(unittest.TestCase):
    def test_exploratory_terrain_configs_use_committed_checkpoint_anchors(self):
        for module_name, expected_path in EXPLORATORY_TERRAIN_CONFIGS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.config["pretrained_path"], expected_path)
                self.assertFalse(module.config["pretrained_path"].startswith("results/"))

    def test_velocity_v2_has_no_historical_results_parent(self):
        module = importlib.import_module("configs.ppo_terrain_velocity_v2")
        self.assertNotIn("pretrained_path", module.config)
        self.assertEqual(module.config["env_id"], "TerrainAnt-v1")


if __name__ == "__main__":
    unittest.main()
