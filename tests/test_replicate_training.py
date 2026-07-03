import os
import tempfile
import unittest

from replicate_training import (
    _best_or_final_model_prefix,
    build_manifest,
)


class ReplicateTrainingTest(unittest.TestCase):
    def test_build_manifest_contains_requested_profiles_and_seeds(self):
        manifest = build_manifest(
            ["terrain_canonical", "damage_canonical"],
            [0, 2],
            timesteps=12345,
            tag="smoke",
        )

        self.assertEqual(manifest["timesteps_override"], 12345)
        self.assertEqual(len(manifest["requested_profiles"]), 2)
        self.assertEqual(len(manifest["replications"]), 4)
        self.assertEqual(
            {(item["profile"], item["seed"]) for item in manifest["replications"]},
            {
                ("terrain_canonical", 0),
                ("terrain_canonical", 2),
                ("damage_canonical", 0),
                ("damage_canonical", 2),
            },
        )

    def test_best_or_final_model_prefix_prefers_best_then_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            best_dir = os.path.join(tmpdir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            best_zip = os.path.join(best_dir, "best_model.zip")
            with open(best_zip, "wb") as f:
                f.write(b"best")
            final_zip = os.path.join(tmpdir, "final_model.zip")
            with open(final_zip, "wb") as f:
                f.write(b"final")

            self.assertEqual(
                _best_or_final_model_prefix(tmpdir),
                os.path.join(tmpdir, "best_model", "best_model"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            final_zip = os.path.join(tmpdir, "final_model.zip")
            with open(final_zip, "wb") as f:
                f.write(b"final")

            self.assertEqual(
                _best_or_final_model_prefix(tmpdir),
                os.path.join(tmpdir, "final_model"),
            )


if __name__ == "__main__":
    unittest.main()
