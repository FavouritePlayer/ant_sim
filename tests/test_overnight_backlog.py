import json
import unittest
from pathlib import Path

from run_overnight_backlog import REPLICATION_ASSETS, summarize_replications


class OvernightBacklogSummaryTest(unittest.TestCase):
    def test_summarize_replications_writes_expected_artifacts(self):
        summary = summarize_replications()

        json_path = REPLICATION_ASSETS / "summary.json"
        md_path = REPLICATION_ASSETS / "SUMMARY.md"
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())

        with json_path.open() as f:
            payload = json.load(f)

        self.assertIn("entries", payload)
        self.assertIn("aggregate", payload)
        self.assertGreaterEqual(len(payload["entries"]), 1)
        self.assertEqual(payload["aggregate"]["damage_canonical"]["n_seeds"], 3)
        self.assertEqual(payload["aggregate"]["terrain_canonical"]["n_seeds"], 3)

        # Returned summary should match the on-disk artifact.
        self.assertEqual(summary["aggregate"], payload["aggregate"])

        md_text = md_path.read_text()
        self.assertIn("Replicated Training Sweep Summary", md_text)
        self.assertIn("damage_canonical", md_text)


if __name__ == "__main__":
    unittest.main()
