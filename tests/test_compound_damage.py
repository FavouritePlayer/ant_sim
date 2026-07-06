"""Tests for compound leg-routing damage policy."""

import unittest

from compound_damage import COMPOUND_ROUTING, CompoundDamageRouter


class CompoundRoutingTest(unittest.TestCase):
    def test_routing_table_covers_all_legs(self):
        self.assertEqual(set(COMPOUND_ROUTING.keys()), {0, 1, 2, 3})

    def test_policy_for_legs_returns_expected_checkpoint(self):
        router = CompoundDamageRouter(
            "checkpoints/damage",
            "results/ppo_damageant_v0_1783295334_seed0_replicate_damage_crossleg_crossleg_gait",
        )
        self.assertEqual(router.policy_for_legs([0])[0], "crossleg")
        self.assertEqual(router.policy_for_legs([1])[0], "specialist")
        self.assertEqual(router.policy_for_legs([2])[0], "crossleg")
        self.assertEqual(router.policy_for_legs([3])[0], "crossleg")

    def test_rejects_multi_leg_amputation(self):
        router = CompoundDamageRouter(
            "checkpoints/damage",
            "results/ppo_damageant_v0_1783295334_seed0_replicate_damage_crossleg_crossleg_gait",
        )
        with self.assertRaises(ValueError):
            router.policy_for_legs([0, 1])


if __name__ == "__main__":
    unittest.main()
