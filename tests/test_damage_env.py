import unittest

import gymnasium as gym
import numpy as np

from envs import register
from envs.damage_ant import LEG_ACTUATORS


class DamageEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register()

    def test_fixed_damage_masks_actuators_and_reports_contacts(self):
        env = gym.make(
            "DamageAnt-v0",
            fixed_disabled_legs=[1],
            min_disabled_legs=1,
            max_disabled_legs=1,
        )
        try:
            env.reset(seed=0)
            ant = env.unwrapped

            self.assertEqual(ant._disabled_legs, [1])
            for actuator_idx in LEG_ACTUATORS[1]:
                self.assertEqual(ant._action_mask[actuator_idx], 0.0)

            active_indices = [
                actuator_idx
                for leg_id, actuator_pair in enumerate(LEG_ACTUATORS)
                if leg_id != 1
                for actuator_idx in actuator_pair
            ]
            for actuator_idx in active_indices:
                self.assertEqual(ant._action_mask[actuator_idx], 1.0)

            _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            self.assertEqual(info["disabled_legs"], [1])
            self.assertEqual(len(info["foot_contacts"]), 4)
        finally:
            env.close()

    def test_set_damage_can_restore_and_reapply_legs(self):
        env = gym.make("DamageAnt-v0", fixed_disabled_legs=[1])
        try:
            env.reset(seed=1)
            ant = env.unwrapped
            ant._steps_since_reset = 123

            ant.set_damage([], reset_tip_grace=True)
            self.assertEqual(ant._disabled_legs, [])
            self.assertEqual(ant._steps_since_reset, 0)
            self.assertTrue(np.allclose(ant._action_mask, 1.0))

            ant.set_damage([2], reset_tip_grace=False)
            self.assertEqual(ant._disabled_legs, [2])
            for actuator_idx in LEG_ACTUATORS[2]:
                self.assertEqual(ant._action_mask[actuator_idx], 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
