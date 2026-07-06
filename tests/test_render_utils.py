"""Tests for torso-tracking demo camera."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from render_utils import LOOKAT_BLEND, clear_tracking_state, render_tracking_frame


class RenderTrackingTest(unittest.TestCase):
    def setUp(self):
        clear_tracking_state()

    def test_render_tracking_frame_updates_smoothed_lookat(self):
        env = MagicMock()
        env.unwrapped.model.body.return_value.id = 1
        env.unwrapped.data.xpos = np.zeros((2, 3))
        env.unwrapped.data.xpos[1] = np.array([1.0, 2.0, 0.5])

        viewer = MagicMock()
        viewer.cam.lookat = np.zeros(3)
        env.unwrapped.mujoco_renderer._get_viewer.return_value = viewer
        viewer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        frame1 = render_tracking_frame(env)
        self.assertEqual(frame1.shape, (480, 640, 3))

        env.unwrapped.data.xpos[1] = np.array([3.0, 2.0, 0.5])
        render_tracking_frame(env)

        expected = LOOKAT_BLEND * np.array([3.0, 2.0, 0.5]) + (1.0 - LOOKAT_BLEND) * np.array(
            [1.0, 2.0, 0.5]
        )
        np.testing.assert_allclose(viewer.cam.lookat, expected, rtol=1e-5, atol=1e-5)

    def test_clear_tracking_state_resets_per_env(self):
        env = MagicMock()
        env.unwrapped.model.body.return_value.id = 1
        env.unwrapped.data.xpos = np.zeros((2, 3))
        viewer = MagicMock()
        viewer.cam.lookat = np.zeros(3)
        env.unwrapped.mujoco_renderer._get_viewer.return_value = viewer
        viewer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        render_tracking_frame(env)
        clear_tracking_state(env)
        env.unwrapped.data.xpos[1] = np.array([5.0, 0.0, 0.5])
        render_tracking_frame(env)
        np.testing.assert_allclose(viewer.cam.lookat, np.array([5.0, 0.0, 0.5]))


if __name__ == "__main__":
    unittest.main()
