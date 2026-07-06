"""Shared demo rendering with smoothed torso tracking."""

from __future__ import annotations

import mujoco
import numpy as np

_LOOKAT: dict[int, np.ndarray] = {}
LOOKAT_BLEND = 0.55
DEMO_DISTANCE = 6.5
DEMO_AZIMUTH = 90.0
DEMO_ELEVATION = -15.0


def clear_tracking_state(env=None) -> None:
    if env is None:
        _LOOKAT.clear()
    else:
        _LOOKAT.pop(id(env), None)


def render_tracking_frame(env) -> np.ndarray:
    """Render with a free camera locked to the ant torso (smoothed lookat)."""
    renderer = env.unwrapped.mujoco_renderer
    viewer = renderer._get_viewer(render_mode="rgb_array")
    cam = viewer.cam
    torso_id = env.unwrapped.model.body("torso").id
    target = env.unwrapped.data.xpos[torso_id].copy()

    key = id(env)
    prev = _LOOKAT.get(key)
    if prev is None:
        _LOOKAT[key] = target.copy()
    else:
        _LOOKAT[key] = LOOKAT_BLEND * target + (1.0 - LOOKAT_BLEND) * prev

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = _LOOKAT[key]
    cam.distance = DEMO_DISTANCE
    cam.azimuth = DEMO_AZIMUTH
    cam.elevation = DEMO_ELEVATION
    return viewer.render(render_mode="rgb_array", camera_id=-1)
