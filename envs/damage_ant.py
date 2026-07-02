"""Flat Ant-v5 with leg amputation and upright-focused reward shaping."""

import os

import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv, DEFAULT_CAMERA_CONFIG

# Actuator order matches Gymnasium Ant-v5 (see ant_v5.py action table).
LEG_ACTUATORS = (
    (0, 1),  # leg 0 — back right  (hip_4, ankle_4)
    (4, 5),  # leg 1 — front right (hip_2, ankle_2)
    (2, 3),  # leg 2 — front left  (hip_1, ankle_1)
    (6, 7),  # leg 3 — back left   (hip_3, ankle_3)
)
LEG_JOINTS = (
    (7, 8),  # back right
    (3, 4),  # front right
    (1, 2),  # front left
    (5, 6),  # back left
)
LEG_GEOMS = (
    ("aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom"),
    ("aux_2_geom", "right_leg_geom", "right_ankle_geom"),
    ("aux_1_geom", "left_leg_geom", "left_ankle_geom"),
    ("aux_3_geom", "back_leg_geom", "third_ankle_geom"),
)
LEG_BODIES = (
    (11, 12, 13),
    (5, 6, 7),
    (2, 3, 4),
    (8, 9, 10),
)
LEG_LABELS = ("back right", "front right", "front left", "back left")
N_LEGS = len(LEG_ACTUATORS)
TORSO_BODY_ID = 1

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
DAMAGE_XML = os.path.join(_ASSET_DIR, "ant_damage.xml")
DAMAGE_CAMERA_CONFIG = {**DEFAULT_CAMERA_CONFIG, "distance": 5.0}


class DamageAntEnv(AntEnv):
    """
    Ant-v5 with leg amputation and reward terms that favour staying upright on 3 legs.

    Amputated legs: invisible, no ground contact, near-zero body mass, joints heavily damped.
    """

    def __init__(
        self,
        max_disabled_legs: int = 1,
        min_disabled_legs: int = 1,
        fixed_disabled_legs: list[int] | None = None,
        upright_reward_weight: float = 2.0,
        height_reward_weight: float = 1.5,
        tilt_penalty_weight: float = 1.0,
        terminate_when_tipped: bool = True,
        min_uprightness: float = 0.45,
        **kwargs,
    ):
        self.max_disabled_legs = int(max(0, min(max_disabled_legs, N_LEGS - 1)))
        self.min_disabled_legs = int(max(0, min(min_disabled_legs, self.max_disabled_legs)))
        self.fixed_disabled_legs = fixed_disabled_legs
        self._upright_reward_weight = float(upright_reward_weight)
        self._height_reward_weight = float(height_reward_weight)
        self._tilt_penalty_weight = float(tilt_penalty_weight)
        self._terminate_when_tipped = bool(terminate_when_tipped)
        self._min_uprightness = float(min_uprightness)
        self._tip_grace_steps = int(kwargs.pop("tip_grace_steps", 80))
        self._steps_since_reset = 0
        self._disabled_legs: list[int] = []
        self._action_mask = np.ones(8, dtype=np.float64)
        self._geom_defaults: dict[int, dict] = {}
        self._body_mass_defaults: dict[int, float] = {}
        self._dof_damping_defaults: dict[int, float] = {}

        if "xml_file" not in kwargs:
            kwargs["xml_file"] = DAMAGE_XML
        if "default_camera_config" not in kwargs:
            kwargs["default_camera_config"] = DAMAGE_CAMERA_CONFIG
        kwargs.setdefault("healthy_z_range", (0.28, 1.0))
        kwargs.setdefault("reset_noise_scale", 0.05)
        super().__init__(**kwargs)
        self._cache_physics_defaults()

    def _cache_physics_defaults(self):
        for names in LEG_GEOMS:
            for name in names:
                gid = self.model.geom(name).id
                self._geom_defaults[gid] = {
                    "rgba": self.model.geom_rgba[gid].copy(),
                    "contype": int(self.model.geom_contype[gid]),
                    "conaffinity": int(self.model.geom_conaffinity[gid]),
                }
        for bodies in LEG_BODIES:
            for bid in bodies:
                self._body_mass_defaults[bid] = float(self.model.body_mass[bid])
        for joints in LEG_JOINTS:
            for j in joints:
                dadr = self.model.jnt_dofadr[j]
                self._dof_damping_defaults[dadr] = float(self.model.dof_damping[dadr])

    def _torso_uprightness(self) -> float:
        z_axis = self.data.xmat[TORSO_BODY_ID].reshape(3, 3)[:, 2]
        return float(np.clip(z_axis[2], -1.0, 1.0))

    def _restore_leg(self, leg_id: int):
        for name in LEG_GEOMS[leg_id]:
            gid = self.model.geom(name).id
            defaults = self._geom_defaults[gid]
            self.model.geom_rgba[gid] = defaults["rgba"]
            self.model.geom_contype[gid] = defaults["contype"]
            self.model.geom_conaffinity[gid] = defaults["conaffinity"]
        for bid in LEG_BODIES[leg_id]:
            self.model.body_mass[bid] = self._body_mass_defaults[bid]
        for j in LEG_JOINTS[leg_id]:
            dadr = self.model.jnt_dofadr[j]
            self.model.dof_damping[dadr] = self._dof_damping_defaults[dadr]

    def _amputate_leg(self, leg_id: int):
        for name in LEG_GEOMS[leg_id]:
            gid = self.model.geom(name).id
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
            self.model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0])
        for bid in LEG_BODIES[leg_id]:
            self.model.body_mass[bid] = 1e-8
        for j in LEG_JOINTS[leg_id]:
            dadr = self.model.jnt_dofadr[j]
            self.model.dof_damping[dadr] = 500.0

    def _freeze_amputated_joints(self):
        for leg_id in self._disabled_legs:
            for j in LEG_JOINTS[leg_id]:
                dadr = self.model.jnt_dofadr[j]
                self.data.qvel[dadr] = 0.0

    def _apply_amputations(self):
        for leg_id in range(N_LEGS):
            self._restore_leg(leg_id)
        for leg_id in self._disabled_legs:
            self._amputate_leg(leg_id)
        self._freeze_amputated_joints()

    def set_damage(self, disabled_legs: list[int]):
        self._disabled_legs = [int(i) for i in disabled_legs if 0 <= int(i) < N_LEGS]
        self._action_mask[:] = 1.0
        for leg_id in self._disabled_legs:
            for idx in LEG_ACTUATORS[leg_id]:
                self._action_mask[idx] = 0.0
        self._apply_amputations()

    def _sample_disabled_legs(self) -> list[int]:
        if self.fixed_disabled_legs is not None:
            return list(self.fixed_disabled_legs)
        lo = self.min_disabled_legs
        hi = self.max_disabled_legs
        n_disable = int(self.np_random.integers(lo, hi + 1))
        if n_disable == 0:
            return []
        return (
            self.np_random.choice(N_LEGS, size=n_disable, replace=False).tolist()
        )

    def reset_model(self):
        self.set_damage(self._sample_disabled_legs())
        self._steps_since_reset = 0
        super().reset_model()
        self._freeze_amputated_joints()
        if self._disabled_legs:
            self.data.qpos[2] = max(float(self.data.qpos[2]), 0.62)
            self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def _get_rew(self, x_velocity: float, action):
        uprightness = self._torso_uprightness()
        height = float(self.data.qpos[2])
        upright_gate = float(np.clip((uprightness - 0.75) / 0.25, 0.0, 1.0))
        height_gate = float(np.clip((height - 0.44) / 0.12, 0.0, 1.0))
        gate = upright_gate * height_gate

        forward_reward = x_velocity * self._forward_reward_weight * gate
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        upright_bonus = self._upright_reward_weight * uprightness
        height_bonus = self._height_reward_weight * max(0.0, height - 0.48)
        flop_penalty = self._tilt_penalty_weight * max(0.0, 0.85 - uprightness) ** 2

        reward = (
            forward_reward
            + healthy_reward
            + upright_bonus
            + height_bonus
            - ctrl_cost
            - contact_cost
            - flop_penalty
        )
        reward_info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_upright": upright_bonus,
            "reward_height": height_bonus,
            "reward_tilt_penalty": flop_penalty,
            "torso_uprightness": uprightness,
            "upright_gate": gate,
        }
        return reward, reward_info

    def step(self, action):
        self._freeze_amputated_joints()
        masked = np.asarray(action, dtype=np.float64) * self._action_mask
        obs, reward, terminated, truncated, info = super().step(masked)
        self._freeze_amputated_joints()
        self._steps_since_reset += 1
        info["disabled_legs"] = list(self._disabled_legs)
        info["torso_uprightness"] = self._torso_uprightness()
        tipped = info["torso_uprightness"] < self._min_uprightness
        if (
            self._terminate_when_tipped
            and tipped
            and self._steps_since_reset > self._tip_grace_steps
        ):
            terminated = True
        return obs, reward, terminated, truncated, info
