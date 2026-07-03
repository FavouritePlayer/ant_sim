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
DAMAGE_CAMERA_CONFIG = {**DEFAULT_CAMERA_CONFIG, "distance": 9.0}


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
        progress_reward_weight: float = 0.0,
        forward_gate_uprightness: float = 0.75,
        forward_gate_height: float = 0.44,
        velocity_tracking_weight: float = 0.0,
        target_speed: float = 0.25,
        backward_penalty_weight: float = 0.25,
        leg_balance_weight: float = 0.0,
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
        self._progress_reward_weight = float(progress_reward_weight)
        self._forward_gate_uprightness = float(forward_gate_uprightness)
        self._forward_gate_height = float(forward_gate_height)
        self._velocity_tracking_weight = float(velocity_tracking_weight)
        self._target_speed = float(target_speed)
        self._backward_penalty_weight = float(backward_penalty_weight)
        self._leg_balance_weight = float(leg_balance_weight)
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

    def _active_leg_joint_speeds(self) -> list[float]:
        speeds = []
        for leg_id in range(N_LEGS):
            if leg_id in self._disabled_legs:
                continue
            leg_speed = 0.0
            for j in LEG_JOINTS[leg_id]:
                dadr = self.model.jnt_dofadr[j]
                leg_speed += abs(float(self.data.qvel[dadr]))
            speeds.append(leg_speed)
        return speeds

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

    def set_damage(self, disabled_legs: list[int], *, reset_tip_grace: bool = False):
        self._disabled_legs = [int(i) for i in disabled_legs if 0 <= int(i) < N_LEGS]
        self._action_mask[:] = 1.0
        for leg_id in self._disabled_legs:
            for idx in LEG_ACTUATORS[leg_id]:
                self._action_mask[idx] = 0.0
        self._apply_amputations()
        if reset_tip_grace:
            self._steps_since_reset = 0

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
        u_gate = float(
            np.clip(
                (uprightness - self._forward_gate_uprightness)
                / max(1e-6, 1.0 - self._forward_gate_uprightness),
                0.0,
                1.0,
            )
        )
        h_gate = float(
            np.clip(
                (height - self._forward_gate_height) / 0.12,
                0.0,
                1.0,
            )
        )
        gate = u_gate * h_gate

        forward_reward = x_velocity * self._forward_reward_weight * gate
        if self._progress_reward_weight > 0 and x_velocity > 0 and uprightness > 0.7:
            forward_reward += x_velocity * self._progress_reward_weight
        if x_velocity < 0:
            forward_reward += x_velocity * self._backward_penalty_weight

        move_factor = float(np.clip(abs(x_velocity) / 0.12, 0.2, 1.0))
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        upright_bonus = self._upright_reward_weight * uprightness * move_factor
        height_bonus = self._height_reward_weight * max(0.0, height - 0.48) * move_factor
        flop_penalty = self._tilt_penalty_weight * max(0.0, 0.85 - uprightness) ** 2

        velocity_tracking = 0.0
        if self._velocity_tracking_weight > 0 and uprightness > 0.65:
            err = x_velocity - self._target_speed
            velocity_tracking = -self._velocity_tracking_weight * err * err

        leg_balance = 0.0
        if self._leg_balance_weight > 0 and x_velocity > 0.05 and uprightness > 0.7:
            leg_speeds = self._active_leg_joint_speeds()
            if len(leg_speeds) >= 2:
                mean_speed = float(np.mean(leg_speeds))
                min_speed = float(np.min(leg_speeds))
                balance_ratio = min_speed / (mean_speed + 1e-6)
                leg_balance = (
                    self._leg_balance_weight
                    * balance_ratio
                    * float(np.clip(x_velocity / 0.2, 0.0, 1.0))
                )

        reward = (
            forward_reward
            + healthy_reward
            + upright_bonus
            + height_bonus
            + velocity_tracking
            + leg_balance
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
            "reward_velocity_tracking": velocity_tracking,
            "reward_leg_balance": leg_balance,
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
