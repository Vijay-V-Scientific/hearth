"""Humanoid robot specification for Project HEARTH (SCAFFOLD).

Defines the canonical full-size humanoid platform HEARTH targets, its
dexterous hands, and the learned-actuator model that closes most of the
sim-to-real gap at the joint level.

This module is pure-Python and importable without Isaac Sim.  The
:class:`ActuatorModel` documents exactly what Isaac Lab
(``isaaclab.robots.AutomotiveActuator`` / a custom ``ActuatorBase`` subclass)
would consume at runtime.

See Project brief §"Robot model and simulation fidelity", Actuators section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class JointSpec:
    """A single actuated joint on the humanoid.

    Attributes:
        name: Joint name as it appears in the vendor URDF/USD.
        dof_index: DOF index within the robot's ``dof`` dimension.
        joint_type: ``revolute``, ``prismatic`` or ``fixed``.
        limit_low_rad / limit_high_rad: Joint position limits (radians or m).
        effort_limit: Maximum joint effort (N or N·m).
        velocity_limit: Maximum joint velocity (rad/s or m/s).
        transmission: ``motor`` or ``direct_drive``.
    """

    name: str
    dof_index: int
    joint_type: str = "revolute"
    limit_low_rad: float = 0.0
    limit_high_rad: float = 0.0
    effort_limit: float = 0.0
    velocity_limit: float = 0.0
    transmission: str = "motor"


@dataclass(frozen=True)
class HandSpec:
    """A dexterous five-finger hand.

    Attributes:
        name: Hand identifier (e.g. ``"right"``).
        finger_count: Number of fingers (5 for the HEARTH hand).
        dof_per_finger: DoF count per finger (4 = underactuated dexterous).
        total_dof: Product of the above; drives the action-space width.
        grasp_mode: ``underactuated`` or ``fully_actuated``.
    """

    name: str
    finger_count: int = 5
    dof_per_finger: int = 4
    grasp_mode: str = "underactuated"

    @property
    def total_dof(self) -> int:
        return self.finger_count * self.dof_per_finger


@dataclass(frozen=True)
class ActuatorModel:
    """Learned actuator network model (MLP over position error + velocity history).

    Replaces an ideal PD controller at each joint with an MLP fitted to real
    motor logs.  This is the primary sim-to-real lever for torque/trajectory
    fidelity (brief §Actuators).  The network weights are *not* shipped here —
    they are trained from motor logs and loaded at runtime by Isaac Lab.

    Attributes:
        kind: ``"learned_mlp"`` (default) or ``"ideal_pd"`` (fallback / test).
        hidden_sizes: MLP hidden layer sizes.
        position_error_scale / velocity_scale: Input normalisation gains.
        history_len: Timesteps of velocity history fed to the MLP.
        latency_ms: Explicit motor latency injected in simulation (ms).
        weights_path: Path to the trained weights (``.pt``).  ``None`` until
            trained; a real run requires this to exist.
    """

    kind: str = "learned_mlp"
    hidden_sizes: tuple[int, ...] = (64, 64)
    position_error_scale: float = 1.0
    velocity_scale: float = 1.0
    history_len: int = 8
    latency_ms: float = 1.0
    weights_path: Optional[str] = None

    def state(self) -> dict[str, object]:
        """Serialisable description for config / checkpoint metadata."""

        return {
            "kind": self.kind,
            "hidden_sizes": list(self.hidden_sizes),
            "position_error_scale": self.position_error_scale,
            "velocity_scale": self.velocity_scale,
            "history_len": self.history_len,
            "latency_ms": self.latency_ms,
            "weights_path": self.weights_path,
            "weights_loaded": self.weights_path is not None,
        }


@dataclass(frozen=True)
class HumanoidRobotSpec:
    """Canonical full-size humanoid platform HEARTH targets.

    The brief lists candidates (Unitree G1/H1 with dexterous hands, or
    Fourier GR-1).  This spec is the *interface* the rest of HEARTH depends on;
    the concrete vendor model is chosen in week 2 based on URDF quality, hand
    availability and actuator documentation (brief §Scope).  Defaults match the
    Unitree G1, which is the leading candidate at time of writing.

    Attributes:
        name: Vendor/model identifier.
        height_m: Standing height in metres (brief: 1.3–1.8 m).
        mass_kg: Standing mass in kg (brief: 25–45 kg).
        dof: Total actuated degrees of freedom (matches ``len(joints)``).
        base_to_com_m: Base-to-centre-of-mass height.
        foot_radius_m: Equivalent foot contact radius for contact filtering.
        joints: Ordered joint specs (index-aligned with ``dof``).
        hands: Dexterous hands (left/right).
        actuator: Shared actuator model applied to all motor joints.
        control_hz: Low-level control loop rate (brief: 500 Hz).
        action_space: Description of the low-level action vector layout.
    """

    name: str = "unitree_g1"
    height_m: float = 1.32
    mass_kg: float = 35.0
    dof: int = 23
    base_to_com_m: float = 0.62
    foot_radius_m: float = 0.08
    joints: tuple[JointSpec, ...] = ()
    hands: tuple[HandSpec, ...] = ()
    actuator: ActuatorModel = field(default_factory=ActuatorModel)
    control_hz: int = 500
    action_space: str = (
        "23 joint position targets (12 legs + 1 waist + 10 arms) + "
        "hand DoF (separate end-effector, not counted in body DoF)"
    )

    @classmethod
    def unitree_g1(cls) -> "HumanoidRobotSpec":
        """Build the Unitree G1 body spec (23 DoF, one dexterous right hand).

        The 23 body DoF match Unitree's G1 basic version "total degrees of
        freedom (excl. end-effector)": 6 per leg (hip pitch/roll/yaw, knee,
        ankle pitch, ankle roll - both revolute, the ankle is a parallel
        mechanism driven in PR mode), 1 waist yaw, and 5 per arm (shoulder
        pitch/roll/yaw, elbow, wrist roll).  Joint limits follow the official
        G1 developer joint table.  The dexterous hand is an end-effector and is
        modelled separately via ``hands`` (not counted in the 23 body DoF).
        Sources: unitree.com/en/g1, support.unitree.com G1 developer guide.
        """

        def leg(side: str, base: int) -> list[JointSpec]:
            return [
                JointSpec(
                    f"{side}_hip_pitch",
                    base,
                    limit_low_rad=-2.53,
                    limit_high_rad=2.88,
                    effort_limit=88.0,
                    velocity_limit=32.0,
                ),
                JointSpec(
                    f"{side}_hip_roll",
                    base + 1,
                    limit_low_rad=-0.5236,
                    limit_high_rad=0.5236,
                    effort_limit=139.0,
                    velocity_limit=20.0,
                ),
                JointSpec(
                    f"{side}_hip_yaw",
                    base + 2,
                    limit_low_rad=-2.7576,
                    limit_high_rad=2.7576,
                    effort_limit=88.0,
                    velocity_limit=32.0,
                ),
                JointSpec(
                    f"{side}_knee",
                    base + 3,
                    limit_low_rad=-0.087267,
                    limit_high_rad=2.8798,
                    effort_limit=139.0,
                    velocity_limit=20.0,
                ),
                JointSpec(
                    f"{side}_ankle_pitch",
                    base + 4,
                    limit_low_rad=-0.87267,
                    limit_high_rad=0.5236,
                    effort_limit=35.0,
                    velocity_limit=37.0,
                ),
                JointSpec(
                    f"{side}_ankle_roll",
                    base + 5,
                    limit_low_rad=-0.2618,
                    limit_high_rad=0.2618,
                    effort_limit=35.0,
                    velocity_limit=37.0,
                ),
            ]

        def arm(side: str, base: int) -> list[JointSpec]:
            return [
                JointSpec(
                    f"{side}_shoulder_pitch",
                    base,
                    limit_low_rad=-3.0892,
                    limit_high_rad=2.6704,
                    effort_limit=25.0,
                    velocity_limit=37.0,
                ),
                JointSpec(
                    f"{side}_shoulder_roll",
                    base + 1,
                    limit_low_rad=-1.5882,
                    limit_high_rad=2.2515,
                    effort_limit=25.0,
                    velocity_limit=37.0,
                ),
                JointSpec(
                    f"{side}_shoulder_yaw",
                    base + 2,
                    limit_low_rad=-2.618,
                    limit_high_rad=2.618,
                    effort_limit=25.0,
                    velocity_limit=37.0,
                ),
                JointSpec(
                    f"{side}_elbow",
                    base + 3,
                    limit_low_rad=-1.0472,
                    limit_high_rad=2.0944,
                    effort_limit=25.0,
                    velocity_limit=37.0,
                ),
                JointSpec(
                    f"{side}_wrist_roll",
                    base + 4,
                    limit_low_rad=-1.9722,
                    limit_high_rad=1.9722,
                    effort_limit=25.0,
                    velocity_limit=37.0,
                ),
            ]

        joints: list[JointSpec] = []
        joints += leg("left", 0)
        joints += leg("right", 6)
        joints.append(
            JointSpec(
                "waist_yaw",
                12,
                limit_low_rad=-2.618,
                limit_high_rad=2.618,
                effort_limit=35.0,
                velocity_limit=32.0,
            )
        )
        joints += arm("left", 13)
        joints += arm("right", 18)

        return cls(
            name="unitree_g1",
            height_m=1.32,
            mass_kg=35.0,
            dof=len(joints),
            base_to_com_m=0.62,
            foot_radius_m=0.08,
            joints=tuple(joints),
            hands=(
                HandSpec("right", finger_count=5, dof_per_finger=4, grasp_mode="underactuated"),
            ),
            control_hz=500,
        )

    @classmethod
    def fourier_gr1(cls) -> "HumanoidRobotSpec":
        """Build a Fourier GR-1 spec (37 DoF, two underactuated hands).

        A second candidate platform from the brief. The GR-1 has more joints
        than the G1; the joint tree is generated programmatically so that
        ``dof`` always equals ``len(joints)`` (the invariant the tests check).
        """

        leg_joints = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
        arm_joints = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        joints: list[JointSpec] = []
        idx = 0
        for side in ("left", "right"):
            for name in leg_joints:
                joints.append(
                    JointSpec(
                        f"{side}_{name}",
                        idx,
                        limit_low_rad=-2.36,
                        limit_high_rad=2.36,
                        effort_limit=200.0,
                        velocity_limit=20.0,
                    )
                )
                idx += 1
        joints.append(
            JointSpec(
                "torso",
                idx,
                limit_low_rad=-0.52,
                limit_high_rad=0.52,
                effort_limit=200.0,
                velocity_limit=20.0,
            )
        )
        idx += 1
        for side in ("left", "right"):
            for name in arm_joints:
                jt = "prismatic" if name.startswith("wrist") else "revolute"
                joints.append(
                    JointSpec(
                        f"{side}_{name}",
                        idx,
                        joint_type=jt,
                        limit_low_rad=-2.0,
                        limit_high_rad=2.0,
                        effort_limit=100.0,
                        velocity_limit=20.0,
                    )
                )
                idx += 1
        # 5 fingers x 2 hands, underactuated prismatic contact DoF.
        for side in ("left", "right"):
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                joints.append(
                    JointSpec(
                        f"{side}_hand_{finger}",
                        idx,
                        joint_type="prismatic",
                        limit_low_rad=-0.47,
                        limit_high_rad=0.47,
                        effort_limit=6.0,
                        velocity_limit=50.0,
                    )
                )
                idx += 1
        return cls(
            name="fourier_gr1",
            height_m=1.65,
            mass_kg=65.0,
            dof=len(joints),
            base_to_com_m=0.85,
            foot_radius_m=0.09,
            joints=tuple(joints),
            hands=(
                HandSpec("left", finger_count=5, dof_per_finger=4, grasp_mode="underactuated"),
                HandSpec("right", finger_count=5, dof_per_finger=4, grasp_mode="underactuated"),
            ),
            control_hz=500,
        )

    @classmethod
    def default(cls) -> "HumanoidRobotSpec":
        """The platform used when the caller does not pick one."""

        return cls.unitree_g1()

    @property
    def num_dof(self) -> int:
        return self.dof

    @property
    def hand_dof(self) -> int:
        return sum(h.total_dof for h in self.hands)

    @property
    def contact_bodies(self) -> tuple[str, ...]:
        """Bodies whose contact forces are used for contact filtering."""

        return ("left_ankle_roll_link", "right_ankle_roll_link")

    def state(self) -> dict[str, object]:
        """Serialisable description for config / checkpoint metadata."""

        return {
            "name": self.name,
            "height_m": self.height_m,
            "mass_kg": self.mass_kg,
            "dof": self.dof,
            "base_to_com_m": self.base_to_com_m,
            "foot_radius_m": self.foot_radius_m,
            "num_joints": len(self.joints),
            "hand_count": len(self.hands),
            "hand_dof": self.hand_dof,
            "control_hz": self.control_hz,
            "actuator": self.actuator.state(),
        }
