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
        dof: Total actuated degrees of freedom (brief: 25–45).
        base_to_com_m: Base-to-centre-of-mass height.
        foot_radius_m: Equivalent foot contact radius for contact filtering.
        joints: Ordered joint specs (index-aligned with ``dof``).
        hands: Dexterous hands (left/right).
        actuator: Shared actuator model applied to all motor joints.
        control_hz: Low-level control loop rate (brief: 500 Hz).
        action_space: Description of the low-level action vector layout.
    """

    name: str = "unitree_g1"
    height_m: float = 1.8
    mass_kg: float = 30.0
    dof: int = 29
    base_to_com_m: float = 0.9
    foot_radius_m: float = 0.08
    joints: tuple[JointSpec, ...] = ()
    hands: tuple[HandSpec, ...] = ()
    actuator: ActuatorModel = field(default_factory=ActuatorModel)
    control_hz: int = 500
    action_space: str = (
        "29 joint position targets (torso+legs+arms) + 8 hand DoF (right), "
        "12 DoF loco-manipulation (torso, arms, legs, hands)"
    )

    @classmethod
    def unitree_g1(cls) -> "HumanoidRobotSpec":
        """Build the default Unitree G1 spec (29 DoF, one dexterous right hand)."""

        return cls(
            name="unitree_g1",
            height_m=1.8,
            mass_kg=30.0,
            dof=29,
            base_to_com_m=0.9,
            foot_radius_m=0.08,
            joints=(
                JointSpec("left_hip_yaw", 0, limit_low_rad=-0.43, limit_high_rad=0.43,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("left_hip_roll", 1, limit_low_rad=-0.21, limit_high_rad=0.21,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("left_hip_pitch", 2, limit_low_rad=-0.52, limit_high_rad=2.36,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("left_knee", 3, limit_low_rad=-2.36, limit_high_rad=0.8,
                          effort_limit=216.0, velocity_limit=13.0),
                JointSpec("ankle_left", 4, joint_type="prismatic", limit_low_rad=-0.52,
                          limit_high_rad=0.52, effort_limit=30.0, velocity_limit=25.0),
                JointSpec("right_hip_yaw", 5, limit_low_rad=-0.43, limit_high_rad=0.43,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("right_hip_roll", 6, limit_low_rad=-0.21, limit_high_rad=0.21,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("right_hip_pitch", 7, limit_low_rad=-0.52, limit_high_rad=2.36,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("right_knee", 8, limit_low_rad=-2.36, limit_high_rad=0.8,
                          effort_limit=216.0, velocity_limit=13.0),
                JointSpec("ankle_right", 9, joint_type="prismatic", limit_low_rad=-0.52,
                          limit_high_rad=0.52, effort_limit=30.0, velocity_limit=25.0),
                JointSpec("torso", 10, limit_low_rad=-0.52, limit_high_rad=0.52,
                          effort_limit=144.0, velocity_limit=20.0),
                JointSpec("left_shoulder_pitch", 11, limit_low_rad=-1.57, limit_high_rad=1.57,
                          effort_limit=44.0, velocity_limit=20.0),
                JointSpec("left_shoulder_roll", 12, limit_low_rad=-1.57, limit_high_rad=1.57,
                          effort_limit=44.0, velocity_limit=20.0),
                JointSpec("left_elbow", 13, limit_low_rad=-0.21, limit_high_rad=2.61,
                          effort_limit=44.0, velocity_limit=20.0),
                JointSpec("right_shoulder_pitch", 14, limit_low_rad=-1.57, limit_high_rad=1.57,
                          effort_limit=44.0, velocity_limit=20.0),
                JointSpec("right_shoulder_roll", 15, limit_low_rad=-1.57, limit_high_rad=1.57,
                          effort_limit=44.0, velocity_limit=20.0),
                JointSpec("right_elbow", 16, limit_low_rad=-0.21, limit_high_rad=2.61,
                          effort_limit=44.0, velocity_limit=20.0),
                # Dexterous right hand: 4 DoF per finger x 5 fingers = 20, but
                # G1 ships with a 2-DoF parallel gripper by default.  The
                # dexterous hand is the week-2 selection; this spec models the
                # underactuated 5-finger variant (8 usable DoF shown below).
                JointSpec("right_hand_thumb", 17, joint_type="prismatic", limit_low_rad=-0.26,
                          limit_high_rad=0.26, effort_limit=6.0, velocity_limit=50.0),
                JointSpec("right_hand_index", 18, joint_type="prismatic", limit_low_rad=-0.47,
                          limit_high_rad=0.47, effort_limit=6.0, velocity_limit=50.0),
                JointSpec("right_hand_middle", 19, joint_type="prismatic", limit_low_rad=-0.47,
                          limit_high_rad=0.47, effort_limit=6.0, velocity_limit=50.0),
                JointSpec("right_hand_ring", 20, joint_type="prismatic", limit_low_rad=-0.47,
                          limit_high_rad=0.47, effort_limit=6.0, velocity_limit=50.0),
                JointSpec("right_hand_pinky", 21, joint_type="prismatic", limit_low_rad=-0.47,
                          limit_high_rad=0.47, effort_limit=6.0, velocity_limit=50.0),
            ),
            hands=(
                HandSpec("right", finger_count=5, dof_per_finger=4, grasp_mode="underactuated"),
            ),
            control_hz=500,
        )

    @property
    def num_dof(self) -> int:
        return self.dof

    @property
    def hand_dof(self) -> int:
        return sum(h.total_dof for h in self.hands)

    @property
    def contact_bodies(self) -> tuple[str, ...]:
        """Bodies whose contact forces are used for contact filtering."""

        return ("ankle_left", "ankle_right")

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
