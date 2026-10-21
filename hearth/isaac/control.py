"""Three-layer control architecture for Project HEARTH (SCAFFOLD).

Closes the gap between the natural-language instruction and the low-level
joint commands.  The three layers map onto the brief's success criteria:

    Layer 2  VLA planner      5–10 Hz   instruction -> sub-task sub-goals
    Layer 1  loco-manipulation 50–100 Hz sub-goal -> whole-body / hand control
    Layer 0  locomotion        500 Hz    joint position targets + whole-body

Each layer is a plain dataclass describing its *inputs, outputs and interface*.
The actual tensors are produced by Isaac Lab at runtime; these classes only
describe the contract so the eval harness can call them without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SubgoalType(str, Enum):
    """The kind of sub-goal the VLA planner emits for the loco-manip layer."""

    NAVIGATE = "navigate"
    OPEN_ARTICULATED = "open_articulated"
    GRASP = "grasp"
    CARRY = "carry"
    FETCH = "fetch"


@dataclass(frozen=True)
class Subgoal:
    """A single sub-goal emitted by the VLA planner (Layer 2).

    Attributes:
        type: :class:`SubgoalType`.
        payload: Semantic target (object name, door handle, room name, colour).
        frame: ``"robot"`` (relative to base) or ``"world"``.
        xyz: Optional target position in the robot base frame (m).
        quatd: Optional target orientation (quaternion, w-xyz).
    """

    type: SubgoalType
    payload: str
    frame: str = "robot"
    xyz: Optional[tuple[float, float, float]] = None
    quatd: Optional[tuple[float, float, float, float]] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "frame": self.frame,
            "xyz": self.xyz,
            "quatd": self.quatd,
        }


@dataclass(frozen=True)
class WholeBodyControlConfig:
    """Layer 0 / Layer 1 whole-body controller configuration.

    Attributes:
        task_rates: Task-space task-function rates for the whole-body controller
            (``"locomotion"``, ``"loco_manip"``, ``"balance"``).
        balance_kp / balance_kd: Balance task proportional / derivative gains.
        task_space_dim: Dimension of the task-space task function.
        contact_filter: Contact-filtered body names for force sensing.
    """

    task_rates: dict[str, float] = field(
        default_factory=lambda: {"locomotion": 1.0, "loco_manip": 1.0, "balance": 10.0}
    )
    balance_kp: float = 100.0
    balance_kd: float = 20.0
    contact_filter: tuple[str, ...] = ("ankle_left", "ankle_right")


@dataclass(frozen=True)
class LocoManipulationConfig:
    """Layer 1 loco-manipulation controller configuration.

    Coordinates the torso, arms and legs so the humanoid can open doors, load
    dishwashers and grasp while standing on one or two feet.
    """

    stance_feet: int = 2
    torso_track_gain: float = 5.0
    arm_impedance: tuple[float, float, float] = (300.0, 30.0, 30.0)
    hand_grasp_force_n: float = 20.0
    contact_filter: tuple[str, ...] = ("ankle_left", "ankle_right")


@dataclass(frozen=True)
class VLAPlanterConfig:
    """Layer 2 VLA planner configuration.

    Attributes:
        model: VLA model identifier (loaded from a checkpoint at runtime).
        plan_hz: Planner invocation rate (brief: >= 5 Hz).
        max_substeps: Maximum sub-steps per instruction.
        horizon: Rollout horizon in timesteps.
    """

    model: str = "hearth_vla_7b"
    plan_hz: int = 10
    max_substeps: int = 15
    horizon: int = 128


class ControlLayers:
    """The three control layers, described as an interface contract.

    This object is the seam the :class:`~hearth.isaac.runner.IsaacLabRunner`
    uses to drive the humanoid.  It does **not** hold Isaac Lab state — it only
    documents and validates the control contract.  Concrete implementations of
    each layer are created inside Isaac Sim.
    """

    def __init__(
        self,
        robot: object,
        whole_body: WholeBodyControlConfig | None = None,
        loco_manip: LocoManipulationConfig | None = None,
        vla: VLAPlanterConfig | None = None,
    ) -> None:
        self.robot = robot
        self.whole_body = whole_body or WholeBodyControlConfig()
        self.loco_manip = loco_manip or LocoManipulationConfig()
        self.vla = vla or VLAPlanterConfig()

    def control_hz(self) -> int:
        """Lowest (fastest) control rate across the layers — the tight loop."""

        return 500  # Layer 0 locomotion, per brief §Success criteria

    def plan_hz(self) -> int:
        """VLA planner rate (brief: >= 5 Hz)."""

        return self.vla.plan_hz

    def plan(
        self,
        instruction: str,
        observed_state: dict[str, object],
    ) -> list[Subgoal]:
        """Layer 2: turn a natural-language instruction into a sub-goal list.

        This is the *interface*.  The real planner loads a VLA checkpoint and
        produces the sub-goals; here we validate the contract and surface the
        intended interface so the eval harness can call it without Isaac Sim.
        """

        if not instruction or not instruction.strip():
            raise ValueError("VLA instruction must be a non-empty string")
        if not isinstance(observed_state, dict):
            raise TypeError("observed_state must be a dict of sensor/observation fields")
        return [
            Subgoal(type=SubgoalType.FETCH, payload=instruction.strip(), frame="world")
        ]

    def __repr__(self) -> str:  # pragma: no cover - repr only
        return (
            f"ControlLayers(robot={getattr(self.robot, 'name', self.robot)!r}, "
            f"control_hz={self.control_hz()}, plan_hz={self.plan_hz()})"
        )
