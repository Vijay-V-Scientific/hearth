"""ROS 2 bridge between the VLA planner and the low-level controller.

The VLA planner (Layer 2) emits :class:`Subgoal` messages at >=5 Hz. A
ROS 2 bridge relays them to the low-level controller (Layer 0/1) running
on the robot's onboard compute. The actual DDS transport needs a ROS 2
install; this module defines the typed message contract and a config
loader so the interface is testable without ROS 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


class BridgeError(ValueError):
    """Raised on invalid bridge configuration or messages."""


@dataclass
class SubgoalMessage:
    """A typed sub-goal message crossing the ROS 2 boundary.

    Attributes:
        seq: Monotonic sequence number.
        subgoal_type: One of ``xyz_position``, ``xyzw_quat``,
            ``grip_width``, ``subgoal_id``.
        position: Position target in the base frame (metres).
        orientation: Orientation as a ``w,x,y,z`` quaternion.
        grip_width: Desired grips width (metres); None if no hand.
        confidence: Planner confidence in [0, 1].
    """

    seq: int
    subgoal_type: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    grip_width: float | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "subgoal_type": self.subgoal_type,
            "position": tuple(self.position),
            "orientation": tuple(self.orientation),
            "grip_width": self.grip_width,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SubgoalMessage":
        try:
            return cls(
                seq=int(data["seq"]),
                subgoal_type=str(data["subgoal_type"]),
                position=tuple(float(x) for x in data["position"]),
                orientation=tuple(float(x) for x in data["orientation"]),
                grip_width=None if data.get("grip_width") is None
                else float(data["grip_width"]),
                confidence=float(data.get("confidence", 1.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeError(f"invalid subgoal message: {exc}") from exc

    def validate(self) -> "SubgoalMessage":
        """Validate fields and return self. Raises BridgeError on error."""
        if not isinstance(self.seq, int) or self.seq < 0:
            raise BridgeError("seq must be a non-negative integer")
        valid_types = {
            "xyz_position",
            "xyzw_quat",
            "grip_width",
            "subgoal_id",
        }
        if self.subgoal_type not in valid_types:
            raise BridgeError(
                f"subgoal_type must be one of {valid_types}, got {self.subgoal_type!r}"
            )
        if len(self.position) != 3:
            raise BridgeError("position must have 3 components")
        if len(self.orientation) != 4:
            raise BridgeError("orientation must have 4 components (w,x,y,z)")
        if not 0.0 <= self.confidence <= 1.0:
            raise BridgeError("confidence must be in [0, 1]")
        return self


def build_subgoal_message(
    seq: int,
    subgoal_type: str,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    grip_width: float | None = None,
    confidence: float = 1.0,
) -> SubgoalMessage:
    """Build and validate a :class:`SubgoalMessage`."""
    return SubgoalMessage(
        seq=seq,
        subgoal_type=subgoal_type,
        position=tuple(float(x) for x in position),
        orientation=tuple(float(x) for x in orientation),
        grip_width=grip_width,
        confidence=float(confidence),
    ).validate()


@dataclass
class BridgeConfig:
    """Configuration for the ROS 2 bridge.

    Attributes:
        topic_planner: Topic the VLA planner publishes sub-goals on.
        topic_controller: Topic the low-level controller subscribes to.
        frame_id: Base frame id for published transforms.
        message_rate_hz: Expected sub-goal publish rate.
        timeout_s: Timeout for a single message round trip.
    """

    topic_planner: str = "/planner/subgoals"
    topic_controller: str = "/controller/subgoals"
    frame_id: str = "base_link"
    message_rate_hz: float = 5.0
    timeout_s: float = 0.5

    def to_dict(self) -> dict:
        return {
            "topic_planner": self.topic_planner,
            "topic_controller": self.topic_controller,
            "frame_id": self.frame_id,
            "message_rate_hz": self.message_rate_hz,
            "timeout_s": self.timeout_s,
        }


@dataclass
class Ros2BridgeConfig:
    """Full bridge configuration including transport details."""

    bridge: BridgeConfig
    qos_profile: str = "reliable"
    use_sim_time: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bridge": self.bridge.to_dict(),
            "qos_profile": self.qos_profile,
            "use_sim_time": self.use_sim_time,
            "extra": dict(self.extra),
        }


def load_bridge_config(path: str | Path) -> Ros2BridgeConfig:
    """Load a :class:`Ros2BridgeConfig` from a YAML file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise BridgeError(f"bridge config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise BridgeError(f"invalid YAML in bridge config: {exc}") from exc
    if not isinstance(raw, dict):
        raise BridgeError("bridge config must be a YAML mapping")

    bridge_raw = raw.get("bridge", {})
    if not isinstance(bridge_raw, dict):
        raise BridgeError("'bridge' must be a mapping")
    bridge = BridgeConfig(
        topic_planner=str(bridge_raw.get("topic_planner", "/planner/subgoals")),
        topic_controller=str(
            bridge_raw.get("topic_controller", "/controller/subgoals")
        ),
        frame_id=str(bridge_raw.get("frame_id", "base_link")),
        message_rate_hz=float(bridge_raw.get("message_rate_hz", 5.0)),
        timeout_s=float(bridge_raw.get("timeout_s", 0.5)),
    )
    return Ros2BridgeConfig(
        bridge=bridge,
        qos_profile=str(raw.get("qos_profile", "reliable")),
        use_sim_time=bool(raw.get("use_sim_time", False)),
        extra={k: v for k, v in raw.items() if k not in ("bridge", "qos_profile", "use_sim_time")},
    )
