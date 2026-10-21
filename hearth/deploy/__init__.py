"""Deployment and sim-to-real calibration for Project HEARTH.

This package contains the pieces needed to move a policy trained in
Isaac Sim to a physical humanoid, and to keep running it over a ROS 2
bridge. The heavy runtime (Isaac Sim, ROS 2, device drivers) lives
outside this environment; every entry point raises ``NotImplementedError``
until a GPU + Isaac Sim 5.x container is available.

Two importable, testable units:

* :mod:`hearth.deploy.calibration` — sim2real domain randomization and
  motor-latency / COM / friction calibration helpers.
* :mod:`hearth.deploy.ros2_bridge` — a typed ROS 2 bridge between the
  VLA planner and the low-level controller.
"""

from .calibration import (
    CalibrationResult,
    CalibrationState,
    DomainRandomization,
    MotorLatencyModel,
    compute_domain_randomization,
    load_calibration_state,
    save_calibration_state,
)
from .ros2_bridge import (
    BridgeConfig,
    Ros2BridgeConfig,
    SubgoalMessage,
    build_subgoal_message,
    load_bridge_config,
)

__all__ = [
    "CalibrationResult",
    "CalibrationState",
    "DomainRandomization",
    "MotorLatencyModel",
    "compute_domain_randomization",
    "load_calibration_state",
    "save_calibration_state",
    "BridgeConfig",
    "Ros2BridgeConfig",
    "SubgoalMessage",
    "build_subgoal_message",
    "load_bridge_config",
]
