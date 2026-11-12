"""Isaac Lab integration for Project HEARTH (SCAFFOLD — importable, integration-ready).

This package is the integration point for the **Isaac Sim / Isaac Lab** runtime
(the only layer that needs a GPU + Isaac Sim installed).  Every module here is
written to be *importable without Isaac Sim present*, so the rest of the repo
and CI stay green on a machine without it.

It plugs into the rest of HEARTH in two ways:

1. **Scene loading.**  ``hearth.scene.generate_scene`` + ``write_usd`` produce a
   ``.usd`` apartment; :func:`hearth.isaac.runner.IsaacLabRunner.load_scene`
   loads that exact file into an Isaac Lab ``Scene``.

2. **Task execution.**  :class:`hearth.isaac.runner.IsaacLabRunner` implements the
   ``TaskRunner`` protocol from ``hearth.eval.benchmark`` (``run_task(task, seed,
   policy) -> TaskResult``), so a real Isaac Lab run drops straight into the eval
   harness with no other code change.

Control architecture (see :mod:`hearth.isaac.control`):

    Layer 0  Locomotion + whole-body control  (500 Hz)
    Layer 1  Loco-manipulation (door/drawer/grasp controllers)
    Layer 2  VLA planner (natural-language instruction -> sub-goals)

Only Isaac Sim / Isaac Lab packages are imported inside functions, so importing
this module never requires them.

Public API (importable without Isaac Sim):
    HumanoidRobotSpec
    ActuatorModel
    ControlLayers
    IsaacLabRunner
    load_scene
"""

from .robot import HumanoidRobotSpec, ActuatorModel
from .control import ControlLayers, Subgoal, SubgoalType
from .runner import IsaacLabRunner, load_scene
from .planner import SceneSummary, PlanResult, plan_task, plan_many, detect_family
from .recovery import (
    Situation,
    RecoveryResult,
    FailureType,
    RecoveryOutcome,
    recover,
    classify_failure,
)
from .grasp import (
    GripperLimits,
    GraspCandidate,
    generate_grasp_candidates,
    rank_grasps,
    best_grasp,
)

__all__ = [
    "HumanoidRobotSpec",
    "ActuatorModel",
    "ControlLayers",
    "IsaacLabRunner",
    "load_scene",
    "Subgoal",
    "SubgoalType",
    "SceneSummary",
    "PlanResult",
    "plan_task",
    "plan_many",
    "detect_family",
    "Situation",
    "RecoveryResult",
    "FailureType",
    "RecoveryOutcome",
    "recover",
    "classify_failure",
    "GripperLimits",
    "GraspCandidate",
    "generate_grasp_candidates",
    "rank_grasps",
    "best_grasp",
]
