"""Isaac Lab runner: the integration point for HEARTH's sim runtime (SCAFFOLD).

:class:`IsaacLabRunner` implements the :class:`hearth.eval.benchmark.TaskRunner`
protocol, so a real Isaac Lab run drops straight into the eval harness with no
other code change.  It is importable without Isaac Sim: every Isaac Lab import
is deferred to inside a function so the import never requires the GPU runtime.

Data flow:

    scene.usd  ->  load_scene()  ->  Isaac Lab Scene
        ->  ControlLayers (locomotion / loco-manip / VLA)
        ->  run_task(task, seed, policy) -> TaskResult   (matches TaskRunner)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from hearth.eval.benchmark import TaskResult

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing isaac at runtime
    from hearth.eval.types import Task
    from hearth.isaac.control import ControlLayers
    from hearth.isaac.robot import HumanoidRobotSpec


@dataclass
class IsaacLabRunner:
    """Drives a humanoid in Isaac Lab and implements ``TaskRunner``.

    Attributes:
        scene_usd: Path to the ``.usd`` apartment to load.
        robot: Humanoid robot spec (see :mod:`hearth.isaac.robot`).
        control: The three :class:`~hearth.isaac.control.ControlLayers`.
        sim_dt_s: Physics timestep in seconds (default 1/500, matches Layer 0).
    """

    scene_usd: str
    robot: object
    control: object
    sim_dt_s: float = 0.002

    # ---- TaskRunner protocol (see hearth/eval/benchmark.py) ---------------
    def run_task(
        self,
        task: object,
        seed: int,
        policy: object | None = None,
    ) -> TaskResult:
        """Run one benchmark task in Isaac Lab.

        This is the :class:`hearth.eval.benchmark.TaskRunner` entry point.  The
        Isaac Lab implementation loads the scene, steps the sim, and returns a
        :class:`~hearth.eval.types.TaskResult`.  The body is deferred to Isaac
        Sim so the module imports cleanly without it; the contract and return
        type are fixed here.
        """

        _require_isaac_runtime()

        # The real implementation would:
        #   1. start the SimulationApp
        #   2. load self.scene_usd into a new Scene
        #   3. spawn self.robot from the USD asset
        #   4. run self.control for `task.max_substeps` sub-steps
        #   5. evaluate success per task.families / task.metrics
        #   6. return TaskResult(...)
        raise NotImplementedError(
            "Isaac Lab runtime requires a GPU + Isaac Sim 5.x container. "
            "The TaskRunner contract is fixed; the body runs only inside the "
            "isaacsim/isaaclab runtime."
        )

    # ---- Scene loading----
    def load_scene(self) -> object:
        """Load ``self.scene_usd`` into an Isaac Lab Scene.

        Deferred to Isaac Sim.  Returns the Isaac Lab ``Scene`` object at
        runtime; the contract is documented here.
        """

        from isaacsim import SimulationApp  # noqa: F401
        import isaaclab.sim as sim_utils  # noqa: F401

        # The real implementation would:
        #   1. start the SimulationApp
        #   2. build a sim_utils.SceneConfiguration pointing at self.scene_usd
        #   3. create sim_utils.create_scene(...) with that config
        raise NotImplementedError(
            "Scene loading requires the isaacsim/isaaclab runtime."
        )


def _require_isaac_runtime() -> None:
    """Raise a clear, consistent error when Isaac Sim / Isaac Lab is absent.

    Every deferred Isaac Lab import funnels through here so the failure mode is
    always the same :class:`NotImplementedError` (never a raw
    :class:`ModuleNotFoundError`), regardless of which package is missing.
    """

    raise NotImplementedError(
        "Isaac Lab runtime requires a GPU + Isaac Sim 5.x container. "
        "This body runs only inside the isaacsim/isaaclab runtime, not on host."
    )


def load_scene(scene_usd: str, robot: object, control: object) -> IsaacLabRunner:
    """Convenience constructor for an :class:`IsaacLabRunner`.

    Args:
        scene_usd: Path to the ``.usd`` apartment (from
            :func:`hearth.scene.generate_scene`).
        robot: :class:`~hearth.isaac.robot.HumanoidRobotSpec`.
        control: :class:`~hearth.isaac.control.ControlLayers`.

    Returns:
        A ready :class:`IsaacLabRunner` implementing ``TaskRunner``.
    """

    return IsaacLabRunner(scene_usd=scene_usd, robot=robot, control=control)
