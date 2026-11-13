"""No-model baseline policy runner for the HEARTH eval harness.

The harness needs *something* to score that is more meaningful than the random
:data:`~hearth.eval.benchmark.MockTaskRunner`.  ``BaselineRunner`` plugs the
symbolic planner (:mod:`hearth.isaac.planner`) and recovery controller
(:mod:`hearth.isaac.recovery`) into the ``TaskRunner`` protocol:

    1. the instruction is decomposed into sub-goals by the planner;
    2. each sub-goal is executed against fixed, hand-written capability
       priors (e.g. navigation 0.97, grasp 0.88) — the "policy";
    3. if a sub-goal fails, the recovery controller gets one chance per
       failure family (no reset) before the sub-goal is marked failed;
    4. the task succeeds only if every substep succeeds (AND, as the brief's
       long-horizon criterion demands).

This gives a deterministic, model-free lower bound: any learned VLA policy
must beat ``baseline_v0``'s success rate on the 50-task benchmark.
Everything runs on a CPU host with no Isaac Sim.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

from hearth.isaac.control import SubgoalType
from hearth.isaac.planner import SceneSummary, plan_task
from hearth.isaac.recovery import Situation, recover

from .types import SubstepResult, Task, TaskResult

# Hand-written per-subgoal success priors (the baseline "policy").  These are
# engineering guesses from the brief's first-draft targets, not learned.
DEFAULT_PRIORS: dict[str, float] = {
    SubgoalType.NAVIGATE.value: 0.97,
    SubgoalType.SEARCH.value: 0.93,
    SubgoalType.STABILIZE.value: 0.99,
    SubgoalType.GRASP.value: 0.88,
    SubgoalType.OPEN_ARTICULATED.value: 0.90,
    SubgoalType.PLACE.value: 0.93,
    SubgoalType.CARRY.value: 0.95,
    SubgoalType.RETREAT.value: 0.97,
    SubgoalType.FETCH.value: 0.88,
}

# Family-level prior for a generic benchmark step that the planner did not
# emit a specific sub-goal for (the reference benchmark uses filler substeps).
FAMILY_PRIORS: dict[str, float] = {
    "tidying": 0.88,
    "kitchen": 0.85,
    "articulated": 0.90,
    "carrying": 0.92,
    "fetch": 0.82,
}

# What kind of failure each sub-goal would exhibit, so recovery is plausible.
_FAILURE_FOR_TYPE: dict[str, str] = {
    SubgoalType.GRASP.value: "dropped_object",
    SubgoalType.OPEN_ARTICULATED.value: "door_closes",
    SubgoalType.NAVIGATE.value: "blocked_path",
    SubgoalType.CARRY.value: "dropped_object",
}


class BaselineRunner:
    """A TaskRunner whose "policy" is the planner + fixed priors + recovery."""

    def __init__(
        self,
        priors: Optional[dict[str, float]] = None,
        allow_recovery: bool = True,
        scene: Optional[SceneSummary] = None,
        name: str = "baseline_v0",
    ) -> None:
        self.priors = dict(DEFAULT_PRIORS)
        self.priors.update({f"__family:{k}": v for k, v in FAMILY_PRIORS.items()})
        if priors:
            self.priors.update(priors)
        self.allow_recovery = allow_recovery
        self.scene = scene or SceneSummary()
        self.name = name

    def _attempt(self, subgoal_type: str, rng: random.Random) -> tuple[bool, bool]:
        """Try one sub-goal once; if it fails, try recovery. -> (ok, recovered)."""
        ok = rng.random() < self.priors.get(subgoal_type, 0.5)
        if ok:
            return True, False
        if not self.allow_recovery:
            return False, False
        failure = _FAILURE_FOR_TYPE.get(subgoal_type)
        if failure is None:
            return False, False
        result = recover(Situation(failure=failure))
        if not result.recovered:
            return False, False
        # Recovery re-enters the sub-goal once more, same prior.
        ok2 = rng.random() < self.priors.get(subgoal_type, 0.5)
        return ok2, ok2

    def run_task(
        self, task: Task, seed: int, policy: Optional[dict[str, Any]] = None
    ) -> TaskResult:
        # Stable seed derived from (task.id, seed).  The built-in hash() is
        # process-randomised by PYTHONHASHSEED, so this must NOT use it: an
        # earlier version did (hash(...) & 0xFFFFFFFF) and the baseline was
        # reproducible only within a single process - the 50-task success rate
        # swung 27/31/33 across PYTHONHASHSEED values.  A cryptographic digest
        # is stable across processes and platforms (matches episode_id_for in
        # benchmark.py).
        rng = random.Random(
            int.from_bytes(hashlib.sha256(f"{task.id}:{seed}".encode()).digest(), "big")
        )
        plan = plan_task(task.instruction, self.scene, family=task.family.value)

        substeps: list[SubstepResult] = []
        task_ok = True
        recovered_count = 0
        # Score the planner's real sub-goal decomposition, NOT an index pairing
        # onto the benchmark's placeholder steps.  The old code did
        # plan.subgoals[k] against the k-th *filler* substep ("Complete subtask
        # 3 of 8"), so "Open the dishwasher" could be scored as a "stabilize"
        # step and extra substeps fell through to a generic prior.  Each result
        # below names the actual sub-goal it is scoring.
        for subgoal in plan.subgoals:
            ok, rec = self._attempt(subgoal.type.value, rng)
            recovered_count += int(rec)
            task_ok = task_ok and ok
            substeps.append(
                SubstepResult(
                    substep=f"{subgoal.type.value}:{subgoal.payload}",
                    success=ok,
                    duration_s=round(rng.uniform(1.0, 30.0), 3),
                    details={
                        "subgoal": subgoal.type.value,
                        "payload": subgoal.payload,
                        "planner_family": plan.family,
                        "recovered": rec,
                    },
                )
            )
        return TaskResult(
            task_id=task.id,
            policy_id=self.name,
            success=task_ok,
            seed=seed,
            duration_s=round(sum(s.duration_s for s in substeps), 3),
            episode_id=f"{self.name}_{task.id}_{seed}",
            substeps=substeps,
            metadata={
                "planned_subgoals": plan.num_subgoals,
                "recoveries_saved": recovered_count,
                "benchmark_substeps": task.num_substeps,
            },
        )
