"""Benchmark loading and execution for the HEARTH eval harness.

``load_benchmark`` turns a YAML file (or a registered name) into a
``Benchmark``.  ``run_benchmark`` executes a benchmark with any object that
implements the ``TaskRunner`` protocol.

A ``TaskRunner`` is the integration point with a real simulator (Isaac Lab)
or a physics-free policy.  It is deliberately tiny so the harness stays
importable without Isaac Sim:

    class TaskRunner(Protocol):
        def run_task(self, task, seed, policy) -> TaskResult: ...

``MockTaskRunner`` simulates a policy's capability so the harness can be
exercised end-to-end on this machine (no Isaac Sim, no GPU required).

Importable without Isaac Sim.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import yaml

from .tasks import BENCHMARK_BUILDERS, get_builder
from .types import (
    Benchmark,
    SubstepResult,
    Task,
    TaskResult,
)


@runtime_checkable
class TaskRunner(Protocol):
    """Runs one task on a policy and returns its result."""

    def run_task(
        self, task: Task, seed: int, policy: Optional[dict[str, Any]] = None
    ) -> TaskResult: ...


@dataclass
class MockPolicy:
    """A policy described by a per-family success capability in [0, 1]."""

    name: str
    capability: dict[str, float] = field(default_factory=dict)

    def capability_for(self, family: str) -> float:
        return self.capability.get(family, self.capability.get("default", 0.0))


class MockTaskRunner:
    """Physics-free runner used for testing the harness itself.

    Success is drawn from the policy's capability per family, then a task is
    successful only if *every* substep succeeds (an AND over substeps, matching
    how a long-horizon task fails on its first missed sub-goal).
    """

    def __init__(self, policy: MockPolicy | None = None) -> None:
        self.policy = policy or MockPolicy(name="mock", capability={"default": 0.5})

    def run_task(
        self, task: Task, seed: int, policy: Optional[dict[str, Any]] = None
    ) -> TaskResult:
        rng = random.Random(seed)
        cap = self.policy.capability_for(task.family.value)

        substeps: list[SubstepResult] = []
        task_ok = True
        for k, sub in enumerate(task.substeps):
            ok = rng.random() < cap
            task_ok = task_ok and ok
            substeps.append(
                SubstepResult(
                    substep=sub.description,
                    success=ok,
                    duration_s=round(rng.uniform(1.0, sub.timeout_s or 60.0), 3),
                    details={"cap": round(cap, 3), "family": task.family.value},
                )
            )
        return TaskResult(
            task_id=task.id,
            policy_id=self.policy.name,
            success=task_ok,
            seed=seed,
            duration_s=round(sum(s.duration_s for s in substeps), 3),
            episode_id=f"mock_{task.id}_{seed}",
            substeps=substeps,
        )


def load_benchmark(source: str | Path) -> Benchmark:
    """Load a benchmark from a YAML file or a registered builder name.

    ``"hearth_50"`` (or any key in :data:`BENCHMARK_BUILDERS`) returns the
    reference benchmark built in code.  Otherwise ``source`` must be a path to
    a YAML file with a ``tasks`` list.
    """
    name = str(source).strip()
    if name in BENCHMARK_BUILDERS:
        return get_builder(name)()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"benchmark file not found: {path}")

    data = yaml.safe_load(path.read_text())
    benchmark = Benchmark.from_dict(data)
    if not benchmark.tasks:
        raise ValueError(f"benchmark '{path}' contains no tasks")
    return benchmark


def run_benchmark(
    benchmark: Benchmark,
    runner: TaskRunner,
    seeds: dict[str, int] | None = None,
    policy: Optional[dict[str, Any]] = None,
) -> TaskResult:
    """Run every task in ``benchmark`` with ``runner``; return the first result.

    Returns a single :class:`TaskResult` for the first task so callers can
    chain; use :func:`collect_benchmark_results` for the full list.
    """
    seed = seeds.get(benchmark.id, 0) if seeds else 0
    return runner.run_task(benchmark.tasks[0], seed, policy)


def collect_benchmark_results(
    benchmark: Benchmark,
    runner: TaskRunner,
    seeds: dict[str, int] | None = None,
    policy: Optional[dict[str, Any]] = None,
) -> list[TaskResult]:
    """Run every task in ``benchmark`` and return all results."""
    results: list[TaskResult] = []
    for task in benchmark.tasks:
        seed = seeds.get(task.id, 0) if seeds else 0
        results.append(runner.run_task(task, seed, policy))
    return results


def episode_id_for(task_id: str, seed: int, policy_id: str) -> str:
    """Deterministic, collision-resistant episode id."""
    h = hashlib.sha1(f"{policy_id}:{task_id}:{seed}".encode()).hexdigest()[:12]
    return f"{policy_id}_{task_id}_{seed}_{h}"
