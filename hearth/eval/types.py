"""Data model for the Project HEARTH evaluation harness.

Everything here is importable without Isaac Sim installed. The classes define
what a *task*, a *task family*, a *benchmark* and a *result* are, plus the
success metrics that family members are scored by.

The eval harness targets objective 4 of the brief:
    "70% success on 50 benchmark tasks of 5-15 sub-steps from a
     natural-language instruction."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskFamily(str, Enum):
    """The five task families in scope for Project HEARTH."""

    TIDYING = "tidying"
    KITCHEN = "kitchen"
    ARTICULATED = "articulated"
    CARRYING = "carrying"
    FETCH = "fetch"

    @property
    def description(self) -> str:
        return {
            TaskFamily.TIDYING: "Pick up clothes, toys and books and put them where they belong.",
            TaskFamily.KITCHEN: "Clear a table, load a dishwasher, put groceries in the fridge.",
            TaskFamily.ARTICULATED: (
                "Open doors, drawers, cabinets, microwave and light switches."
            ),
            TaskFamily.CARRYING: "Move a box, laundry basket or chair between rooms.",
            TaskFamily.FETCH: (
                "Bring a requested object, including search when it is not in view."
            ),
        }[self]


@dataclass
class Substep:
    """A single sub-goal within a multi-step task."""

    description: str
    success_condition: str
    timeout_s: Optional[float] = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "success_condition": self.success_condition,
            "timeout_s": self.timeout_s,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Substep":
        return cls(
            description=d["description"],
            success_condition=d["success_condition"],
            timeout_s=d.get("timeout_s"),
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class Task:
    """A single benchmark task, described from a natural-language instruction."""

    id: str
    family: TaskFamily
    instruction: str
    substeps: list[Substep]
    difficulty: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_substeps(self) -> int:
        return len(self.substeps)

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.substeps) or 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family.value,
            "instruction": self.instruction,
            "difficulty": self.difficulty,
            "substeps": [s.to_dict() for s in self.substeps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(
            id=d["id"],
            family=TaskFamily(d["family"]),
            instruction=d["instruction"],
            difficulty=d.get("difficulty", "medium"),
            substeps=[Substep.from_dict(s) for s in d.get("substeps", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass
class Benchmark:
    """A named collection of tasks (e.g. the 50-task long-horizon benchmark)."""

    id: str
    version: int
    tasks: list[Task]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        families = {t.family for t in self.tasks}
        self.families: list[TaskFamily] = sorted(families, key=lambda f: f.value)

    @property
    def avg_substeps(self) -> float:
        return sum(t.num_substeps for t in self.tasks) / max(len(self.tasks), 1)

    def tasks_by_family(self, family: TaskFamily) -> list[Task]:
        return [t for t in self.tasks if t.family == family]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "families": [f.value for f in self.families],
            "num_tasks": len(self.tasks),
            "avg_substeps": round(self.avg_substeps, 2),
            "tasks": [t.to_dict() for t in self.tasks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Benchmark":
        return cls(
            id=d["id"],
            version=d.get("version", 1),
            tasks=[Task.from_dict(t) for t in d.get("tasks", [])],
            description=d.get("description", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SubstepResult:
    """Outcome of a single substep within a run."""

    substep: str
    success: bool
    duration_s: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "substep": self.substep,
            "success": self.success,
            "duration_s": round(self.duration_s, 3),
            "details": self.details,
        }


@dataclass
class TaskResult:
    """The outcome of running one Task on one policy for one episode."""

    task_id: str
    policy_id: str
    success: bool
    seed: int
    duration_s: float
    episode_id: str
    substeps: list[SubstepResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        """Fraction of weighted substeps completed successfully."""
        num = sum(s.weight for s in self.substeps if s.success)
        return num / self.task.total_weight if hasattr(self, "task") else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy_id": self.policy_id,
            "success": self.success,
            "seed": self.seed,
            "duration_s": round(self.duration_s, 3),
            "episode_id": self.episode_id,
            "substeps": [s.to_dict() for s in self.substeps],
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Aggregated results across all tasks/policies for a benchmark run."""

    benchmark_id: str
    policy_id: str
    results: list[TaskResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # TaskResult needs to know its task to compute weighted_score;
        # attach it lazily when the benchmark is available.
        self._task_index: dict[str, Task] = {}

    def attach_tasks(self, benchmark: Benchmark) -> None:
        self._task_index = {t.id: t for t in benchmark.tasks}
        for r in self.results:
            r.task = self._task_index.get(r.task_id, _PlaceholderTask(r.task_id))

    @property
    def num_tasks(self) -> int:
        return len(self.results)

    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def rate_by_family(self, family: TaskFamily) -> float:
        group = [r for r in self.results if _family_of(r.task_id, self._task_index) == family]
        if not group:
            return 0.0
        return sum(1 for r in group if r.success) / len(group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "policy_id": self.policy_id,
            "num_tasks": self.num_tasks,
            "success_rate": round(self.success_rate(), 4),
            "by_family": {
                f.value: round(self.rate_by_family(f), 4) for f in TaskFamily
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }


class _PlaceholderTask:
    """Stand-in so TaskResult.total_weight works before attach_tasks()."""

    def __init__(self, task_id: str) -> None:
        self.id = task_id


def _family_of(task_id: str, index: dict[str, Task]) -> Optional[TaskFamily]:
    t = index.get(task_id)
    return t.family if t else None
