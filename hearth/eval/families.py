"""Task-family definitions and builders for the HEARTH eval harness.

Each family knows how to build its benchmark tasks from a small set of
parameters (a "recipe") and how to describe the success metric that its
members are scored by.  These builders turn a YAML task spec into a concrete
``Task`` object; the YAML files in ``configs/eval`` are the source of truth
for the benchmark, and this module is the reference implementation.

Importable without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import Benchmark, Substep, Task, TaskFamily


@dataclass
class FamilySpec:
    """Static description of a task family and its success metric."""

    family: TaskFamily
    metric: str
    description: str
    substep_template: str


FAMILY_METRICS: dict[TaskFamily, FamilySpec] = {
    TaskFamily.TIDYING: FamilySpec(
        TaskFamily.TIDYING,
        "object placed in correct target location",
        "Pick up clothes, toys and books and put them where they belong.",
        "move {obj} to {target}",
    ),
    TaskFamily.KITCHEN: FamilySpec(
        TaskFamily.KITCHEN,
        "item placed in correct appliance slot",
        "Clear a table, load a dishwasher, put groceries in the fridge.",
        "place {obj} into {appliance}",
    ),
    TaskFamily.ARTICULATED: FamilySpec(
        TaskFamily.ARTICULATED,
        "articulated object reached target state (open/closed/on)",
        "Open doors, drawers, cabinets, microwave and light switches.",
        "{action} {obj} to {state}",
    ),
    TaskFamily.CARRYING: FamilySpec(
        TaskFamily.CARRYING,
        "carried object released at the destination",
        "Move a box, laundry basket or chair between rooms.",
        "carry {obj} from {src} to {dst}",
    ),
    TaskFamily.FETCH: FamilySpec(
        TaskFamily.FETCH,
        "requested object delivered to the requester",
        "Bring a requested object, including search when it is not in view.",
        "bring {obj} to {requester}",
    ),
}


def family_metric(family: TaskFamily) -> str:
    """Return the success metric string for a task family."""
    return FAMILY_METRICS[family].metric


def build_tasks_from_recipe(
    family: TaskFamily,
    n: int,
    substeps_per_task: tuple[int, int] = (5, 15),
    start_id: int = 1,
    difficulty: str = "medium",
    extra_substep: Callable[[int, int], Substep] | None = None,
) -> list[Task]:
    """Build ``n`` tasks of a family following the brief's 5-15 substep range.

    ``substeps_per_task`` is an inclusive ``(low, high)`` range; the number of
    substeps for each task is drawn deterministically from ``n`` so the same
    recipe always yields the same benchmark.

    ``extra_substep`` is an optional hook that returns a family-specific
    ``Substep`` for task index ``i`` and substep index ``j``; if omitted a
    generic substep is used.
    """
    low, high = substeps_per_task
    tasks: list[Task] = []
    for i in range(n):
        # Deterministic pseudo-random substep count within [low, high].
        span = high - low + 1
        n_sub = low + ((i * 7 + 3) % span)
        substeps: list[Substep] = []
        for j in range(n_sub):
            if extra_substep is not None:
                substeps.append(extra_substep(i, j))
            else:
                substeps.append(
                    Substep(
                        description=f"Complete subtask {j + 1} of {n_sub}",
                        success_condition=family_metric(family),
                        timeout_s=60.0 + j * 10.0,
                    )
                )
        tasks.append(
            Task(
                id=f"{family.value}_{start_id + i:04d}",
                family=family,
                instruction=f"[{family.value}] {FAMILY_METRICS[family].description} "
                f"(task {i + 1})",
                substeps=substeps,
                difficulty=difficulty,
            )
        )
    return tasks


def build_50_task_benchmark() -> Benchmark:
    """Construct the 50-task long-horizon benchmark from the brief.

    Distribution mirrors objective 4's target across the five families:
    kitchen 14, fetch 12, articulated 10, tidying 7, carrying 7 = 50.
    """
    recipes = [
        (TaskFamily.KITCHEN, 14),
        (TaskFamily.FETCH, 12),
        (TaskFamily.ARTICULATED, 10),
        (TaskFamily.TIDYING, 7),
        (TaskFamily.CARRYING, 7),
    ]
    tasks: list[Task] = []
    for family, n in recipes:
        tasks.extend(build_tasks_from_recipe(family, n))
    return Benchmark(
        id="hearth_50",
        version=1,
        tasks=tasks,
        description="50-task long-horizon household benchmark (brief objective 4).",
        metadata={"objective": 4, "target_success_rate": 0.70},
    )
