"""Task registry and validation for the HEARTH eval harness.

Provides the reference benchmark builders (delegating to ``families``) plus
validation of a benchmark against the brief's structural constraints:
    * each task has 5-15 substeps (objective 4)
    * every task belongs to an in-scope family
    * task ids are unique

Importable without Isaac Sim.
"""

from __future__ import annotations

from typing import Callable

from .families import build_50_task_benchmark
from .types import Benchmark, Task, TaskFamily

# Registry of named benchmark builders.
BENCHMARK_BUILDERS: dict[str, Callable[[], Benchmark]] = {
    "hearth_50": build_50_task_benchmark,
}


def get_builder(name: str) -> Callable[[], Benchmark]:
    if name not in BENCHMARK_BUILDERS:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARK_BUILDERS)}"
        )
    return BENCHMARK_BUILDERS[name]


def valid_family(name: str) -> bool:
    """Return True if ``name`` is an in-scope task family."""
    try:
        TaskFamily(name)
        return True
    except ValueError:
        return False


def validate_benchmark(benchmark: Benchmark) -> list[str]:
    """Validate a benchmark against the brief's structural constraints.

    Returns a list of human-readable problems; empty means valid.
    """
    problems: list[str] = []
    seen_ids: set[str] = set()

    for task in benchmark.tasks:
        if task.id in seen_ids:
            problems.append(f"duplicate task id: {task.id}")
        seen_ids.add(task.id)

        if not valid_family(task.family.value):
            problems.append(f"task {task.id}: unknown family {task.family!r}")
            continue

        n = task.num_substeps
        if not (5 <= n <= 15):
            problems.append(
                f"task {task.id}: {n} substeps, brief requires 5-15"
            )

    if not benchmark.tasks:
        problems.append("benchmark contains no tasks")
    return problems


def is_valid_benchmark(benchmark: Benchmark) -> bool:
    return not validate_benchmark(benchmark)
