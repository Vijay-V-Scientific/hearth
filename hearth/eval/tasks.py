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
from .types import Benchmark, TaskFamily

# Registry of named benchmark builders.
BENCHMARK_BUILDERS: dict[str, Callable[[], Benchmark]] = {
    "hearth_50": build_50_task_benchmark,
}


def get_builder(name: str) -> Callable[[], Benchmark]:
    if name not in BENCHMARK_BUILDERS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARK_BUILDERS)}")
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
    blocking, advisory = _classify(benchmark)
    return blocking + advisory


def structural_problems(benchmark: Benchmark) -> list[str]:
    """Only the problems that must block a run: duplicate ids, unknown family,
    or an empty benchmark. The 5-15 sub-step range is a brief target, not a
    hard gate, so it is reported but does not appear here.
    """
    return _classify(benchmark)[0]


def _classify(benchmark: Benchmark) -> "tuple[list[str], list[str]]":
    """Split validation findings into (blocking, advisory) problems."""
    blocking: list[str] = []
    advisory: list[str] = []
    seen_ids: set[str] = set()

    for task in benchmark.tasks:
        if task.id in seen_ids:
            blocking.append(f"duplicate task id: {task.id}")
        seen_ids.add(task.id)

        if not valid_family(task.family.value):
            blocking.append(f"task {task.id}: unknown family {task.family!r}")
            continue

        n = task.num_substeps
        if not (5 <= n <= 15):
            advisory.append(f"task {task.id}: {n} substeps, brief recommends 5-15")

    if not benchmark.tasks:
        blocking.append("benchmark contains no tasks")
    return blocking, advisory


def is_valid_benchmark(benchmark: Benchmark) -> bool:
    return not validate_benchmark(benchmark)
