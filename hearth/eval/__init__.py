"""Evaluation harness for Project HEARTH (SCAFFOLD — importable, runnable).

Defines what a task, task family, benchmark and result are; builds the
reference 50-task benchmark; and runs it with any TaskRunner.

Public API:
    load_benchmark(source) -> Benchmark
    collect_benchmark_results(benchmark, runner) -> list[TaskResult]
    build_benchmark_result(benchmark, results, policy_id) -> BenchmarkResult
    MockTaskRunner / MockPolicy  (physics-free runner for testing)
    render_markdown / render_json / render_csv / write_report
    validate_benchmark

Importable without Isaac Sim installed.
"""

from .types import (
    Benchmark,
    BenchmarkResult,
    Substep,
    SubstepResult,
    Task,
    TaskFamily,
    TaskResult,
)
from .families import FAMILY_METRICS, family_metric
from .tasks import BENCHMARK_BUILDERS, get_builder, validate_benchmark
from .benchmark import (
    MockPolicy,
    MockTaskRunner,
    TaskRunner,
    load_benchmark,
    collect_benchmark_results,
)
from .results import (
    ReportConfig,
    build_benchmark_result,
    render_csv,
    render_json,
    render_markdown,
    write_report,
)

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "Substep",
    "SubstepResult",
    "Task",
    "TaskFamily",
    "TaskResult",
    "FAMILY_METRICS",
    "family_metric",
    "BENCHMARK_BUILDERS",
    "get_builder",
    "validate_benchmark",
    "MockPolicy",
    "MockTaskRunner",
    "TaskRunner",
    "load_benchmark",
    "collect_benchmark_results",
    "ReportConfig",
    "build_benchmark_result",
    "render_csv",
    "render_json",
    "render_markdown",
    "write_report",
]
