"""Aggregation and reporting for the HEARTH eval harness.

Turns a list of :class:`TaskResult` into a :class:`BenchmarkResult` and
renders it to markdown, JSON or CSV for the GitHub milestone record.

Importable without Isaac Sim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import episode_id_for
from .types import Benchmark, TaskFamily, TaskResult, BenchmarkResult


@dataclass
class ReportConfig:
    """Options controlling how a benchmark run is summarised."""

    include_substeps: bool = True
    include_metadata: bool = True
    precision: int = 4


def build_benchmark_result(
    benchmark: Benchmark,
    results: list[TaskResult],
    policy_id: str,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Wrap raw task results in a :class:`BenchmarkResult` with task context."""
    br = BenchmarkResult(
        benchmark_id=benchmark.id,
        policy_id=policy_id,
        results=results,
        metadata=metadata or {},
    )
    br.attach_tasks(benchmark)
    return br


def render_markdown(
    benchmark: Benchmark,
    result: BenchmarkResult,
    config: ReportConfig | None = None,
) -> str:
    """Render a human-readable markdown report of a benchmark run."""
    cfg = config or ReportConfig()
    lines: list[str] = []
    lines.append(f"# Benchmark `{result.benchmark_id}` — policy `{result.policy_id}`")
    lines.append("")
    lines.append(f"- Tasks run: **{result.num_tasks}**")
    lines.append(f"- Success rate: **{result.success_rate() * 100:.1f}%**")
    lines.append(f"- Benchmark target: {result.metadata.get('target_success_rate', 0.0) * 100:.0f}%")
    lines.append("")

    # Per-family table.
    lines.append("## Success by family")
    lines.append("")
    lines.append("| Family | Tasks | Success |")
    lines.append("| --- | ---: | ---: |")
    for family in TaskFamily:
        group = [r for r in result.results if r.task_id in _family_index(benchmark, family)]
        if not group:
            continue
        ok = sum(1 for r in group if r.success)
        lines.append(
            f"| {family.value} | {len(group)} | {ok / len(group) * 100:.1f}% |"
        )
    lines.append("")

    # Per-task detail.
    lines.append("## Task detail")
    lines.append("")
    lines.append("| Task | Family | Difficulty | Substeps | Result |")
    lines.append("| --- | --- | --- | ---: | :---: |")
    for r in result.results:
        t = r.task
        status = "✅" if r.success else "❌"
        detail = f"{t.num_substeps}/{t.num_substeps}"
        if not cfg.include_substeps:
            detail = "—"
        lines.append(
            f"| `{r.task_id}` | {t.family.value} | {t.difficulty} | {detail} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_json(
    benchmark: Benchmark,
    result: BenchmarkResult,
    config: ReportConfig | None = None,
) -> str:
    cfg = config or ReportConfig()
    data = result.to_dict()
    data["benchmark"] = {
        "id": benchmark.id,
        "num_tasks": len(benchmark.tasks),
        "avg_substeps": benchmark.avg_substeps,
    }
    if not cfg.include_metadata:
        data.pop("metadata", None)
    return json.dumps(data, indent=2)


def render_csv(
    benchmark: Benchmark,
    result: BenchmarkResult,
    config: ReportConfig | None = None,
) -> str:
    cfg = config or ReportConfig()
    header = "task_id,policy_id,success,seed,duration_s,family,difficulty,substeps"
    rows = [header]
    for r in result.results:
        t = r.task
        row = [
            r.task_id,
            r.policy_id,
            str(r.success),
            str(r.seed),
            f"{r.duration_s:.3f}",
            t.family.value,
            t.difficulty,
            str(t.num_substeps),
        ]
        rows.append(",".join(row))
    return "\n".join(rows)


def write_report(
    benchmark: Benchmark,
    result: BenchmarkResult,
    out_dir: Path,
    fmt: str = "md",
) -> Path:
    """Write a report of the given ``fmt`` to ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "md":
        content = render_markdown(benchmark, result)
        path = out_dir / f"{result.benchmark_id}_{result.policy_id}.md"
    elif fmt == "json":
        content = render_json(benchmark, result)
        path = out_dir / f"{result.benchmark_id}_{result.policy_id}.json"
    elif fmt == "csv":
        content = render_csv(benchmark, result)
        path = out_dir / f"{result.benchmark_id}_{result.policy_id}.csv"
    else:
        raise ValueError(f"unknown report format: {fmt!r}")
    path.write_text(content)
    return path


def _family_index(benchmark: Benchmark, family: TaskFamily) -> set[str]:
    return {t.id for t in benchmark.tasks if t.family == family}
