"""Aggregation and reporting for the HEARTH eval harness.

Turns a list of :class:`TaskResult` into a :class:`BenchmarkResult` and
renders it to markdown, JSON or CSV for the GitHub milestone record.

Importable without Isaac Sim.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Benchmark, BenchmarkResult, TaskFamily, TaskResult


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
    lines.append(f"- Tasks: **{result.num_tasks}**")
    lines.append(f"- Episodes (task x seed): **{result.num_episodes}**")
    lines.append(f"- Success rate (per episode): **{result.success_rate() * 100:.1f}%**")
    # The pass target is declared on the benchmark, not the result. The old
    # code read result.metadata (never populated), so every report printed
    # "Benchmark target: 0%". Fall back to the result only if a caller set it.
    target = benchmark.metadata.get(
        "target_success_rate",
        result.metadata.get("target_success_rate"),
    )
    if target is not None:
        lines.append(f"- Benchmark target: {float(target) * 100:.0f}%")
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
        lines.append(f"| {family.value} | {len(group)} | {ok / len(group) * 100:.1f}% |")
    lines.append("")

    # Per-task detail.
    lines.append("## Task detail")
    lines.append("")
    lines.append("| Task | Family | Difficulty | Substeps | Episodes | Result |")
    lines.append("| --- | --- | --- | ---: | ---: | :---: |")
    # Group by task so several seeds of the same task become one honest row
    # (episodes = successful/total runs), instead of one row per episode.
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in result.results:
        grouped.setdefault(r.task_id, []).append(r)
    for task_id, group in grouped.items():
        t = group[0].task
        # build_benchmark_result() -> attach_tasks() guarantees this.
        assert t is not None, f"task not attached for {task_id}"
        n_runs = len(group)
        n_ok = sum(1 for r in group if r.success)
        passes = n_ok / n_runs
        status = "✅" if passes >= 0.5 else "❌"
        episodes = f"{n_ok}/{n_runs}" if n_runs > 1 else ("success" if n_ok else "fail")
        if cfg.include_substeps:
            # Real successful substeps (mean across runs), not num/num. The old
            # code printed f"{num}/{num}" so a task that failed its first
            # substep still read "8/8".
            mean_ok = sum(sum(1 for s in r.substeps if s.success) for r in group) / n_runs
            detail = f"{mean_ok:.1f}/{t.num_substeps}"
        else:
            detail = "—"
        lines.append(
            f"| `{task_id}` | {t.family.value} | {t.difficulty} | "
            f"{detail} | {episodes} | {status} |"
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
    header = "task_id,policy_id,success,seed,duration_s,family,difficulty,substeps"
    rows = [header]
    for r in result.results:
        t = r.task
        # build_benchmark_result() -> attach_tasks() guarantees this.
        assert t is not None, f"task not attached for {r.task_id}"
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
