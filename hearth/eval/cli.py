"""Command-line interface for the HEARTH eval harness.

    hearth-eval run hearth_50 --policy default --seeds 0,1,2
    hearth-eval load configs/eval/benchmark_50.yaml
    hearth-eval report hearth_50 mock --format md --out ./eval_reports

Importable without Isaac Sim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import (
    MockPolicy,
    MockTaskRunner,
    TaskRunner,
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
from .tasks import validate_benchmark
from .types import TaskFamily


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hearth-eval",
        description="Project HEARTH evaluation harness.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark with a runner")
    run.add_argument("source", help="benchmark name (e.g. hearth_50) or YAML path")
    run.add_argument(
        "--runner",
        default="mock",
        help="runner name: mock (default). Real Isaac Lab runner is a scaffold.",
    )
    run.add_argument("--policy", default="default", help="policy name")
    run.add_argument("--seeds", default="0", help="comma-separated seeds per task")
    run.add_argument("--out", help="directory to write reports to")
    run.add_argument(
        "--format",
        choices=["md", "json", "csv"],
        default="md",
        help="report format (default md)",
    )

    load = sub.add_parser("load", help="validate + describe a benchmark")
    load.add_argument("source", help="benchmark name or YAML path")

    report = sub.add_parser("report", help="render a report for a run")
    report.add_argument("source", help="benchmark name or YAML path")
    report.add_argument("--policy", default="mock")
    report.add_argument("--format", choices=["md", "json", "csv"], default="md")
    report.add_argument("--out", help="output directory")

    families = sub.add_parser("families", help="list task families and metrics")

    return p


def _parse_seeds(seeds: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in seeds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                out[f"task_{i}"] = i
        else:
            out["default"] = int(tok)
    return out


def _make_runner(name: str) -> TaskRunner:
    if name == "mock":
        return MockTaskRunner(MockPolicy(name="mock", capability={"default": 0.5}))
    raise ValueError(
        f"unknown runner '{name}'. Only 'mock' is available here; "
        "the Isaac Lab runner is a scaffold integration point."
    )


def _load(source: str):
    from .benchmark import load_benchmark

    return load_benchmark(source)


def cmd_run(args: argparse.Namespace) -> int:
    benchmark = _load(args.source)
    problems = validate_benchmark(benchmark)
    if problems:
        print(f"WARNING: benchmark has issues:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)

    runner = _make_runner(args.runner)
    seeds = _parse_seeds(args.seeds)
    results = collect_benchmark_results(benchmark, runner, seeds=seeds, policy={args.policy})
    br = build_benchmark_result(benchmark, results, args.policy)

    print(f"Ran {br.num_tasks} tasks for policy '{args.policy}'.")
    print(f"Success rate: {br.success_rate() * 100:.1f}%")
    for family in TaskFamily:
        rate = br.rate_by_family(family)
        if rate > 0:
            print(f"  {family.value}: {rate * 100:.1f}%")

    if args.out:
        path = write_report(benchmark, br, Path(args.out), args.format)
        print(f"Report written to {path}")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    benchmark = _load(args.source)
    problems = validate_benchmark(benchmark)
    print(f"benchmark {benchmark.id} (v{benchmark.version})")
    print(f"  tasks: {len(benchmark.tasks)}")
    print(f"  families: {', '.join(f.value for f in benchmark.families)}")
    print(f"  avg substeps: {benchmark.avg_substeps}")
    if problems:
        print("  problems:", file=sys.stderr)
        for pr in problems:
            print(f"    - {pr}", file=sys.stderr)
        return 1
    print("  validation: OK")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    benchmark = _load(args.source)
    runner = _make_runner("mock")
    results = collect_benchmark_results(benchmark, runner)
    br = build_benchmark_result(benchmark, results, args.policy)
    if args.out:
        path = write_report(benchmark, br, Path(args.out), args.format)
        print(f"Report written to {path}")
    else:
        if args.format == "json":
            print(render_json(benchmark, br))
        elif args.format == "csv":
            print(render_csv(benchmark, br))
        else:
            print(render_markdown(benchmark, br))
    return 0


def cmd_families(args: argparse.Namespace) -> int:
    from .families import FAMILY_METRICS

    for family, spec in FAMILY_METRICS.items():
        print(f"{family.value}: {spec.description}")
        print(f"    metric: {spec.metric}")
    return 0


_COMMANDS = {
    "run": cmd_run,
    "load": cmd_load,
    "report": cmd_report,
    "families": cmd_families,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
