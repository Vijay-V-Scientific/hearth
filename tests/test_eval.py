"""Smoke tests for the Project HEARTH eval harness (no Isaac Sim required)."""

from hearth.eval import (
    Benchmark,
    TaskFamily,
    MockPolicy,
    MockTaskRunner,
    build_benchmark_result,
    collect_benchmark_results,
    family_metric,
    load_benchmark,
    render_csv,
    render_json,
    render_markdown,
    validate_benchmark,
)
from hearth.eval.families import build_50_task_benchmark
from hearth.eval.tasks import BENCHMARK_BUILDERS


def test_50_task_benchmark_matches_brief():
    b = build_50_task_benchmark()
    assert len(b.tasks) == 50, f"expected 50 tasks, got {len(b.tasks)}"
    # Every task has 5-15 substeps (objective 4 constraint).
    for t in b.tasks:
        assert 5 <= t.num_substeps <= 15, f"{t.id}: {t.num_substeps} substeps"
    # All families are in scope.
    assert set(b.families) == set(TaskFamily)


def test_family_distribution():
    b = build_50_task_benchmark()
    counts = {f: len(b.tasks_by_family(f)) for f in TaskFamily}
    assert counts == {
        TaskFamily.KITCHEN: 14,
        TaskFamily.FETCH: 12,
        TaskFamily.ARTICULATED: 10,
        TaskFamily.TIDYING: 7,
        TaskFamily.CARRYING: 7,
    }, counts


def test_builder_registry_has_50():
    assert "hearth_50" in BENCHMARK_BUILDERS
    assert len(BENCHMARK_BUILDERS["hearth_50"]().tasks) == 50


def test_validate_passes_for_50():
    b = build_50_task_benchmark()
    assert validate_benchmark(b) == []


def test_validate_flags_bad_benchmark():
    b = Benchmark(id="bad", version=1, tasks=[], description="")
    problems = validate_benchmark(b)
    assert problems, "empty benchmark should be flagged"

    from hearth.eval.types import Substep, Task
    b2 = Benchmark(
        id="bad2",
        version=1,
        tasks=[
            Task(
                id="t1",
                family=TaskFamily.FETCH,
                instruction="short",
                substeps=[Substep("s", "ok")],
            )
        ],
    )
    problems2 = validate_benchmark(b2)
    assert any("5-15" in p for p in problems2), problems2


def test_run_and_report():
    b = build_50_task_benchmark()
    runner = MockTaskRunner(MockPolicy(name="cap0.6", capability={"default": 0.6}))
    results = collect_benchmark_results(b, runner)
    assert len(results) == 50
    br = build_benchmark_result(b, results, "cap0.6")
    assert 0.0 <= br.success_rate() <= 1.0
    assert br.num_tasks == 50

    md = render_markdown(b, br)
    assert "Success by family" in md
    assert "kitchen" in md
    assert "Success rate:" in md

    js = render_json(b, br)
    assert '"num_tasks": 50' in js
    csv = render_csv(b, br)
    assert csv.splitlines()[0] == "task_id,policy_id,success,seed,duration_s,family,difficulty,substeps"


def test_load_benchmark_by_name():
    b = load_benchmark("hearth_50")
    assert isinstance(b, Benchmark)
    assert len(b.tasks) == 50
