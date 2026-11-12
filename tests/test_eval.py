"""Smoke tests for the Project HEARTH eval harness (no Isaac Sim required)."""

from hearth.eval import (
    Benchmark,
    MockPolicy,
    MockTaskRunner,
    TaskFamily,
    build_benchmark_result,
    collect_benchmark_results,
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
    assert (
        csv.splitlines()[0]
        == "task_id,policy_id,success,seed,duration_s,family,difficulty,substeps"
    )


def test_mock_runner_seeds_per_task_not_per_benchmark():
    """Regression: MockTaskRunner must seed per (task.id, seed), not per seed.

    Seeding with the bare ``seed`` gave every task the same ``Random(seed)``
    stream, so on a fixed-seed run all equal-length tasks shared one success
    pattern (only 11 of 50 patterns were distinct) and the benchmark collapsed
    to 0/50 at cap 0.7. Each task must get its own stable, distinct stream.
    """
    b = build_50_task_benchmark()
    runner = MockTaskRunner(MockPolicy(name="m", capability={"default": 0.7}))
    results = collect_benchmark_results(b, runner)  # all tasks seed 0

    patterns = [tuple(s.success for s in r.substeps) for r in results]
    # The collapse left only 11 distinct patterns (every equal-length task ran
    # the same stream). With per-task seeding almost every task differs; short
    # tasks can still share a success pattern by chance (a 2**5 space holds 12
    # length-5 tasks), so assert "far above 11" rather than exactly 50.
    assert len(set(patterns)) >= 40, (
        f"expected ~all distinct task outcomes, got {len(set(patterns))} (collapsed runner gave 11)"
    )
    # The run must not collapse to 0 or 50: independent per-task streams at
    # cap 0.7 give a realistic small nonzero count (the collapse gave 0).
    n_success = sum(1 for r in results if r.success)
    assert 0 < n_success < 50, f"benchmark collapsed to {n_success}/50"

    # And it is stable when re-run with the same seed.
    again = collect_benchmark_results(b, runner)
    assert [r.substeps[i].success for r in again for i in range(len(r.substeps))] == [
        r.substeps[i].success for r in results for i in range(len(r.substeps))
    ]


def test_load_benchmark_by_name():
    b = load_benchmark("hearth_50")
    assert isinstance(b, Benchmark)
    assert len(b.tasks) == 50


def test_seeds_are_applied_to_all_tasks():
    from hearth.eval.benchmark import _seed_for
    from hearth.eval.cli import _parse_seeds

    b = build_50_task_benchmark()
    runner = MockTaskRunner(MockPolicy(name="cap0.6", capability={"default": 0.6}))

    # No seeds -> all episodes use seed 0.
    r0 = collect_benchmark_results(b, runner)
    assert all(r.seed == 0 for r in r0)

    # A single default seed applies to every task.
    r5 = collect_benchmark_results(b, runner, seeds=_parse_seeds("5"))
    assert all(r.seed == 5 for r in r5)

    # A range seeds the first three tasks by index (0,1,2); the default
    # fallback only applies to tasks beyond the range.
    rng = _parse_seeds("0-2")
    for i, t in enumerate(b.tasks[:3]):
        assert _seed_for(t, i, rng) == i
    assert _seed_for(b.tasks[3], 3, rng) == rng["default"]
    assert _seed_for(b.tasks[3], 3, rng) == 2
    # No seeds -> seed 0.
    assert _seed_for(b.tasks[0], 0, {}) == 0
