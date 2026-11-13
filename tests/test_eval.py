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
    assert "Success rate (per episode):" in md
    # Single seed -> one episode per task.
    assert "Episodes (task x seed): **50**" in md

    js = render_json(b, br)
    assert '"num_tasks": 50' in js
    assert '"num_episodes": 50' in js
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


def test_seeds_run_each_task_once_per_seed():
    """Regression: --seeds must run EVERY task at EACH seed (README promise).

    The old parser mapped a task's position to a seed (task 0 -> seed 0,
    task 1 -> seed 1, ...), so ``--seeds 0,1,2`` ran 50 tasks *once each* at
    a different seed, not 50 tasks three times. Now N seeds => N runs per task.
    """
    from hearth.eval.cli import _parse_seeds

    b = build_50_task_benchmark()
    runner = MockTaskRunner(MockPolicy(name="cap0.6", capability={"default": 0.6}))

    # No seeds -> one episode per task at seed 0.
    r0 = collect_benchmark_results(b, runner)
    assert len(r0) == 50
    assert all(r.seed == 0 for r in r0)

    # A single seed value still runs one episode per task, at that seed.
    r5 = collect_benchmark_results(b, runner, seeds=_parse_seeds("5"))
    assert all(r.seed == 5 for r in r5)
    assert len(r5) == 50

    # --seeds 0,1,2 -> every task runs three times, once at each seed.
    three = collect_benchmark_results(b, runner, seeds=_parse_seeds("0,1,2"))
    assert len(three) == 150
    by_task: dict[str, set[int]] = {}
    for r in three:
        by_task.setdefault(r.task_id, set()).add(r.seed)
    assert all(seeds == {0, 1, 2} for seeds in by_task.values())
    assert len(by_task) == 50  # 50 tasks, each 3x = 150 episodes

    # A range token expands to the full set.
    assert _parse_seeds("0-2") == [0, 1, 2]
    # Duplicates / unsorted input normalise to a sorted unique list.
    assert _parse_seeds("2,0,1,0") == [0, 1, 2]
    # Empty spec falls back to a single seed 0 run.
    assert _parse_seeds("") == [0]


def test_report_shows_real_target_and_real_substep_count():
    """Regression: the report used to print 'target 0%' and '8/8' on failed rows.

    The pass target is declared on ``benchmark.metadata`` (objective 4 -> 0.70),
    not on ``result.metadata``; and the per-task detail printed
    ``num_substeps/num_substeps`` so a task that failed its first substep still
    read "8/8". Both must reflect reality.
    """
    from hearth.eval import build_benchmark_result, collect_benchmark_results, render_markdown
    from hearth.eval.baseline import BaselineRunner

    b = load_benchmark("hearth_50")
    res = collect_benchmark_results(b, BaselineRunner(), seeds={"default": 0})
    br = build_benchmark_result(b, res, "baseline_v0")
    md = render_markdown(b, br)

    # The real 70% target from benchmark.metadata, not "0%".
    assert "Benchmark target: 70%" in md
    assert "Benchmark target: 0%" not in md

    # A failing task must NOT print "N/N" (all substeps succeeded).
    for line in md.splitlines():
        if line.startswith("| `") and line.rstrip().endswith("❌ |"):
            # detail column is "x.y/N"; the integer part must be < N (a real
            # failure means at least one substep did not succeed).
            import re

            m = re.search(r"\| ([\d.]+)/(\d+) \|", line)
            assert m is not None, line
            ok, total = float(m.group(1)), int(m.group(2))
            assert ok < total, f"failed row still shows {ok}/{total} (all-pass): {line}"


def test_make_runner_supports_baseline():
    """Regression: --runner baseline must be selectable from the CLI."""
    from hearth.eval.baseline import BaselineRunner
    from hearth.eval.cli import _make_runner

    assert isinstance(_make_runner("mock"), MockTaskRunner)
    assert isinstance(_make_runner("baseline"), BaselineRunner)
    import pytest

    with pytest.raises(ValueError, match="baseline"):
        _make_runner("isaaclab")


def test_policy_name_is_passed_through_to_episodes():
    """Regression: --policy must label every episode, not just the report.

    Both runners used to hardcode their internal name (mock / baseline_v0) on
    every TaskResult, so ``--policy mine`` labelled the summary but every
    episode still read 'mock' / 'baseline_v0' and the episode ids were wrong.
    """
    from hearth.eval.baseline import BaselineRunner

    b = build_50_task_benchmark()

    res = collect_benchmark_results(b, MockTaskRunner(), seeds=[0], policy={"name": "mine"})
    assert all(r.policy_id == "mine" for r in res)
    assert all(r.episode_id.startswith("mine_") for r in res)

    res2 = collect_benchmark_results(b, BaselineRunner(), seeds=[0], policy={"name": "sweep3"})
    assert all(r.policy_id == "sweep3" for r in res2)
    assert all(r.episode_id.startswith("sweep3_") for r in res2)

    # Without an override the runners keep their own default names.
    def_res = collect_benchmark_results(b, BaselineRunner(), seeds=[0])
    assert all(r.policy_id == "baseline_v0" for r in def_res)
