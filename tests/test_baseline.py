"""Tests for the no-model baseline runner wired to the eval harness."""

from __future__ import annotations

import os
import subprocess
import sys

from hearth.eval import (
    BaselineRunner,
    build_benchmark_result,
    collect_benchmark_results,
    load_benchmark,
)
from hearth.eval.baseline import DEFAULT_PRIORS, FAMILY_PRIORS
from hearth.eval.types import Substep, Task, TaskFamily


def _toy_task(family: TaskFamily, n: int = 5) -> Task:
    return Task(
        id=f"{family.value}_test_0001",
        family=family,
        instruction=f"[{family.value}] do the thing",
        substeps=[
            Substep(description=f"step {j}", success_condition="done", timeout_s=30.0)
            for j in range(n)
        ],
    )


def test_runner_scores_planned_subgoals_not_positional_filler():
    """Regression: substeps used to be matched to the benchmark by POSITION.

    The task declares 6 placeholder substeps ("Complete subtask k"), but the
    planner emits 5 real sub-goals. The old runner paired plan.subgoals[k] to
    the k-th filler step, so "Open the dishwasher" could be scored as a
    "stabilize" step and the extra filler fell through to a generic prior. The
    runner now scores the planner's real decomposition, so there is one result
    per *planned sub-goal*, each labelled with its actual type and payload.
    """
    task = _toy_task(TaskFamily.TIDYING, 6)
    res = BaselineRunner().run_task(task, seed=0)
    assert res.task_id == task.id
    # One result per real sub-goal (5), not per placeholder (6).
    assert len(res.substeps) == res.metadata["planned_subgoals"]
    assert res.metadata["planned_subgoals"] != task.num_substeps
    assert res.success in (True, False)
    assert res.policy_id == "baseline_v0"
    # Every scored substep names the concrete sub-goal it evaluated.
    for s in res.substeps:
        assert s.details["subgoal"] in {
            "navigate",
            "search",
            "grasp",
            "carry",
            "place",
            "open_articulated",
            "recover",
            "reset",
            "stabilize",
        }
        assert ":" in s.substep  # "type:payload", not "Complete subtask 3 of 8"
    # A tidying plan must contain a grasp and a place, never a filler "8/8".
    types = [s.details["subgoal"] for s in res.substeps]
    assert "grasp" in types and "place" in types


def test_runner_is_deterministic_per_seed():
    task = _toy_task(TaskFamily.KITCHEN)
    a = BaselineRunner().run_task(task, seed=7)
    b = BaselineRunner().run_task(task, seed=7)
    assert [s.success for s in a.substeps] == [s.success for s in b.substeps]


def test_runner_stable_across_pythonhashseed():
    """The baseline seed must be stable across processes (PYTHONHASHSEED).

    Regression guard: an earlier version seeded with the built-in ``hash()``,
    which Python randomises per process, so the baseline was reproducible only
    within one process - the 50-task success rate swung between 27, 31 and 33
    across PYTHONHASHSEED values.  Re-running in a subprocess with different
    hash seeds must give an identical outcome signature.
    """
    # Use the real 50-task benchmark so the signature contains both successes
    # and failures (a toy task can come out all-True and be a weak probe).
    code = (
        "from hearth.eval.baseline import BaselineRunner\n"
        "from hearth.eval.families import build_50_task_benchmark\n"
        "from hearth.eval.benchmark import collect_benchmark_results\n"
        "bm = build_50_task_benchmark()\n"
        "res = collect_benchmark_results(bm, BaselineRunner(), "
        "seeds={'default': 0})\n"
        "print(sum(x.success for x in res), "
        "round(sum(x.duration_s for x in res), 6))\n"
    )
    sigs = []
    env_base = dict(os.environ)
    env_base["PYTHONPATH"] = os.getcwd()
    for hs in ("0", "1", "12345"):
        env = dict(env_base, PYTHONHASHSEED=hs)
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        sigs.append(out.stdout.strip())
    assert sigs[0] == sigs[1] == sigs[2], (
        f"baseline outcome changed with PYTHONHASHSEED (unstable seed):\n{sigs}"
    )
    # Sanity: the 50-task run produced a non-trivial mix of outcomes.
    n_success = int(sigs[0].split()[0])
    assert 0 < n_success < 50, f"expected a mix of outcomes, got {n_success}/50"


def test_perfect_priors_guarantee_success():
    task = _toy_task(TaskFamily.FETCH)
    perfect = {k: 1.0 for k in DEFAULT_PRIORS}
    perfect.update({f"__family:{k}": 1.0 for k in FAMILY_PRIORS})
    res = BaselineRunner(priors=perfect).run_task(task, seed=1)
    assert res.success is True
    assert all(s.success for s in res.substeps)


def test_zero_priors_cause_failure():
    task = _toy_task(TaskFamily.TIDYING)
    zeros = {k: 0.0 for k in DEFAULT_PRIORS}
    zeros.update({f"__family:{k}": 0.0 for k in FAMILY_PRIORS})
    res = BaselineRunner(priors=zeros).run_task(task, seed=1)
    assert res.success is False


def test_recovery_lifts_success_over_no_recovery():
    """Recovery's retry attempt must strictly beat a policy that cannot recover.

    With recovery re-entering the sub-goal once more, a moderate per-attempt
    prior p yields success probability 2p - p^2 instead of p, so over many
    seeds recovery-on beats recovery-off by a reliable, non-flaky margin.
    """
    task = _toy_task(TaskFamily.TIDYING)
    grasp_prone = dict(DEFAULT_PRIORS)
    grasp_prone["grasp"] = 0.3
    grasp_prone.update({f"__family:{k}": 1.0 for k in FAMILY_PRIORS})

    with_recovery = BaselineRunner(priors=grasp_prone, allow_recovery=True)
    no_recovery = BaselineRunner(priors=grasp_prone, allow_recovery=False)

    saved = sum(with_recovery.run_task(task, seed=s).success for s in range(400))
    plain = sum(no_recovery.run_task(task, seed=s).success for s in range(400))
    assert saved > plain


def test_over_the_reference_50_task_benchmark():
    bm = load_benchmark("hearth_50")
    results = collect_benchmark_results(bm, BaselineRunner(), seeds={"default": 0})
    agg = build_benchmark_result(bm, results, "baseline_v0")
    agg.attach_tasks(bm)
    assert agg.num_tasks == 50
    rate = agg.success_rate()
    # A deterministic baseline lower bound, not the 0.70 target.
    assert 0.0 < rate < 1.0
    # Every family is exercised.
    from hearth.eval.types import TaskFamily as TF

    for fam in TF:
        assert agg.to_dict()["by_family"][fam.value] is not None
