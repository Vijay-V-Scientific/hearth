"""Tests for the failure-recovery controller (objective 5)."""

from __future__ import annotations

import pytest

from hearth.isaac.control import SubgoalType
from hearth.isaac.recovery import (
    FailureType,
    RecoveryOutcome,
    RecoveryResult,
    Situation,
    classify_failure,
    recover,
)


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("dropped_object", FailureType.DROPPED_OBJECT),
        ("object_dropped", FailureType.DROPPED_OBJECT),
        ("blocked_path", FailureType.BLOCKED_PATH),
        ("stuck", FailureType.BLOCKED_PATH),
        ("door_closes", FailureType.DOOR_CLOSES),
        ("balance_loss", FailureType.BALANCE_LOSS),
        ("fell over", FailureType.BALANCE_LOSS),
        ("", FailureType.UNKNOWN),
        ("quantum_flip", FailureType.UNKNOWN),
    ],
)
def test_classify_failure(tag, expected):
    assert classify_failure(Situation(failure=tag)) == expected


@pytest.mark.parametrize(
    "failure",
    [
        "dropped_object",
        "blocked_path",
        "door_closes",
        "balance_loss",
    ],
)
def test_handled_failures_recover_without_reset(failure):
    res = recover(
        Situation(
            failure=failure,
            object_name="mug",
            target_object="door",
            blocked_target="kitchen",
        )
    )
    assert isinstance(res, RecoveryResult)
    assert res.recovered
    assert res.outcome == RecoveryOutcome.RECOVERED
    # The brief's hard requirement: recover WITHOUT a reset.
    assert res.needs_reset is False
    assert len(res.subgoals) >= 1


def test_dropped_object_strategy_regrasps():
    res = recover(Situation(failure="dropped_object", object_name="cup"))
    types = [s.type for s in res.subgoals]
    assert SubgoalType.SEARCH in types     # find where it fell
    assert SubgoalType.GRASP in types      # re-grasp
    assert any("re-grasp" in s.payload for s in res.subgoals)


def test_blocked_path_strategy_replans_navigation():
    res = recover(Situation(failure="blocked_path", blocked_target="fridge"))
    types = [s.type for s in res.subgoals]
    assert SubgoalType.RETREAT in types
    assert SubgoalType.NAVIGATE in types
    assert any("alt-route:fridge" in s.payload for s in res.subgoals)


def test_door_closes_strategy_reopens():
    res = recover(Situation(failure="door_closes", target_object="cabinet"))
    assert any(
        s.type == SubgoalType.OPEN_ARTICULATED and "cabinet" in s.payload
        for s in res.subgoals
    )


def test_no_failure_yields_no_action():
    res = recover(Situation(failure=None))
    assert res.outcome == RecoveryOutcome.NO_ACTION
    assert res.subgoals == []
    assert res.needs_reset is False


def test_unknown_failure_needs_reset():
    res = recover(Situation(failure="meteor_strike"))
    assert res.outcome == RecoveryOutcome.UNHANDLED
    assert res.needs_reset is True


def test_exhausted_attempts_need_reset():
    res = recover(Situation(failure="dropped_object", object_name="mug", attempt=3))
    assert res.outcome == RecoveryOutcome.UNHANDLED
    assert res.needs_reset is True


def test_result_serialises():
    res = recover(Situation(failure="door_closes", target_object="door"))
    d = res.to_dict()
    assert d["failure"] == "door_closes"
    assert d["outcome"] == "recovered"
    assert d["needs_reset"] is False
    assert d["num_subgoals"] == len(res.subgoals)
