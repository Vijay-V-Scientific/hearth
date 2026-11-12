"""Tests for the Layer-2 instruction planner (objective 4)."""

from __future__ import annotations

import pytest

from hearth.isaac.control import Subgoal, SubgoalType
from hearth.isaac.planner import (
    PlanResult,
    SceneSummary,
    detect_family,
    plan_many,
    plan_task,
)

SCENE = SceneSummary(
    rooms=("kitchen", "bedroom", "living", "hallway"),
    object_locations={"red mug": "kitchen", "laundry basket": "bedroom"},
    visible_objects=("chair",),
    requester_room="living",
)


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("Bring me the red mug from the kitchen", "fetch"),
        ("Fetch the towel", "fetch"),
        ("Clear the table and load the dishwasher", "kitchen"),
        ("Put the cups in the fridge", "kitchen"),
        ("Open the cabinet", "articulated"),
        ("Close the door", "articulated"),
        ("Carry the box to the bedroom", "carrying"),
        ("Move the laundry basket to the hall", "carrying"),
        ("Tidy the toys where they belong", "tidying"),
        ("Pick up the clothes", "tidying"),
    ],
)
def test_detect_family(instruction, expected):
    assert detect_family(instruction) == expected


@pytest.mark.parametrize(
    "instruction,family",
    [
        ("Bring me the red mug from the kitchen", "fetch"),
        ("Clear the dinner table and load the cups into the dishwasher", "kitchen"),
        ("Open the cabinet above the counter", "articulated"),
        ("Carry the box to the bedroom", "carrying"),
        ("Put the toys where they belong", "tidying"),
    ],
)
def test_plan_is_in_the_5_15_subgoal_range(instruction, family):
    result = plan_task(instruction, SCENE, family=family)
    assert isinstance(result, PlanResult)
    assert result.family == family
    assert 5 <= result.num_subgoals <= 15
    assert all(isinstance(s, Subgoal) for s in result.subgoals)


def test_grasp_family_plans_grasp_and_place():
    r = plan_task("Put the book on the shelf", SCENE)
    types = [s.type for s in r.subgoals]
    assert SubgoalType.GRASP in types
    assert SubgoalType.PLACE in types


def test_fetch_searches_when_object_not_visible():
    r = plan_task("Bring me the red mug", SCENE)
    types = [s.type for s in r.subgoals]
    assert SubgoalType.SEARCH in types
    assert SubgoalType.GRASP in types


def test_articulated_emits_open_or_close():
    op = plan_task("Open the cabinet", SCENE, family="articulated")
    assert any(s.type == SubgoalType.OPEN_ARTICULATED and "open" in s.payload for s in op.subgoals)
    cl = plan_task("Close the cabinet", SCENE, family="articulated")
    assert any(s.type == SubgoalType.OPEN_ARTICULATED and "close" in s.payload for s in cl.subgoals)


def test_planner_is_deterministic():
    a = plan_task("Bring me the red mug from the kitchen", SCENE)
    b = plan_task("Bring me the red mug from the kitchen", SCENE)
    assert [s.to_dict() for s in a.subgoals] == [s.to_dict() for s in b.subgoals]


def test_room_resolved_from_instruction():
    r = plan_task("Carry the box to the bedroom", SCENE, family="carrying")
    payloads = " ".join(s.payload for s in r.subgoals)
    assert "bedroom" in payloads


def test_plan_many_matches_one_by_one():
    ins = ["Open the door", "Tidy the toys", "Fetch the cup"]
    plans = plan_many(ins, SCENE)
    assert len(plans) == 3
    for p, i in zip(plans, ins):
        assert p.instruction == i
