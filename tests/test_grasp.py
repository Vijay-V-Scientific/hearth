"""Tests for geometric grasp candidate generation and ranking (objective 3)."""

from __future__ import annotations

import math

import pytest

from hearth.isaac.grasp import (
    GripperLimits,
    best_grasp,
    generate_grasp_candidates,
    rank_grasps,
)
from hearth.scene.types import ObjectConfig


def _obj(category, size, position=(0.0, 0.0, 0.8)):
    return ObjectConfig(category=category, position=position, size=size)


def test_mug_produces_valid_candidates():
    cands = generate_grasp_candidates(_obj("mug", (0.08, 0.08, 0.10)))
    assert cands
    for c in cands:
        assert GripperLimits().min_aperture <= c.width <= GripperLimits().max_aperture
        assert c.depth <= GripperLimits().max_depth


def test_candidates_sorted_by_quality_descending():
    cands = generate_grasp_candidates(_obj("bottle", (0.06, 0.06, 0.20)))
    qualities = [c.quality for c in cands]
    assert qualities == sorted(qualities, reverse=True)


def test_smallest_dimension_grasp_preferred():
    # A 4 cm wide box: the 0.04-width grasp must be the best (narrowest).
    best = best_grasp(_obj("remote", (0.04, 0.14, 0.02)))
    assert best is not None
    assert best.width == pytest.approx(0.02, abs=1e-9) or best.width == pytest.approx(
        0.04, abs=1e-9
    )
    all_widths = [c.width for c in generate_grasp_candidates(_obj("remote", (0.04, 0.14, 0.02)))]
    assert all_widths == sorted(all_widths)  # narrowest first


def test_tiny_object_is_ungraspable():
    # Smaller than a fingertip: no candidate fits.
    assert best_grasp(_obj("bead", (0.005, 0.005, 0.005))) is None


def test_large_object_is_ungraspable_in_one_grasp():
    assert best_grasp(_obj("sofa", (1.5, 0.8, 0.6))) is None


def test_gripper_limits_are_respected():
    obj = _obj("book", (0.15, 0.03, 0.21))
    tight = GripperLimits(max_aperture=0.03)
    cands = generate_grasp_candidates(obj, tight)
    assert all(c.width <= 0.03 for c in cands)


def test_quality_in_unit_range():
    cands = generate_grasp_candidates(_obj("apple", (0.07, 0.07, 0.07)))
    for c in cands:
        assert 0.0 <= c.quality <= 1.0


def test_axes_are_unit_vectors():
    cands = generate_grasp_candidates(_obj("cup", (0.09, 0.09, 0.10)))
    for c in cands:
        assert math.isclose(math.sqrt(sum(a * a for a in c.axis)), 1.0, rel_tol=1e-6)


def test_empty_candidates_rank_to_empty():
    assert rank_grasps([], _obj("bead", (0.005, 0.005, 0.005))) == []
