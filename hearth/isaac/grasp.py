"""Geometric grasp candidate generation and ranking (objective 3, part 1).

The brief's objective 3:

    "85% grasp success on 500 held-out household objects (rigid,
     articulated, deformable)."

A full grasp pipeline learns contact points end-to-end; that needs GPU and
Isaac.  But the *geometry* of grasping — where can a parallel/anthropomorphic
gripper physically close around an object, and which of those spots is best —
is computable here from the object bounding boxes the scene generator already
produces (:class:`~hearth.scene.types.ObjectConfig`).  This module provides
that baseline: generate antipodal grasp candidates on a box, filter the ones
the hand cannot fit, and rank them with a simple quality score.

This is the no-model baseline a learned grasp network is scored against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from hearth.scene.types import ObjectConfig


@dataclass(frozen=True)
class GripperLimits:
    """Physical limits of the gripper, in metres.

    Attributes:
        max_aperture: Largest opening between fingertips.
        min_aperture: Smallest closing width (a thin object can be pinched).
        max_depth: How far fingertips extend past the palm.
        tip_force: Nominal normal force per fingertip (N), for ranking.
    """

    max_aperture: float = 0.085   # ~8.5 cm, typical anthropomorphic hand
    min_aperture: float = 0.0
    max_depth: float = 0.075
    tip_force: float = 15.0


# A humanoid hand cannot usefully grasp an object much larger than itself in
# every axis, nor one smaller than a fingertip.
_MIN_GRASPABLE = 0.02   # 2 cm


@dataclass(frozen=True)
class GraspCandidate:
    """One candidate grasp: two antipodal contact points and a quality score.

    Attributes:
        center: Midpoint of the two contacts (world space, metres).
        axis: Unit vector along the grasp (from one contact to the other).
        width: Distance between the contacts (aperture needed, metres).
        depth: How deep the fingertips sit (metres).
        quality: Heuristic score in [0, 1]; higher is easier to grasp.
    """

    center: tuple[float, float, float]
    axis: tuple[float, float, float]
    width: float
    depth: float
    quality: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "center": [round(v, 4) for v in self.center],
            "axis": [round(v, 4) for v in self.axis],
            "width": round(self.width, 4),
            "depth": round(self.depth, 4),
            "quality": round(self.quality, 4),
        }


def _mid(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _box_corners(
    center: tuple[float, float, float], size: tuple[float, float, float]
) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    sx, sy, sz = [s / 2.0 for s in size]
    return [
        (cx + dx * sx, cy + dy * sy, cz + dz * sz)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]


def _antipodal_pairs(
    center: tuple[float, float, float], size: tuple[float, float, float]
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], float]]:
    """Face-to-face width candidates along each of the three box axes.

    Returns (p1, p2, width) triples for grasping across the object's smallest,
    middle and largest dimensions.  Grasping across the smallest dimension is
    usually best (least aperture, most stable pinch).
    """
    pairs = []
    cx, cy, cz = center
    for axis, dim in enumerate(size):
        p1 = list(center)
        p2 = list(center)
        p1[axis] = cx if axis == 0 else (cy if axis == 1 else cz)
        p1[axis] = center[axis] - dim / 2.0
        p2[axis] = center[axis] + dim / 2.0
        pairs.append((tuple(p1), tuple(p2), dim))
    return pairs


def generate_grasp_candidates(
    obj: ObjectConfig,
    gripper: GripperLimits = GripperLimits(),
) -> list[GraspCandidate]:
    """Generate valid antipodal grasp candidates for one box object.

    A candidate is kept only if its required aperture fits inside the
    gripper's [min_aperture, max_aperture] range and the fingertips reach the
    contacts within ``max_depth``.
    """
    center = tuple(obj.position)
    size = tuple(obj.size)
    candidates: list[GraspCandidate] = []
    for p1, p2, width in _antipodal_pairs(center, size):
        if width < gripper.min_aperture or width > gripper.max_aperture:
            continue
        if width < _MIN_GRASPABLE:
            continue
        # Fingertip depth needed is roughly the half-width plus a margin.
        depth = width / 2.0 + 0.01
        if depth > gripper.max_depth:
            continue
        axis = tuple((p2[i] - p1[i]) / max(width, 1e-9) for i in range(3))
        # Normalise a pure-axis vector (one component is +/-1 already).
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis = tuple(a / norm for a in axis)
        candidates.append(
            GraspCandidate(
                center=_mid(p1, p2), axis=axis, width=width, depth=depth
            )
        )
    return rank_grasps(candidates, obj)


def _compactness(obj: ObjectConfig) -> float:
    """How close the object's bounding box is to a cube (1) vs a sliver (0).

    A compact object is easier to grasp from more directions.
    """
    s = sorted(obj.size)
    if s[-1] <= 0:
        return 0.0
    return s[0] / s[-1]


def rank_grasps(
    candidates: list[GraspCandidate],
    obj: ObjectConfig,
    gripper: GripperLimits = GripperLimits(),
) -> list[GraspCandidate]:
    """Score and sort grasp candidates best-first.

    Quality rewards:
      * small aperture (a pinch needs less force to hold; closer to a fingertip)
      * a narrow grasp relative to the gripper's maximum (more margin)
      * objects that are not paper-thin slivers (compactness)
      * grasps near the object's top surface (easier approach from above)
    """
    if not candidates:
        return []
    compact = _compactness(obj)
    top_z = obj.position[2] + obj.size[2] / 2.0
    scored: list[GraspCandidate] = []
    for c in candidates:
        # Narrower apertures are better: linear in [0, max_aperture].
        aperture_term = 1.0 - (c.width / gripper.max_aperture)
        # Approach-from-above bonus: contact higher on the object is easier.
        above_term = 0.5 + 0.5 * ((c.center[2] - obj.position[2]) / max(obj.size[2], 1e-9))
        q = 0.55 * aperture_term + 0.25 * compact + 0.20 * above_term
        q = max(0.0, min(1.0, q))
        scored.append(
            GraspCandidate(
                center=c.center, axis=c.axis, width=c.width,
                depth=c.depth, quality=q,
            )
        )
    scored.sort(key=lambda g: g.quality, reverse=True)
    return scored


def best_grasp(
    obj: ObjectConfig,
    gripper: GripperLimits = GripperLimits(),
) -> Optional[GraspCandidate]:
    """Return the highest-quality grasp for one object, or None if ungraspable."""
    ranked = generate_grasp_candidates(obj, gripper)
    return ranked[0] if ranked else None
