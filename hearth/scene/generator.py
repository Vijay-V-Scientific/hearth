"""Procedural apartment + object placement generator (DONE).

Algorithm:
    1. Partition the apartment rectangle into rooms by recursive binary
       subdivision (a simple, robust BSP that guarantees all rooms are
       rectangular and connected via shared edges).
    2. Within each room, place objects by rejection sampling against the
       min-object-spacing rule and against room bounds.
    3. Assign task-relevant objects (doors, drawers, cabinets, tables) first.

The geometry is deterministic given a seed.
"""

from __future__ import annotations

import random
from .types import ApartmentConfig, ObjectConfig, RoomConfig, Scene, SceneMetadata

# Semantic categories with (typical size, fixed) priors. Sizes are in metres.
_ROOM_TYPES = ("living", "kitchen", "bedroom", "bathroom", "hallway", "dining")

# Objects that can be task-relevant (hinged/sliding/movable furniture).
_TASK_OBJECTS = {
    "door": ((0.08, 0.9, 2.1), True),
    "drawer": ((0.5, 0.4, 0.5), True),
    "cabinet": ((0.6, 0.5, 0.8), True),
    "table": ((1.2, 0.6, 0.75), False),
    "chair": ((0.45, 0.45, 0.45), False),
}

# Generic placeable objects (non-task, for richness).
_PLACABLES = {
    "sofa": ((0.9, 2.0, 0.8), False),
    "plant": ((0.3, 0.3, 1.0), False),
    "rug": ((2.0, 3.0, 0.02), True),
    "lamp": ((0.3, 0.3, 1.5), False),
}


def _subdivide(rng: random.Random, x0: float, y0: float, w: float, d: float,
               depth: int, rooms: list[RoomConfig]) -> None:
    """Recursively split the rectangle [x0,x0+w] x [y0,y0+d] into rooms."""
    if depth <= 0:
        name = rng.choice(_ROOM_TYPES)
        rooms.append(RoomConfig(name=name, size_x=w, size_y=d, origin_x=x0, origin_y=y0))
        return
    # Prefer splitting along the longer axis; fall back to random.
    if w >= d:
        if w < 1.0:
            name = rng.choice(_ROOM_TYPES)
            rooms.append(RoomConfig(name=name, size_x=w, size_y=d, origin_x=x0, origin_y=y0))
            return
        t = rng.uniform(0.3, 0.7)
        cut = x0 + w * t
        _subdivide(rng, x0, y0, cut - x0, d, depth - 1, rooms)
        _subdivide(rng, cut, y0, w - (cut - x0), d, depth - 1, rooms)
    else:
        if d < 1.0:
            name = rng.choice(_ROOM_TYPES)
            rooms.append(RoomConfig(name=name, size_x=w, size_y=d, origin_x=x0, origin_y=y0))
            return
        t = rng.uniform(0.3, 0.7)
        cut = y0 + d * t
        _subdivide(rng, x0, y0, w, cut - y0, depth - 1, rooms)
        _subdivide(rng, x0, cut, w, d - (cut - y0), depth - 1, rooms)


def _place_in_room(rng: random.Random, room: RoomConfig,
                   category: str, fixed: bool,
                   min_spacing: float, placed: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Rejection-sample a centre position in the room, respecting spacing."""
    size, _fixed = _TASK_OBJECTS.get(category, _PLACABLES.get(category, ((0.5, 0.5, 0.5), False)))
    sx, sy, _ = size
    max_iter = 200
    for _ in range(max_iter):
        cx = rng.uniform(room.origin_x + sx / 2 + 0.1, room.origin_x + room.size_x - sx / 2 - 0.1)
        cy = rng.uniform(room.origin_y + sy / 2 + 0.1, room.origin_y + room.size_y - sy / 2 - 0.1)
        if cx < cy:  # degenerate guard
            continue
        ok = True
        for px, py in placed:
            if abs(px - cx) < min_spacing and abs(py - cy) < min_spacing:
                ok = False
                break
        if ok:
            placed.append((cx, cy))
            return (cx, cy, 0.0)
    return None


def generate_scene(cfg: ApartmentConfig) -> Scene:
    """Generate a Scene from an ApartmentConfig.

    Args:
        cfg: Apartment description.

    Returns:
        A populated Scene (rooms + objects + metadata).
    """
    rng = cfg.rng
    if rng is None:
        rng = random.Random(cfg.seed)

    # 1. Rooms via BSP subdivision.
    rooms: list[RoomConfig] = []
    target_depth = max(1, cfg.room_count - 1)
    _subdivide(rng, 0.0, 0.0, cfg.width, cfg.depth, target_depth, rooms)
    # If we got fewer rooms than requested, split a couple more.
    guard = 0
    while len(rooms) < cfg.room_count and guard < 20:
        # Find the largest room and split it.
        largest = max(rooms, key=lambda r: r.size_x * r.size_y)
        rooms.remove(largest)
        t = rng.uniform(0.3, 0.7)
        cut_x = largest.origin_x + largest.size_x * t
        rooms.append(RoomConfig(name=rng.choice(_ROOM_TYPES), size_x=cut_x - largest.origin_x,
                                size_y=largest.size_y, origin_x=largest.origin_x, origin_y=largest.origin_y))
        rooms.append(RoomConfig(name=rng.choice(_ROOM_TYPES), size_x=largest.size_x - (cut_x - largest.origin_x),
                                size_y=largest.size_y, origin_x=cut_x, origin_y=largest.origin_y))
        guard += 1

    total_area = cfg.width * cfg.depth

    # 2. Place task-relevant objects first (spread across rooms).
    placed: list[tuple[str, ObjectConfig]] = []
    placed_xy: list[tuple[float, float]] = []
    task_budget = cfg.task_objects_per_scene
    room_i = 0
    while len(placed) < task_budget and rooms:
        room = rooms[room_i % len(rooms)]
        room_i += 1
        category = rng.choice(list(_TASK_OBJECTS.keys()))
        fixed = _TASK_OBJECTS[category][1]
        pos = _place_in_room(rng, room, category, fixed,
                             cfg.min_object_spacing, placed_xy)
        if pos is None:
            continue
        orient = rng.uniform(0, 360) if category in ("drawer", "cabinet") else 0.0
        size = _TASK_OBJECTS[category][0]
        placed.append((category, ObjectConfig(
            category=category, position=pos,
            size=size, orientation_deg=orient, fixed=fixed, task_relevant=True)))

    # 3. Place generic placables to fill rooms.
    for room in rooms:
        for _ in range(rng.randint(1, 4)):
            category = rng.choice(list(_PLACABLES.keys()))
            fixed = _PLACABLES[category][1]
            pos = _place_in_room(rng, room, category, fixed,
                                 cfg.min_object_spacing, placed_xy)
            if pos is None:
                continue
            size = _PLACABLES[category][0]
            placed.append((category, ObjectConfig(
                category=category, position=pos, size=size,
                orientation_deg=0.0, fixed=fixed, task_relevant=False)))

    num_task = sum(1 for _, o in placed if o.task_relevant)
    metadata = SceneMetadata(
        scene_id=f"scene_{cfg.seed:06d}",
        seed=cfg.seed,
        num_rooms=len(rooms),
        total_area=total_area,
        num_objects=len(placed),
        num_task_objects=num_task,
    )

    return Scene(rooms=rooms, objects=placed, metadata=metadata)
