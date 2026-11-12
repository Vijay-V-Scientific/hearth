"""Data types for the procedural scene generator.

These are plain dataclasses so they are cheap to construct, compare, and test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RoomConfig:
    """One room in an apartment.

    Attributes:
        name: Human label, e.g. "kitchen".
        size_x, size_y, height: Interior dimensions in metres.
        origin_x, origin_y: World-space floor origin in metres (bottom-left corner).
    """

    name: str
    size_x: float
    size_y: float
    height: float = 2.8
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class ObjectConfig:
    """A single placed object with a category and bounding box.

    Attributes:
        category: Semantic category, e.g. "table", "chair", "door".
        position: (x, y, z) centre of the bounding box in world space (metres).
        size: (sx, sy, sz) bounding-box extents in metres.
        orientation_deg: Yaw about the vertical axis, degrees.
        fixed: Whether the object is kinematically fixed to the ground plane.
        task_relevant: Whether this object participates in a task (e.g. a target door).
    """

    category: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    orientation_deg: float = 0.0
    fixed: bool = True
    task_relevant: bool = False


@dataclass
class SceneMetadata:
    """Bookkeeping about a generated scene.

    Attributes:
        scene_id: Stable unique id for the scene.
        seed: RNG seed used to generate the scene (for reproducibility).
        num_rooms: Number of rooms.
        total_area: Floor area of the apartment in square metres.
        num_objects: Number of placed objects.
        num_task_objects: Number of task-relevant objects.
    """

    scene_id: str
    seed: int
    num_rooms: int
    total_area: float
    num_objects: int
    num_task_objects: int


@dataclass
class ApartmentConfig:
    """Full description of an apartment to generate.

    Attributes:
        width, depth: Overall apartment footprint in metres.
        room_count: Number of rooms to generate.
        rng: Optional RNG (for tests / reproducibility).
        min_object_spacing: Minimum separation between placed objects in metres.
        seed: Base RNG seed.
        task_objects_per_scene: Number of task-relevant objects per scene.
    """

    width: float
    depth: float
    room_count: int = 4
    rng: Optional[random.Random] = field(default=None)
    min_object_spacing: float = 0.5
    seed: int = 0
    task_objects_per_scene: int = 3

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.seed)


@dataclass
class Scene:
    """A generated scene: rooms, placed objects, and metadata.

    Attributes:
        rooms: List of rooms in the apartment.
        objects: List of (category, ObjectConfig) tuples placed in the scene.
        metadata: Bookkeeping about the generated scene.
    """

    rooms: List[RoomConfig]
    objects: List[tuple[str, ObjectConfig]]
    metadata: SceneMetadata
