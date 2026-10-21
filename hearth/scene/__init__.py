"""Procedural scene generator (DONE — runnable + tested on this machine).

Generates procedurally generated apartment scenes and writes them to USD for
loading into Isaac Sim. The USD writer is real and tested; the Isaac Sim
import path is a clean SCAFFOLD integration point (Isaac Sim is not on this box).

Public API:
    generate_scene(cfg) -> Scene
    write_usd(scene, path) -> None
"""

from .types import (
    ApartmentConfig,
    RoomConfig,
    ObjectConfig,
    Scene,
    SceneMetadata,
)
from .generator import generate_scene
from .writer import write_usd

__all__ = [
    "ApartmentConfig",
    "RoomConfig",
    "ObjectConfig",
    "Scene",
    "SceneMetadata",
    "generate_scene",
    "write_usd",
]
