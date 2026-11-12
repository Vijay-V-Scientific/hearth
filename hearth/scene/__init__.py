"""Procedural scene generator (DONE — runnable + tested on this machine).

Generates procedurally generated apartment scenes and writes them to USD for
loading into Isaac Sim. The USD writer is real and tested; the Isaac Sim
import path is a clean SCAFFOLD integration point (Isaac Sim is not on this box).

Public API:
    generate_scene(cfg) -> Scene
    write_usd(scene, path) -> None
    write_usd_from_config(cfg, out_path) -> str
"""

from .generator import generate_scene
from .types import (
    ApartmentConfig,
    ObjectConfig,
    RoomConfig,
    Scene,
    SceneMetadata,
)
from .writer import write_usd, write_usd_from_config

__all__ = [
    "ApartmentConfig",
    "RoomConfig",
    "ObjectConfig",
    "Scene",
    "SceneMetadata",
    "generate_scene",
    "write_usd",
    "write_usd_from_config",
]
