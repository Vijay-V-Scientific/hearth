"""USD writer for generated scenes (DONE — tested; Isaac Sim path is SCAFFOLD).

Writes a self-contained USD stage with:
    - A room hierarchy (walls/floor/ceiling as simple meshes).
    - Placed objects as prim boxes with category metadata.
    - Custom stage attributes for scene metadata (id, seed, areas).

The output is valid USD readable by usdview / Isaac Sim. The Isaac Sim import
helper is a SCAFFOLD integration point (Isaac Sim is not installed here).
"""

from __future__ import annotations

import os
from typing import Optional
from .types import Scene

# Approximate room surface colours (rgb), for visualisation in usdview.
_ROOM_COLOURS = {
    "living": (0.85, 0.82, 0.72),
    "kitchen": (0.9, 0.9, 0.9),
    "bedroom": (0.75, 0.78, 0.85),
    "bathroom": (0.88, 0.92, 0.92),
    "hallway": (0.8, 0.78, 0.72),
    "dining": (0.82, 0.7, 0.6),
}


def _room_colour(name: str) -> tuple[float, float, float]:
    return _ROOM_COLOURS.get(name, (0.75, 0.75, 0.75))


def _write_room(stage, room_path, room, colour) -> None:
    from pxr import UsdGeom, Gf

    x0, y0 = room.origin_x, room.origin_y
    w, d, h = room.size_x, room.size_y, room.height
    # Floor slab.
    floor = UsdGeom.Mesh.Define(stage, f"{room_path}/floor")
    corners = [
        Gf.Vec3f(float(x0), float(y0), 0.0),
        Gf.Vec3f(float(x0 + w), float(y0), 0.0),
        Gf.Vec3f(float(x0 + w), float(y0 + d), 0.0),
        Gf.Vec3f(float(x0), float(y0 + d), 0.0),
    ]
    floor.CreatePointsAttr(corners)
    # Quad face: four vertices, single quad.
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    _apply_material(stage, f"{room_path}/floor", colour)


def _apply_material(stage, prim_path, colour) -> None:
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, prim_path + "/material")
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if prim is not None and prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


def _write_object(stage, obj_path, category, config) -> None:
    from pxr import UsdGeom, Gf, Sdf

    x, y, z = config.position
    sx, sy, sz = config.size
    box = UsdGeom.Cube.Define(stage, obj_path)
    box.CreateSizeAttr(max(float(sx), float(sy), float(sz)))
    box.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.4, 0.2)])
    # Tag task-relevant objects.
    if config.task_relevant:
        box.GetPrim().CreateAttribute("hearth_task_object", Sdf.ValueTypeNames.String).Set(category)
    # Optional yaw.
    if config.orientation_deg:
        import math
        from pxr import Sdf
        prim = box.GetPrim()
        rot = prim.CreateAttribute("hearth_yaw", Sdf.ValueTypeNames.Quatd)
        theta = math.radians(config.orientation_deg)
        rot.Set(Gf.Quatd(math.cos(theta / 2), Gf.Vec3d(0, 0, math.sin(theta / 2))))


def write_usd(scene: Scene, path: str) -> str:
    """Write a Scene to a USD file.

    Args:
        scene: The scene to write.
        path: Output .usd path (created if missing).

    Returns:
        The path written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    from pxr import Usd, UsdGeom, Sdf

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetMetadata("metersPerUnit", 1.0)  # metres; Isaac Sim sets its own scale on load

    root = stage.DefinePrim("/hearth_scene")
    root.CreateAttribute("hearth_scene_id", Sdf.ValueTypeNames.String).Set(scene.metadata.scene_id)
    root.CreateAttribute("hearth_seed", Sdf.ValueTypeNames.Int).Set(scene.metadata.seed)
    root.CreateAttribute("hearth_num_rooms", Sdf.ValueTypeNames.Int).Set(scene.metadata.num_rooms)
    root.CreateAttribute("hearth_total_area_m2", Sdf.ValueTypeNames.Float).Set(scene.metadata.total_area)
    root.CreateAttribute("hearth_num_objects", Sdf.ValueTypeNames.Int).Set(scene.metadata.num_objects)
    root.CreateAttribute("hearth_num_task_objects", Sdf.ValueTypeNames.Int).Set(scene.metadata.num_task_objects)

    rooms_prim = stage.DefinePrim("/hearth_scene/rooms")
    for room in scene.rooms:
        room_prim = f"/hearth_scene/rooms/{room.name}"
        _write_room(stage, room_prim, room, _room_colour(room.name))

    objects_prim = stage.DefinePrim("/hearth_scene/objects")
    objects_path = objects_prim.GetPath().pathString
    for category, obj in scene.objects:
        obj_path = f"{objects_path}/{category}"
        _write_object(stage, obj_path, category, obj)

    stage.Save()
    return path


def write_usd_from_config(cfg, out_path: str) -> str:
    """Convenience: build a scene from an ApartmentConfig and write it.

    Args:
        cfg: An ApartmentConfig.
        out_path: Output .usd path.

    Returns:
        The path written.
    """
    from .generator import generate_scene

    scene = generate_scene(cfg)
    return write_usd(scene, out_path)
