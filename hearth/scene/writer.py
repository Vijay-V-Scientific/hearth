"""USD writer for generated scenes (tested; Isaac Sim path is SCAFFOLD).

Writes a self-contained USD stage with:
    - A room hierarchy: a floor slab plus perimeter walls (so rooms are
      physically bounded, not unbounded quads).
    - Placed objects as unit cubes given a real translate/rotate/scale Xform,
      so each object sits at its world position and yaw instead of all piled
      at the origin.
    - A ``hearth_category`` string on every object (task-relevant and generic)
      plus a ``hearth_task_object`` marker on task objects.

The output is valid USD readable by usdview / Isaac Sim. The Isaac Sim import
helper is a SCAFFOLD integration point (Isaac Sim is not installed here).
"""

from __future__ import annotations

import math
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

_WALL_THICKNESS = 0.1
_OBJECT_DISPLAY = (0.6, 0.4, 0.2)


def _room_colour(name: str) -> tuple[float, float, float]:
    return _ROOM_COLOURS.get(name, (0.75, 0.75, 0.75))


def _make_box(stage, path: str, centre, size, colour) -> "object":
    """Define a unit cube given a world-space centre and full extents.

    The cube is scaled from a unit (side 1, centred at the local origin) box,
    so ``size`` is the true bounding-box extent and ``centre`` is the world
    position of the box centre. Returns the prim.
    """
    from pxr import UsdGeom, Gf

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    # Composed as translate * rotate * scale: scale the unit cube to its
    # extents, yaw it about its own centre, then move the centre to world
    # position. Listing them translate-first gives that composition.
    xform.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xform.AddRotateZOp().Set(0.0)
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    return cube


def _write_room(stage, room_path, room, colour) -> None:
    from pxr import UsdGeom, Gf

    x0, y0 = room.origin_x, room.origin_y
    w, d, h = room.size_x, room.size_y, room.height
    t = _WALL_THICKNESS

    # Floor slab (planar quad at z = 0).
    floor = UsdGeom.Mesh.Define(stage, f"{room_path}/floor")
    corners = [
        Gf.Vec3f(float(x0), float(y0), 0.0),
        Gf.Vec3f(float(x0 + w), float(y0), 0.0),
        Gf.Vec3f(float(x0 + w), float(y0 + d), 0.0),
        Gf.Vec3f(float(x0), float(y0 + d), 0.0),
    ]
    floor.CreatePointsAttr(corners)
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    _apply_material(stage, f"{room_path}/floor", colour)

    # Perimeter walls: four thin boxes enclosing the room footprint so the
    # room is a physically bounded volume rather than a free-floating quad.
    walls = (
        ("wall_south", (x0 + w / 2, y0 + t / 2, h / 2), (w, t, h)),
        ("wall_north", (x0 + w / 2, y0 + d - t / 2, h / 2), (w, t, h)),
        ("wall_west", (x0 + t / 2, y0 + d / 2, h / 2), (t, d, h)),
        ("wall_east", (x0 + w - t / 2, y0 + d / 2, h / 2), (t, d, h)),
    )
    for name, centre, size in walls:
        _make_box(stage, f"{room_path}/{name}", centre, size, colour)


def _apply_material(stage, prim_path, colour) -> None:
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, prim_path + "/material")
    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if prim is not None and prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


def _write_object(stage, obj_path, category, config) -> None:
    from pxr import Sdf

    x, y, z = config.position
    sx, sy, sz = config.size

    box = _make_box(stage, obj_path, (float(x), float(y), float(z)),
                    (float(sx), float(sy), float(sz)), _OBJECT_DISPLAY)

    prim = box.GetPrim()
    # Category on every object (generic placables included, not just task ones).
    prim.CreateAttribute("hearth_category", Sdf.ValueTypeNames.String).Set(category)
    # Real yaw as a rotate op so orientation affects the geometry, not just a
    # metadata tag. Applied about the object centre (see _make_box op order).
    if config.orientation_deg:
        from pxr import UsdGeom
        ops = UsdGeom.Xformable(box).GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                op.Set(float(config.orientation_deg))
                break
    # Task-object marker.
    if config.task_relevant:
        prim.CreateAttribute("hearth_task_object", Sdf.ValueTypeNames.String).Set(category)


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

    stage.DefinePrim("/hearth_scene/rooms")
    for i, room in enumerate(scene.rooms):
        # Unique prim path per room: same-typed rooms (e.g. two "kitchen"s)
        # would otherwise collide and overwrite each other in the stage.
        room_prim = f"/hearth_scene/rooms/room_{i}"
        _write_room(stage, room_prim, room, _room_colour(room.name))

    objects_prim = stage.DefinePrim("/hearth_scene/objects")
    objects_path = objects_prim.GetPath().pathString
    for i, (category, obj) in enumerate(scene.objects):
        # Unique prim path per object: objects of the same category would
        # otherwise collide (same prim path) and overwrite each other.
        obj_path = f"{objects_path}/obj_{i}"
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
