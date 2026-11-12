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

import os
from typing import Any

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


def _make_box(stage, path: str, centre, size, colour) -> "Any":
    """Define a unit cube given a world-space centre and full extents.

    The cube is scaled from a unit (side 1, centred at the local origin) box,
    so ``size`` is the true bounding-box extent and ``centre`` is the world
    position of the box centre. Returns the prim.
    """
    from pxr import Gf, UsdGeom

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


def _split_span(lo: float, hi: float, gaps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Solid ``[lo, hi]`` minus ``gaps`` -> list of solid sub-spans (may be empty)."""
    gaps = sorted(g for g in gaps if g[1] > lo and g[0] < hi)
    pieces: list[tuple[float, float]] = []
    cur = lo
    for gs, ge in gaps:
        if gs > cur:
            pieces.append((cur, gs))
        cur = max(cur, ge)
    if cur < hi:
        pieces.append((cur, hi))
    return pieces


_DOOR_HEIGHT = 2.1
_DOOR_LEAF_COLOUR = (0.45, 0.3, 0.15)


def _wall_gaps_for(doorways, room_index, axis: str, edge: float):
    """Doorways whose opening lies on this wall plane.

    A doorway's plane coordinate is ``edge`` and its normal axis is ``axis``;
    its opening runs ``along_start..along_end``. It belongs to this room if the
    room index is one of the doorway's two incident rooms.
    """
    hits = []
    for dw in doorways:
        if dw.axis != axis or room_index not in dw.rooms:
            continue
        if abs(dw.fixed - edge) > 1e-6:
            continue
        hits.append(dw)
    return hits


def _write_room(stage, room_path, room, colour, index, doorways) -> None:
    """One room: a floor slab plus four walls, each split at its doorways.

    Rooms used to be sealed (four unbroken walls, no openings) and a "door" was
    a stray box dropped wherever placement put it. Walls are now cut open at
    real doorways between rooms, and a door leaf is fitted flush in each opening
    (drawn once per doorway, from the lower-indexed room).
    """
    from pxr import Gf, Sdf, UsdGeom

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

    # Each wall: normal axis, the shared-edge plane coordinate, the box centre
    # along the normal (inset by t/2), and the wall's along-axis span. Walls are
    # split into solid pieces at every doorway gap and the door leaf drawn in
    # the opening (only from the lower room index, so a shared wall gets one).
    wall_defs = (
        ("wall_south", "y", y0, y0 + t / 2, x0, x0 + w),
        ("wall_north", "y", y0 + d, y0 + d - t / 2, x0, x0 + w),
        ("wall_west", "x", x0, x0 + t / 2, y0, y0 + d),
        ("wall_east", "x", x0 + w, x0 + w - t / 2, y0, y0 + d),
    )
    for name, axis, edge, center, span_lo, span_hi in wall_defs:
        dws = _wall_gaps_for(doorways, index, axis, edge)
        gaps = [(dw.along_start, dw.along_end) for dw in dws]
        solid = _split_span(span_lo, span_hi, gaps)
        for k, (a, b) in enumerate(solid):
            if axis == "y":
                centre = ((a + b) / 2, center, h / 2)
                size = (b - a, t, h)
            else:
                centre = (center, (a + b) / 2, h / 2)
                size = (t, b - a, h)
            _make_box(stage, f"{room_path}/{name}_{k}", centre, size, colour)
        # Door leaf flush in the opening. Drawn once per doorway, from its
        # lower room index, so the coincident walls of the two rooms do not
        # stack two leaves in the same gap.
        for dw in sorted(dws, key=lambda d: d.along_start):
            if dw.rooms[0] != index:
                continue
            gs, ge = dw.along_start, dw.along_end
            if axis == "y":
                centre = ((gs + ge) / 2, center, _DOOR_HEIGHT / 2)
                size = (ge - gs, t * 0.6, _DOOR_HEIGHT)
            else:
                centre = (center, (gs + ge) / 2, _DOOR_HEIGHT / 2)
                size = (t * 0.6, ge - gs, _DOOR_HEIGHT)
            leaf = _make_box(
                stage,
                f"{room_path}/door_{name}_{int(gs * 1000)}",
                centre,
                size,
                _DOOR_LEAF_COLOUR,
            )
            leaf.GetPrim().CreateAttribute("hearth_doorway", Sdf.ValueTypeNames.String).Set(
                "closed"
            )


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

    box = _make_box(
        stage,
        obj_path,
        (float(x), float(y), float(z)),
        (float(sx), float(sy), float(sz)),
        _OBJECT_DISPLAY,
    )

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

    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetMetadata("metersPerUnit", 1.0)  # metres; Isaac Sim sets its own scale on load

    root = stage.DefinePrim("/hearth_scene")
    root.CreateAttribute("hearth_scene_id", Sdf.ValueTypeNames.String).Set(scene.metadata.scene_id)
    root.CreateAttribute("hearth_seed", Sdf.ValueTypeNames.Int).Set(scene.metadata.seed)
    root.CreateAttribute("hearth_num_rooms", Sdf.ValueTypeNames.Int).Set(scene.metadata.num_rooms)
    root.CreateAttribute("hearth_total_area_m2", Sdf.ValueTypeNames.Float).Set(
        scene.metadata.total_area
    )
    root.CreateAttribute("hearth_num_objects", Sdf.ValueTypeNames.Int).Set(
        scene.metadata.num_objects
    )
    root.CreateAttribute("hearth_num_task_objects", Sdf.ValueTypeNames.Int).Set(
        scene.metadata.num_task_objects
    )

    stage.DefinePrim("/hearth_scene/rooms")
    for i, room in enumerate(scene.rooms):
        # Unique prim path per room: same-typed rooms (e.g. two "kitchen"s)
        # would otherwise collide and overwrite each other in the stage.
        room_prim = f"/hearth_scene/rooms/room_{i}"
        _write_room(stage, room_prim, room, _room_colour(room.name), i, scene.doorways)

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
