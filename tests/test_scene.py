"""Tests for the procedural scene generator and USD writer (pxr available)."""

import pytest

from hearth.scene import (
    ApartmentConfig,
    generate_scene,
    write_usd,
    write_usd_from_config,
)


def test_scene_is_deterministic_given_seed():
    cfg = ApartmentConfig(width=8.0, depth=6.0, seed=42)
    s1 = generate_scene(cfg)
    s2 = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=42))
    assert s1.metadata.scene_id == s2.metadata.scene_id
    assert [c.category for _, c in s1.objects] == [c.category for _, c in s2.objects]
    for (_, a), (_, b) in zip(s1.objects, s2.objects):
        assert a.position == b.position


def test_different_seed_changes_scene():
    s1 = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=1))
    s2 = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=2))
    assert s1.metadata.scene_id != s2.metadata.scene_id


def test_metadata_counts():
    cfg = ApartmentConfig(width=8.0, depth=6.0, seed=7)
    s = generate_scene(cfg)
    assert s.metadata.num_rooms == len(s.rooms)
    assert s.metadata.num_objects == len(s.objects)
    assert s.metadata.num_task_objects == sum(1 for _, o in s.objects if o.task_relevant)
    assert s.metadata.total_area == cfg.width * cfg.depth
    assert s.metadata.num_task_objects <= cfg.task_objects_per_scene


def test_fixed_and_movable_flags_match_registry():
    from hearth.scene.generator import _PLACABLES, _TASK_OBJECTS

    cfg = ApartmentConfig(width=8.0, depth=6.0, seed=3)
    s = generate_scene(cfg)
    for cat, obj in s.objects:
        if cat in _TASK_OBJECTS:
            assert obj.fixed == _TASK_OBJECTS[cat][1]
        elif cat in _PLACABLES:
            assert obj.fixed == _PLACABLES[cat][1]


def test_write_usd_roundtrip(tmp_path):
    out = tmp_path / "scenes" / "a.usd"
    write_usd_from_config(ApartmentConfig(width=8.0, depth=6.0, seed=5), str(out))
    assert out.exists()
    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    root = stage.GetPrimAtPath("/hearth_scene")
    assert root.IsValid()
    assert root.GetAttribute("hearth_scene_id").Get()
    assert root.GetAttribute("hearth_num_objects").Get() == len(
        generate_scene(ApartmentConfig(seed=5)).objects
    )


def test_write_usd_handles_same_category_objects(tmp_path):
    # Multiple objects of the same category must not collide in prim paths.
    cfg = ApartmentConfig(width=12.0, depth=10.0, seed=9, task_objects_per_scene=6)
    s = generate_scene(cfg)
    out = tmp_path / "multi.usd"
    write_usd(s, str(out))
    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    # Count object boxes written under /hearth_scene/objects
    objs = stage.GetPrimAtPath("/hearth_scene/objects").GetChildren()
    assert len(objs) == len(s.objects)


# --- Reuse-the-same-config determinism (regression) -------------------------


def test_reusing_config_object_is_repeatable():
    # The bug: ApartmentConfig stored a live RNG on itself, so calling
    # generate_scene(cfg) twice advanced it and yielded a different apartment.
    cfg = ApartmentConfig(width=8.0, depth=6.0, seed=11)
    s1 = generate_scene(cfg)
    s2 = generate_scene(cfg)  # same object again
    sig1 = [(category, obj.position) for category, obj in s1.objects]
    sig2 = [(category, obj.position) for category, obj in s2.objects]
    assert sig1 == sig2
    assert sig1  # and it actually generated something


# --- USD geometry the brief actually needs--


def _stage(tmp_path, seed=5):
    from pxr import Usd

    out = tmp_path / f"s{seed}.usd"
    write_usd_from_config(ApartmentConfig(width=9.0, depth=7.0, seed=seed), str(out))
    return Usd.Stage.Open(str(out))


def test_write_usd_has_physics_scene_and_colliders(tmp_path):
    # The scene used to be inert geometry (no physics), so it was unusable in a
    # sim: nothing rested on the floor and the robot fell through the world. It
    # now carries a gravity-bearing PhysicsScene and real collision.
    from pxr import UsdPhysics

    stage = _stage(tmp_path)
    ps = stage.GetPrimAtPath("/physicsScene")
    assert ps.IsValid() and ps.GetTypeName() == "PhysicsScene"
    gravity = ps.GetAttribute("physics:gravity").Get()
    assert gravity is not None and gravity[2] < 0.0

    colliders = rigid = movable_seen = 0

    def walk(p):
        nonlocal colliders, rigid, movable_seen
        if p.IsValid():
            if p.HasAPI(UsdPhysics.CollisionAPI):
                colliders += 1
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid += 1
        for c in p.GetChildren():
            walk(c)

    walk(stage.GetPseudoRoot())
    assert colliders > 0, "no colliders authored"
    assert rigid > 0, "no movable rigid bodies authored"
    # Floor must be a static collider, not a falling rigid body.
    floor = stage.GetPrimAtPath("/hearth_scene/rooms/room_0/floor")
    assert floor.HasAPI(UsdPhysics.CollisionAPI)
    assert not floor.HasAPI(UsdPhysics.RigidBodyAPI)
    # Every movable rigid body carries a sane positive mass.
    for obj in stage.GetPrimAtPath("/hearth_scene/objects").GetChildren():
        if obj.HasAPI(UsdPhysics.RigidBodyAPI):
            movable_seen += 1
            mass = obj.GetAttribute("physics:mass").Get()
            assert mass is not None and 0.0 < mass <= 500.0
    assert movable_seen > 0


def test_write_usd_positions_objects_not_at_origin(tmp_path):
    # Objects used to all sit at the world origin because no translate op was
    # ever applied. Read the local transforms back and require distinct,
    # non-origin placements.
    from pxr import UsdGeom

    stage = _stage(tmp_path)
    objs = list(stage.GetPrimAtPath("/hearth_scene/objects").GetChildren())
    assert objs
    translations = []
    for ch in objs:
        m = UsdGeom.Xformable(ch).GetLocalTransformation()
        t = m.ExtractTranslation()
        translations.append((t[0], t[1], t[2]))
    # At least one object must be off the origin in x or y.
    assert any(abs(x) > 1e-6 or abs(y) > 1e-6 for x, y, _ in translations)
    # And placements must differ from one another (not all collapsed).
    assert len(set(translations)) > 1


def test_write_usd_records_category_on_every_object(tmp_path):
    stage = _stage(tmp_path)
    objs = list(stage.GetPrimAtPath("/hearth_scene/objects").GetChildren())
    assert objs
    for ch in objs:
        assert ch.HasAttribute("hearth_category")
        assert isinstance(ch.GetAttribute("hearth_category").Get(), str)


def test_write_usd_rooms_have_walls(tmp_path):
    # Rooms used to be a bare floor quad with no enclosure. Walls are now split
    # at doorways, so they carry a _<k> suffix; each side must still be present.
    stage = _stage(tmp_path)
    children = {
        c.GetName() for c in stage.GetPrimAtPath("/hearth_scene/rooms/room_0").GetChildren()
    }
    assert "floor" in children
    for side in ("wall_north", "wall_south", "wall_east", "wall_west"):
        assert any(c.startswith(side) for c in children), f"{side} missing"


def test_rooms_connect_through_doorways():
    # Rooms were sealed boxes; a stray "door" box sat in the middle of one and
    # the robot could never leave a room. Doorways in shared walls must connect
    # every room to the rest of the apartment.
    for seed in range(50):
        s = generate_scene(ApartmentConfig(width=8.0, depth=6.0, room_count=4, seed=seed))
        n = len(s.rooms)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for dw in s.doorways:
            a, b = dw.rooms
            parent[find(a)] = find(b)
        assert n == 1 or len({find(i) for i in range(n)}) == 1, f"seed {seed} not connected"


def test_write_usd_doorways_are_openings_with_leaves(tmp_path):
    # A doorway must cut a gap in the wall (so the wall is split into >=2 pieces
    # where a leaf sits) and a door leaf prim must exist flush in the opening.
    s = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=1))
    assert s.doorways
    out = tmp_path / "doors.usd"
    write_usd(s, str(out))
    from pxr import Usd

    stage = Usd.Stage.Open(str(out))
    leaves = []
    walls = []

    def walk(p):
        n = p.GetName()
        if n.startswith("door_"):
            leaves.append(p)
        if n.startswith("wall_"):
            walls.append(p.GetPath().pathString)
        for c in p.GetChildren():
            walk(c)

    walk(stage.GetPseudoRoot())
    # Exactly one leaf per doorway.
    assert len(leaves) == len(s.doorways)
    # Each leaf carries the closed/openable marker.
    for leaf in leaves:
        assert leaf.HasAttribute("hearth_doorway")
    # Doorways split walls: far more wall pieces than the 4-per-room a sealed
    # apartment would have.
    assert len(walls) > 4 * len(s.rooms)


def test_write_usd_yaw_is_a_real_rotation(tmp_path):
    # orientation_deg used to be recorded but never applied. A 90-degree yaw
    # must rotate the object's local x-axis onto ~+y.
    from pxr import Usd, UsdGeom

    from hearth.scene.types import ObjectConfig, Scene, SceneMetadata

    obj = ObjectConfig("cabinet", (2.0, 1.0, 0.5), (0.4, 0.4, 1.8), orientation_deg=90.0)
    sc = Scene(rooms=[], objects=[("cabinet", obj)], metadata=SceneMetadata("x", 0, 0, 0.0, 1, 1))
    out = tmp_path / "yaw.usd"
    write_usd(sc, str(out))
    stage = Usd.Stage.Open(str(out))
    ch = list(stage.GetPrimAtPath("/hearth_scene/objects").GetChildren())[0]
    m = UsdGeom.Xformable(ch).GetLocalTransformation()
    # Gf.Matrix4d indexes m[row][col]. The first column is the image of the
    # object's local x-axis (its x-extent, 0.4). With no yaw that entry sits on
    # the diagonal (m[0][0]); a 90-degree yaw moves it onto the y-axis, so
    # m[0][0] goes to zero and |m[1][0]| picks up the full 0.4. Reading the raw
    # entries avoids any convention/scale coupling in the extraction helpers.
    assert m[0][0] == pytest.approx(0.0, abs=1e-6)  # x-extent no longer along x
    assert abs(m[1][0]) == pytest.approx(0.4, abs=1e-6)  # ...now along y
    assert m[2][0] == pytest.approx(0.0, abs=1e-6)  # still flat in z
    # The scale factor is preserved under rotation (column norm is invariant).
    colnorm_x = (m[0][0] ** 2 + m[1][0] ** 2 + m[2][0] ** 2) ** 0.5
    assert colnorm_x == pytest.approx(0.4, abs=1e-6)


def test_write_usd_object_scale_matches_extent(tmp_path):
    # The unit cube must be scaled so its world extents equal the config size.
    from pxr import Usd, UsdGeom

    from hearth.scene.types import ObjectConfig, Scene, SceneMetadata

    obj = ObjectConfig("box", (3.0, 4.0, 0.25), (0.5, 0.6, 0.7))
    sc = Scene(rooms=[], objects=[("box", obj)], metadata=SceneMetadata("x", 0, 0, 0.0, 1, 1))
    out = tmp_path / "scale.usd"
    write_usd(sc, str(out))
    stage = Usd.Stage.Open(str(out))
    ch = list(stage.GetPrimAtPath("/hearth_scene/objects").GetChildren())[0]
    # The cube is a unit cube scaled by the config size and moved to the
    # configured centre. Gf.Matrix4d indexes m[row][col] with translation in
    # row 3; with no yaw each axis column norm equals the configured extent.
    m = UsdGeom.Xformable(ch).GetLocalTransformation()
    centre = m.ExtractTranslation()
    assert (centre[0], centre[1], centre[2]) == pytest.approx((3.0, 4.0, 0.25), abs=1e-6)

    def colnorm(j):
        return (m[0][j] ** 2 + m[1][j] ** 2 + m[2][j] ** 2) ** 0.5

    assert (colnorm(0), colnorm(1), colnorm(2)) == pytest.approx((0.5, 0.6, 0.7), abs=1e-6)


# --- objects rest on the floor, and do not overlap (regression) --------------


def test_objects_rest_on_floor_not_half_buried():
    # Every object used to be returned with its centre at z = 0, so each box sat
    # half underground. The base (z - sz/2) must now be at ground level.
    for seed in range(20):
        s = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=seed))
        for _, o in s.objects:
            assert o.position[2] == pytest.approx(o.size[2] / 2.0, abs=1e-9)


def test_scene_objects_do_not_overlap():
    # Spacing used to compare centre points against a fixed gap and ignore
    # object size, so large and small objects piled up. Across many scenes,
    # axis-aligned boxes (excluding thin floor rugs) must not intersect.
    for seed in range(50):
        s = generate_scene(ApartmentConfig(width=8.0, depth=6.0, seed=seed))
        boxes = [
            (x, y, z, sx, sy, sz)
            for _, o in s.objects
            for (x, y, z) in [o.position]
            for (sx, sy, sz) in [o.size]
            if sz > 0.05
        ]
        for i in range(len(boxes)):
            xi, yi, zi, sxi, syi, szi = boxes[i]
            for j in range(i + 1, len(boxes)):
                xj, yj, zj, sxj, syj, szj = boxes[j]
                inter = (
                    abs(xi - xj) < (sxi + sxj) / 2 - 1e-9
                    and abs(yi - yj) < (syi + syj) / 2 - 1e-9
                    and abs(zi - zj) < (szi + szj) / 2 - 1e-9
                )
                assert not inter, f"overlap in seed {seed}: {boxes[i]} vs {boxes[j]}"
