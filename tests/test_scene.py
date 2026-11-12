"""Tests for the procedural scene generator and USD writer (pxr available)."""

import hearth.scene as scene
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
    from hearth.scene.generator import _TASK_OBJECTS, _PLACABLES
    cfg = ApartmentConfig(width=8.0, depth=6.0, seed=3)
    s = generate_scene(cfg)
    for cat, obj in s.objects:
        if cat in _TASK_OBJECTS:
            assert obj.fixed == _TASK_OBJECTS[cat][1]
        elif cat in _PLACABLES:
            assert obj.fixed == _PLACABLES[cat][1]


def test_write_usd_roundtrip(tmp_path):
    out = tmp_path / "scenes" / "a.usd"
    path = write_usd_from_config(ApartmentConfig(width=8.0, depth=6.0, seed=5), str(out))
    assert out.exists()
    from pxr import Usd
    stage = Usd.Stage.Open(str(out))
    root = stage.GetPrimAtPath("/hearth_scene")
    assert root.IsValid()
    assert root.GetAttribute("hearth_scene_id").Get()
    assert root.GetAttribute("hearth_num_objects").Get() == len(generate_scene(ApartmentConfig(seed=5)).objects)


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
