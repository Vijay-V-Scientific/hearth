"""Tests for the Isaac layer that do NOT need the Isaac Sim runtime.

These cover the parts of the Isaac layer that are importable on host: the
robot spec consistency, the control-layer frequency contract, the deferred
runtime error path, and the constructor helpers.
"""

import pytest

from hearth.isaac import (
    ControlLayers,
    HumanoidRobotSpec,
    IsaacLabRunner,
    load_scene,
)


def test_g1_dof_matches_num_joints():
    for spec in (
        HumanoidRobotSpec.unitree_g1(),
        HumanoidRobotSpec.fourier_gr1(),
        HumanoidRobotSpec.default(),
    ):
        assert spec.dof == len(spec.joints), (
            f"{spec.name}: dof={spec.dof} joints={len(spec.joints)}"
        )
        assert spec.state()["dof"] == spec.state()["num_joints"]


def test_g1_state_consistent():
    spec = HumanoidRobotSpec.unitree_g1()
    state = spec.state()
    assert state["dof"] == 23
    assert state["num_joints"] == 23


def test_g1_physical_spec_matches_official():
    """Pin the G1 spec to Unitree's published numbers (basic version).

    Corrected from an earlier version that claimed 1.8 m / 30 kg / 22 DoF
    with a single prismatic ankle. Official G1 (excl. end-effector):
    1320 mm standing, ~35 kg with battery, 23 DoF = 6/leg (hip pitch/roll/
    yaw, knee, ankle pitch + ankle roll, both revolute), 1 waist yaw,
    5/arm. Sources: unitree.com/en/g1, support.unitree.com G1 developer guide.
    """
    spec = HumanoidRobotSpec.unitree_g1()
    assert spec.height_m == pytest.approx(1.32)
    assert spec.mass_kg == pytest.approx(35.0)
    assert spec.dof == 23
    assert len(spec.joints) == 23

    names = [j.name for j in spec.joints]
    # Every leg has both ankle joints, and they are revolute (not prismatic).
    for side in ("left", "right"):
        assert f"{side}_ankle_pitch" in names
        assert f"{side}_ankle_roll" in names
    for j in spec.joints:
        if "ankle" in j.name:
            assert j.joint_type == "revolute", f"{j.name} must be revolute"

    # 6 leg + 1 waist + 5 per arm == 23, and DOF indices are contiguous.
    legs = [j for j in spec.joints if any(k in j.name for k in ("hip", "knee", "ankle"))]
    assert len(legs) == 12
    assert sum(1 for j in spec.joints if j.name == "waist_yaw") == 1
    arms = [j for j in spec.joints if any(k in j.name for k in ("shoulder", "elbow", "wrist"))]
    assert len(arms) == 10
    assert sorted(j.dof_index for j in spec.joints) == list(range(23))
    # Hands are an end-effector, modelled separately, not in body DoF.
    assert spec.hand_dof == 20
    assert "ankle_roll_link" in spec.contact_bodies[0]


def test_control_layer_frequencies():
    layers = ControlLayers()
    # Objective 6: Layer 0 at 500 Hz, VLA planner at >=5 Hz.
    assert layers.control_hz(0) == 500
    assert layers.control_hz(1) == pytest.approx(50.0)
    assert layers.plan_hz(2) == pytest.approx(5.0)
    assert layers.control_hz(0) > layers.control_hz(1) > layers.control_hz(2)


def test_runner_constructs_without_runtime():
    runner = IsaacLabRunner(
        scene_usd="/tmp/scenes/a.usd",
        robot=HumanoidRobotSpec.unitree_g1(),
        control=ControlLayers(),
    )
    assert runner.scene_usd == "/tmp/scenes/a.usd"
    assert runner.sim_dt_s == 0.002


def test_load_scene_helper():
    runner = load_scene(
        "/tmp/scenes/a.usd",
        HumanoidRobotSpec.unitree_g1(),
        ControlLayers(),
    )
    assert isinstance(runner, IsaacLabRunner)
    assert runner.robot is not None


def test_run_task_defers_to_runtime():
    runner = IsaacLabRunner(
        scene_usd="/tmp/scenes/a.usd",
        robot=HumanoidRobotSpec.unitree_g1(),
        control=ControlLayers(),
    )
    with pytest.raises(NotImplementedError):
        runner.run_task(task=None, seed=0)
