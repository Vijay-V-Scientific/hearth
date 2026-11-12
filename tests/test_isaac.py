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
        assert spec.dof == len(spec.joints), f"{spec.name}: dof={spec.dof} joints={len(spec.joints)}"
        assert spec.state()["dof"] == spec.state()["num_joints"]


def test_g1_state_consistent():
    spec = HumanoidRobotSpec.unitree_g1()
    state = spec.state()
    assert state["dof"] == 22
    assert state["num_joints"] == 22


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
