"""Tests for deploy: calibration and ROS 2 bridge."""

import json

import numpy as np
import pytest

from hearth.deploy.calibration import (
    CalibrationError,
    CalibrationResult,
    CalibrationState,
    DomainRandomization,
    MotorLatencyModel,
    load_calibration_state,
    save_calibration_state,
)
from hearth.deploy.ros2_bridge import (
    BridgeError,
    BridgeConfig,
    Ros2BridgeConfig,
    SubgoalMessage,
    build_subgoal_message,
    load_bridge_config,
)

_TMP = "tests/_tmp_calibration.json"


def _cleanup(path: str) -> None:
    try:
        __import__("pathlib").Path(path).unlink()
    except OSError:
        pass


def test_domain_randomization_sample_shape():
    dr = DomainRandomization()
    sample = dr.sample()
    assert len(sample["com_offset"]) == 3
    assert 0.9 <= sample["mass_scale"] <= 1.1
    assert 0.7 <= sample["friction_scale"] <= 1.3


def test_motor_latency_fit_reduces_residual():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-0.1, 0.1, 200)
    vel = rng.uniform(-1.0, 1.0, 200)
    true_gain, true_lat, true_bias = 1.3, 0.02, -0.005
    measured = pos * true_gain - vel * true_lat * 0.01 + true_bias
    model = MotorLatencyModel()
    residual = model.fit(pos, vel, measured)
    assert model.fit_residual is not None
    assert residual < 1e-9
    assert model.gain == pytest.approx(true_gain, rel=1e-3)
    assert model.bias == pytest.approx(true_bias, rel=1e-3)


def test_motor_latency_requires_shape_match():
    with pytest.raises(CalibrationError):
        MotorLatencyModel().fit(np.zeros(3), np.zeros(2), np.zeros(3))


def test_calibration_save_load_roundtrip(tmp_path):
    state = CalibrationState(
        robot_id="g1-001",
        com=(0.02, -0.01, 0.35),
        total_mass=42.5,
        joint_offsets={"left_knee": 0.001},
        friction={"foot": 1.2},
        motor_models={"left_knee": MotorLatencyModel(latency_s=0.01, gain=1.1)},
        timestamp="2026-10-21T00:00:00Z",
    )
    path = tmp_path / "calib.json"
    save_calibration_state(state, path)
    loaded = load_calibration_state(path)
    assert loaded.robot_id == "g1-001"
    assert loaded.com == (0.02, -0.01, 0.35)
    assert loaded.motor_models["left_knee"].gain == pytest.approx(1.1)
    # Verify the file is valid JSON with expected keys
    with path.open() as fh:
        data = json.load(fh)
    assert data["robot_id"] == "g1-001"


def test_calibration_result_pass_and_fail():
    ok = CalibrationResult("fall_rate", before=2.0, after=0.5, target=1.0)
    assert ok.passed is True
    bad = CalibrationResult("fall_rate", before=2.0, after=1.5, target=1.0)
    assert bad.passed is False
    no_target = CalibrationResult("fall_rate", before=2.0, after=1.5)
    assert no_target.passed is True


def test_subgoal_message_validate():
    msg = build_subgoal_message(
        seq=0,
        subgoal_type="xyz_position",
        position=(0.3, -0.1, 0.5),
        orientation=(1.0, 0.0, 0.0, 0.0),
        grip_width=0.08,
        confidence=0.95,
    )
    assert msg.validate() is msg
    assert msg.to_dict()["seq"] == 0


def test_subgoal_message_invalid_type():
    with pytest.raises(BridgeError):
        build_subgoal_message(
            seq=1,
            subgoal_type="not_a_type",
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )


def test_subgoal_message_bad_confidence():
    with pytest.raises(BridgeError):
        build_subgoal_message(
            seq=2,
            subgoal_type="subgoal_id",
            position=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            confidence=1.5,
        )


def test_bridge_config_load(tmp_path):
    cfg_path = tmp_path / "bridge.yaml"
    cfg_path.write_text(
        "bridge:\n"
        "  topic_planner: /planner/subgoals\n"
        "  topic_controller: /controller/subgoals\n"
        "  frame_id: base_link\n"
        "  message_rate_hz: 5.0\n"
        "  timeout_s: 0.5\n"
        "qos_profile: reliable\n"
    )
    cfg = load_bridge_config(cfg_path)
    assert isinstance(cfg, Ros2BridgeConfig)
    assert cfg.bridge.topic_planner == "/planner/subgoals"
    assert cfg.bridge.message_rate_hz == pytest.approx(5.0)
    assert cfg.qos_profile == "reliable"
