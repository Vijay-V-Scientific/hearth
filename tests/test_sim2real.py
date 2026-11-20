"""Unit tests for the sim-vs-real joint-replay gate (brief gate G1)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hearth.sim2real import GATE_G1_MAX_RMSE_DEG, JointTrajectory, joint_replay_rmse

NAMES = ["left_hip_pitch_joint", "left_knee_joint", "right_elbow_joint"]


def _noise(base: np.ndarray, deg: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base + rng.normal(0.0, math.radians(deg), size=base.shape)


def _base(seed: int = 0, t: int = 200) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, size=(t, len(NAMES)))


def test_identical_trajectories_rmse_zero_and_pass() -> None:
    a = JointTrajectory(joint_names=list(NAMES), positions=_base(0))
    rep = joint_replay_rmse(a, a)
    assert rep.passed
    assert rep.mean_rmse_deg == pytest.approx(0.0, abs=1e-12)
    assert rep.num_frames == 200 and rep.num_joints == 3


def test_independent_small_noise_passes_the_gate() -> None:
    base = _base(1)
    sim = JointTrajectory(joint_names=list(NAMES), positions=_noise(base, 1.0, 2))
    real = JointTrajectory(joint_names=list(NAMES), positions=_noise(base, 1.0, 3))
    # independent ~1 deg scatter on each side -> RMSE ~ sqrt(2) deg < 3 deg gate
    rep = joint_replay_rmse(sim, real)
    assert rep.mean_rmse_deg < GATE_G1_MAX_RMSE_DEG


def test_known_offset_below_threshold_passes() -> None:
    sim = JointTrajectory(joint_names=list(NAMES), positions=np.zeros((50, 3)))
    real = JointTrajectory(joint_names=list(NAMES), positions=np.full((50, 3), math.radians(2.0)))
    rep = joint_replay_rmse(sim, real)
    assert rep.mean_rmse_deg == pytest.approx(2.0, abs=1e-9)
    assert rep.passed  # 2 deg is inside the 3 deg gate


def test_known_offset_above_threshold_fails() -> None:
    sim = JointTrajectory(joint_names=list(NAMES), positions=np.zeros((50, 3)))
    real = JointTrajectory(joint_names=list(NAMES), positions=np.full((50, 3), math.radians(5.0)))
    rep = joint_replay_rmse(sim, real)
    assert rep.mean_rmse_deg == pytest.approx(5.0, abs=1e-9)
    assert rep.max_joint in NAMES
    assert not rep.passed  # 5 deg exceeds the 3 deg gate


def test_joints_aligned_by_name_not_by_order() -> None:
    pos = _base(7, t=30)
    sim = JointTrajectory(joint_names=list(NAMES), positions=pos)
    perm = [2, 0, 1]
    real = JointTrajectory(joint_names=[NAMES[i] for i in perm], positions=pos[:, perm])
    assert joint_replay_rmse(sim, real).mean_rmse_deg == pytest.approx(0.0, abs=1e-12)


def test_subset_of_joints_is_allowed() -> None:
    pos = _base(4, t=10)
    sim = JointTrajectory(joint_names=list(NAMES), positions=pos)
    real_two = JointTrajectory(joint_names=NAMES[:2], positions=pos[:, :2])
    assert joint_replay_rmse(sim, real_two).num_joints == 2


def test_no_common_joints_raises() -> None:
    sim = JointTrajectory(joint_names=["a"], positions=np.zeros((5, 1)))
    real = JointTrajectory(joint_names=["b"], positions=np.zeros((5, 1)))
    with pytest.raises(ValueError, match="no joint names"):
        joint_replay_rmse(sim, real)


def test_bad_shape_and_nan_rejected() -> None:
    with pytest.raises(ValueError, match="joint names"):
        JointTrajectory(joint_names=["a", "b"], positions=np.zeros((5, 3)))
    pos = np.zeros((5, 2))
    pos[0, 0] = math.nan
    with pytest.raises(ValueError, match="NaN"):
        JointTrajectory(joint_names=["a", "b"], positions=pos)


def test_load_json_log(tmp_path: Path) -> None:
    p = tmp_path / "real.json"
    p.write_text(json.dumps({"joint_names": NAMES, "positions": [[0.1, 0.2, 0.3]] * 2}))
    traj = JointTrajectory.load(p)
    assert traj.joint_names == NAMES
    assert traj.positions.shape == (2, 3)
    assert joint_replay_rmse(traj, traj).passed
