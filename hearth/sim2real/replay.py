"""Sim-vs-real joint-replay gate (brief gate G1).

G1 asks one measurable question: *does the simulator track the real robot?*
The procedure from the brief is to replay a recording of real joint commands
into the simulated robot and require the realised joint angles to match the
real robot's angles to within **3 degrees RMSE**.

This module is the pure comparison: it takes the sim trajectory and the real
trajectory (joint names + angles over time) and reports per-joint and mean
RMSE and whether the gate passes. It needs no Isaac Sim, no GPU and no robot,
so it is fully unit-tested on synthetic data. The Isaac driver in
``scripts/hearth_g1_replay.py`` is what feeds it real data.

Import convention: joint angles are in **radians** (the convention of the
robot models and of URDF); the report is given in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GATE_G1_MAX_RMSE_DEG = 3.0
"""The brief's G1 acceptance threshold: mean per-joint RMSE must be below this."""


@dataclass
class JointTrajectory:
    """A sequence of joint angles: ``positions`` is ``(T, J)`` in radians."""

    joint_names: list[str]
    positions: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        if pos.ndim != 2:
            raise ValueError(f"positions must be 1-D or 2-D, got {pos.ndim}-D")
        if pos.shape[1] != len(self.joint_names):
            raise ValueError(
                f"{pos.shape[1]} columns of positions but {len(self.joint_names)} joint names"
            )
        if np.any(~np.isfinite(pos)):
            raise ValueError("positions contain NaN or Inf")
        self.positions = pos

    @classmethod
    def load(cls, path: str | Path) -> "JointTrajectory":
        """Load a log as ``.json`` or ``.npz`` (see examples/g1_real_log_format...)."""
        p = Path(path)
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=True) as z:
                names = [str(n) for n in z["joint_names"]]
                pos = z["positions"]
        elif p.suffix == ".json":
            import json

            data: dict[str, Any] = json.loads(p.read_text())
            names = [str(n) for n in data["joint_names"]]
            pos = np.asarray(data["positions"], dtype=float)
        else:
            raise ValueError(f"unsupported log suffix: {p.suffix!r} (use .json or .npz)")
        return cls(joint_names=names, positions=pos)


@dataclass
class ReplayReport:
    """Outcome of a sim-vs-real joint-replay comparison (all angles in degrees)."""

    threshold_deg: float
    per_joint_rmse_deg: dict[str, float]
    mean_rmse_deg: float
    max_joint: str
    max_rmse_deg: float
    num_frames: int
    num_joints: int

    @property
    def passed(self) -> bool:
        return self.mean_rmse_deg < self.threshold_deg


def joint_replay_rmse(
    sim: JointTrajectory,
    real: JointTrajectory,
    *,
    threshold_deg: float = GATE_G1_MAX_RMSE_DEG,
) -> ReplayReport:
    """RMSE (degrees) between simulated and real joint angles, aligned by name.

    Joints present in both trajectories are compared (a subset of the real
    recording is fine). Frames are aligned from the start over the shorter of
    the two recordings.
    """
    sim_idx = {n: i for i, n in enumerate(sim.joint_names)}
    real_idx = {n: i for i, n in enumerate(real.joint_names)}
    common = sorted(set(sim_idx) & set(real_idx))
    if not common:
        raise ValueError("sim and real trajectories share no joint names")

    n_frames = min(sim.positions.shape[0], real.positions.shape[0])
    if n_frames == 0:
        raise ValueError("an empty trajectory has no frames to compare")

    s = sim.positions[:n_frames, [sim_idx[n] for n in common]]
    r = real.positions[:n_frames, [real_idx[n] for n in common]]
    err_deg = np.degrees(s - r)
    per_joint = np.sqrt(np.mean(err_deg**2, axis=0))

    per_map = {name: float(v) for name, v in zip(common, per_joint)}
    max_j = common[int(np.argmax(per_joint))]
    return ReplayReport(
        threshold_deg=float(threshold_deg),
        per_joint_rmse_deg=per_map,
        mean_rmse_deg=float(np.mean(per_joint)),
        max_joint=max_j,
        max_rmse_deg=float(per_joint[int(np.argmax(per_joint))]),
        num_frames=int(n_frames),
        num_joints=len(common),
    )
