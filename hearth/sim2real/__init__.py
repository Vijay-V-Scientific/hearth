"""Sim-to-real comparison tools (brief gates). Importable without Isaac Sim."""

from .replay import (
    GATE_G1_MAX_RMSE_DEG,
    JointTrajectory,
    ReplayReport,
    joint_replay_rmse,
)

__all__ = [
    "GATE_G1_MAX_RMSE_DEG",
    "JointTrajectory",
    "ReplayReport",
    "joint_replay_rmse",
]
