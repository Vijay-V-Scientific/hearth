"""Data types for the HEARTH data pipeline and storage format.

Defines the canonical record schema for episodes collected during policy
training / teleoperation, plus the storage container type. Plain dataclasses
so this module is importable without Isaac Sim or h5py.

Public API:
    EpisodeRecord, Observation, Action, TaskSignal, StorageFormat
    STORAGE_FORMATS (valid storage format names)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class StorageFormat(str, enum.Enum):
    """On-disk storage container format for episode data."""

    HDF5 = "hdf5"
    PARQUET = "parquet"
    JSONL = "jsonl"

    def extension(self) -> str:
        return {"hdf5": ".h5", "parquet": ".parquet", "jsonl": ".jsonl"}[self.value]


@dataclass
class Observation:
    """A single observation vector for one timestep."""

    step: int
    qpos: list[float]
    qvel: list[float]
    image: list[float] | None = None  # flattened RGB, optional
    text_embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "qpos": self.qpos,
            "qvel": self.qvel,
            "image": self.image,
            "text_embedding": self.text_embedding,
        }


@dataclass
class Action:
    """A single action (end-effector / joint targets) for one timestep."""

    step: int
    joint_targets: list[float]
    gripper: list[float] | None = None
    raw: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "joint_targets": self.joint_targets,
            "gripper": self.gripper,
            "raw": self.raw,
        }


@dataclass
class TaskSignal:
    """A scalar task progress / reward signal for one timestep."""

    step: int
    name: str
    value: float

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "name": self.name, "value": self.value}


@dataclass
class EpisodeRecord:
    """A complete episode: a trajectory of observations, actions, signals.

    This is the canonical record schema. The storage backends (h5/parquet/jsonl)
    serialize a list of EpisodeRecords to disk.
    """

    episode_id: str
    robot: str
    task: str
    seed: int
    dt: float
    observations: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    signals: list[TaskSignal] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "robot": self.robot,
            "task": self.task,
            "seed": self.seed,
            "dt": self.dt,
            "length": self.length,
            "observations": [o.to_dict() for o in self.observations],
            "actions": [a.to_dict() for a in self.actions],
            "signals": [s.to_dict() for s in self.signals],
            "metadata": self.metadata,
        }


STORAGE_FORMATS = {fmt.value: fmt for fmt in StorageFormat}
