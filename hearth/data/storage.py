"""Storage backends for HEARTH episode records.

Writes and reads EpisodeRecords to HDF5, Parquet and JSONL. The HDF5 backend
is the production format (used by Isaac Lab / RL training pipelines). The
Parquet and JSONL backends are SCAFFOLD: importable without Isaac Sim, useful
for small episodes, inspection and CI.

Public API:
    save_episodes(path, records, fmt="hdf5") -> Path
    load_episodes(path) -> list[EpisodeRecord]
    storage_format(path) -> StorageFormat
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .types import EpisodeRecord, StorageFormat

if TYPE_CHECKING:
    from pathlib import Path


def storage_format(path: str | Path) -> StorageFormat:
    """Detect the storage format of a file by its extension."""
    ext = Path(path).suffix.lower()
    for fmt in StorageFormat:
        if fmt.extension() == ext:
            return fmt
    raise ValueError(f"unknown storage format for file: {path}")


def save_episodes(
    path: str | Path,
    records: Sequence[EpisodeRecord],
    fmt: str | StorageFormat = StorageFormat.HDF5,
) -> Path:
    """Serialize records to disk in the requested format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = StorageFormat(fmt)
    if fmt == StorageFormat.HDF5:
        _save_hdf5(p, records)
    elif fmt == StorageFormat.PARQUET:
        _save_parquet(p, records)
    elif fmt == StorageFormat.JSONL:
        _save_jsonl(p, records)
    else:
        raise ValueError(f"unsupported storage format: {fmt}")
    return p


def load_episodes(path: str | Path) -> list[EpisodeRecord]:
    """Load records from disk, detecting the format from the extension."""
    p = Path(path)
    fmt = storage_format(p)
    if fmt == StorageFormat.HDF5:
        return _load_hdf5(p)
    elif fmt == StorageFormat.PARQUET:
        return _load_parquet(p)
    elif fmt == StorageFormat.JSONL:
        return _load_jsonl(p)
    raise ValueError(f"unsupported storage format: {fmt}")


#-----------
# HDF5 (production)
#-----------


def _save_hdf5(path: Path, records: Sequence[EpisodeRecord]) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - h5py present in this env
        raise ImportError(
            "h5py is required to write HDF5 files. Install it with: pip install h5py"
        ) from exc

    import numpy as np

    with h5py.File(path, "w") as f:
        for rec in records:
            g = f.create_group(rec.episode_id)
            g.attrs["robot"] = rec.robot
            g.attrs["task"] = rec.task
            g.attrs["seed"] = rec.seed
            g.attrs["dt"] = rec.dt
            g.create_dataset("observations/step", data=np.array([o.step for o in rec.observations], dtype=np.int32))
            g.create_dataset("observations/qpos", data=np.array([o.qpos for o in rec.observations], dtype=np.float32))
            g.create_dataset("observations/qvel", data=np.array([o.qvel for o in rec.observations], dtype=np.float32))
            g.create_dataset("actions/joint_targets", data=np.array([a.joint_targets for a in rec.actions], dtype=np.float32))
            g.create_dataset("actions/gripper", data=np.array([a.gripper for a in rec.actions], dtype=np.float32))
            from collections import defaultdict
            _by_name: dict[str, list[float]] = defaultdict(list)
            for s in rec.signals:
                _by_name[s.name].append(s.value)
            for name, values in _by_name.items():
                g.create_dataset(f"signals/{name}", data=np.array(values, dtype=np.float32))
            for k, v in rec.metadata.items():
                g.attrs[f"meta_{k}"] = v


def _load_hdf5(path: Path) -> list[EpisodeRecord]:
    import h5py

    import numpy as np

    from .types import Action, Observation, TaskSignal

    records: list[EpisodeRecord] = []
    with h5py.File(path, "r") as f:
        for episode_id, g in f.items():
            obs_steps = g["observations/step"][...]
            obs = [
                Observation(
                    step=int(obs_steps[i]),
                    qpos=np.asarray(g["observations/qpos"][...][i]).tolist(),
                    qvel=np.asarray(g["observations/qvel"][...][i]).tolist(),
                )
                for i in range(len(obs_steps))
            ]
            jt = g["actions/joint_targets"][...]
            gripper = g["actions/gripper"][...]
            actions = [
                Action(
                    step=int(obs_steps[i]),
                    joint_targets=np.asarray(jt[i]).tolist(),
                    gripper=np.asarray(gripper[i]).tolist(),
                )
                for i in range(len(obs_steps))
            ]
            signals = []
            for k, v in g["signals"].items():
                vals = np.asarray(v)[...]
                for i in range(len(obs_steps)):
                    signals.append(TaskSignal(step=int(obs_steps[i]), name=k, value=float(vals[i])))
            meta = {
                k[len("meta_"):]: v for k, v in g.attrs.items() if k.startswith("meta_")
            }
            rec = EpisodeRecord(
                episode_id=episode_id,
                robot=str(g.attrs["robot"]),
                task=str(g.attrs["task"]),
                seed=int(g.attrs["seed"]),
                dt=float(g.attrs["dt"]),
                observations=obs,
                actions=actions,
                signals=signals,
                metadata=meta,
            )
            records.append(rec)
    return records


#-----------
# Parquet (scaffold)
#-----------


def _save_parquet(path: Path, records: Sequence[EpisodeRecord]) -> None:
    import pandas as pd

    rows = [rec.to_dict() for rec in records]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, engine="pyarrow")


def _load_parquet(path: Path) -> list[EpisodeRecord]:
    import pandas as pd

    from .types import Action, Observation, TaskSignal

    df = pd.read_parquet(path)
    records: list[EpisodeRecord] = []
    for _, row in df.iterrows():
        obs = [
            Observation(step=int(o["step"]), qpos=o["qpos"], qvel=o["qvel"])
            for o in row["observations"]
        ]
        actions = [
            Action(step=int(a["step"]), joint_targets=a["joint_targets"], gripper=a["gripper"])
            for a in row["actions"]
        ]
        signals = [
            TaskSignal(step=int(s["step"]), name=s["name"], value=s["value"])
            for s in row["signals"]
        ]
        records.append(
            EpisodeRecord(
                episode_id=row["episode_id"],
                robot=row["robot"],
                task=row["task"],
                seed=int(row["seed"]),
                dt=float(row["dt"]),
                observations=obs,
                actions=actions,
                signals=signals,
                metadata=row.get("metadata", {}),
            )
        )
    return records


#-----------
# JSONL (scaffold)
#-----------


def _save_jsonl(path: Path, records: Sequence[EpisodeRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def _load_jsonl(path: Path) -> list[EpisodeRecord]:
    import json

    from .types import Action, Observation, TaskSignal

    records: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            obs = [Observation(step=o["step"], qpos=o["qpos"], qvel=o["qvel"]) for o in d["observations"]]
            actions = [
                Action(step=a["step"], joint_targets=a["joint_targets"], gripper=a["gripper"])
                for a in d["actions"]
            ]
            signals = [TaskSignal(step=s["step"], name=s["name"], value=s["value"]) for s in d["signals"]]
            records.append(
                EpisodeRecord(
                    episode_id=d["episode_id"],
                    robot=d["robot"],
                    task=d["task"],
                    seed=int(d["seed"]),
                    dt=d["dt"],
                    observations=obs,
                    actions=actions,
                    signals=signals,
                    metadata=d.get("metadata", {}),
                )
            )
    return records
