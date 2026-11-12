"""Tests for episode storage (HDF5, Parquet, JSONL round-trips)."""

import numpy as np
import pytest

from hearth.data import (
    Action,
    EpisodeRecord,
    Observation,
    StorageFormat,
    TaskSignal,
    save_episodes,
    storage_format,
)
from hearth.data.storage import load_episodes


def _sample_record():
    obs = [
        Observation(step=i, qpos=[float(i), 0.1, 0.2], qvel=[0.0, 0.1, 0.2])
        for i in range(3)
    ]
    actions = [
        Action(step=i, joint_targets=[float(i) / 10.0, 0.05], gripper=float(i) / 3.0)
        for i in range(3)
    ]
    signals = [
        TaskSignal(step=0, name="approach", value=1.0),
        TaskSignal(step=1, name="grasp", value=1.0),
    ]
    return EpisodeRecord(
        episode_id="ep_001",
        robot="unitree_g1",
        task="fetch_mug",
        seed=12,
        dt=0.002,
        observations=obs,
        actions=actions,
        signals=signals,
        metadata={"difficulty": "easy"},
    )


@pytest.mark.parametrize("fmt", [StorageFormat.JSONL, StorageFormat.PARQUET])
def test_storage_roundtrip(tmp_path, fmt):
    rec = _sample_record()
    ext = fmt.extension()
    path = tmp_path / f"eps{ext}"
    save_episodes(path, [rec], fmt=fmt)
    assert path.exists()
    assert storage_format(path) == fmt
    loaded = load_episodes(path)
    assert len(loaded) == 1
    lr = loaded[0]
    assert lr.episode_id == rec.episode_id
    assert lr.robot == rec.robot
    assert lr.task == rec.task
    assert lr.seed == rec.seed
    assert lr.dt == rec.dt
    assert lr.length == 3
    assert lr.metadata == rec.metadata
    assert len(lr.observations) == 3
    assert len(lr.actions) == 3
    assert len(lr.signals) == 2
    # Values survive the round-trip.
    assert lr.observations[1].qpos[0] == pytest.approx(1.0)
    assert lr.actions[1].joint_targets[0] == pytest.approx(0.1)
    assert lr.actions[1].gripper[0] == pytest.approx(1.0 / 3.0)
    assert lr.signals[0].name == "approach"
    assert lr.signals[1].value == pytest.approx(1.0)


def test_load_detects_format(tmp_path):
    rec = _sample_record()
    p_jsonl = tmp_path / "a.jsonl"
    p_parquet = tmp_path / "a.parquet"
    save_episodes(p_jsonl, [rec], fmt=StorageFormat.JSONL)
    save_episodes(p_parquet, [rec], fmt=StorageFormat.PARQUET)
    assert storage_format(p_jsonl) == StorageFormat.JSONL
    assert storage_format(p_parquet) == StorageFormat.PARQUET
    # Cross-load: JSONL file loads fine via load_episodes.
    assert len(load_episodes(p_jsonl)) == 1
    assert len(load_episodes(p_parquet)) == 1


def test_storage_format_unknown(tmp_path):
    p = tmp_path / "unknown.txt"
    p.write_text("not a record")
    with pytest.raises(ValueError):
        storage_format(p)


def test_hdf5_unavailable(tmp_path):
    # h5py is not installed in this env; the HDF5 path must raise a clear error.
    rec = _sample_record()
    p = tmp_path / "eps.h5"
    with pytest.raises(ImportError):
        save_episodes(p, [rec], fmt=StorageFormat.HDF5)
