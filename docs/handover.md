# Project HEARTH — Handover Guide

This document tells a new team member how to get HEARTH running.

## Prerequisites

- NVIDIA GPU (RTX 5090 or better) with the latest driver.
- CUDA 12.x toolkit.
- NVIDIA Isaac Sim 5.x (see the Docker section for the base image).
- Python 3.10–3.12.

## What is already working

The following pieces are real, runnable, and tested on a machine without Isaac
Sim. They are the parts of the project that ship today:

- `hearth/scene/` — procedural apartment generator with a valid `.usd` writer.
- `hearth/eval/` — 50-task benchmark, `hearth-eval` CLI, md/json/csv reports.
- `hearth/data/` — episode storage (HDF5, Parquet, JSONL) with schema validation.
- `hearth/training/` — PPO / diffusion / VLA configs + loader + validation.
- `hearth/deploy/` — domain randomization, motor latency, calibration, ROS 2 bridge.
- `docker/` — Isaac Sim container + a lightweight tests image + CI.

Everything that needs Isaac Sim, the robot, or ROS 2 is marked `SCAFFOLD` in its
docstring and raises `NotImplementedError` until that runtime is present.

## Running the tests

```bash
# On this machine (no GPU needed)
python -m pytest -q

# In a container (Isaac Sim 5.x)
docker/run.sh tests
```

## Running the CLI tools

```bash
python -m hearth.scene.cli generate --seed 0 --out /tmp/hearth/scene0
python -m hearth.eval.cli --config configs/eval/benchmark_50.yaml
python -m hearth.data.cli --config configs/data/episode_schema.yaml
python -m hearth.training.cli --layer 0
```

## Getting the Isaac Sim container up

```bash
docker/run.sh build   # first run builds the image
docker/run.sh run     # launches Isaac Sim with the project bind-mounted
```

## Sim-to-real

See `docs/architecture.md` for the full flow. The short version:

1. Train in Isaac Sim with domain randomization.
2. Export the policy (onnx/torch).
3. Run `calibration.py` against the real robot to fit motor latency, center of
   mass, and friction.
4. Deploy. No fine-tuning beyond the ~10-min calibration.

## The six objectives

Objective 4 (70% success on 50 benchmark tasks) is the primary eval target.
The 50 tasks live in `hearth/eval/tasks.py`. The other five are measured in
simulation and on hardware.

## Where things are

- Architecture overview: `docs/architecture.md`
- Milestones: the GitHub milestones page (linked from the README)
- CI: `.github/workflows/`
