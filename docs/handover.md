# Project HEARTH — Handover Guide

This document tells a new team member how to get HEARTH running.

## Prerequisites

- NVIDIA GPU (RTX 5090 or better) with the latest driver.
- CUDA 12.x toolkit.
- NVIDIA Isaac Sim 5.x (see the Docker section for the base image).
- Python 3.11–3.12.

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
# On this machine (no GPU needed, but you must install the deps with `pip install -e ".[dev]"`)
python -m pytest -q

# In the lightweight test container (no GPU, no Isaac Sim)
docker compose -f docker/docker-compose.yml run --rm tests
```

## Running the CLI tools

```bash
python -m hearth.scene.cli generate --seed 0 --out /tmp/hearth/scene0.usda
python -m hearth.eval.cli run hearth_50 --seeds 0,1,2
python -m hearth.data.cli save examples/episode_sample.jsonl /tmp/episode.parquet

# Training is config-only here (no CLI): the layer configs load and validate as
# dataclasses via hearth/training/config.py.
python -c "from hearth.training.config import load_policy_config; print(load_policy_config(0).to_dict())"
```

## Getting the Isaac Sim container up

```bash
# Auto-builds on first use and bind-mounts the project (headless by default).
docker compose -f docker/docker-compose.yml build hearth
docker compose -f docker/docker-compose.yml up hearth

# Or the wrapper, which forwards its args to the container with --gpus all:
./docker/run.sh python -m hearth.scene.cli generate --seed 0 --out /tmp/scene0.usda
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
