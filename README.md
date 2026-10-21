# Project HEARTH
**H**umanoid **E**mbodied **A**gent for **R**eal-world **T**ask **H**andling

An autonomous humanoid household-manipulation research stack: a full-size humanoid walks through procedurally generated apartments and completes multi-step tasks from spoken natural-language instructions.

> **Working title:** HEARTH · **Owner:** @Vijay · **Date:** 21 Oct 2026

---

## TL;DR

- **One monorepo**, three RL/control layers, one evaluation harness, one data pipeline.
- **Trained in simulation** (NVIDIA Isaac Sim + Isaac Lab, PhysX GPU), **transferred to hardware** with no fine-tuning beyond a ~10-min calibration.
- This repo ships the **real, runnable pieces that don't need Isaac Sim** (scene generator, evaluation harness, data format, configs, CI, Docker, docs) plus **clean integration points** for everything that does.
- What is *done and tested here* vs *scaffolding / needs hardware* is marked explicitly in every file.

## The six objectives (first-draft targets for the kickoff)

| # | Objective | Success criterion | Measured in |
|---|-----------|-------------------|-------------|
| 1 | Robust locomotion | 0.0–1.5 m/s, turns, 15 cm steps; < 1 fall / 1,000 m; survives 150 N lateral push | Sim + real |
| 2 | Loco-manipulation | Opens 90% of unseen doors/drawers/cabinances standing on 1–2 feet | Sim + real |
| 3 | Dexterous grasping | 85% grasp success on 500 held-out household objects | Sim + real |
| 4 | Long-horizon tasks | 70% success on 50 benchmark tasks, 5–15 sub-steps, from a language instruction | Sim |
| 5 | Recovery | Recovers 80% of injected failures without a reset | Sim |
| 6 | Real-time control | Low-level policy 500 Hz; VLA planner ≥ 5 Hz on onboard compute | Sim + real |
## The three-layer control architecture

## The three-layer control architecture

## Repository map

## How to read this repo

- **"DONE"** = real, runnable, and tested on a machine without Isaac Sim (this one: RTX 5090, torch).
- **"SCAFFOLD"** = correct, importable, integration-ready, but the heavy runtime (Isaac Sim, the robot) is elsewhere.
- Every file that is scaffolding states, in a docstring, exactly what it plugs into.

## Quickstart

```bash
python -m pip install -e .            # install the package + deps
python -m hearth.scene.cli generate --seed 0 --out /tmp/hearth/scene0
python -m hearth.eval.cli --config configs/eval/benchmark_50.yaml
python -m hearth.data.cli --config configs/data/episode_schema.yaml
```

## Milestones

See [Milestones](https://github.com/Vijay-V-Scientific/hearth/milestones) on GitHub.

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Repo scaffolding, docs, config, CI | done |
| 2 | Procedural scene generator | done |
| 3 | Isaac Lab extension scaffolding | pending |
| 4 | Data pipeline + storage format | pending |
| 5 | Training configs (PPO/diffusion/VLA) | pending |
| 6 | Evaluation harness | done |
| 7 | Sim-to-real + deployment | pending |
| 8 | Docker, datasheet, architecture, handover | pending |

## License

Proprietary working repo for Project HEARTH. See `LICENSE`.
