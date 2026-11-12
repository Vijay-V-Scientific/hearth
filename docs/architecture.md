# Project HEARTH — Architecture

> Working document for the HEARTH research stack. Last updated: 21 Oct 2026.

## 1. Goal

Train a policy that teaches a full-size humanoid to walk through a cluttered
apartment and complete multi-step household tasks from a spoken instruction
(e.g. "clear the dinner table and load the cups into the dishwasher"), trained
in simulation and transferred to hardware with no fine-tuning beyond a short
calibration.

## 2. The three-layer control architecture

The humanoid is driven by three stacked controllers. Each layer is a plain
dataclass contract in `hearth/isaac/control.py`; the actual tensors are
produced by Isaac Lab at runtime.

| Layer | Rate | Interface | Responsibility |
|-------|------|-----------|----------------|
| Layer 2 — VLA planner | 5–10 Hz | `Subgoal` list | instruction → semantic sub-goals (navigate / open / grasp / carry / fetch) |
| Layer 1 — loco-manipulation | 50–100 Hz | whole-body joint targets | sub-goal → whole-body + five-finger hand control |
| Layer 0 — locomotion | 500 Hz | joint position targets + WBC | balance, foot placement, slip recovery |

The low-level layer runs at 500 Hz on the onboard compute; the VLA planner runs
at ≥ 5 Hz. Between them, a typed message passes over a ROS 2 bridge
(`hearth/deploy/ros2_bridge.py`).
````
## 3. Package layout

```
hearth/
├── __init__.py
├── scene/        Procedural apartment generation (M2)
│   ├── types.py      Room, furniture, scene primitives
│   ├── writer.py     Valid .usd writer (verified round-trip)
│   ├── generator.py  Procedural layout + domain randomization
│   └── cli.py        `python -m hearth.scene.cli`
├── isaac/        Isaac Lab integration points (M3)
│   ├── control.py    Three-layer control contracts
│   ├── robot.py      Humanoid / hand spec (Unitree G1 default)
│   └── runner.py     TaskRunner (deferred Isaac Lab imports)
├── data/         Episode storage + schema (M4)
│   ├── types.py      Episode / state / action schema
│   ├── storage.py    HDF5 / Parquet / JSONL writers + schema validation
│   └── cli.py        `python -m hearth.data.cli`
├── eval/         Evaluation harness (M6)
│   ├── types.py      TaskFamily, TaskSpec, TaskResult
│   ├── tasks.py      50 benchmark task definitions
│   ├── benchmark.py  TaskRunner + result collection
│   ├── results.py    ReportConfig + md/json/csv writers
│   ├── families.py   Family distribution
│   └── cli.py        `python -m hearth.eval.cli`
├── training/     Training configs (M5)
│   ├── config.py     PolicyConfig / LocoManipDiffusionConfig / VlaPlannerConfig
│   └── __init__.py
└── deploy/       Sim-to-real + deployment (M7)
    ├── calibration.py    Domain randomization, motor latency, calibration
    └── ros2_bridge.py    Typed ROS 2 SubgoalMessage bridge
