"""Training configuration package for Project HEARTH.

Provides typed config loaders for the three RL/control layers:

* Layer 0 — PPO locomotion (500 Hz low-level joint control)
* Layer 1 — diffusion loco-manipulation (whole-body + five-finger)
* Layer 2 — VLA planner (5-10 Hz instruction -> sub-goals)

Configs are plain YAML. This package loads and validates them without
requiring Isaac Sim, torch, or a GPU so they can be checked in CI.
"""

from .config import (
    LocoManipDiffusionConfig,
    PolicyConfig,
    VlaPlannerConfig,
    load_policy_config,
    load_all_policy_configs,
    validate_policy_config,
)

__all__ = [
    "PolicyConfig",
    "VlaPlannerConfig",
    "LocoManipDiffusionConfig",
    "load_policy_config",
    "load_all_policy_configs",
    "validate_policy_config",
]
