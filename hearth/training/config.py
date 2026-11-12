"""Typed loaders and validators for HEARTH training configs.

Every policy layer has one config file under ``configs/training/``.
These loaders read YAML, apply defaults, run sanity checks, and raise
``ConfigError`` on anything the benchmarks rely on (control rates,
objectives, target success rates).

The whole module imports without torch, jax, or Isaac Sim so CI can
validate configs on any machine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#-----------
# Paths
#-----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "training"

DEFAULT_CONFIG_DIR = CONFIG_DIR


class ConfigError(Exception):
    """Raised when a training config is missing or invalid."""


#-----------
# Config objects
#-----------

@dataclass
class PolicyConfig:
    """Base class for every policy-layer config.

    Attributes:
        name: Human-readable policy name.
        layer: Control layer index (0 locomotion, 1 loco-manip, 2 planner).
        algorithm: Training algorithm label.
        control_rate_hz: Low-level control rate the policy drives.
        objectives: Per-objective target success rates (matches benchmark).
        notes: Free-form design notes.
    """

    name: str
    layer: int
    algorithm: str
    control_rate_hz: int
    objectives: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "algorithm": self.algorithm,
            "control_rate_hz": self.control_rate_hz,
            "objectives": dict(self.objectives),
            "notes": self.notes,
        }


@dataclass
class LocoManipDiffusionConfig(PolicyConfig):
    """Layer 1 — diffusion loco-manipulation policy.

    Diffusion policies output a trajectory of whole-body actions and are
    the natural fit for contact-rich door/drawer/pick tasks.
    """

    horizon: int = 64
    action_scale: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.995
    num_samples: int = 64
    seed: int = 0
    train: dict[str, Any] = field(default_factory=dict)
    augment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            horizon=self.horizon,
            action_scale=self.action_scale,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            num_samples=self.num_samples,
            seed=self.seed,
            train=self.train,
            augment=self.augment,
        )
        return d


@dataclass
class VlaPlannerConfig(PolicyConfig):
    """Layer 2 — vision-language-action planner.

    Produces sub-goal poses (``Subgoal``) at 5-10 Hz from a spoken or
    typed natural-language instruction.
    """

    model_name: str = "hearth-vla-7b"
    subgoal_rate_hz: int = 5
    context_window: int = 4096
    use_language_encoder: bool = True
    action_horizon: int = 16
    seed: int = 0
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            model_name=self.model_name,
            subgoal_rate_hz=self.subgoal_rate_hz,
            context_window=self.context_window,
            use_language_encoder=self.use_language_encoder,
            action_horizon=self.action_horizon,
            seed=self.seed,
            train=self.train,
            eval=self.eval,
        )
        return d


#-----------
# Loaders
#-----------

def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    return data


def load_policy_config(
    layer: int, config_dir: Path | None = None
) -> PolicyConfig:
    """Load the config for a single control layer.

    Args:
        layer: Layer index (0, 1, or 2).
        config_dir: Optional override directory (defaults to
            ``configs/training/``).

    Returns:
        A typed config object.

    Raises:
        ConfigError: On missing file, bad YAML, or validation failure.
    """
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    path = config_dir / f"layer_{layer}.yaml"
    raw = _read_yaml(path)
    cfg = validate_policy_config(raw, layer=layer, source=path)
    return cfg


def _build_layer1(raw: dict[str, Any]) -> LocoManipDiffusionConfig:
    req = raw.get("train", {})
    aug = raw.get("augment", {})
    return LocoManipDiffusionConfig(
        name=raw.get("name", "hearth-loco-manip-diffusion"),
        layer=1,
        algorithm=raw.get("algorithm", "diffusion"),
        control_rate_hz=int(raw.get("control_rate_hz", 50)),
        objectives={k: float(v) for k, v in raw.get("objectives", {}).items()},
        notes=raw.get("notes", ""),
        horizon=int(raw.get("horizon", 64)),
        action_scale=float(raw.get("action_scale", 1.0)),
        use_ema=bool(raw.get("use_ema", True)),
        ema_decay=float(raw.get("ema_decay", 0.995)),
        num_samples=int(raw.get("num_samples", 64)),
        seed=int(raw.get("seed", 0)),
        train=dict(req),
        augment=dict(aug),
    )


def _build_layer2(raw: dict[str, Any]) -> VlaPlannerConfig:
    req = raw.get("train", {})
    evl = raw.get("eval", {})
    return VlaPlannerConfig(
        name=raw.get("name", "hearth-vla-planner"),
        layer=2,
        algorithm=raw.get("algorithm", "vla"),
        control_rate_hz=int(raw.get("control_rate_hz", 1000)),
        objectives={k: float(v) for k, v in raw.get("objectives", {}).items()},
        notes=raw.get("notes", ""),
        model_name=raw.get("model_name", "hearth-vla-7b"),
        subgoal_rate_hz=int(raw.get("subgoal_rate_hz", 5)),
        context_window=int(raw.get("context_window", 4096)),
        use_language_encoder=bool(raw.get("use_language_encoder", True)),
        action_horizon=int(raw.get("action_horizon", 16)),
        seed=int(raw.get("seed", 0)),
        train=dict(req),
        eval=dict(evl),
    )


def validate_policy_config(
    raw: dict[str, Any],
    layer: int,
    source: Path | None = None,
) -> PolicyConfig:
    """Validate a raw config dict and return a typed config object.

    Args:
        raw: Parsed YAML mapping.
        layer: Layer index (0, 1, or 2).
        source: Optional path used in error messages.

    Returns:
        A typed config object.

    Raises:
        ConfigError: On any structural or value problem.
    """
    label = str(source) if source else f"layer {layer}"
    if layer not in (0, 1, 2):
        raise ConfigError(f"unknown layer {layer} in {label}")

    algorithm = raw.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ConfigError(f"algorithm must be a non-empty string in {label}")

    control_rate = raw.get("control_rate_hz")
    if not isinstance(control_rate, int) or control_rate <= 0:
        raise ConfigError(
            f"control_rate_hz must be a positive integer in {label}"
        )

    objectives = raw.get("objectives", {})
    if not isinstance(objectives, dict) or not objectives:
        raise ConfigError(f"objectives must be a non-empty mapping in {label}")
    clean_obj: dict[str, float] = {}
    for key, val in objectives.items():
        if not isinstance(key, str):
            raise ConfigError(f"objective key must be a string in {label}")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ConfigError(
                f"objective '{key}' must be a number in {label}"
            )
        if not 0.0 <= float(val) <= 1.0:
            raise ConfigError(
                f"objective '{key}' must be in [0, 1] in {label}"
            )
        clean_obj[key] = float(val)

    if layer == 0:
        return PolicyConfig(
            name=raw.get("name", "hearth-ppo-locomotion"),
            layer=0,
            algorithm=algorithm,
            control_rate_hz=control_rate,
            objectives=clean_obj,
            notes=raw.get("notes", ""),
        )
    if layer == 1:
        return _build_layer1(raw)
    return _build_layer2(raw)


def load_all_policy_configs(
    config_dir: Path | None = None,
) -> dict[int, PolicyConfig]:
    """Load and validate all three layer configs.

    Returns:
        Mapping of layer index to typed config.

    Raises:
        ConfigError: On the first invalid config found.
    """
    result: dict[int, PolicyConfig] = {}
    for layer in (0, 1, 2):
        result[layer] = load_policy_config(layer, config_dir=config_dir)
    return result
