"""Sim-to-real calibration for Project HEARTH.

The sim-to-real gap is where most policies fail. This module provides:

* :class:`DomainRandomization` — scene-level randomization knobs (mass,
  friction, lighting, texture) used during training to broaden the
  distribution the policy sees.
* :class:`MotorLatencyModel` — fits motor latency to real motor logs
  (per the project brief: "learned actuator networks fitted to real
  motor logs, plus explicit motor latency").
* :class:`CalibrationState` / :class:`CalibrationResult` — the measured
  real-robot parameters (center of mass, friction, joint offsets) and
  the residuals after calibration.

All runtime (Isaac Sim, real robot) work raises ``NotImplementedError``
until a GPU + Isaac Sim 5.x container is present. The math and the
state management are importable and tested without a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class NotSupportedError(RuntimeError):
    """Raised when a calibration step needs Isaac Sim / a real robot."""


class CalibrationError(ValueError):
    """Raised on invalid calibration inputs."""


@dataclass
class DomainRandomization:
    """Per-parameter scene randomization ranges used in training.

    Attributes:
        mass_scale: Multiplicative mass perturbation range (min, max).
        friction_scale: Static friction perturbation range.
        com_offset: Centre-of-mass offset std in metres, per axis.
        lighting_range: Ambient lighting multiplicative range.
        texture_noise: Whether texture noise is applied.
        joint_damping: Joint damping perturbation range.
        joint_offset: Joint position offset range (radians).
    """

    mass_scale: tuple[float, float] = (0.9, 1.1)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lighting_range: tuple[float, float] = (0.6, 1.4)
    texture_noise: bool = True
    joint_damping: tuple[float, float] = (0.8, 1.2)
    joint_offset: tuple[float, float] = (-0.02, 0.02)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Draw one random sample of the randomization parameters."""
        rng = rng or np.random.default_rng()
        return {
            "mass_scale": float(rng.uniform(*self.mass_scale)),
            "friction_scale": float(rng.uniform(*self.friction_scale)),
            "com_offset": tuple(
                float(rng.uniform(self.com_offset[0], self.com_offset[1]))
                for _ in range(3)
            ),
            "lighting": float(rng.uniform(*self.lighting_range)),
            "texture_noise": self.texture_noise,
            "joint_damping": float(rng.uniform(*self.joint_damping)),
            "joint_offset": float(rng.uniform(*self.joint_offset)),
        }


@dataclass
class MotorLatencyModel:
    """Fits motor latency to real motor logs.

    The brief calls for "learned actuator networks (an MLP over position
    error and velocity history) fitted to real motor logs, plus explicit
    motor latency". This class holds the fitted parameters and evaluates
    a simple linear latency+gain model. The full MLP lives in the
    Isaac Sim pipeline; this is the importable, testable core.

    Attributes:
        latency_s: Estimated motor latency in seconds.
        gain: Actuator gain correction.
        bias: Actuator bias offset.
        fit_residual: RMS residual of the fit (None if not fit yet).
    """

    latency_s: float = 0.01
    gain: float = 1.0
    bias: float = 0.0
    fit_residual: float | None = None

    def predict(
        self, position_error: float, velocity: float
    ) -> float:
        """Predict the corrected torque-like command."""
        delayed_error = position_error - self.gain * velocity * self.latency_s
        return delayed_error + self.bias

    def fit(
        self,
        position_errors: np.ndarray,
        velocities: np.ndarray,
        measured: np.ndarray,
    ) -> float:
        """Fit gain/bias/latency to real motor logs via least squares.

        Args:
            position_errors: Measured joint position errors.
            velocities: Measured joint velocities.
            measured: Measured commanded torques.

        Returns:
            The RMS residual of the fit.

        Raises:
            CalibrationError: On mismatched input shapes.
        """
        position_errors = np.asarray(position_errors, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        measured = np.asarray(measured, dtype=float)
        if not (
            position_errors.shape == velocities.shape == measured.shape
        ):
            raise CalibrationError(
                "position_errors, velocities, and measured must share shape"
            )
        if position_errors.size < 2:
            raise CalibrationError(
                "need at least two samples to fit the motor latency model"
            )
        design = np.stack(
            [position_errors, -velocities * 0.01, np.ones_like(position_errors)],
            axis=1,
        )
        # Solve design @ [gain, latency_scaled, bias] = measured
        coeffs, _, _, _ = np.linalg.lstsq(design, measured, rcond=None)
        self.gain = float(coeffs[0])
        self.latency_s = float(abs(coeffs[1]))
        self.bias = float(coeffs[2])
        predicted = design @ coeffs
        residual = float(np.sqrt(np.mean((measured - predicted) ** 2)))
        self.fit_residual = residual
        return residual


@dataclass
class CalibrationState:
    """Measured real-robot calibration parameters.

    Attributes:
        robot_id: Identifier of the physical robot.
        com: Estimated centre of mass in the base frame (metres).
        total_mass: Estimated total mass (kg).
        joint_offsets: Per-joint position offsets (radians).
        friction: Estimated static friction per contact.
        motor_models: Per-joint motor latency models keyed by joint name.
        timestamp: ISO-8601 creation timestamp.
    """

    robot_id: str
    com: tuple[float, float, float]
    total_mass: float
    joint_offsets: dict[str, float]
    friction: dict[str, float]
    motor_models: dict[str, MotorLatencyModel]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "com": tuple(self.com),
            "total_mass": self.total_mass,
            "joint_offsets": dict(self.joint_offsets),
            "friction": dict(self.friction),
            "motor_models": {
                k: {
                    "latency_s": v.latency_s,
                    "gain": v.gain,
                    "bias": v.bias,
                    "fit_residual": v.fit_residual,
                }
                for k, v in self.motor_models.items()
            },
            "timestamp": self.timestamp,
        }


@dataclass
class CalibrationResult:
    """Outcome of a calibration run.

    Attributes:
        metric_name: Name of the measured metric (e.g. "fall_rate").
        before: Value before calibration.
        after: Value after calibration.
        target: Success-criterion target, if any.
        passed: Whether the post-calibration value meets the target.
    """

    metric_name: str
    before: float
    after: float
    target: float | None = None
    minimize: bool = True

    @property
    def passed(self) -> bool:
        if self.target is None:
            return True
        if self.minimize:
            return self.after <= self.target
        return self.after >= self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "before": self.before,
            "after": self.after,
            "target": self.target,
            "passed": self.passed,
        }


def compute_domain_randomization(
    rng: np.random.Generator | None = None,
) -> DomainRandomization:
    """Return a default domain randomization object (importable core)."""
    return DomainRandomization()


def save_calibration_state(
    state: CalibrationState, path: str | Path
) -> Path:
    """Write a :class:`CalibrationState` to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2)
    return path


def load_calibration_state(path: str | Path) -> CalibrationState:
    """Read a :class:`CalibrationState` from a JSON file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise CalibrationError(f"calibration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid calibration JSON: {exc}") from exc

    motor_models: dict[str, MotorLatencyModel] = {}
    for name, mv in data.get("motor_models", {}).items():
        motor_models[name] = MotorLatencyModel(
            latency_s=float(mv["latency_s"]),
            gain=float(mv["gain"]),
            bias=float(mv["bias"]),
            fit_residual=mv.get("fit_residual"),
        )
    return CalibrationState(
        robot_id=data["robot_id"],
        com=tuple(data["com"]),
        total_mass=float(data["total_mass"]),
        joint_offsets=dict(data["joint_offsets"]),
        friction=dict(data["friction"]),
        motor_models=motor_models,
        timestamp=data["timestamp"],
    )
