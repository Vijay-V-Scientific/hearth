"""Tests for training config loaders and validators."""

import pytest

from hearth.training.config import (
    ConfigError,
    LocoManipDiffusionConfig,
    PolicyConfig,
    VlaPlannerConfig,
    load_all_policy_configs,
    load_policy_config,
    validate_policy_config,
)


def test_load_all_three_layers():
    configs = load_all_policy_configs()
    assert set(configs) == {0, 1, 2}
    assert isinstance(configs[0], PolicyConfig)
    assert isinstance(configs[1], LocoManipDiffusionConfig)
    assert isinstance(configs[2], VlaPlannerConfig)


def test_layer0_is_ppo_500hz():
    cfg = load_policy_config(0)
    assert cfg.algorithm == "ppo"
    assert cfg.control_rate_hz == 500
    assert 0.0 <= cfg.objectives["locomotion"] <= 1.0


def test_layer1_diffusion_horizon():
    cfg = load_policy_config(1)
    assert cfg.algorithm == "diffusion"
    assert cfg.horizon == 64
    assert cfg.use_ema is True
    assert cfg.ema_decay == pytest.approx(0.995)


def test_layer2_vla_subgoal_rate():
    cfg = load_policy_config(2)
    assert cfg.algorithm == "vla"
    assert cfg.subgoal_rate_hz == 5
    assert cfg.model_name == "hearth-vla-7b"
    assert cfg.action_horizon == 16


def test_validate_rejects_bad_control_rate():
    with pytest.raises(ConfigError):
        validate_policy_config(
            {"algorithm": "ppo", "control_rate_hz": 0, "objectives": {"x": 1.0}},
            layer=0,
        )


def test_validate_rejects_objective_out_of_range():
    with pytest.raises(ConfigError):
        validate_policy_config(
            {"algorithm": "ppo", "control_rate_hz": 500,
             "objectives": {"x": 1.5}},
            layer=0,
        )


def test_validate_rejects_unknown_layer():
    with pytest.raises(ConfigError):
        validate_policy_config(
            {"algorithm": "x", "control_rate_hz": 1, "objectives": {"x": 1.0}},
            layer=3,
        )
