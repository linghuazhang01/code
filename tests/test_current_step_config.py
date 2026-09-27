"""Current-step head/tail selection configuration and eval isolation."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_selection_scoring import (
    normalize_selection_mode,
    validate_loss_teacher_confidence_selection_contract,
)
from mopd_verl.domain_gradient.occurrence_config import validate_occurrence_actor
from mopd_verl.launch import build_command
from mopd_verl.settings import AuditConfig, MOPDConfig, load_config
from mopd_verl.verl_audit import MOPDAuditLogger


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / (
    "configs/token_selection/math_code/taxonomy/"
    "mopd_math_code_toploss_m05_c01_fixed4_4gpu.yaml"
)
TC = "top_teacher_confidence"


def _load(tmp_path: Path, audit: dict, **overrides: dict) -> MOPDConfig:
    path = tmp_path / "current_step.yaml"
    path.write_text(yaml.safe_dump({
        "extends": str(BASE),
        "audit": {"control_token_online_selection_timing": "current_step", **audit},
        **overrides,
    }), encoding="utf-8")
    return load_config(path)


def _metadata(config: MOPDConfig, tmp_path: Path, mode: str = "train") -> dict:
    # Parse the actual launch arguments, including Hydra map serialization.
    audit = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (
            argument.split("=", 1)
            for argument in build_command(config)
            if argument.startswith("+mopd_audit.")
        )
    }
    audit["output_dir"] = str(tmp_path)
    return MOPDAuditLogger({"mopd_audit": audit}).full_gradient_meta(
        mode, 1
    )["mopd_full_gradient"]


def test_legacy_defaults_are_unchanged() -> None:
    audit = AuditConfig()
    assert audit.control_token_online_selection_timing == "next_step"
    assert audit.control_token_tail_top_p == 0
    assert audit.control_token_tail_top_p_by_domain == {}
    assert audit.control_token_tail_weight == 1
    assert audit.control_token_tail_selection_mode == "bottom_loss"
    config = load_config(BASE)
    logger = MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
    domain = DomainGradientConfig.from_meta(
        logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    )
    assert domain.control_token_online_selection_unit == "token_id"
    assert domain.control_token_online_selection_timing == "next_step"
    assert domain.control_token_tail_top_p_by_domain == ()


@pytest.mark.parametrize("unit", ["token_id", "occurrence"])
@pytest.mark.parametrize("head_mode,tail_mode", [
    ("top_loss", "bottom_loss"),
    (TC, "bottom_teacher_confidence"),
    ("top_loss_teacher_confidence", "bottom_loss"),
    ("top_loss_teacher_confidence", "bottom_loss_teacher_confidence"),
])
def test_yaml_launch_logger_and_domain_metadata_chain(
    tmp_path: Path, unit: str, head_mode: str, tail_mode: str
) -> None:
    config = _load(tmp_path, {
        "control_token_online_selection_unit": unit,
        "control_token_online_selection_mode": head_mode,
        "control_token_tail_selection_mode": tail_mode,
        "control_token_tail_top_p": 0.1,
        "control_token_tail_top_p_by_domain": {"math": 0.0},
        "control_token_tail_weight": 0.5,
    })
    metadata = _metadata(config, tmp_path)
    domain = DomainGradientConfig.from_meta(metadata)
    assert domain.control_token_online_selection_timing == "current_step"
    assert domain.control_token_online_selection_unit == unit
    assert domain.control_token_online_selection_mode == head_mode
    assert domain.control_token_tail_selection_mode == tail_mode
    assert domain.control_token_tail_top_p == 0.1
    assert domain.control_token_tail_top_p_by_domain == (("math", 0.0),)
    assert domain.control_token_tail_weight == 0.5


@pytest.mark.parametrize("unit", ["token_id", "occurrence"])
def test_eval_disables_current_step_and_tail(tmp_path: Path, unit: str) -> None:
    config = _load(tmp_path, {
        "control_token_online_selection_unit": unit,
        "control_token_online_selection_mode": TC,
        "control_token_online_selection_mode_by_domain": {"math": TC, "code": "top_loss"},
        "control_token_tail_selection_mode": "bottom_teacher_confidence",
        "control_token_tail_top_p": 0.1,
        "control_token_tail_top_p_by_domain": {"code": 0.2},
        "control_token_tail_weight": 0,
    })
    metadata = _metadata(config, tmp_path, "eval")
    domain = DomainGradientConfig.from_meta(metadata)
    assert not domain.control_token_online_selection_enabled
    assert domain.control_token_online_selection_timing == "next_step"
    assert domain.control_token_online_selection_unit == "token_id"
    assert domain.control_token_online_selection_mode == "top_loss"
    assert set(domain.online_selection_mode_map().values()) == {"top_loss"}
    assert domain.control_token_tail_selection_mode == "bottom_loss"
    assert domain.control_token_tail_top_p == 0
    assert domain.control_token_tail_top_p_by_domain == ()
    assert domain.control_token_tail_weight == 1


@pytest.mark.parametrize("field,value", [
    ("control_token_online_selection_timing", "unknown"),
    ("control_token_tail_top_p", -0.1),
    ("control_token_tail_top_p", 1.1),
    ("control_token_tail_top_p", float("nan")),
    ("control_token_tail_weight", -0.1),
    ("control_token_tail_weight", 1.1),
    ("control_token_tail_weight", float("inf")),
    ("control_token_tail_selection_mode", "unknown"),
    ("control_token_tail_top_p_by_domain", {"math": float("nan")}),
    ("control_token_tail_top_p_by_domain", {"science": 0.1}),
])
def test_invalid_values_fail_yaml_and_worker_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    config = _load(tmp_path, {})
    metadata = _metadata(config, tmp_path)
    with pytest.raises((TypeError, ValueError)):
        _load(tmp_path, {field: value})
    with pytest.raises((TypeError, ValueError)):
        DomainGradientConfig.from_meta({**metadata, field: value})


@pytest.mark.parametrize("audit", [
    {"control_token_tail_top_p": 0.1, "control_token_tail_weight": 0.5},
    {"control_token_tail_top_p_by_domain": {"code": 0.1}, "control_token_tail_weight": 0},
    {"control_token_online_selection_mode": TC},
])
def test_new_selection_requires_current_step(tmp_path: Path, audit: dict) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, {**audit, "control_token_online_selection_timing": "next_step"})


@pytest.mark.parametrize("actor", [
    {"ppo_mini_batch_size": 264},
    {"ppo_epochs": 2},
    {"entropy_coeff": 0.01},
])
def test_current_step_token_ids_keep_actor_guards(tmp_path: Path, actor: dict) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, {}, actor=actor)


def test_pure_teacher_confidence_is_fixed_one_step() -> None:
    assert normalize_selection_mode(TC) == TC
    for weight, interval, window in (("loss_ratio", 1, 1), ("fixed", 2, 1), ("fixed", 1, 2)):
        with pytest.raises(ValueError):
            validate_loss_teacher_confidence_selection_contract(TC, weight, interval, window)


def test_direct_actor_rejects_sequence_parallelism() -> None:
    with pytest.raises(ValueError):
        validate_occurrence_actor({"ulysses_sequence_parallel_size": 2})


@pytest.mark.parametrize("value", [None, {}, []])
def test_empty_tail_domain_overrides_normalize_to_empty(tmp_path: Path, value: object) -> None:
    config = _load(tmp_path, {})
    metadata = _metadata(config, tmp_path)
    metadata["control_token_tail_top_p_by_domain"] = value
    assert DomainGradientConfig.from_meta(metadata).control_token_tail_top_p_by_domain == ()
    logger = MOPDAuditLogger({"mopd_audit": {**asdict(config.audit),
                                           "control_token_tail_top_p_by_domain": value}})
    assert logger.control_token_tail_top_p_by_domain == {}


def test_duplicate_tail_domain_overrides_are_rejected(tmp_path: Path) -> None:
    config = _load(tmp_path, {})
    value = [("math", .1), ("math", .2)]
    metadata = _metadata(config, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        DomainGradientConfig.from_meta({**metadata, "control_token_tail_top_p_by_domain": value})
    with pytest.raises(ValueError, match="duplicate"):
        MOPDAuditLogger({"mopd_audit": {**asdict(config.audit),
                                      "control_token_tail_top_p_by_domain": value}})


def test_active_next_step_example_is_isolated_and_launchable(tmp_path: Path) -> None:
    profiles = sorted(BASE.parent.glob("mopd_math_code_next_step_toploss_*_4gpu.yaml"))
    assert len(profiles) == 1
    configs = [load_config(path) for path in profiles]
    original = load_config(BASE)
    for getter in (
        lambda cfg: cfg.runtime.wandb_run_id,
        lambda cfg: cfg.trainer.experiment_name,
        lambda cfg: cfg.trainer.default_local_dir,
        lambda cfg: cfg.audit.output_dir,
        lambda cfg: cfg.paper_eval.output_dir,
        lambda cfg: cfg.huggingface_checkpoint.path_prefix,
    ):
        identities = [getter(cfg) for cfg in [original, *configs]]
        assert len(identities) == len(set(identities))
    for config in configs:
        assert config.data.train_batch_size == config.actor.ppo_mini_batch_size == 528
        assert config.worker_placement.actor_rollout.n_gpus_per_node == 4
        assert config.trainer.n_gpus_per_node == 4
        assert 0 < config.trainer.total_training_steps <= 65
        assert not config.huggingface_checkpoint.private
        assert max(config.huggingface_checkpoint.steps) <= config.trainer.total_training_steps
        assert config.runtime.wandb_resume == "never"
        command = build_command(config)
        assert "trainer.resume_mode=disable" in command
        assert not any("resume_from_path=" in argument for argument in command)
        assert "+mopd_audit.control_token_online_selection_timing=next_step" in command
        domain = DomainGradientConfig.from_meta(_metadata(config, tmp_path))
        assert domain.control_token_online_selection_timing == "next_step"
        DomainGradientConfig.from_meta(_metadata(config, tmp_path, "eval"))


def test_verl_stub_keeps_string_mocks_bound_to_live_modules() -> None:
    import importlib
    import sys
    from unittest.mock import patch

    import torch
    from test_domain_gradient_optimization_contracts import (
        DomainGradientOptimizationContractTests,
    )

    package = importlib.import_module("mopd_verl.full_gradient")
    helper = DomainGradientOptimizationContractTests()
    name = "mopd_verl.full_gradient.actor_loss"
    original = sys.modules.get(name)
    for _ in range(2):
        with helper._stubbed_verl(torch):
            with patch(name + ".build_actor_micro_batch_loss") as mocked:
                live = importlib.import_module(name)
                assert package.actor_loss is live
                assert live.build_actor_micro_batch_loss is mocked
        assert sys.modules.get(name) is original
        assert getattr(package, "actor_loss", None) is original
