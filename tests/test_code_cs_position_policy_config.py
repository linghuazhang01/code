"""Config and checkpoint contracts for dynamic Code C+S position eligibility."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import test_current_step_runtime as runtime
from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.occurrence_config import (
    effective_code_cs_position_policy,
    validate_occurrence_config,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


CONFIG = Path("configs/token_selection/math_code/taxonomy/") / (
    "mopd_math_code_next_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml"
)


def _profile(tmp_path: Path, *, mode: str, gate: bool | None) -> object:
    path = tmp_path / f"{mode}-{gate}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "extends": str(CONFIG.resolve()),
                "audit": {
                    "versioned_cs_selection_mode_by_domain": {
                        "math": "position_fixed",
                        "code": mode,
                    },
                    "code_cs_position_gate_enabled": gate,
                },
            }
        )
    )
    return load_config(path)


def _meta(config: object, tmp_path: Path, step: int) -> dict:
    logger = MOPDAuditLogger(
        {"mopd_audit": {**asdict(config.audit), "output_dir": str(tmp_path)}}
    )
    return logger.full_gradient_meta("train", step)["mopd_full_gradient"]


def _actor(config: object, checkpoint: dict | None = None) -> SimpleNamespace:
    group = {}
    if checkpoint is not None:
        group["mopd_online_control_selection_state"] = checkpoint
    return SimpleNamespace(
        actor_module=torch.nn.Linear(1, 1),
        actor_optimizer=SimpleNamespace(param_groups=[group]),
        config={"policy_loss": asdict(config.actor)},
    )


@pytest.mark.parametrize("mode", ["top_loss", "top_teacher_confidence"])
@pytest.mark.parametrize(
    ("gate", "expected"),
    [(None, "dynamic_ungated"), (False, "dynamic_ungated"),
     (True, "dynamic_gated")],
)
def test_code_position_policy_round_trips_through_launch_and_meta(
    tmp_path: Path, mode: str, gate: bool | None, expected: str,
) -> None:
    config = _profile(tmp_path, mode=mode, gate=gate)
    assert effective_code_cs_position_policy(config.audit) == expected
    rendered = "null" if gate is None else str(gate).lower()
    assert f"+mopd_audit.code_cs_position_gate_enabled={rendered}" in build_command(config)
    gradient_config = DomainGradientConfig.from_meta(_meta(config, tmp_path, 1))
    assert gradient_config.code_cs_position_gate_enabled is gate
    assert effective_code_cs_position_policy(gradient_config) == expected


@pytest.mark.parametrize("gate", [False, True])
def test_non_train_meta_disables_code_gate_with_online_selection(
    tmp_path: Path, gate: bool,
) -> None:
    config = _profile(tmp_path, mode="top_loss", gate=gate)
    logger = MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
    for mode in ("eval", "validation"):
        meta = logger.full_gradient_meta(mode, 1)["mopd_full_gradient"]
        assert meta["code_cs_position_gate_enabled"] is None
        gradient_config = DomainGradientConfig.from_meta(meta)
        assert not gradient_config.control_token_online_selection_enabled
        assert effective_code_cs_position_policy(gradient_config) is None


@pytest.mark.parametrize("gate", [False, True])
def test_fixed_code_rejects_even_explicit_false(tmp_path: Path, gate: bool) -> None:
    with pytest.raises(ValueError, match="requires dynamic Code C\\+S"):
        _profile(tmp_path, mode="position_fixed", gate=gate)


@pytest.mark.parametrize("invalid", [0, "false", 1])
def test_code_gate_requires_boolean_or_null(tmp_path: Path, invalid: object) -> None:
    base = load_config(CONFIG).audit
    with pytest.raises(TypeError, match="bool or null"):
        validate_occurrence_config(
            replace(
                base,
                versioned_cs_selection_mode_by_domain={
                    "math": "position_fixed", "code": "top_loss",
                },
                code_cs_position_gate_enabled=invalid,
            )
        )


def test_gate_rejects_current_step_position_fixed_and_legacy(tmp_path: Path) -> None:
    base = load_config(CONFIG).audit
    with pytest.raises(ValueError, match="current-step Code structure_only"):
        validate_occurrence_config(
            replace(base, control_token_online_selection_timing="current_step",
                    code_cs_position_gate_enabled=False)
        )
    with pytest.raises(ValueError, match="requires active V4/V5"):
        validate_occurrence_config(
            replace(base, token_taxonomy_version="legacy",
                    structure_token_loss_weighting_enabled=False,
                    code_cs_position_gate_enabled=False)
        )


def test_checkpoint_policy_is_signed_and_legacy_fixed_only(tmp_path: Path) -> None:
    dynamic = _profile(tmp_path, mode="top_loss", gate=False)
    fixed = load_config(CONFIG)
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        dynamic_state = DomainGradientAudit(
            _actor(dynamic), _meta(dynamic, tmp_path, 1)
        )._online_control_selection_state
        assert dynamic_state.code_cs_position_policy == "dynamic_ungated"
        signed = dynamic_state.as_dict()
        assert signed["code_cs_position_policy"] == "dynamic_ungated"
        assert dynamic_state.from_mapping(signed) == dynamic_state

        same = _profile(tmp_path, mode="top_loss", gate=None)
        restored = DomainGradientAudit(
            _actor(same, signed), _meta(same, tmp_path, 2)
        )
        assert restored._online_control_selection_state == dynamic_state

        gated = _profile(tmp_path, mode="top_loss", gate=True)
        with pytest.raises(ValueError, match="does not match"):
            DomainGradientAudit(
                _actor(gated, signed), _meta(gated, tmp_path, 2)
            )

        unsigned_dynamic = dict(signed)
        unsigned_dynamic.pop("code_cs_position_policy")
        with pytest.raises(ValueError, match="does not match"):
            DomainGradientAudit(
                _actor(dynamic, unsigned_dynamic), _meta(dynamic, tmp_path, 2)
            )

        fixed_state = DomainGradientAudit(
            _actor(fixed), _meta(fixed, tmp_path, 1)
        )._online_control_selection_state
        assert fixed_state.code_cs_position_policy == "position_fixed"
        unsigned_fixed = fixed_state.as_dict()
        unsigned_fixed.pop("code_cs_position_policy")
        restored_fixed = DomainGradientAudit(
            _actor(fixed, unsigned_fixed), _meta(fixed, tmp_path, 2)
        )
        assert restored_fixed._online_control_selection_state.candidate_map() == (
            fixed_state.candidate_map()
        )
        assert (
            restored_fixed._online_control_selection_state.code_cs_position_policy
            == "position_fixed"
        )
        assert restored_fixed._online_control_selection_state.as_dict()[
            "code_cs_position_policy"
        ] == "position_fixed"


def test_gated_code_jsonl_reports_source_budget_and_applied_positions(
    tmp_path: Path,
) -> None:
    config = _profile(tmp_path, mode="top_loss", gate=True)
    logger = MOPDAuditLogger({"mopd_audit": {
        **asdict(config.audit),
        "output_dir": str(tmp_path),
        "control_token_online_min_mean_occurrences_per_step": 0.0,
        "control_token_online_top_p_by_domain": {"math": 0.05, "code": 0.25},
    }})
    batch = runtime._batch([[704, 1590, 704, 1590]], ["code"])
    batch.batch["mopd_control_position_mask"] = torch.tensor([[1, 1, 0, 1]])
    batch.batch["mopd_structure_position_mask"] = torch.tensor([[0, 1, 0, 0]])
    raw = torch.tensor([[5.0, 1.0, 100.0, 100.0]])
    mask = batch.batch["response_mask"]

    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        actor = _actor(config)
        first = DomainGradientAudit(
            actor, logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
        )
        first_metrics = first.observe_completed_step(
            [batch], [raw], [mask],
            selector_token_loss_batches=[raw],
            selector_token_loss_mask_batches=[mask],
        )
        assert first._online_control_selection_state.active_map()["code"] == (704,)
        assert first_metrics[
            "code/token_weight/source_eligible_control_occurrence_count"
        ] == 1.0
        assert first_metrics[
            "code/token_weight/source_eligible_structure_occurrence_count"
        ] == 1.0
        assert first_metrics["code/token_weight/top_p_target_occurrence_count"] == 1.0
        assert first_metrics["code/token_weight/top_p_selected_occurrence_count"] == 1.0
        assert first_metrics["code/token_weight/top_p_occurrence_shortfall"] == 0.0

        checkpoint = actor.actor_optimizer.param_groups[0][
            "mopd_online_control_selection_state"
        ]
        second = DomainGradientAudit(
            _actor(config, checkpoint),
            logger.full_gradient_meta("train", 2)["mopd_full_gradient"],
        )
        second_metrics = second.observe_completed_step(
            [batch], [raw], [mask],
            selector_token_loss_batches=[raw],
            selector_token_loss_mask_batches=[mask],
        )
        assert second_metrics[
            "code/token_weight/applied_control_weighted_position_count"
        ] == 1.0
        assert second_metrics[
            "code/token_weight/applied_structure_weighted_position_count"
        ] == 0.0

    rows = [
        json.loads(line)
        for path in sorted(tmp_path.rglob("versioned_token_selection.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[-1]["code_cs_position_policy"] == "dynamic_gated"
    code = rows[-1]["domains"]["code"]
    assert code["position_policy"] == "dynamic_gated"
    assert code["source_valid_response_token_count"] == 4
    assert code["source_eligible_control_occurrence_count"] == 1
    assert code["source_eligible_structure_occurrence_count"] == 1
    assert code["top_p_target_occurrence_count"] == 1
    assert code["top_p_selected_occurrence_count"] == 1
    assert code["top_p_occurrence_shortfall"] == 0
    assert code["applied_control_weighted_position_count"] == 1
    assert code["applied_structure_weighted_position_count"] == 0
