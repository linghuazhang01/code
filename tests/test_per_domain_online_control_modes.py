"""Deterministic coverage for mixed online Control-token modes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_selection_scoring import (
    normalize_online_selection_mode_by_domain,
    normalize_online_weight_mode_by_domain,
    validate_online_control_mode_contracts,
)
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


BASE_CONFIG = Path(__file__).resolve().parents[1] / (
    "configs/token_selection/math/taxonomy/"
    "top32kl_next_step_full_taxonomy_split_topp0p05_i1_w1_5gpu_b256.yaml"
)


def test_mode_maps_use_scalar_fallback_and_exact_domain_keys() -> None:
    domains = ("math", "code")

    assert normalize_online_selection_mode_by_domain(
        domains,
        {},
        "top_logp_diff",
    ) == (("math", "top_logp_diff"), ("code", "top_logp_diff"))
    assert normalize_online_weight_mode_by_domain(
        domains,
        {},
        "paired",
    ) == (("math", "paired"), ("code", "paired"))

    with pytest.raises(ValueError, match="exactly match"):
        normalize_online_selection_mode_by_domain(
            domains,
            {"math": "top_loss"},
            "top_loss",
        )
    with pytest.raises(ValueError, match="exactly match"):
        normalize_online_weight_mode_by_domain(
            domains,
            {"math": "fixed", "code": "fixed", "science": "fixed"},
            "fixed",
        )


@pytest.mark.parametrize(
    ("selection", "weight", "window", "match"),
    [
        ("top_loss", "paired", 1, "paired-signal"),
        ("top_logp_diff", "loss_ratio", 1, "loss-ratio"),
        ("top_speed", "fixed", 1, "at least 2"),
        ("top_q_loss_entropy", "paired", 1, "Q selection"),
    ],
)
def test_mode_contracts_are_checked_per_domain(
    selection: str,
    weight: str,
    window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_online_control_mode_contracts(
            ("math", "code"),
            selection_mode="top_loss",
            weight_mode="fixed",
            selection_mode_by_domain={"math": selection, "code": "top_loss"},
            weight_mode_by_domain={"math": weight, "code": "fixed"},
            interval=1,
            window=window,
        )


def test_domain_gradient_config_stores_effective_mode_maps() -> None:
    config = DomainGradientConfig.from_meta(
        {
            "domains": ["math", "code"],
            "control_token_online_selection_mode": "top_loss",
            "control_token_online_weight_mode": "fixed",
            "control_token_online_selection_mode_by_domain": {
                "math": "top_loss",
                "code": "top_logp_diff",
            },
            "control_token_online_weight_mode_by_domain": {
                "math": "fixed",
                "code": "fixed",
            },
        }
    )

    assert config.online_selection_mode_map() == {
        "math": "top_loss",
        "code": "top_logp_diff",
    }
    assert config.online_weight_mode_map() == {"math": "fixed", "code": "fixed"}
    config.validate()


def test_mixed_state_dispatches_ranking_and_weight_modes() -> None:
    domains = ("math", "code", "science")
    state = initial_online_control_selection_state(
        domains,
        {"math": (10, 11), "code": (20, 21), "science": (30, 31)},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=1.0,
        top_k=1,
        selection_mode_by_domain={
            "math": "top_loss",
            "code": "top_kl_student_entropy",
            "science": "top_loss",
        },
        weight_mode_by_domain={
            "math": "fixed",
            "code": "paired",
            "science": "loss_ratio",
        },
    )

    outcome, state = update_online_control_selection(
        state,
        {
            "math": {10: (4.0, 1), 11: (1.0, 1)},
            "code": {20: (3.0, 1), 21: (1.0, 1)},
            "science": {30: (8.0, 1), 31: (2.0, 1)},
        },
        step=1,
        valid_token_counts={domain: 2 for domain in domains},
        valid_score_sums={"math": 5.0, "code": 4.0, "science": 10.0},
        loss_ratio_max_weight=4.0,
    )

    assert outcome.audit_triggered
    assert state.active_map() == {"math": (10,), "code": (20,), "science": (30,)}
    assert state.active_weight_map() == {
        "math": {},
        "code": {20: 4.0},
        "science": {30: 4.0},
    }
    assert state.selection_mode_map() == {
        "math": "top_loss",
        "code": "top_kl_student_entropy",
        "science": "top_loss",
    }
    assert state.weight_mode_map() == {
        "math": "fixed",
        "code": "paired",
        "science": "loss_ratio",
    }

    payload = json.loads(json.dumps(state.as_dict()))
    assert payload["schema_version"] == 12
    assert OnlineControlSelectionState.from_mapping(payload) == state

    legacy = state.as_dict()
    legacy["selection_mode_by_domain"] = None
    legacy["weight_mode_by_domain"] = None
    legacy["selection_mode"] = "top_loss"
    legacy["weight_mode"] = "fixed"
    legacy["active_token_weights"] = tuple((domain, ()) for domain in domains)
    legacy["schema_version"] = 10
    migrated = OnlineControlSelectionState.from_mapping(legacy)
    assert migrated.selection_mode_map() == {domain: "top_loss" for domain in domains}
    assert migrated.weight_mode_map() == {domain: "fixed" for domain in domains}


def test_runtime_dispatches_mixed_score_families_in_one_batch() -> None:
    domains = ("loss", "speed", "logp", "q", "kl", "teacher")
    candidate_ids = (101, 102, 103, 104, 105, 106)
    token_ids = torch.tensor(
        [[101, 901], [102, 902], [103, 903], [104, 904], [105, 905], [106, 906]],
        dtype=torch.long,
    )
    configured_loss = torch.tensor(
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]
    )
    response_mask = torch.ones_like(configured_loss, dtype=torch.bool)
    student_entropy = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )
    teacher_entropy = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [0.0, 2.0], [0.0, 2.0]]
    )
    selection_modes = {
        "loss": "top_loss",
        "speed": "top_speed",
        "logp": "top_logp_diff",
        "q": "top_q_loss_entropy",
        "kl": "top_kl_student_entropy",
        "teacher": "top_teacher_confidence_student_entropy",
    }
    logp_loss = configured_loss.clone()
    logp_loss[2] = torch.tensor([-5.0, 0.0])
    kl_loss = configured_loss.clone()
    kl_loss[4] = torch.tensor([4.0, 1.0])

    result = global_candidate_loss_statistics_with_valid_counts(
        (token_ids,),
        (configured_loss,),
        (response_mask,),
        (domains,),
        domains=domains,
        domain_candidate_token_ids={
            domain: (token_id,) for domain, token_id in zip(domains, candidate_ids)
        },
        selection_mode="top_loss",
        selection_mode_by_domain=selection_modes,
        selection_loss_batches_by_domain={"logp": (logp_loss,), "kl": (kl_loss,)},
        selection_loss_mask_batches_by_domain={
            "logp": (response_mask,),
            "kl": (response_mask,),
        },
        student_entropy_batches=(student_entropy,),
        teacher_entropy_batches=(teacher_entropy,),
    )

    assert result.by_domain["loss"][101] == (2.0, 1)
    assert result.by_domain["speed"][102] == (2.0, 1)
    assert result.by_domain["logp"][103] == (5.0, 1)
    assert result.by_domain["q"][104][1] == 1
    assert result.by_domain["q"][104][0] == pytest.approx(26.0 / 9.0)
    assert result.by_domain["kl"][105][0] == pytest.approx(2.0)
    assert result.by_domain["teacher"][106][0] == pytest.approx(2.0)
    assert result.valid_token_counts == {domain: 2 for domain in domains}
    assert result.q_normalization_stats is not None
    assert result.q_normalization_stats["q"]["q_all_valid_mean_abs_loss"] == 1.5


@pytest.mark.parametrize(
    "selection_mode",
    [
        "top_kl_student_entropy",
        "top_teacher_confidence_student_entropy",
    ],
)
def test_scalar_paired_runtime_uses_full_packed_channels(
    selection_mode: str,
) -> None:
    token_ids = torch.tensor([[10, 99]], dtype=torch.long)
    configured_loss = torch.tensor([[2.0, 1.0]])
    response_mask = torch.ones_like(configured_loss, dtype=torch.bool)

    result = global_candidate_loss_statistics_with_valid_counts(
        (token_ids,),
        (configured_loss,),
        (response_mask,),
        (("math",),),
        domains=("math",),
        candidate_token_ids=(10,),
        selection_mode=selection_mode,
        student_entropy_batches=(torch.tensor([[1.0, 1.0]]),),
        teacher_entropy_batches=(torch.tensor([[0.0, 1.0]]),),
    )

    assert result.by_domain == {"math": {10: (3.0, 1)}}
    assert result.valid_token_counts == {"math": 2}


def test_config_launcher_and_metadata_preserve_effective_mode_maps(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "per_domain_modes.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(BASE_CONFIG),
                "audit": {
                    "control_token_online_selection_mode_by_domain": {
                        "math": "top_logp_diff"
                    },
                    "control_token_online_weight_mode_by_domain": {"math": "fixed"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.audit.control_token_online_selection_mode_by_domain == {
        "math": "top_logp_diff"
    }
    assert config.audit.control_token_online_weight_mode_by_domain == {"math": "fixed"}
    command = build_command(config)
    selection_override = next(
        value
        for value in command
        if value.startswith(
            "+mopd_audit.control_token_online_selection_mode_by_domain="
        )
    )
    assert selection_override.endswith("{math: 'top_logp_diff'}")

    audit_overrides = {
        key.removeprefix("+mopd_audit."): yaml.safe_load(value)
        for key, value in (
            argument.split("=", 1)
            for argument in command
            if argument.startswith("+mopd_audit.")
        )
    }
    audit_overrides["output_dir"] = str(tmp_path)
    logger = MOPDAuditLogger({"mopd_audit": audit_overrides})
    metadata = logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    assert metadata["control_token_online_selection_mode"] == "top_loss"
    assert metadata["control_token_online_weight_mode"] == "fixed"
    assert metadata["control_token_online_selection_mode_by_domain"] == {
        "math": "top_logp_diff"
    }
    assert metadata["control_token_online_weight_mode_by_domain"] == {"math": "fixed"}


def test_mixed_training_mask_uses_scalar_fixed_and_stored_variable_weights() -> None:
    import test_domain_gradient_optimization_contracts as contracts

    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        actor = SimpleNamespace(actor_optimizer=SimpleNamespace(param_groups=[{}]))
        audit = DomainGradientAudit(
            actor,
            {
                "domains": ["math", "code", "science"],
                "control_token_loss_weighting_enabled": True,
                "control_token_loss_weight": 4.0,
                "control_token_candidate_ids": [],
                "domain_control_token_candidate_ids": {
                    "math": [10],
                    "code": [20],
                    "science": [30],
                },
                "control_token_online_selection_enabled": True,
                "control_token_online_audit_interval_steps": 1,
                "control_token_online_window_steps": 1,
                "control_token_online_min_mean_occurrences_per_step": 1.0,
                "control_token_online_top_k": 1,
                "control_token_online_selection_mode_by_domain": {
                    "math": "top_loss",
                    "code": "top_kl_student_entropy",
                    "science": "top_loss",
                },
                "control_token_online_weight_mode_by_domain": {
                    "math": "fixed",
                    "code": "paired",
                    "science": "loss_ratio",
                },
            },
        )
        audit._applied_online_control_token_ids = {
            "math": (10,),
            "code": (20,),
            "science": (30,),
        }
        audit._applied_online_control_token_weights = {
            "math": {},
            "code": {20: 3.0},
            "science": {30: 2.5},
        }
        micro_batch = SimpleNamespace(
            batch={
                "responses": torch.tensor([[10, 99], [20, 99], [30, 99]]),
                "response_mask": torch.ones(3, 2, dtype=torch.bool),
            },
            non_tensor_batch={"domain": ["math", "code", "science"]},
        )
        weights = audit.training_gradient_mask(micro_batch)

    torch.testing.assert_close(
        weights,
        torch.tensor([[4.0, 1.0], [3.0, 1.0], [2.5, 1.0]]),
    )
