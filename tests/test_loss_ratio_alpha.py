"""Loss-ratio alpha contracts from scalar selection through actor resume."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.control_selection_scoring import (
    selected_to_other_loss_ratio_weight,
    validate_scaled_loss_ratio_weight,
)
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.settings import AuditConfig, load_config


BASE_CONFIG = Path(__file__).resolve().parents[1] / (
    "configs/token_selection/math/"
    "a_next_step_expanded_pruned_v3_unified_i1_w1_k25_5gpu_4a1t_b256.yaml"
)
INVALID_ALPHA = (0.0, -0.5, float("nan"), float("inf"), -float("inf"))


def _ratio(alpha: float = 1.0, **overrides: Any) -> Any:
    arguments = dict(
        selected_loss_abs_sum=6.0,
        selected_occurrence_count=2,
        valid_loss_abs_sum=12.0,
        valid_occurrence_count=8,
        max_weight=4.0,
    )
    return selected_to_other_loss_ratio_weight(
        **{**arguments, **overrides},
        alpha=alpha,
    )


def _state(alpha: float | None = None, mode: str = "loss_ratio") -> Any:
    return initial_online_control_selection_state(
        ("math",),
        (10, 20),
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=1.0,
        top_k=1,
        weight_mode=mode,
        selection_mode=("top_kl_student_entropy" if mode == "paired" else "top_loss"),
        **({} if alpha is None else {"loss_ratio_alpha": alpha}),
    )


def _advance(state: Any, step: int, selected_mean: float = 3.0) -> Any:
    return update_online_control_selection(
        state,
        {"math": {10: (10 * selected_mean, 10), 20: (20.0, 20)}},
        step=step,
        valid_token_counts={"math": 50},
        valid_score_sums={"math": 10 * selected_mean + 40},
        loss_ratio_max_weight=4.0,
    )


def _meta(alpha: float = 1.0, mode: str = "loss_ratio") -> dict[str, Any]:
    return {
        "domains": ["math"],
        "control_token_loss_weighting_enabled": True,
        "control_token_loss_weight": 4.0,
        "control_token_normalize_per_domain": True,
        "control_token_candidate_ids": [10, 20],
        "control_token_online_selection_enabled": True,
        "control_token_online_audit_interval_steps": 1,
        "control_token_online_window_steps": 1,
        "control_token_online_min_mean_occurrences_per_step": 1.0,
        "control_token_online_top_k": 1,
        "control_token_online_selection_mode": (
            "top_kl_student_entropy" if mode == "paired" else "top_loss"
        ),
        "control_token_online_weight_mode": mode,
        "control_token_loss_ratio_alpha": alpha,
    }


def _settings(tmp_path: Path, alpha: float | None, mode: str = "loss_ratio") -> Any:
    audit: dict[str, Any] = {
        "control_token_online_selection_mode": (
            "top_kl_student_entropy" if mode == "paired" else "top_loss"
        ),
        "control_token_online_weight_mode": mode,
    }
    if alpha is not None:
        audit["control_token_loss_ratio_alpha"] = alpha
    path = tmp_path / "alpha.yaml"
    path.write_text(yaml.safe_dump({"extends": str(BASE_CONFIG), "audit": audit}))
    return load_config(path)


@pytest.mark.parametrize(
    "selected_mean,other_mean,alpha,cap,raw,clipped,scaled",
    [
        (3, 1, 1, 4, 3, 3, 3),
        (3, 1, 2, 4, 3, 3, 6),
        (10, 1, 2, 4, 10, 4, 8),
        (3, 1, 0.25, 4, 3, 3, 0.75),
        (0.25, 1, 0.25, 4, 0.25, 1, 0.25),
        (0, 1, 2, 4, 0, 1, 2),
        (3, 0, 2, 4, 3e12, 4, 8),
        (3, 1, 2, 1, 3, 1, 2),
    ],
)
def test_helper_scales_after_legacy_clip_without_recapping(
    selected_mean: float,
    other_mean: float,
    alpha: float,
    cap: float,
    raw: float,
    clipped: float,
    scaled: float,
) -> None:
    result = _ratio(
        alpha,
        selected_loss_abs_sum=2 * selected_mean,
        valid_loss_abs_sum=2 * selected_mean + 6 * other_mean,
        max_weight=cap,
    )
    assert result.raw_ratio == pytest.approx(raw)
    assert result.clipped_weight == pytest.approx(clipped)
    assert result.scaled_weight == pytest.approx(scaled)
    assert result.selected_mean_abs_loss == pytest.approx(selected_mean)
    assert result.other_mean_abs_loss == pytest.approx(other_mean)


@pytest.mark.parametrize("alpha", [0.25, 1.0, 2.0, sys.float_info.max])
@pytest.mark.parametrize("selected,valid", [(0, 4), (4, 4), (0, 0)])
def test_empty_group_preserves_neutral_fallback(
    alpha: float,
    selected: int,
    valid: int,
) -> None:
    result = _ratio(
        alpha,
        selected_loss_abs_sum=float(selected),
        selected_occurrence_count=selected,
        valid_loss_abs_sum=float(valid),
        valid_occurrence_count=valid,
    )
    assert result.raw_ratio is None
    assert result.clipped_weight == result.scaled_weight == 1.0


def test_omitted_alpha_preserves_legacy_defaults(tmp_path: Path) -> None:
    result = selected_to_other_loss_ratio_weight(
        selected_loss_abs_sum=6,
        selected_occurrence_count=2,
        valid_loss_abs_sum=12,
        valid_occurrence_count=8,
        max_weight=4,
    )
    assert result == _ratio(1.0)
    assert _state().loss_ratio_alpha == 1.0
    assert AuditConfig().control_token_loss_ratio_alpha == 1.0
    assert _settings(tmp_path, None).audit.control_token_loss_ratio_alpha == 1.0
    meta = _meta()
    meta.pop("control_token_loss_ratio_alpha")
    assert DomainGradientConfig.from_meta(meta).control_token_loss_ratio_alpha == 1.0


@pytest.mark.parametrize("alpha", [0.25, 1.0, 2.0])
def test_state_checkpoint_next_update_scales_once(alpha: float) -> None:
    outcome, state = _advance(_state(alpha), 1)
    result = outcome.domain_results[0]
    assert result.raw_selected_to_other_loss_ratio == pytest.approx(3.0)
    assert result.selected_unscaled_loss_ratio_weight == pytest.approx(3.0)
    assert result.selected_raw_loss_ratio_weight == pytest.approx(3 * alpha)
    assert state.active_weight_map() == {"math": {10: 3 * alpha}}
    payload = json.loads(json.dumps(state.as_dict()))
    assert payload["schema_version"] == 10
    assert payload["loss_ratio_alpha"] == alpha
    restored = OnlineControlSelectionState.from_mapping(payload)
    assert restored == state
    expected_outcome, expected_state = _advance(state, 2, selected_mean=2.0)
    resumed_outcome, resumed_state = _advance(restored, 2, selected_mean=2.0)
    assert (resumed_outcome, resumed_state) == (expected_outcome, expected_state)
    assert resumed_state.active_weight_map() == {"math": {10: 2 * alpha}}
    duplicate, unchanged = _advance(resumed_state, 2, selected_mean=10.0)
    assert duplicate.duplicate_step and unchanged == resumed_state
    assert unchanged.as_dict()["loss_ratio_alpha"] == alpha


@pytest.mark.parametrize("schema", [8, 9])
def test_legacy_checkpoint_missing_alpha_defaults_to_one(schema: int) -> None:
    _, original = _advance(_state(), 1)
    payload = json.loads(json.dumps(original.as_dict()))
    payload.pop("loss_ratio_alpha", None)
    payload["schema_version"] = schema
    restored = OnlineControlSelectionState.from_mapping(payload)
    assert restored.loss_ratio_alpha == 1.0
    assert restored.active_weight_map() == {"math": {10: 3.0}}
    assert restored.as_dict()["schema_version"] == 10
    _, next_state = _advance(restored, 2, selected_mean=2.0)
    assert next_state.active_weight_map() == {"math": {10: 2.0}}
    assert next_state.loss_ratio_alpha == 1.0


@pytest.mark.parametrize("alpha", [0.25, 2.0])
@pytest.mark.parametrize("selected_present", [False, True])
def test_neutral_fallback_survives_checkpoint(
    alpha: float,
    selected_present: bool,
) -> None:
    outcome, state = update_online_control_selection(
        _state(alpha),
        {"math": {10: (3.0, 1)} if selected_present else {}},
        step=1,
        valid_token_counts={"math": 1},
        valid_score_sums={"math": 3.0},
        loss_ratio_max_weight=4.0,
    )
    result = outcome.domain_results[0]
    assert result.raw_selected_to_other_loss_ratio is None
    assert result.selected_raw_loss_ratio_weight == 1.0
    assert result.selected_unscaled_loss_ratio_weight == 1.0
    restored = OnlineControlSelectionState.from_mapping(
        json.loads(json.dumps(state.as_dict()))
    )
    assert restored == state
    assert restored.loss_ratio_alpha == alpha
    assert restored.active_weight_map() == {
        "math": {10: 1.0} if selected_present else {},
    }


@pytest.mark.parametrize("alpha", INVALID_ALPHA)
@pytest.mark.parametrize(
    "boundary",
    ["helper", "state", "checkpoint", "meta", "settings"],
)
def test_invalid_alpha_is_rejected_at_every_entry(
    alpha: float,
    boundary: str,
    tmp_path: Path,
) -> None:
    payload = _state().as_dict()
    payload["loss_ratio_alpha"] = alpha
    operations = {
        "helper": lambda: _ratio(alpha),
        "state": lambda: _state(alpha),
        "checkpoint": lambda: OnlineControlSelectionState.from_mapping(payload),
        "meta": lambda: DomainGradientConfig.from_meta(_meta(alpha)),
        "settings": lambda: _settings(tmp_path, alpha),
    }
    with pytest.raises(ValueError, match="(?i)alpha"):
        operations[boundary]()


@pytest.mark.parametrize("mode", ["fixed", "paired"])
@pytest.mark.parametrize("alpha", [0.25, 2.0])
def test_non_ratio_modes_reject_nondefault_alpha(
    mode: str,
    alpha: float,
    tmp_path: Path,
) -> None:
    for operation in (
        lambda: _state(alpha, mode),
        lambda: DomainGradientConfig.from_meta(_meta(alpha, mode)),
        lambda: _settings(tmp_path, alpha, mode),
        lambda: OnlineControlSelectionState.from_mapping(
            {**_state(None, mode).as_dict(), "loss_ratio_alpha": alpha}
        ),
    ):
        with pytest.raises(ValueError, match="(?i)alpha"):
            operation()


@pytest.mark.parametrize("alpha", [sys.float_info.max / 2, sys.float_info.max])
def test_scaled_weight_overflow_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="(?i)(finite|overflow)"):
        _ratio(alpha)
    with pytest.raises(ValueError, match="(?i)(finite|overflow)"):
        _advance(_state(alpha), 1)


def test_raw_ratio_overflow_keeps_legacy_cap_before_scaling() -> None:
    result = _ratio(
        2.0,
        selected_loss_abs_sum=1e308,
        selected_occurrence_count=1,
        valid_loss_abs_sum=1e308,
        valid_occurrence_count=2,
    )
    assert result.raw_ratio == result.clipped_weight == 4.0
    assert result.scaled_weight == 8.0


@pytest.mark.parametrize("alpha", [1e39, 1e-50])
def test_scaled_weight_must_survive_runtime_float32(alpha: float) -> None:
    with pytest.raises(ValueError, match="float32"):
        _ratio(alpha)


def test_schema10_requires_alpha_field() -> None:
    _, state = _advance(_state(1.5), 1)
    payload = state.as_dict()
    payload.pop("loss_ratio_alpha")
    with pytest.raises(ValueError, match="loss_ratio_alpha"):
        OnlineControlSelectionState.from_mapping(payload)


def test_scaled_weight_rejects_values_that_round_down_to_float32_max() -> None:
    float32_max = float.fromhex("0x1.fffffep+127")
    validate_scaled_loss_ratio_weight(float32_max)
    with pytest.raises(ValueError, match="float32"):
        validate_scaled_loss_ratio_weight(3.4028235004135232e38)
