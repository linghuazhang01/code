"""Serializable state for dynamic MOPD domain budgeting."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainBudgetState:
    """Causal state; observations at step t affect later actor batches."""

    schema_version: int = 2
    validation_updates: int = 0
    variance_updates: int = 0
    last_validation_step: int | None = None
    last_variance_step: int | None = None
    initial_gaps: dict[str, float] = field(default_factory=dict)
    gap_ema: dict[str, float] = field(default_factory=dict)
    normalized_gaps: dict[str, float] = field(default_factory=dict)
    student_scores: dict[str, float] = field(default_factory=dict)
    raw_variances: dict[str, float] = field(default_factory=dict)
    log_variance_ema: dict[str, float] = field(default_factory=dict)
    target_contributions: dict[str, float] = field(default_factory=dict)
    desired_sampling: dict[str, float] = field(default_factory=dict)
    observed_sampling: dict[str, float] = field(default_factory=dict)
    effective_sampling: dict[str, float] = field(default_factory=dict)
    valid_fractions: dict[str, float] = field(default_factory=dict)
    loss_scales: dict[str, float] = field(default_factory=dict)


def _validate_domain_mapping(
    name: str,
    values: dict[str, float],
    expected_domains: set[str],
    *,
    require_all: bool,
    positive: bool = False,
    non_negative: bool = False,
) -> None:
    keys = set(values)
    if (require_all and keys != expected_domains) or not keys <= expected_domains:
        raise ValueError(f"Invalid domains in restored {name}: {sorted(keys)}.")
    for domain, value in values.items():
        numeric = float(value)
        if (
            not math.isfinite(numeric)
            or (positive and numeric <= 0.0)
            or (non_negative and numeric < 0.0)
        ):
            raise ValueError(
                f"Restored {name}.{domain} must be finite"
                f"{' and positive' if positive else ''}"
                f"{' and non-negative' if non_negative else ''}."
            )


def validate_domain_budget_state(
    state: DomainBudgetState,
    domains: tuple[str, ...],
) -> None:
    """Reject corrupt or incompatible controller checkpoint state."""

    expected = set(domains)
    for name, counter_value in (
        ("validation_updates", state.validation_updates),
        ("variance_updates", state.variance_updates),
    ):
        if (
            isinstance(counter_value, bool)
            or not isinstance(counter_value, int)
            or counter_value < 0
        ):
            raise ValueError(f"Restored {name} must be a non-negative integer.")
    for name, step_value in (
        ("last_validation_step", state.last_validation_step),
        ("last_variance_step", state.last_variance_step),
    ):
        if step_value is not None and (
            isinstance(step_value, bool)
            or not isinstance(step_value, int)
            or step_value < 0
        ):
            raise ValueError(f"Restored {name} must be null or a non-negative integer.")
    for name, values in (
        ("target_contributions", state.target_contributions),
        ("desired_sampling", state.desired_sampling),
    ):
        _validate_domain_mapping(
            name,
            values,
            expected,
            require_all=True,
            positive=True,
        )
        total = sum(float(value) for value in values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Restored {name} must sum to one.")

    optional_mappings = (
        ("initial_gaps", state.initial_gaps, True),
        ("gap_ema", state.gap_ema, True),
        ("normalized_gaps", state.normalized_gaps, True),
        ("student_scores", state.student_scores, False),
        ("raw_variances", state.raw_variances, True),
        ("log_variance_ema", state.log_variance_ema, False),
    )
    for name, values, non_negative in optional_mappings:
        _validate_domain_mapping(
            name,
            values,
            expected,
            require_all=False,
            non_negative=non_negative,
        )
    capability_key_sets = {
        frozenset(values)
        for values in (
            state.initial_gaps,
            state.gap_ema,
            state.normalized_gaps,
            state.student_scores,
        )
    }
    if capability_key_sets not in ({frozenset()}, {frozenset(expected)}):
        raise ValueError(
            "Restored capability mappings must be jointly empty or cover all domains."
        )
    capability_present = capability_key_sets == {frozenset(expected)}
    if (
        state.validation_updates < 0
        or capability_present != (state.validation_updates > 0)
        or capability_present != (state.last_validation_step is not None)
    ):
        raise ValueError(
            "Restored validation counters, step, and capability mappings are inconsistent."
        )
    if set(state.raw_variances) != set(state.log_variance_ema):
        raise ValueError(
            "Restored raw_variances and log_variance_ema must cover the same domains."
        )
    variance_present = bool(state.raw_variances)
    if (
        state.variance_updates < 0
        or variance_present != (state.variance_updates > 0)
        or variance_present != (state.last_variance_step is not None)
    ):
        raise ValueError(
            "Restored variance counters, step, and variance mappings are inconsistent."
        )
    has_applied_batch = bool(
        state.observed_sampling
        or state.effective_sampling
        or state.valid_fractions
        or state.loss_scales
    )
    for name, values in (
        ("observed_sampling", state.observed_sampling),
        ("effective_sampling", state.effective_sampling),
        ("valid_fractions", state.valid_fractions),
        ("loss_scales", state.loss_scales),
    ):
        _validate_domain_mapping(
            name,
            values,
            expected,
            require_all=has_applied_batch,
            positive=True,
        )
    if has_applied_batch:
        if not math.isclose(
            sum(float(value) for value in state.observed_sampling.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Restored observed sampling must sum to one.")
        if not math.isclose(
            sum(float(value) for value in state.effective_sampling.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("Restored effective sampling must sum to one.")
        active_mass = sum(
            float(state.observed_sampling[domain])
            * float(state.valid_fractions[domain])
            for domain in domains
        )
        if active_mass <= 0.0:
            raise ValueError("Restored active sampling mass must be positive.")
        for domain in domains:
            if float(state.valid_fractions[domain]) > 1.0 + 1e-8:
                raise ValueError(
                    f"Restored valid fraction exceeds one for domain {domain!r}."
                )
            expected_effective = (
                float(state.observed_sampling[domain])
                * float(state.valid_fractions[domain])
                / active_mass
            )
            if not math.isclose(
                expected_effective,
                float(state.effective_sampling[domain]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Restored nominal, valid, and effective sampling disagree "
                    f"for domain {domain!r}."
                )
            realized = float(state.effective_sampling[domain]) * float(
                state.loss_scales[domain]
            )
            if not math.isclose(
                realized,
                float(state.target_contributions[domain]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"Restored q=p*lambda closure failed for domain {domain!r}."
                )


def persist_controller_payload(
    output_dir: Path,
    payload: dict[str, Any],
    event: str,
    step: int,
    extra: dict[str, Any],
) -> None:
    """Atomically persist the latest controller state and append its history."""

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    temporary_path = output_dir / ".state.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary_path, state_path)
    row = {"event": event, "step": int(step), **payload, **extra}
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
