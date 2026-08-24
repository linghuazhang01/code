from __future__ import annotations

from pathlib import Path

import pytest

from mopd_verl.config_profiles import list_config_profiles
from mopd_verl.launch import build_command, format_command
from mopd_verl.settings import MOPDConfig, load_config
from mopd_verl.token_baselines import token_baseline_method
from mopd_verl.topk_distill import configured_distill_loss_name

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATHS = tuple(
    sorted(
        set((ROOT / "configs" / "baselines").glob("*.yaml"))
        | set((ROOT / "configs").rglob("*baseline*.yaml"))
    )
)


def _baseline_references() -> tuple[str, ...]:
    references: list[str] = []
    for path in BASELINE_PATHS:
        profiles = list_config_profiles(path)
        if profiles:
            references.extend(f"{path}::{profile}" for profile in profiles)
        else:
            references.append(str(path))
    return tuple(references)


BASELINE_REFERENCES = _baseline_references()


def _canonical_baseline_references() -> tuple[str, ...]:
    return tuple(
        str(path)
        for path in sorted(
            (ROOT / "configs" / "baselines" / "canonical").glob("*.yaml")
        )
    )


CANONICAL_BASELINE_REFERENCES = _canonical_baseline_references()
LIGHTWEIGHT_AUDIT_FLAGS = (
    "log_sample_level",
    "response_level_enabled",
    "log_validation_metrics",
    "full_gradient_enabled",
    "sequence_masked_target_enabled",
    "sample_gradient_enabled",
    "token_gap_enabled",
    "token_gap_vocab_vector_enabled",
    "entropy_enabled",
    "entropy_vocab_vector_enabled",
    "topk_teacher_student_cross_entropy_vocab_enabled",
    "logp_vector_enabled",
    "logp_abs_vector_enabled",
    "token_gradient_enabled",
    "dynamic_domain_loss_weighting_enabled",
    "control_token_loss_weighting_enabled",
    "control_token_online_selection_enabled",
    "control_token_phase_gate_enabled",
    "control_token_span_weighting_enabled",
    "control_token_speed_weighting_enabled",
    "all_domain_shared_token_loss_weighting_enabled",
    "gradient_fingerprint_enabled",
)


def _expected_loss_variance_signal(config: MOPDConfig) -> str:
    signal = configured_distill_loss_name(config.actor)
    baseline_method = token_baseline_method(config.actor)
    if baseline_method != "none":
        signal = f"{signal}+{baseline_method}_token_weighting"
    return signal


@pytest.mark.parametrize("config_reference", BASELINE_REFERENCES)
def test_every_baseline_records_domain_metrics(config_reference: str) -> None:
    config = load_config(config_reference)
    expected_domains = set(config.data.domain_sampling_weights)

    assert config.audit.enabled
    assert set(config.audit.domains) == expected_domains
    assert config.audit.output_dir != "mopd_audit"
    assert config.audit.loss_variance_signal == _expected_loss_variance_signal(
        config
    )

    rendered = format_command(build_command(config))
    assert "+mopd_audit.enabled=true" in rendered
    assert "+mopd_audit.output_dir=" in rendered
    assert "+mopd_audit.loss_variance_signal=" in rendered


def test_baseline_domain_metric_outputs_are_unique() -> None:
    output_dirs = [
        load_config(config_reference).audit.output_dir
        for config_reference in BASELINE_REFERENCES
    ]

    assert len(output_dirs) == len(set(output_dirs))


@pytest.mark.parametrize(
    "config_reference",
    CANONICAL_BASELINE_REFERENCES,
)
def test_canonical_baseline_domain_metrics_are_lightweight(
    config_reference: str,
) -> None:
    config = load_config(config_reference)
    audit = config.audit

    if audit.enabled:
        for field_name in LIGHTWEIGHT_AUDIT_FLAGS:
            assert not getattr(audit, field_name), field_name

    assert not config.domain_budgeting.enabled
    assert not config.paper_eval.enabled
