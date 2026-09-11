"""Per-domain Top-P configuration and selector behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


BASE_CONFIG = Path(__file__).resolve().parents[1] / (
    "configs/token_selection/math_code_science/taxonomy/"
    "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_"
    "toploss_topp05_fixed4_b528_colocated.yaml"
)
DOMAINS = ("math", "code", "science")


def test_selector_uses_domain_specific_top_p() -> None:
    top_p_by_domain = {"math": 0.10, "code": 0.50, "science": 0.20}
    state = initial_online_control_selection_state(
        DOMAINS,
        {"math": (10,), "code": (20,), "science": (30,)},
        audit_interval_steps=1,
        window_steps=1,
        min_mean_occurrences_per_step=1.0,
        top_k=1,
        budget_mode="top_p",
        top_p=0.05,
        top_p_by_domain=top_p_by_domain,
    )

    outcome, next_state = update_online_control_selection(
        state,
        {
            "math": {10: (10.0, 10)},
            "code": {20: (20.0, 50)},
            "science": {30: (30.0, 20)},
        },
        step=1,
        valid_token_counts={domain: 100 for domain in DOMAINS},
    )

    assert outcome.audit_triggered
    results = {result.domain: result for result in outcome.domain_results}
    assert {
        domain: results[domain].target_occurrence_count for domain in DOMAINS
    } == {"math": 10, "code": 50, "science": 20}
    assert {
        domain: results[domain].selected_occurrence_fraction for domain in DOMAINS
    } == {"math": 0.10, "code": 0.50, "science": 0.20}
    assert next_state.top_p_map() == top_p_by_domain
    assert OnlineControlSelectionState.from_mapping(next_state.as_dict()) == next_state


def test_per_domain_top_p_requires_exact_domain_keys() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        initial_online_control_selection_state(
            DOMAINS,
            {domain: (index,) for index, domain in enumerate(DOMAINS, start=10)},
            audit_interval_steps=1,
            window_steps=1,
            min_mean_occurrences_per_step=1.0,
            top_k=1,
            top_p_by_domain={"math": 0.02, "code": 0.05},
        )


def test_config_and_launcher_preserve_per_domain_top_p(tmp_path: Path) -> None:
    config_path = tmp_path / "per_domain_top_p.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(BASE_CONFIG),
                "audit": {
                    "control_token_online_top_p_by_domain": {
                        "math": 0.02,
                        "code": 0.05,
                        "science": 0.10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.audit.control_token_online_top_p_by_domain == {
        "math": 0.02,
        "code": 0.05,
        "science": 0.10,
    }
    command = build_command(config)
    override = next(
        argument
        for argument in command
        if argument.startswith(
            "+mopd_audit.control_token_online_top_p_by_domain="
        )
    )
    assert override.endswith("{code: 0.05, math: 0.02, science: 0.1}")


def test_config_rejects_unknown_per_domain_top_p_key(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_per_domain_top_p.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": str(BASE_CONFIG),
                "audit": {
                    "control_token_online_top_p_by_domain": {
                        "math": 0.02,
                        "code": 0.05,
                        "science": 0.10,
                        "biology": 0.10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly match"):
        load_config(config_path)
