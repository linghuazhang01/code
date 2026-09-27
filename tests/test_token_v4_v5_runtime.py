"""Versioned Control position gates affect both source scoring and application."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import test_current_step_runtime as runtime
import torch
from test_structure_positions import _Tokenizer

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.occurrence import occurrence_mask
from mopd_verl.domain_gradient.structure_positions import attach_structure_position_mask
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger


def test_actor_batch_selection_retains_both_trainer_position_masks() -> None:
    # This production boundary drops non-whitelisted tensors before microbatch
    # scoring; checking it guards a failure that isolated selector tests miss.
    source = Path("third_party/verl/verl/workers/actor/dp_actor.py").read_text()
    selection = source.split("    def update_policy(", 1)[1].split(
        "        data = data.select(", 1
    )[0]
    for name in ("mopd_control_position_mask", "mopd_structure_position_mask"):
        assert f'"{name}"' in selection


def _versioned_audit(weight: int = 4) -> object:
    config = load_config(
        f"configs/token_selection/math_code/taxonomy/mopd_math_code_next_step_"
        f"token_v5_toploss_m05_c01_fixed{weight}_4gpu_colocated.yaml"
    )
    # Historical same-step runtime behavior is tested with an explicit override.
    logger = MOPDAuditLogger({"mopd_audit": {
        **asdict(config.audit), "control_token_online_selection_timing": "current_step",
    }})
    domain = DomainGradientConfig.from_meta(
        logger.full_gradient_meta("train", 1)["mopd_full_gradient"]
    )
    audit = runtime._audit()
    audit.config = replace(
        domain,
        control_token_online_min_mean_occurrences_per_step=0,
        control_token_online_top_p=0.1,
        control_token_online_top_p_by_domain=(),
    )
    return audit


@pytest.mark.parametrize("weight", [4, 8])
def test_code_control_inside_fences_cannot_score_or_receive_selected_id_weight(
    weight: int,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        audit = _versioned_audit(weight)
        batch = runtime._batch([[704, 333, 9000, 704, 333, 9001, 704, 333]], ["code"])
        tokenizer = _Tokenizer(
            {704: "so", 333: "if", 9000: "\n```python\n", 9001: "\n```\n"}
        )
        trainer_config = {
            **asdict(audit.config),
            "domain_structure_token_ids": dict(audit.config.domain_structure_token_ids),
        }
        attach_structure_position_mask(batch, tokenizer, trainer_config)
        metrics = runtime._prepare(
            audit,
            [batch],
            [
                runtime._result(
                    batch, [[4.0, 1.0, 0.0, 1000.0, 100000.0, 0.0, 4.0, 1.0]]
                )
            ],
        )
        expected = torch.tensor([[weight, 1.0, 1.0, 1.0, 1.0, 1.0, weight, 1.0]])
        expected /= expected.mean()
        torch.testing.assert_close(occurrence_mask(audit, batch), expected)
        assert metrics["code/current_step/head/selected_count"] == 2
        assert metrics["code/current_step/eligible_count"] == 4
        assert metrics["code/current_step/valid_count"] == 8
        # The ID has three total occurrences but only two eligible occurrences.
        audit.config = replace(
            audit.config, control_token_online_min_mean_occurrences_per_step=2
        )
        metrics = runtime._prepare(
            audit,
            [batch],
            [
                runtime._result(
                    batch, [[4.0, 1.0, 0.0, 1000.0, 100000.0, 0.0, 4.0, 1.0]]
                )
            ],
        )
        assert metrics["code/current_step/head/selected_count"] == 0
        torch.testing.assert_close(occurrence_mask(audit, batch), torch.ones(1, 8))


def test_versioned_prepass_fails_closed_without_control_mask() -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        audit = _versioned_audit()
        batch = runtime._batch([[704]], ["code"])
        batch.batch["mopd_structure_position_mask"] = torch.zeros(1, 1)
        with pytest.raises(ValueError, match="mopd_control_position_mask"):
            runtime._prepare(audit, [batch], [runtime._result(batch, [[4.0]])])


@pytest.mark.parametrize("domain", ["math", "code"])
@pytest.mark.parametrize("weight", [4, 8])
def test_answer_structure_is_outside_toploss_budget_and_normalized_once(
    domain: str,
    weight: int,
) -> None:
    with runtime.existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(
        torch
    ):
        audit = _versioned_audit(weight)
        batch = runtime._batch([[704, 9000, 16688, 9001, 16688, 9002, 704]], [domain])
        tokenizer = _Tokenizer(
            {704: "so", 9000: "\n##", 16688: " conclusion", 9001: "\nplain", 9002: "\n"}
        )
        attach_structure_position_mask(
            batch,
            tokenizer,
            {
                **asdict(audit.config),
                "domain_structure_token_ids": dict(
                    audit.config.domain_structure_token_ids
                ),
            },
        )
        metrics = runtime._prepare(
            audit,
            [batch],
            [runtime._result(batch, [[4.0, 0.0, 100000.0, 0.0, 100000.0, 0.0, 4.0]])],
        )
        expected = torch.tensor([[weight, 1.0, weight, 1.0, 1.0, 1.0, weight]])
        expected /= expected.mean()
        torch.testing.assert_close(occurrence_mask(audit, batch), expected)
        assert metrics[f"{domain}/current_step/head/selected_count"] == 2
