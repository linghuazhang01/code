"""Coverage for the normalized loss + chosen-token-confidence selector."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / (
    "configs/token_selection/math_code_science/taxonomy/"
    "mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_"
    "topp_m05_c02_s02_code_structure_only_science_loss_teacherconf_"
    "fixed4_b528_colocated.yaml"
)


def test_loss_teacher_confidence_score_normalizes_token_type_means() -> None:
    result = global_candidate_loss_statistics_with_valid_counts(
        (torch.tensor([[10, 20, 30]], dtype=torch.long),),
        (torch.tensor([[1.0, 2.0, 3.0]]),),
        (torch.ones(1, 3, dtype=torch.bool),),
        (("science",),),
        domains=("science",),
        candidate_token_ids=(10, 20, 30),
        selection_mode="top_loss_teacher_confidence",
        teacher_log_prob_batches=(torch.tensor([[-3.0, -2.0, -1.0]]),),
    )

    assert result.by_domain["science"] == pytest.approx(
        {
            10: (0.0, 1),
            20: (1.25, 1),
            30: (3.0, 1),
        }
    )
    assert result.valid_token_counts == {"science": 3}
    assert result.q_normalization_stats is not None
    assert result.q_normalization_stats["science"] == pytest.approx(
        {
            "loss_teacher_confidence_eligible_token_count": 3,
            "loss_teacher_confidence_loss_q02": 1.04,
            "loss_teacher_confidence_loss_q98": 2.96,
            "loss_teacher_confidence_teacher_logp_q02": -2.96,
            "loss_teacher_confidence_teacher_logp_q98": -1.04,
        }
    )


def test_loss_teacher_confidence_runtime_does_not_require_student_entropy() -> None:
    token_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    configured_loss = torch.tensor([[1.0, 2.0, 3.0]])
    response_mask = torch.ones_like(configured_loss, dtype=torch.bool)
    teacher_log_prob = torch.tensor([[-3.0, -2.0, -1.0]])

    result = global_candidate_loss_statistics_with_valid_counts(
        (token_ids,),
        (configured_loss,),
        (response_mask,),
        (("science",),),
        domains=("science",),
        candidate_token_ids=(10, 20, 30),
        selection_mode="top_loss_teacher_confidence",
        teacher_log_prob_batches=(teacher_log_prob,),
    )

    assert result.by_domain["science"] == pytest.approx(
        {
            10: (0.0, 1),
            20: (1.25, 1),
            30: (3.0, 1),
        }
    )
    assert result.valid_token_counts == {"science": 3}


def test_loss_teacher_confidence_runtime_requires_teacher_log_probs() -> None:
    with pytest.raises(ValueError, match="chosen-token log-probability"):
        global_candidate_loss_statistics_with_valid_counts(
            (torch.tensor([[10]], dtype=torch.long),),
            (torch.tensor([[1.0]]),),
            (torch.ones(1, 1, dtype=torch.bool),),
            (("science",),),
            domains=("science",),
            candidate_token_ids=(10,),
            selection_mode="top_loss_teacher_confidence",
        )


def test_mixed_domain_runtime_keeps_math_on_plain_top_loss() -> None:
    result = global_candidate_loss_statistics_with_valid_counts(
        (torch.tensor([[10, 20, 30], [10, 20, 30]], dtype=torch.long),),
        (torch.tensor([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),),
        (torch.ones(2, 3, dtype=torch.bool),),
        (("math", "science"),),
        domains=("math", "science"),
        candidate_token_ids=(10, 20, 30),
        selection_mode="top_loss",
        selection_mode_by_domain={
            "math": "top_loss",
            "science": "top_loss_teacher_confidence",
        },
        teacher_log_prob_batches=(
            torch.tensor([[-1.0, -1.0, -1.0], [-3.0, -2.0, -1.0]]),
        ),
    )

    assert result.by_domain["math"] == pytest.approx(
        {10: (3.0, 1), 20: (1.0, 1), 30: (2.0, 1)}
    )
    assert result.by_domain["science"] == pytest.approx(
        {10: (0.0, 1), 20: (1.25, 1), 30: (3.0, 1)}
    )
    assert result.valid_token_counts == {"math": 3, "science": 3}


def test_four_gpu_profile_has_requested_domain_policy() -> None:
    config = load_config(CONFIG_PATH)

    assert config.runtime.slurm_allocation_gpus == 4
    assert config.trainer.n_gpus_per_node == 4
    assert config.data.train_batch_size == 528
    assert config.audit.control_token_online_top_p_by_domain == {
        "math": 0.05,
        "code": 0.02,
        "science": 0.02,
    }
    assert config.audit.control_token_loss_weight == 4.0
    assert config.audit.control_token_online_min_mean_occurrences_per_step == 20.0
    assert config.audit.control_token_online_strict_occurrence_gate is True
    assert config.audit.control_token_online_selection_mode_by_domain == {
        "math": "top_loss",
        "code": "top_loss",
        "science": "top_loss_teacher_confidence",
    }
    assert config.audit.control_token_online_weight_mode_by_domain == {
        "math": "fixed",
        "code": "fixed",
        "science": "fixed",
    }
    assert set(config.audit.domain_control_token_candidate_groups["code"]) == {
        "Structure"
    }

    command = build_command(config)
    assert any(
        argument.endswith(
            "{math: 'top_loss', code: 'top_loss', "
            "science: 'top_loss_teacher_confidence'}"
        )
        for argument in command
        if argument.startswith(
            "+mopd_audit.control_token_online_selection_mode_by_domain="
        )
    )
