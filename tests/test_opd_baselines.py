from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from mopd_verl.full_gradient.loss_support import exopd_policy_gradient_rewards
from mopd_verl.token_baselines import (
    build_token_baseline_weights,
    fire_opd_trajectory_weights,
    validate_token_baseline_config,
)
from mopd_verl.topk_distill import (
    distill_loss_builder,
    topk_distill_loss_matrix,
    uses_exopd_loss,
)

@contextmanager
def _stubbed_verl() -> Iterator[None]:
    verl_module = ModuleType("verl")
    verl_module.__path__ = []
    verl_module.DataProto = object
    utils_module = ModuleType("verl.utils")
    utils_module.__path__ = []
    device_module = ModuleType("verl.utils.device")
    device_module.get_device_id = lambda: torch.device("cpu")
    trainer_module = ModuleType("verl.trainer")
    trainer_module.__path__ = []
    ppo_module = ModuleType("verl.trainer.ppo")
    ppo_module.__path__ = []
    core_algos_module = ModuleType("verl.trainer.ppo.core_algos")

    def aggregate(
        loss_mat: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_agg_mode: str,
    ) -> torch.Tensor:
        del loss_agg_mode
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    core_algos_module.agg_loss = aggregate
    core_algos_module.get_policy_loss_fn = lambda _mode: None
    core_algos_module.kl_penalty = lambda **_kwargs: None
    isolated_names = ("mopd_verl.full_gradient.actor_loss",)
    saved_modules = {
        name: sys.modules.pop(name)
        for name in isolated_names
        if name in sys.modules
    }
    try:
        with patch.dict(
            sys.modules,
            {
                "verl": verl_module,
                "verl.trainer": trainer_module,
                "verl.trainer.ppo": ppo_module,
                "verl.trainer.ppo.core_algos": core_algos_module,
                "verl.utils": utils_module,
                "verl.utils.device": device_module,
            },
        ):
            yield
    finally:
        for name in isolated_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def test_entropy_topk_selects_high_entropy_and_preserves_row_mean() -> None:
    result = build_token_baseline_weights(
        student_entropy=torch.tensor([[0.1, 0.9, 0.7, 0.2]]),
        response_mask=torch.ones(1, 4),
        policy_loss_config={
            "token_baseline_method": "entropy",
            "token_baseline_retention_ratio": 0.5,
            "token_baseline_selection_mode": "topk",
        },
    )

    torch.testing.assert_close(
        result.selected_mask,
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.weights,
        torch.tensor([[0.0, 2.0, 2.0, 0.0]]),
    )
    torch.testing.assert_close(result.weights.mean(dim=-1), torch.ones(1))


def test_entropy_sampling_retains_exact_budget() -> None:
    torch.manual_seed(7)
    result = build_token_baseline_weights(
        student_entropy=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        response_mask=torch.ones(1, 5),
        policy_loss_config={
            "token_baseline_method": "entropy_select",
            "token_baseline_retention_ratio": 0.4,
            "token_baseline_selection_mode": "sample",
        },
    )

    assert int(result.selected_mask.sum().item()) == 2
    torch.testing.assert_close(result.weights.mean(dim=-1), torch.ones(1))


def test_tip_soft_or_combines_entropy_and_topk_disagreement() -> None:
    result = build_token_baseline_weights(
        student_entropy=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        divergence=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        response_mask=torch.ones(1, 4),
        policy_loss_config={
            "token_baseline_method": "tip",
            "token_baseline_retention_ratio": 0.5,
            "token_baseline_selection_mode": "topk",
        },
    )

    torch.testing.assert_close(
        result.score,
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.selected_mask,
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
    )


def test_tip_requires_topk_divergence_objective() -> None:
    with pytest.raises(ValueError, match="TIP-TopK32 requires"):
        validate_token_baseline_config(
            {"token_baseline_method": "tip_topk32"},
            uses_topk_distillation=False,
        )


def test_tip_weights_the_actual_topk_actor_loss_matrix() -> None:
    with _stubbed_verl():
        from mopd_verl.full_gradient.actor_loss import (
            build_actor_micro_batch_loss,
        )

        student_topk = torch.tensor(
            [
                [
                    [-0.1, -2.5],
                    [-0.8, -0.9],
                    [-2.5, -0.1],
                    [-0.7, -0.8],
                ]
            ],
            requires_grad=True,
        )
        teacher_topk = torch.tensor(
            [
                [
                    [-0.2, -1.8],
                    [-0.2, -1.8],
                    [-0.2, -1.8],
                    [-0.2, -1.8],
                ]
            ]
        )
        student_entropy = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        class MicroBatch:
            batch = {
                "response_mask": torch.ones(1, 4),
                "old_log_probs": torch.full((1, 4), -1.0),
                "advantages": torch.zeros(1, 4),
                "student_entropy": student_entropy,
                "math_teacher_topk_ids": torch.tensor(
                    [[[0, 1], [0, 1], [0, 1], [0, 1]]]
                ),
                "math_teacher_topk_logprobs": teacher_topk,
            }
            non_tensor_batch: dict[str, object] = {}
            meta_info = {"temperature": 1.0}

            def to(self, _device: object) -> "MicroBatch":
                return self

        class Actor:
            config = {
                "entropy_coeff": 0.0,
                "kl_loss_coef": 0.0,
                "loss_agg_mode": "token-mean",
                "use_kl_loss": False,
                "policy_loss": {
                    "distill_loss_builder": "topk_kl",
                    "distill_mode": "topk_renormalized_reverse_kl",
                    "topk_distill_enabled": True,
                    "topk_distill_support_source": "teacher",
                    "topk_distill_tail_bucket": False,
                    "token_baseline_method": "tip_topk32",
                    "token_baseline_retention_ratio": 0.5,
                    "token_baseline_selection_mode": "topk",
                },
            }

            def _forward_micro_batch(
                self,
                _model_inputs: dict[str, object],
                **_kwargs: object,
            ) -> tuple[object, ...]:
                return (
                    None,
                    torch.zeros(1, 4, requires_grad=True),
                    None,
                    None,
                    student_topk,
                )

        result = build_actor_micro_batch_loss(
            Actor(),
            MicroBatch(),
            loss_scale_factor=1.0,
            on_policy=False,
            include_metrics=True,
        )

    divergence = topk_distill_loss_matrix(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
        mode="topk_renormalized_reverse_kl",
        include_tail=False,
        temperature=1.0,
    )
    baseline = build_token_baseline_weights(
        student_entropy=student_entropy,
        divergence=divergence,
        response_mask=torch.ones(1, 4),
        policy_loss_config=Actor.config["policy_loss"],
    )
    expected = (divergence * baseline.weights).mean()

    torch.testing.assert_close(result.loss, expected)
    assert result.metrics["actor/token_baseline_selected_ratio"] == 0.5


def test_fire_token_weight_matches_confidence_confusion_formula() -> None:
    result = build_token_baseline_weights(
        student_entropy=torch.tensor([[1.0, 2.0]]),
        teacher_entropy=torch.tensor([[1.0, 2.0]]),
        response_mask=torch.ones(1, 2),
        policy_loss_config={
            "token_baseline_method": "fire_opd",
            "fire_opd_teacher_confidence_alpha": 1.0,
            "fire_opd_student_confusion_beta": 1.0,
        },
    )

    raw = torch.tensor([[2.25, 2.0]])
    expected = raw / raw.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(result.weights, expected)
    torch.testing.assert_close(result.weights.mean(dim=-1), torch.ones(1))


def test_fire_filter_is_global_and_preserves_valid_token_mean() -> None:
    response_mask = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    teacher_log_probs = torch.tensor(
        [
            [-5.0, -5.0],
            [-4.0, -4.0],
            [-3.0, -3.0],
            [-2.0, -2.0],
            [-1.0, -1.0],
        ]
    )

    weights = fire_opd_trajectory_weights(
        teacher_chosen_log_probs=teacher_log_probs,
        response_mask=response_mask,
        drop_ratio=0.2,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor([0.0, 1.125, 1.125, 1.125, 1.125]),
    )
    token_weight_mean = (
        weights.unsqueeze(-1) * response_mask
    ).sum() / response_mask.sum()
    torch.testing.assert_close(token_weight_mean, torch.tensor(1.0))


def test_exopd_advantage_uses_frozen_student_reference() -> None:
    old = torch.tensor([[-2.0, -3.0]])
    teacher = torch.tensor([[-1.0, -1.5]])
    base = torch.tensor([[-2.5, -2.5]])

    actual = exopd_policy_gradient_rewards(
        {
            "math_teacher_log_prob": teacher,
            "base_log_prob": base,
        },
        {
            "lambda_vals": 1.25,
            "multi_teacher_distill": False,
        },
        old,
    )
    expected = 1.25 * (teacher - base) - (old - base)

    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="base_log_prob"):
        exopd_policy_gradient_rewards(
            {"math_teacher_log_prob": teacher},
            {"lambda_vals": 1.25},
            old,
        )


def test_exopd_has_an_explicit_builder() -> None:
    assert distill_loss_builder({"distill_loss_builder": "exopd"}) == "exopd"
    assert uses_exopd_loss({"distill_loss_builder": "extrapolated_opd"})
