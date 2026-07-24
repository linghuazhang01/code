"""Small helpers shared by the unified actor loss."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from mopd_verl.full_gradient.config import _cfg_get
from mopd_verl.full_gradient.labels import _TEACHER_LABEL_KEY
from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_SPARSE,
    TOPK_SUPPORT_SOURCE_STUDENT,
    TOPK_SUPPORT_SOURCE_TEACHER,
    chosen_token_policy_gradient_reward_matrix,
    select_teacher_log_prob_tensor,
    select_teacher_tensor_by_domain,
    topk_distill_logprob_mode,
    topk_distill_support_source,
    topk_distill_uses_renormalized_support,
)


@dataclass(frozen=True)
class ActorMicroBatchLossResult:
    loss: torch.Tensor
    metrics: dict[str, Any]
    teacher_student_cross_entropy: torch.Tensor | None = None
    configured_token_loss: torch.Tensor | None = None
    configured_token_loss_mask: torch.Tensor | None = None


ACTOR_LOSS_CONTRIBUTION_METRICS: Final[frozenset[str]] = frozenset(
    {
        "actor/kl_loss",
        "actor/pg_loss",
        "actor/student_suffix_token_count",
        "actor/teacher_prefix_forward_kl_loss",
        "actor/teacher_prefix_token_count",
        "actor/topk_distill_loss",
    }
)


def partition_actor_micro_batch_metrics(
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split additive mini-batch contributions from per-micro observations."""

    contributions: dict[str, Any] = {}
    observations: dict[str, Any] = {}
    for key, value in metrics.items():
        target = (
            contributions
            if key in ACTOR_LOSS_CONTRIBUTION_METRICS
            else observations
        )
        target[key] = value
    return contributions, observations


def aggregate_actor_micro_batch_metrics(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
    """Sum weighted loss contributions once per optimizer mini-batch.

    The driver averages metric rows. Emitting every already-weighted loss
    contribution separately would divide the mini-batch loss by the number of
    micro-batches a second time.
    """

    contribution_totals: dict[str, float] = {}
    observation_rows: list[dict[str, Any]] = []
    for row in rows:
        contributions, observations = partition_actor_micro_batch_metrics(row)
        for key, value in contributions.items():
            contribution_totals[key] = contribution_totals.get(key, 0.0) + float(
                value
            )
        if observations:
            observation_rows.append(observations)
    return contribution_totals, tuple(observation_rows)


def active_kl_loss(config: Any) -> tuple[bool, float]:
    kl_coef = float(_cfg_get(config, "kl_loss_coef", 0.0) or 0.0)
    enabled = bool(_cfg_get(config, "use_kl_loss", False)) and kl_coef != 0.0
    return enabled, kl_coef


def topk_runtime_config(policy_loss_cfg: Any) -> tuple[bool, str]:
    use_renormalized_support = topk_distill_uses_renormalized_support(
        policy_loss_cfg
    )
    effective_mode = (
        TOPK_LOGPROB_MODE_SPARSE
        if use_renormalized_support
        else topk_distill_logprob_mode(policy_loss_cfg)
    )
    return use_renormalized_support, effective_mode


def selected_teacher_log_prob(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> torch.Tensor:
    if "math_teacher_log_prob" not in model_inputs:
        if "ref_log_prob" in model_inputs:
            return model_inputs["ref_log_prob"]
        raise ValueError(
            "Reverse-KL advantages require a teacher/ref log-prob tensor."
        )
    return select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)


def selected_topk_support(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_source = topk_distill_support_source(policy_loss_cfg)
    if support_source == TOPK_SUPPORT_SOURCE_STUDENT:
        if "student_topk_ids" not in model_inputs:
            raise ValueError("Student top-k distillation requires student_topk_ids.")
        key = "math_teacher_student_topk_logprobs"
        if key not in model_inputs:
            raise ValueError(f"Student top-k distillation requires {key}.")
        teacher_values = select_teacher_tensor_by_domain(
            model_inputs,
            policy_loss_cfg,
            suffix="student_topk_logprobs",
        )
        return model_inputs["student_topk_ids"], teacher_values
    if support_source != TOPK_SUPPORT_SOURCE_TEACHER:
        raise ValueError(f"Unsupported top-k support source: {support_source!r}.")
    for key in ("math_teacher_topk_ids", "math_teacher_topk_logprobs"):
        if key not in model_inputs:
            raise ValueError(f"Top-k distillation requires {key}.")
    return (
        select_teacher_tensor_by_domain(
            model_inputs,
            policy_loss_cfg,
            suffix="topk_ids",
        ),
        select_teacher_tensor_by_domain(
            model_inputs,
            policy_loss_cfg,
            suffix="topk_logprobs",
        ),
    )


def actor_advantages(
    actor: Any,
    model_inputs: dict[str, Any],
    old_log_prob: torch.Tensor,
) -> torch.Tensor:
    policy_cfg = _cfg_get(actor.config, "policy_loss", {})
    if not bool(_cfg_get(policy_cfg, "only_reverse_kl_advantages", False)):
        return model_inputs["advantages"]

    math_teacher = selected_teacher_log_prob(model_inputs, policy_cfg)
    base = model_inputs.get("base_log_prob")
    weight = float(_cfg_get(policy_cfg, "lambda_vals", 1.0))
    multi_teacher = bool(_cfg_get(policy_cfg, "multi_teacher_distill", False))

    if base is not None:
        if multi_teacher and _TEACHER_LABEL_KEY in model_inputs:
            teacher = selected_teacher_log_prob(model_inputs, policy_cfg)
            reverse_kl = (
                old_log_prob - teacher
                if weight == 1.0
                else (old_log_prob - base) - (teacher - base) * weight
            )
        elif multi_teacher:
            reverse_kl = old_log_prob - math_teacher
        else:
            reverse_kl = (
                old_log_prob - math_teacher
                if weight == 1.0
                else (old_log_prob - base) - (math_teacher - base) * weight
            )
    elif multi_teacher and _TEACHER_LABEL_KEY in model_inputs:
        reverse_kl = old_log_prob - selected_teacher_log_prob(
            model_inputs,
            policy_cfg,
        )
    else:
        reverse_kl = old_log_prob - math_teacher
    return -reverse_kl


def policy_gradient_rewards(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
    old_log_prob: torch.Tensor,
) -> torch.Tensor:
    teacher = selected_teacher_log_prob(model_inputs, policy_loss_cfg)
    return chosen_token_policy_gradient_reward_matrix(
        student_log_probs=old_log_prob,
        teacher_log_probs=teacher,
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    detached_mask = mask.detach().float()
    denominator = detached_mask.sum().clamp(min=1.0)
    masked_value = torch.where(
        detached_mask != 0,
        value.detach().float(),
        torch.zeros_like(detached_mask),
    )
    result = (masked_value * detached_mask).sum() / denominator
    return float(result.cpu().item())


def gate_tensor_gradient(
    value: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    """Keep forward values unchanged while selecting a gradient contribution."""

    if value is None:
        return None
    gate = mask.to(device=value.device, dtype=value.dtype)
    while gate.ndim < value.ndim:
        gate = gate.unsqueeze(-1)
    # Express the straight-through estimator around an exact zero residual.
    # This keeps the forward tensor bitwise identical even in BF16 while the
    # derivative with respect to ``value`` is scaled by ``gate``.
    return value.detach() + (value - value.detach()) * gate


def floating_response_gradient_mask(
    weights: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve weights above one even when the response mask is boolean."""

    return weights.to(
        device=response_mask.device,
        dtype=torch.float32,
    ) * response_mask.to(dtype=torch.float32)
