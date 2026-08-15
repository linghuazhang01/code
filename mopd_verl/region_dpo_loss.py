"""Region-level DPO objective over naturally sampled sibling rerollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional


REGION_DPO_INPUT_KEYS = (
    "region_dpo_responses",
    "region_dpo_input_ids",
    "region_dpo_attention_mask",
    "region_dpo_position_ids",
    "region_dpo_reference_log_probs",
    "region_dpo_loss_mask",
    "region_dpo_pair_mask",
)


@dataclass(frozen=True)
class RegionDPOLossResult:
    """Scalar auxiliary loss and detached logging metrics."""

    loss: torch.Tensor
    metrics: dict[str, float]


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def region_dpo_enabled(policy_loss_config: Any) -> bool:
    """Return whether the Region-DPO actor auxiliary is enabled."""

    return bool(
        _cfg_get(policy_loss_config, "region_dpo_enabled", False)
    )


def region_dpo_pair_loss(
    *,
    current_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    beta: float,
) -> RegionDPOLossResult:
    """Compute length-summed DPO over ``[batch, point, branch, token]``.

    Branch index 0 is preferred and branch index 1 is rejected. The reference
    probabilities come from the frozen behavior policy that sampled each
    rerollout, so this is an online DPO update without a second reference model.
    """

    expected_shape = current_log_probs.shape
    if current_log_probs.ndim != 4 or expected_shape[2] != 2:
        raise ValueError(
            "Region-DPO log probabilities must have shape "
            "[batch, point, 2, token]."
        )
    for name, tensor in (
        ("reference_log_probs", reference_log_probs),
        ("loss_mask", loss_mask),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Region-DPO {name} shape {tuple(tensor.shape)} does not "
                f"match {tuple(expected_shape)}."
            )
    if pair_mask.shape != expected_shape[:2]:
        raise ValueError(
            "Region-DPO pair_mask must have shape [batch, point]."
        )
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Region-DPO beta must be finite and positive.")

    mask = loss_mask.to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    reference = reference_log_probs.detach().to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    branch_log_ratios = ((current_log_probs - reference) * mask).sum(
        dim=-1
    )
    preference_logits = float(beta) * (
        branch_log_ratios[..., 0] - branch_log_ratios[..., 1]
    )
    active = pair_mask.to(
        device=current_log_probs.device,
        dtype=current_log_probs.dtype,
    )
    active_count = active.sum()
    if not bool(active_count.detach().item()):
        zero = current_log_probs.sum() * 0.0
        return RegionDPOLossResult(
            loss=zero,
            metrics={
                "actor/region_dpo_loss": 0.0,
                "actor/region_dpo_pair_count": 0.0,
                "actor/region_dpo_preference_accuracy": 0.0,
                "actor/region_dpo_logit_mean": 0.0,
            },
        )

    per_pair_loss = -functional.logsigmoid(preference_logits)
    pair_count_per_rollout = active.sum(dim=-1)
    loss_per_rollout = (per_pair_loss * active).sum(dim=-1) / (
        pair_count_per_rollout.clamp_min(1.0)
    )
    loss = loss_per_rollout.mean()
    accuracy = (
        (preference_logits.detach() > 0).to(active.dtype) * active
    ).sum() / active_count
    logit_mean = (
        preference_logits.detach() * active
    ).sum() / active_count
    return RegionDPOLossResult(
        loss=loss,
        metrics={
            "actor/region_dpo_loss": float(loss.detach().item()),
            "actor/region_dpo_pair_count": float(
                active_count.detach().item()
            ),
            "actor/region_dpo_preference_accuracy": float(accuracy.item()),
            "actor/region_dpo_logit_mean": float(logit_mean.item()),
        },
    )


def _active_branches(
    tensor: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Select active ``[batch, point]`` entries and retain two branches."""

    if tensor.ndim < 3:
        raise ValueError(
            "Packed Region-DPO tensors require at least three dimensions."
        )
    active_indices = pair_mask.detach().bool().nonzero(as_tuple=False)
    return tensor[active_indices[:, 0], active_indices[:, 1]]


def _flatten_active_branches(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten active-pair and chosen/rejected dimensions."""

    if tensor.ndim < 3 or tensor.shape[1] != 2:
        raise ValueError(
            "Active Region-DPO tensors require shape [pair, 2, ...]."
        )
    return tensor.reshape((-1, *tensor.shape[2:]))


def _scatter_active_branches(
    active_tensor: torch.Tensor,
    *,
    pair_mask: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    """Restore active branches to ``[batch, point, 2, ...]`` layout."""

    expected_active = int(pair_mask.detach().bool().sum().item())
    if active_tensor.shape[0] != expected_active:
        raise ValueError(
            "Active Region-DPO branch count does not match pair_mask."
        )
    flat_template = torch.zeros_like(
        template,
        dtype=active_tensor.dtype,
        device=active_tensor.device,
    ).reshape((-1, *template.shape[2:]))
    active_indices = pair_mask.detach().bool().flatten().nonzero(
        as_tuple=False
    ).squeeze(-1)
    restored = flat_template.index_copy(
        0,
        active_indices.to(device=active_tensor.device),
        active_tensor,
    )
    return restored.reshape(template.shape)


def build_region_dpo_actor_loss(
    *,
    actor: Any,
    model_inputs: dict[str, Any],
    policy_loss_config: Any,
    temperature: float,
    loss_scale_factor: float,
) -> RegionDPOLossResult:
    """Run the auxiliary candidate forward and return weighted Region-DPO."""

    weight = float(
        _cfg_get(policy_loss_config, "region_dpo_loss_weight", 0.1)
    )
    beta = float(_cfg_get(policy_loss_config, "region_dpo_beta", 0.1))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError(
            "Region-DPO loss weight must be finite and positive."
        )
    if not all(key in model_inputs for key in REGION_DPO_INPUT_KEYS):
        zero = model_inputs["responses"].sum() * 0.0
        return RegionDPOLossResult(
            loss=zero,
            metrics={
                "actor/region_dpo_loss": 0.0,
                "actor/region_dpo_weighted_loss": 0.0,
                "actor/region_dpo_pair_count": 0.0,
                "actor/region_dpo_preference_accuracy": 0.0,
                "actor/region_dpo_logit_mean": 0.0,
            },
        )

    pair_mask = model_inputs["region_dpo_pair_mask"]
    if not bool(pair_mask.detach().bool().any().item()):
        zero = model_inputs["responses"].sum() * 0.0
        return RegionDPOLossResult(
            loss=zero,
            metrics={
                "actor/region_dpo_loss": 0.0,
                "actor/region_dpo_weighted_loss": 0.0,
                "actor/region_dpo_pair_count": 0.0,
                "actor/region_dpo_preference_accuracy": 0.0,
                "actor/region_dpo_logit_mean": 0.0,
            },
        )

    active_responses = _active_branches(
        model_inputs["region_dpo_responses"], pair_mask
    )
    region_inputs = {
        "responses": _flatten_active_branches(active_responses),
        "input_ids": _flatten_active_branches(
            _active_branches(
                model_inputs["region_dpo_input_ids"], pair_mask
            )
        ),
        "attention_mask": _flatten_active_branches(
            _active_branches(
                model_inputs["region_dpo_attention_mask"], pair_mask
            )
        ),
        "position_ids": _flatten_active_branches(
            _active_branches(
                model_inputs["region_dpo_position_ids"], pair_mask
            )
        ),
    }
    _entropy, current_log_probs = actor._forward_micro_batch(
        region_inputs,
        temperature=float(temperature),
        calculate_entropy=False,
        calculate_log_probs=True,
    )
    active_shape = active_responses.shape
    active_current_log_probs = current_log_probs.reshape(active_shape)
    current_log_probs = _scatter_active_branches(
        active_current_log_probs,
        pair_mask=pair_mask,
        template=model_inputs["region_dpo_reference_log_probs"],
    )
    result = region_dpo_pair_loss(
        current_log_probs=current_log_probs,
        reference_log_probs=model_inputs[
            "region_dpo_reference_log_probs"
        ],
        loss_mask=model_inputs["region_dpo_loss_mask"],
        pair_mask=pair_mask,
        beta=beta,
    )
    weighted_loss = result.loss * weight
    metrics = dict(result.metrics)
    metrics["actor/region_dpo_loss"] *= float(loss_scale_factor)
    metrics["actor/region_dpo_weighted_loss"] = float(
        weighted_loss.detach().item()
    ) * float(loss_scale_factor)
    metrics["actor/region_dpo_beta"] = beta
    metrics["actor/region_dpo_loss_weight"] = weight
    rewards = model_inputs.get("region_dpo_rewards")
    if rewards is not None:
        active_rewards = _active_branches(rewards, pair_mask)
        reward_margin = active_rewards[..., 0] - active_rewards[..., 1]
        metrics["actor/region_dpo_reward_margin"] = float(
            reward_margin.detach().mean().item()
        )
    return RegionDPOLossResult(loss=weighted_loss, metrics=metrics)
