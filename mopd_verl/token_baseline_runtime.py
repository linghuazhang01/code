"""Driver-to-actor transport for batch-scoped OPD baseline scores."""

from __future__ import annotations

from typing import Any

import torch

from mopd_verl.token_baselines import (
    TOKEN_BASELINE_FIRE_OPD,
    TOKEN_BASELINE_TIP_TOPK32,
    TokenBaselineResult,
    build_token_baseline_weights,
    token_baseline_method,
)
from mopd_verl.topk_distill import select_teacher_tensor_by_domain


PRECOMPUTED_WEIGHT_KEY = "token_baseline_precomputed_weight"
PRECOMPUTED_SCORE_KEY = "token_baseline_precomputed_score"
PRECOMPUTED_SELECTED_KEY = "token_baseline_precomputed_selected_mask"
PRECOMPUTED_KEYS = (
    PRECOMPUTED_WEIGHT_KEY,
    PRECOMPUTED_SCORE_KEY,
    PRECOMPUTED_SELECTED_KEY,
)


def requires_batch_precompute(policy_loss_config: Any) -> bool:
    """Return whether scoring uses statistics spanning the rollout batch."""

    return token_baseline_method(policy_loss_config) in {
        TOKEN_BASELINE_TIP_TOPK32,
        TOKEN_BASELINE_FIRE_OPD,
    }


def build_precomputed_baseline_tensors(
    model_inputs: dict[str, Any],
    policy_loss_config: Any,
) -> dict[str, torch.Tensor]:
    """Build detached weights once on the complete driver-side batch."""

    method = token_baseline_method(policy_loss_config)
    if method not in {TOKEN_BASELINE_TIP_TOPK32, TOKEN_BASELINE_FIRE_OPD}:
        return {}
    if "student_entropy" not in model_inputs:
        raise ValueError(f"{method} requires cached full-vocabulary student entropy.")
    if "response_mask" not in model_inputs:
        raise ValueError(f"{method} requires response_mask.")

    teacher_entropy = None
    divergence = None
    if method == TOKEN_BASELINE_FIRE_OPD:
        teacher_entropy = select_teacher_tensor_by_domain(
            model_inputs,
            policy_loss_config,
            suffix="entropy",
        )
    else:
        divergence = select_teacher_tensor_by_domain(
            model_inputs,
            policy_loss_config,
            suffix="topk_divergence",
        )

    response_mask = model_inputs["response_mask"]
    if "student_suffix_mask" in model_inputs:
        response_mask = response_mask * model_inputs["student_suffix_mask"]
    result = build_token_baseline_weights(
        student_entropy=model_inputs["student_entropy"],
        teacher_entropy=teacher_entropy,
        divergence=divergence,
        response_mask=response_mask,
        policy_loss_config=policy_loss_config,
        trajectory_weight=model_inputs.get("token_baseline_trajectory_weight"),
    )
    return {
        PRECOMPUTED_WEIGHT_KEY: result.weights.detach(),
        PRECOMPUTED_SCORE_KEY: result.score.detach(),
        PRECOMPUTED_SELECTED_KEY: result.selected_mask.detach(),
    }


def load_precomputed_baseline_result(
    model_inputs: dict[str, Any],
    response_mask: torch.Tensor,
) -> TokenBaselineResult | None:
    """Validate and load a driver-computed result from an actor microbatch."""

    present = [key in model_inputs for key in PRECOMPUTED_KEYS]
    if not any(present):
        return None
    if not all(present):
        missing = [key for key, exists in zip(PRECOMPUTED_KEYS, present) if not exists]
        raise ValueError(f"Incomplete precomputed token baseline tensors: {missing}.")

    tensors = [model_inputs[key].detach().float() for key in PRECOMPUTED_KEYS]
    for key, tensor in zip(PRECOMPUTED_KEYS, tensors):
        if tensor.shape != response_mask.shape:
            raise ValueError(
                f"{key} must match response_mask, got {tuple(tensor.shape)} and "
                f"{tuple(response_mask.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains non-finite values.")
    if (tensors[0] < 0).any() or (tensors[2] < 0).any():
        raise ValueError("Precomputed token baseline weights/mask must be non-negative.")
    return TokenBaselineResult(
        weights=tensors[0],
        score=tensors[1],
        selected_mask=tensors[2],
    )
