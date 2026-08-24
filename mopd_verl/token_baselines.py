"""Token-selection and token-weighting baselines for OPD training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mopd_verl.token_baseline_math import (
    checked_matrix,
    clip_masked_upper_quantile,
    finite_non_negative,
    masked_batch_minmax,
    normalize_by_batch_max,
    normalize_per_sequence,
    select_positions,
)
from mopd_verl.topk_distill import cfg_get


TOKEN_BASELINE_NONE = "none"
TOKEN_BASELINE_ENTROPY = "entropy"
TOKEN_BASELINE_TIP_TOPK32 = "tip_topk32"
TOKEN_BASELINE_FIRE_OPD = "fire_opd"
TOKEN_BASELINES = {
    TOKEN_BASELINE_NONE,
    TOKEN_BASELINE_ENTROPY,
    TOKEN_BASELINE_TIP_TOPK32,
    TOKEN_BASELINE_FIRE_OPD,
}
TOKEN_BASELINE_ALIASES = {
    "disabled": TOKEN_BASELINE_NONE,
    "off": TOKEN_BASELINE_NONE,
    "entropy_select": TOKEN_BASELINE_ENTROPY,
    "tip": TOKEN_BASELINE_TIP_TOPK32,
    "fire": TOKEN_BASELINE_FIRE_OPD,
    "fire-opd": TOKEN_BASELINE_FIRE_OPD,
}

TOKEN_SELECTION_TOPK = "topk"
TOKEN_SELECTION_SAMPLE = "sample"
TOKEN_SELECTION_MODES = {TOKEN_SELECTION_TOPK, TOKEN_SELECTION_SAMPLE}
TIP_ENTROPY_CLIP_QUANTILE = 0.98


@dataclass(frozen=True)
class TokenBaselineResult:
    """Detached token weights and observability tensors for one baseline."""

    weights: torch.Tensor
    score: torch.Tensor
    selected_mask: torch.Tensor


def token_baseline_method(policy_loss_config: Any) -> str:
    method = str(
        cfg_get(
            policy_loss_config,
            "token_baseline_method",
            TOKEN_BASELINE_NONE,
        )
        or TOKEN_BASELINE_NONE
    ).strip().lower()
    method = TOKEN_BASELINE_ALIASES.get(method, method)
    if method not in TOKEN_BASELINES:
        raise ValueError(
            "token_baseline_method must be one of "
            f"{sorted(TOKEN_BASELINES)} or aliases "
            f"{sorted(TOKEN_BASELINE_ALIASES)}, got {method!r}."
        )
    return method


def token_baseline_retention_ratio(policy_loss_config: Any) -> float:
    value = float(
        cfg_get(policy_loss_config, "token_baseline_retention_ratio", 0.5)
    )
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            "token_baseline_retention_ratio must be finite and in (0, 1], "
            f"got {value}."
        )
    return value


def token_baseline_selection_mode(policy_loss_config: Any) -> str:
    mode = str(
        cfg_get(
            policy_loss_config,
            "token_baseline_selection_mode",
            TOKEN_SELECTION_TOPK,
        )
    ).strip().lower()
    if mode not in TOKEN_SELECTION_MODES:
        raise ValueError(
            "token_baseline_selection_mode must be one of "
            f"{sorted(TOKEN_SELECTION_MODES)}, got {mode!r}."
        )
    return mode


def fire_opd_teacher_confidence_alpha(policy_loss_config: Any) -> float:
    return finite_non_negative(
        cfg_get(
            policy_loss_config,
            "fire_opd_teacher_confidence_alpha",
            1.0,
        ),
        "fire_opd_teacher_confidence_alpha",
    )


def fire_opd_student_confusion_beta(policy_loss_config: Any) -> float:
    return finite_non_negative(
        cfg_get(
            policy_loss_config,
            "fire_opd_student_confusion_beta",
            1.0,
        ),
        "fire_opd_student_confusion_beta",
    )


def fire_opd_trajectory_drop_ratio(policy_loss_config: Any) -> float:
    value = float(
        cfg_get(policy_loss_config, "fire_opd_trajectory_drop_ratio", 0.2)
    )
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(
            "fire_opd_trajectory_drop_ratio must be finite and in [0, 1), "
            f"got {value}."
        )
    return value


def fire_opd_filter_trajectories(policy_loss_config: Any) -> bool:
    return bool(
        cfg_get(policy_loss_config, "fire_opd_filter_trajectories", False)
    )


def uses_token_baseline(policy_loss_config: Any) -> bool:
    return token_baseline_method(policy_loss_config) != TOKEN_BASELINE_NONE


def uses_fire_opd_baseline(policy_loss_config: Any) -> bool:
    return token_baseline_method(policy_loss_config) == TOKEN_BASELINE_FIRE_OPD


def token_baseline_requires_student_entropy(policy_loss_config: Any) -> bool:
    return token_baseline_method(policy_loss_config) in {
        TOKEN_BASELINE_ENTROPY,
        TOKEN_BASELINE_TIP_TOPK32,
        TOKEN_BASELINE_FIRE_OPD,
    }


def token_baseline_requires_teacher_entropy(policy_loss_config: Any) -> bool:
    return uses_fire_opd_baseline(policy_loss_config)


def token_baseline_requires_divergence(policy_loss_config: Any) -> bool:
    return token_baseline_method(policy_loss_config) == TOKEN_BASELINE_TIP_TOPK32


def validate_token_baseline_config(
    policy_loss_config: Any,
    *,
    uses_topk_distillation: bool,
) -> None:
    """Fail early for unsupported or numerically invalid combinations."""

    method = token_baseline_method(policy_loss_config)
    token_baseline_retention_ratio(policy_loss_config)
    token_baseline_selection_mode(policy_loss_config)
    fire_opd_teacher_confidence_alpha(policy_loss_config)
    fire_opd_student_confusion_beta(policy_loss_config)
    fire_opd_trajectory_drop_ratio(policy_loss_config)
    if method == TOKEN_BASELINE_TIP_TOPK32 and not uses_topk_distillation:
        raise ValueError(
            "TIP-TopK32 requires distill_loss_builder=topk_kl because its "
            "disagreement score is the per-token TopK KL."
        )


def build_token_baseline_weights(
    *,
    student_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    policy_loss_config: Any,
    divergence: torch.Tensor | None = None,
    teacher_entropy: torch.Tensor | None = None,
    trajectory_weight: torch.Tensor | None = None,
) -> TokenBaselineResult:
    """Build mean-preserving detached weights for the configured baseline."""

    method = token_baseline_method(policy_loss_config)
    mask = response_mask.detach().to(dtype=torch.float32)
    entropy = checked_matrix("student_entropy", student_entropy, mask)
    if method == TOKEN_BASELINE_NONE:
        return TokenBaselineResult(
            weights=mask,
            score=torch.zeros_like(mask),
            selected_mask=mask,
        )
    if method == TOKEN_BASELINE_ENTROPY:
        score = entropy
        selected = select_positions(
            score=score,
            mask=mask,
            retention_ratio=token_baseline_retention_ratio(policy_loss_config),
            selection_mode=token_baseline_selection_mode(policy_loss_config),
        )
        weights = normalize_per_sequence(selected, mask)
        return TokenBaselineResult(weights, score, selected)
    if method == TOKEN_BASELINE_TIP_TOPK32:
        if divergence is None:
            raise ValueError("TIP-TopK32 requires a per-token divergence matrix.")
        disagreement = checked_matrix("divergence", divergence, mask)
        clipped_entropy = clip_masked_upper_quantile(
            entropy,
            mask,
            quantile=TIP_ENTROPY_CLIP_QUANTILE,
        )
        entropy_score = masked_batch_minmax(clipped_entropy, mask)
        disagreement_score = masked_batch_minmax(disagreement, mask)
        score = (
            entropy_score
            + disagreement_score
            - entropy_score * disagreement_score
        )
        selected = select_positions(
            score=score,
            mask=mask,
            retention_ratio=token_baseline_retention_ratio(policy_loss_config),
            selection_mode=token_baseline_selection_mode(policy_loss_config),
        )
        weights = normalize_per_sequence(selected, mask)
        return TokenBaselineResult(weights, score, selected)
    if method == TOKEN_BASELINE_FIRE_OPD:
        if teacher_entropy is None:
            raise ValueError("FiRe-OPD requires full-vocabulary teacher entropy.")
        teacher = checked_matrix("teacher_entropy", teacher_entropy, mask)
        teacher_confidence = 1.0 - normalize_by_batch_max(teacher, mask)
        student_confusion = normalize_by_batch_max(entropy, mask)
        score = (
            1.0
            + fire_opd_teacher_confidence_alpha(policy_loss_config)
            * teacher_confidence
        ) * (
            1.0
            + fire_opd_student_confusion_beta(policy_loss_config)
            * student_confusion
        )
        weights = normalize_per_sequence(score * mask, mask)
        if trajectory_weight is not None:
            row_weight = trajectory_weight.detach().float()
            if row_weight.ndim == 1:
                row_weight = row_weight.unsqueeze(-1)
            if row_weight.shape != (mask.shape[0], 1):
                raise ValueError(
                    "trajectory_weight must have shape [batch] or [batch, 1], "
                    f"got {tuple(row_weight.shape)}."
                )
            weights = weights * row_weight.to(device=weights.device)
            selected = mask * (row_weight > 0).to(mask.dtype)
        else:
            selected = mask
        return TokenBaselineResult(weights, score, selected)
    raise AssertionError(f"Unhandled token baseline method: {method}")


def fire_opd_trajectory_weights(
    *,
    teacher_chosen_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    drop_ratio: float,
) -> torch.Tensor:
    """Return exact bottom-p filter weights for sequence-mean accumulation."""

    mask = response_mask.detach().float()
    teacher_logp = checked_matrix(
        "teacher_chosen_log_probs",
        teacher_chosen_log_probs,
        mask,
    )
    if not math.isfinite(float(drop_ratio)) or not 0.0 <= float(drop_ratio) < 1.0:
        raise ValueError("drop_ratio must be finite and in [0, 1).")
    lengths = mask.sum(dim=-1)
    valid_rows = lengths > 0
    trajectory_scores = (teacher_logp * mask).sum(dim=-1) / lengths.clamp(
        min=1.0
    )
    trajectory_scores = torch.where(
        valid_rows,
        trajectory_scores,
        torch.full_like(trajectory_scores, torch.inf),
    )
    valid_count = int(valid_rows.sum().item())
    if valid_count == 0:
        return torch.zeros_like(trajectory_scores, dtype=torch.float32)
    drop_count = int(math.floor(float(drop_ratio) * valid_count))
    keep = valid_rows.to(dtype=torch.float32)
    if drop_count > 0:
        ranked = torch.argsort(trajectory_scores.float(), stable=True)
        keep[ranked[:drop_count]] = 0.0
    kept_count = int(keep.sum().item())
    if kept_count == 0:
        return valid_rows.to(dtype=torch.float32)
    return keep * (float(valid_count) / float(kept_count))
