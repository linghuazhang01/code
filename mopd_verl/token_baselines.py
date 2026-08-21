"""Token-selection and token-weighting baselines for OPD training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

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
    return _finite_non_negative(
        cfg_get(
            policy_loss_config,
            "fire_opd_teacher_confidence_alpha",
            1.0,
        ),
        "fire_opd_teacher_confidence_alpha",
    )


def fire_opd_student_confusion_beta(policy_loss_config: Any) -> float:
    return _finite_non_negative(
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
    entropy = _checked_matrix("student_entropy", student_entropy, mask)
    if method == TOKEN_BASELINE_NONE:
        return TokenBaselineResult(
            weights=mask,
            score=torch.zeros_like(mask),
            selected_mask=mask,
        )
    if method == TOKEN_BASELINE_ENTROPY:
        score = entropy
        selected = _select_positions(
            score=score,
            mask=mask,
            retention_ratio=token_baseline_retention_ratio(policy_loss_config),
            selection_mode=token_baseline_selection_mode(policy_loss_config),
        )
        weights = _normalize_per_sequence(selected, mask)
        return TokenBaselineResult(weights, score, selected)
    if method == TOKEN_BASELINE_TIP_TOPK32:
        if divergence is None:
            raise ValueError("TIP-TopK32 requires a per-token divergence matrix.")
        disagreement = _checked_matrix("divergence", divergence, mask)
        entropy_score = _masked_minmax(entropy, mask)
        disagreement_score = _masked_minmax(disagreement, mask)
        score = (
            entropy_score
            + disagreement_score
            - entropy_score * disagreement_score
        )
        selected = _select_positions(
            score=score,
            mask=mask,
            retention_ratio=token_baseline_retention_ratio(policy_loss_config),
            selection_mode=token_baseline_selection_mode(policy_loss_config),
        )
        weights = _normalize_per_sequence(selected, mask)
        return TokenBaselineResult(weights, score, selected)
    if method == TOKEN_BASELINE_FIRE_OPD:
        if teacher_entropy is None:
            raise ValueError("FiRe-OPD requires full-vocabulary teacher entropy.")
        teacher = _checked_matrix("teacher_entropy", teacher_entropy, mask)
        teacher_confidence = 1.0 - _normalize_by_sequence_max(teacher, mask)
        student_confusion = _normalize_by_sequence_max(entropy, mask)
        score = (
            1.0
            + fire_opd_teacher_confidence_alpha(policy_loss_config)
            * teacher_confidence
        ) * (
            1.0
            + fire_opd_student_confusion_beta(policy_loss_config)
            * student_confusion
        )
        weights = _normalize_per_sequence(score * mask, mask)
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
    """Return valid-token-mean-one weights for FiRe trajectory filtering."""

    mask = response_mask.detach().float()
    teacher_logp = _checked_matrix(
        "teacher_chosen_log_probs",
        teacher_chosen_log_probs,
        mask,
    )
    if not math.isfinite(float(drop_ratio)) or not 0.0 <= float(drop_ratio) < 1.0:
        raise ValueError("drop_ratio must be finite and in [0, 1).")
    lengths = mask.sum(dim=-1).clamp(min=1.0)
    trajectory_scores = (teacher_logp * mask).sum(dim=-1) / lengths
    if float(drop_ratio) == 0.0 or trajectory_scores.numel() <= 1:
        return torch.ones_like(trajectory_scores, dtype=torch.float32)
    threshold = torch.quantile(trajectory_scores.float(), float(drop_ratio))
    keep = (trajectory_scores >= threshold).float()
    kept_token_count = (mask * keep.unsqueeze(-1)).sum()
    if float(kept_token_count.item()) <= 0.0:
        return torch.ones_like(trajectory_scores, dtype=torch.float32)
    scale = mask.sum() / kept_token_count
    return keep * scale


def _finite_non_negative(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative, got {parsed}.")
    return parsed


def _checked_matrix(
    label: str,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    matrix = value.detach().float().to(device=mask.device)
    if matrix.shape != mask.shape:
        raise ValueError(
            f"{label} must match response_mask, got "
            f"{tuple(matrix.shape)} and {tuple(mask.shape)}."
        )
    if not torch.isfinite(matrix[mask.bool()]).all():
        raise ValueError(f"{label} contains non-finite values on valid tokens.")
    return torch.where(mask.bool(), matrix, torch.zeros_like(matrix))


def _masked_minmax(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    row_min = torch.where(valid, value, torch.inf).amin(dim=-1, keepdim=True)
    row_max = torch.where(valid, value, -torch.inf).amax(dim=-1, keepdim=True)
    row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
    row_max = torch.where(torch.isfinite(row_max), row_max, row_min)
    denominator = (row_max - row_min).clamp(min=1e-12)
    normalized = (value - row_min) / denominator
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def _normalize_by_sequence_max(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.bool()
    row_max = torch.where(valid, value, -torch.inf).amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.ones_like(row_max))
    normalized = value / row_max.clamp(min=1e-12)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def _normalize_per_sequence(
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_weights = weights.detach().float() * mask
    valid_count = mask.sum(dim=-1, keepdim=True)
    weight_sum = masked_weights.sum(dim=-1, keepdim=True)
    scale = valid_count / weight_sum.clamp(min=1e-12)
    normalized = masked_weights * scale
    return torch.where(weight_sum > 0, normalized, torch.zeros_like(normalized))


def _select_positions(
    *,
    score: torch.Tensor,
    mask: torch.Tensor,
    retention_ratio: float,
    selection_mode: str,
) -> torch.Tensor:
    selected = torch.zeros_like(mask)
    for row_index in range(int(mask.shape[0])):
        valid_indices = torch.nonzero(
            mask[row_index].bool(),
            as_tuple=False,
        ).squeeze(-1)
        valid_count = int(valid_indices.numel())
        if valid_count == 0:
            continue
        selected_count = max(1, int(math.floor(retention_ratio * valid_count)))
        selected_count = min(selected_count, valid_count)
        valid_scores = score[row_index].index_select(0, valid_indices).float()
        if selection_mode == TOKEN_SELECTION_TOPK:
            local_indices = torch.topk(
                valid_scores,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
        elif selection_mode == TOKEN_SELECTION_SAMPLE:
            probabilities = valid_scores.clamp(min=0.0)
            if float(probabilities.sum().item()) <= 0.0:
                probabilities = torch.ones_like(probabilities)
            local_indices = torch.multinomial(
                probabilities,
                num_samples=selected_count,
                replacement=False,
            )
        else:
            raise AssertionError(f"Unhandled token selection mode: {selection_mode}")
        selected[row_index, valid_indices.index_select(0, local_indices)] = 1.0
    return selected
