"""Exact full-vocabulary mathematics for the native TIP baseline.

TIP scores response tokens with normalized student entropy and reverse KL,
combines their batch-normalized values with Soft-OR, and retains the largest
``floor(rho * response_length)`` scores independently for every rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mopd_verl.token_baseline_math import (
    clip_masked_upper_quantile,
    masked_batch_minmax,
)


@dataclass(frozen=True)
class TIPSelection:
    """Detached TIP selector outputs on a padded response batch."""

    selected: torch.Tensor
    score: torch.Tensor
    clipped_entropy: torch.Tensor


def _validate_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> None:
    if teacher_logits.ndim != 2 or student_logits.ndim != 2:
        raise ValueError("TIP logits must have shape [tokens, vocabulary].")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher and student TIP logits must have identical shapes, got "
            f"{tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}."
        )
    if student_logits.shape[-1] < 2:
        raise ValueError("TIP requires a vocabulary containing at least two tokens.")


def normalized_entropy_per_token(
    student_logits: torch.Tensor,
    *,
    chunk_size: int = 512,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return ``H(P_student) / log(|V|)`` for every token."""

    if student_logits.ndim != 2:
        raise ValueError("Student logits must have shape [tokens, vocabulary].")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if student_logits.shape[-1] < 2:
        raise ValueError("TIP requires a vocabulary containing at least two tokens.")

    parts: list[torch.Tensor] = []
    normalizer = math.log(student_logits.shape[-1])
    for logits in student_logits.split(chunk_size, dim=0):
        log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
        probs = log_probs.exp()
        parts.append(-(probs * log_probs).sum(dim=-1) / normalizer)
    if not parts:
        return student_logits.new_empty((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def full_vocab_reverse_kl_per_token(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    chunk_size: int = 512,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return exact ``KL(P_student || P_teacher)`` for every token.

    Teacher logits are treated as frozen while gradients are preserved through
    the student logits. Computation is token-chunked but never truncates the
    vocabulary support.
    """

    _validate_logits(teacher_logits, student_logits)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    parts: list[torch.Tensor] = []
    for start in range(0, student_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, student_logits.shape[0])
        student_log_probs = F.log_softmax(
            student_logits[start:stop].float() / temperature,
            dim=-1,
        )
        teacher_log_probs = F.log_softmax(
            teacher_logits[start:stop].detach().float() / temperature,
            dim=-1,
        )
        student_probs = student_log_probs.exp()
        parts.append(
            (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        )
    if not parts:
        return student_logits.new_empty((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _stable_topk_mask(scores: torch.Tensor, count: int) -> torch.Tensor:
    selected = torch.zeros_like(scores, dtype=torch.bool)
    if count <= 0:
        return selected
    if count >= scores.numel():
        selected.fill_(True)
        return selected
    order = torch.argsort(scores, descending=True, stable=True)
    selected[order[:count]] = True
    return selected


@torch.no_grad()
def select_tip_tokens(
    entropy: torch.Tensor,
    divergence: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    retention_ratio: float = 0.5,
    entropy_clip_quantile: float = 0.98,
) -> TIPSelection:
    """Apply the paper TIP selector to a complete padded rollout batch."""

    if entropy.shape != divergence.shape or entropy.shape != response_mask.shape:
        raise ValueError("TIP entropy, divergence, and response_mask shapes must match.")
    if entropy.ndim != 2:
        raise ValueError("TIP selector inputs must have shape [batch, response].")
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError("retention_ratio must be in (0, 1].")
    if not 0.0 < entropy_clip_quantile <= 1.0:
        raise ValueError("entropy_clip_quantile must be in (0, 1].")

    mask = response_mask.bool()
    valid_entropy = entropy[mask].float()
    valid_divergence = divergence[mask].float()
    if valid_entropy.numel() == 0:
        raise ValueError("TIP selection requires at least one valid response token.")
    if not torch.isfinite(valid_entropy).all() or not torch.isfinite(
        valid_divergence
    ).all():
        raise ValueError("TIP selector inputs must be finite on valid tokens.")

    clipped_entropy = clip_masked_upper_quantile(
        entropy.float(),
        mask,
        quantile=entropy_clip_quantile,
    )
    entropy_hat = masked_batch_minmax(clipped_entropy, mask)
    divergence_hat = masked_batch_minmax(divergence.float(), mask)
    score = entropy_hat + divergence_hat - entropy_hat * divergence_hat
    score = score.masked_fill(~mask, 0.0)

    selected = torch.zeros_like(mask)
    for row in range(mask.shape[0]):
        valid_indices = torch.nonzero(mask[row], as_tuple=False).flatten()
        keep_count = math.floor(retention_ratio * valid_indices.numel())
        row_selected = _stable_topk_mask(score[row, valid_indices], keep_count)
        selected[row, valid_indices[row_selected]] = True

    return TIPSelection(
        selected=selected,
        score=score,
        clipped_entropy=clipped_entropy,
    )
