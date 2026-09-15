"""Robust token-type scoring for loss plus chosen-token confidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


LOWER_QUANTILE = 0.02
UPPER_QUANTILE = 0.98


def _robust_minmax(
    values: "torch.Tensor",
    normalization_mask: "torch.Tensor",
) -> tuple["torch.Tensor", float, float]:
    """Clip to Q2--Q98 and min-max normalize one token-type signal."""

    import torch

    selected = values[normalization_mask]
    if selected.numel() == 0:
        return torch.zeros_like(values), 0.0, 0.0
    quantiles = torch.quantile(
        selected.to(dtype=torch.float64),
        torch.tensor(
            [LOWER_QUANTILE, UPPER_QUANTILE],
            device=selected.device,
            dtype=torch.float64,
        ),
    )
    lower, upper = quantiles.unbind()
    width = upper - lower
    if width <= 1e-12:
        normalized = torch.zeros_like(values)
    else:
        normalized = (values.clamp(min=lower, max=upper) - lower) / width
    return normalized, float(lower), float(upper)


def loss_teacher_confidence_type_scores(
    *,
    loss_abs_sums: "torch.Tensor",
    teacher_logp_sums: "torch.Tensor",
    counts: "torch.Tensor",
    normalization_mask: "torch.Tensor",
) -> tuple["torch.Tensor", dict[str, float]]:
    """Return ``L_hat + C_hat + L_hat*C_hat`` for each token type.

    Chosen-token confidence is ``p_T(y_t | c_t)``.  It is aggregated as
    mean teacher chosen-token log-probability per token type and normalized
    in log space for numerical stability.  The transformation is monotonic:
    larger normalized values always mean larger teacher confidence.
    """

    import torch

    tensors = (loss_abs_sums, teacher_logp_sums, counts, normalization_mask)
    if (
        any(tensor.ndim != 1 for tensor in tensors)
        or len({tuple(tensor.shape) for tensor in tensors}) != 1
    ):
        raise ValueError(
            "Loss-confidence sums, counts, and normalization mask must be "
            "aligned rank-1 tensors."
        )
    observed = counts > 0
    effective_mask = normalization_mask.to(dtype=torch.bool) & observed
    if not torch.isfinite(loss_abs_sums[observed]).all() or bool(
        (loss_abs_sums[observed] < 0.0).any()
    ):
        raise ValueError(
            "Loss-confidence absolute loss sums must be finite and non-negative."
        )
    if not torch.isfinite(teacher_logp_sums[observed]).all():
        raise ValueError("Teacher chosen-token log-probability sums must be finite.")

    safe_counts = counts.clamp(min=1).to(dtype=torch.float64)
    mean_loss = loss_abs_sums.to(dtype=torch.float64) / safe_counts
    mean_teacher_logp = (teacher_logp_sums.to(dtype=torch.float64) / safe_counts).clamp(
        max=0.0
    )
    normalized_loss, loss_q02, loss_q98 = _robust_minmax(
        mean_loss,
        effective_mask,
    )
    normalized_confidence, logp_q02, logp_q98 = _robust_minmax(
        mean_teacher_logp,
        effective_mask,
    )
    scores = (
        normalized_loss
        + normalized_confidence
        + normalized_loss * normalized_confidence
    )
    scores = torch.where(observed, scores, torch.zeros_like(scores))
    return scores, {
        "loss_teacher_confidence_eligible_token_count": float(effective_mask.sum()),
        "loss_teacher_confidence_loss_q02": loss_q02,
        "loss_teacher_confidence_loss_q98": loss_q98,
        "loss_teacher_confidence_teacher_logp_q02": logp_q02,
        "loss_teacher_confidence_teacher_logp_q98": logp_q98,
    }
