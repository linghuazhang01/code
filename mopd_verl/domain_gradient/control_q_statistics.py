"""Exact occurrence-Q moments, normalized after global step reduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QStatistics:
    by_domain: dict[str, dict[int, tuple[float, int]]]
    valid_token_counts: dict[str, int]
    valid_score_sums: dict[str, float]
    normalization: dict[str, dict[str, float]]


@torch.no_grad()
def global_q_statistics(
    token_id_batches: Sequence[torch.Tensor],
    loss_batches: Sequence[torch.Tensor],
    mask_batches: Sequence[torch.Tensor],
    label_batches: Sequence[Sequence[str]],
    *,
    domains: Sequence[str],
    domain_candidates: Mapping[str, Sequence[int]],
    student_entropy_batches: Sequence[torch.Tensor] | None,
) -> QStatistics:
    """Pool sum(L), sum(H), sum(L*H), count before deriving sum(Q).

    The penultimate column includes ALL valid occurrences, not just candidates.
    The last column carries numerical-error flags through the same collective,
    so a bad value on one rank cannot strand peers at the reduction.
    """

    if student_entropy_batches is None or len(student_entropy_batches) != len(
        loss_batches
    ):
        raise ValueError("Q selection requires aligned student entropy batches.")
    candidates = tuple(
        sorted({int(t) for ids in domain_candidates.values() for t in ids})
    )
    device = loss_batches[0].device
    candidate_ids = torch.tensor(candidates, device=device, dtype=torch.long)
    packed = torch.zeros(
        (len(domains), 4, len(candidates) + 2), device=device, dtype=torch.float64
    )
    allowed = torch.tensor(
        [[t in domain_candidates[d] for t in candidates] for d in domains],
        device=device,
        dtype=torch.bool,
    )
    for ids, losses, mask, labels, entropy in zip(
        token_id_batches,
        loss_batches,
        mask_batches,
        label_batches,
        student_entropy_batches,
        strict=True,
    ):
        if ids.ndim != 2 or not (
            ids.shape == losses.shape == mask.shape == entropy.shape
        ):
            raise ValueError("Q token IDs, loss, mask and student entropy must align.")
        if len(labels) != ids.shape[0]:
            raise ValueError("Q domain labels must align with response rows.")
        valid = mask.to(device=device, dtype=torch.bool)
        loss = losses.detach().to(device=device, dtype=torch.float64).abs()
        ent = entropy.detach().to(device=device, dtype=torch.float64)
        token_ids = ids.to(device=device, dtype=torch.long)
        for di, domain in enumerate(domains):
            rows = torch.tensor(
                [str(label) == domain for label in labels], device=device
            ).bool()
            selected = valid & rows.unsqueeze(-1)
            lval, hval = loss[selected], ent[selected]
            product = lval * hval
            good = (
                torch.isfinite(lval)
                & torch.isfinite(hval)
                & (hval >= 0)
                & torch.isfinite(product)
            )
            packed[di, 0, -1].add_((~good).sum())
            packed[di, 1, -1].add_((~torch.isfinite(lval)).sum())
            packed[di, 2, -1].add_((~torch.isfinite(hval)).sum())
            packed[di, 3, -1].add_((torch.isfinite(hval) & (hval < 0)).sum())
            moments = torch.stack((lval, hval, product, torch.ones_like(lval)))
            moments = torch.where(good.unsqueeze(0), moments, torch.zeros_like(moments))
            packed[di, :, -2].add_(moments.sum(dim=1))
            positions = torch.searchsorted(candidate_ids, token_ids[selected])
            safe = positions.clamp(max=len(candidates) - 1)
            matched = (
                (positions < len(candidates))
                & candidate_ids[safe].eq(token_ids[selected])
                & allowed[di, safe]
            )
            packed[di].scatter_add_(
                1, positions[matched].expand(4, -1), moments[:, matched]
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    if not torch.isfinite(packed).all() or bool((packed[:, 0, -1] > 0).any()):
        diagnostics = {
            domain: dict(zip(
                ("invalid_occurrences", "nonfinite_loss", "nonfinite_entropy",
                 "negative_finite_entropy"),
                packed[di, :, -1].cpu().tolist(),
                strict=True,
            ))
            for di, domain in enumerate(domains)
        }
        raise ValueError(
            "Q requires finite loss and non-negative finite student entropy and moments."
            f" Global input counts: {diagnostics}; "
            f"nonfinite_reduced_entries={int((~torch.isfinite(packed)).sum())}."
        )
    totals = packed[:, :, -2]
    if bool((totals[:, 3] <= 0).any()):
        raise ValueError("Q requires non-empty all-valid occurrences in each domain.")
    means = totals[:, :2] / totals[:, 3:4]
    if not torch.isfinite(means).all() or bool((means <= 0).any()):
        raise ValueError(
            "Q requires positive finite all-valid mean loss and mean entropy."
        )
    # Sequential division avoids introducing overflow via mean(L)*mean(H).
    sums = (
        packed[:, 0] / means[:, 0:1]
        + packed[:, 1] / means[:, 1:2]
        + packed[:, 2] / means[:, 0:1] / means[:, 1:2]
    )
    if not torch.isfinite(sums).all():
        raise ValueError("Q scores must remain finite after normalization.")
    values, counts, mean_values = sums.cpu(), packed[:, 3].cpu(), means.cpu()
    by_domain = {
        domain: {
            token: (float(values[di, ci]), int(counts[di, ci]))
            for ci, token in enumerate(candidates)
            if token in domain_candidates[domain] and counts[di, ci] > 0
        }
        for di, domain in enumerate(domains)
    }
    return QStatistics(
        by_domain=by_domain,
        valid_token_counts={
            domain: int(counts[di, -2]) for di, domain in enumerate(domains)
        },
        valid_score_sums={
            domain: float(values[di, -2]) for di, domain in enumerate(domains)
        },
        normalization={
            domain: {
                "q_all_valid_mean_abs_loss": float(mean_values[di, 0]),
                "q_all_valid_mean_student_entropy": float(mean_values[di, 1]),
                "q_all_valid_mean_score": float(values[di, -2] / counts[di, -2]),
            }
            for di, domain in enumerate(domains)
        },
    )
