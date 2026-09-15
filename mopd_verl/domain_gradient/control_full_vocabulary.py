"""Dense distributed statistics for full-vocabulary Top-Loss selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FullVocabularyLossStatistics:
    """Sparse CPU view of globally reduced full-vocabulary loss statistics."""

    by_domain: dict[str, dict[int, tuple[float, int]]]
    valid_token_counts: dict[str, int]
    valid_score_sums: dict[str, float]


@torch.no_grad()
def global_full_vocabulary_loss_statistics(
    token_id_batches: Sequence[torch.Tensor],
    loss_batches: Sequence[torch.Tensor],
    mask_batches: Sequence[torch.Tensor],
    label_batches: Sequence[Sequence[str]],
    *,
    domains: Sequence[str],
    vocab_size: int,
) -> FullVocabularyLossStatistics:
    """Aggregate every valid response token ID on a fixed tokenizer-vocab axis."""

    lengths = {
        len(token_id_batches),
        len(loss_batches),
        len(mask_batches),
        len(label_batches),
    }
    if lengths != {len(token_id_batches)} or not token_id_batches:
        raise ValueError(
            "Full-vocabulary observer batches must be non-empty and aligned."
        )
    if vocab_size < 1:
        raise ValueError("Full-vocabulary selection requires a positive vocab_size.")

    normalized_domains = tuple(str(domain) for domain in domains)
    device = loss_batches[0].device
    prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[str]]] = []
    invalid = torch.zeros(2, device=device, dtype=torch.long)
    for token_ids, loss, mask, labels in zip(
        token_id_batches,
        loss_batches,
        mask_batches,
        label_batches,
        strict=True,
    ):
        if token_ids.shape != loss.shape or loss.shape != mask.shape:
            raise ValueError(
                "Full-vocabulary token IDs, configured loss, and mask must align."
            )
        if len(labels) != int(loss.shape[0]):
            raise ValueError(
                "Full-vocabulary domain labels must align with batch rows."
            )
        ids = token_ids.to(device=device, dtype=torch.long)
        scores = loss.to(device=device, dtype=torch.float64).abs()
        valid = mask.to(device=device, dtype=torch.bool)
        invalid[0].add_((valid & ((ids < 0) | (ids >= vocab_size))).sum())
        invalid[1].add_((valid & ~torch.isfinite(scores)).sum())
        prepared.append((ids, scores, valid, labels))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.SUM)
    if bool((invalid > 0).any()):
        raise ValueError(
            "Full-vocabulary selection requires finite configured loss and token "
            f"IDs in [0, {vocab_size}); global invalid counts="
            f"{{'token_id': {int(invalid[0])}, 'loss': {int(invalid[1])}}}."
        )

    packed = torch.zeros(
        (len(normalized_domains), 2, vocab_size),
        device=device,
        dtype=torch.float64,
    )
    for ids, scores, valid, labels in prepared:
        for domain_index, domain in enumerate(normalized_domains):
            domain_rows = torch.tensor(
                [str(label) == domain for label in labels],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            selected = valid & domain_rows
            selected_ids = ids[selected]
            if selected_ids.numel() == 0:
                continue
            packed[domain_index, 0].scatter_add_(
                0,
                selected_ids,
                scores[selected],
            )
            packed[domain_index, 1].scatter_add_(
                0,
                selected_ids,
                torch.ones_like(selected_ids, dtype=torch.float64),
            )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)

    packed_cpu = packed.cpu()
    by_domain: dict[str, dict[int, tuple[float, int]]] = {}
    valid_token_counts: dict[str, int] = {}
    valid_score_sums: dict[str, float] = {}
    for domain_index, domain in enumerate(normalized_domains):
        score_sums = packed_cpu[domain_index, 0]
        counts = packed_cpu[domain_index, 1]
        observed_ids = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
        by_domain[domain] = {
            int(token_id): (float(score_sums[token_id]), int(counts[token_id]))
            for token_id in observed_ids
        }
        valid_token_counts[domain] = int(counts.sum())
        valid_score_sums[domain] = float(score_sums.sum())
    return FullVocabularyLossStatistics(
        by_domain=by_domain,
        valid_token_counts=valid_token_counts,
        valid_score_sums=valid_score_sums,
    )
