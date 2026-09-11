"""Composition of token occurrences actually amplified in a completed step."""

from collections import Counter
from collections.abc import Mapping, Sequence

import torch

from mopd_verl.domain_gradient.frozen_taxonomy import (
    CONTROL_TOKEN_IDS,
    STRUCTURE_TOKEN_IDS,
)


def amplified_token_source_metrics(
    token_id_batches: Sequence[torch.Tensor],
    valid_mask_batches: Sequence[torch.Tensor],
    gradient_mask_batches: Sequence[torch.Tensor],
    label_batches: Sequence[Sequence[str]],
    *,
    domains: Sequence[str],
    domain_weights: Mapping[str, float],
    sequence_parallel_size: int = 1,
) -> dict[str, float]:
    """Merge rank histograms before computing occurrence and unique-ID shares.

    A token is amplified when its production multiplier exceeds its domain
    multiplier. Only valid response loss positions count. Global unique IDs
    are deduplicated across both ranks and domains.
    """
    if sequence_parallel_size < 1:
        raise ValueError("Sequence parallel size must be positive.")
    local: dict[str, Counter[int]] = {domain: Counter() for domain in domains}
    for ids, valid, weights, labels in zip(
        token_id_batches, valid_mask_batches, gradient_mask_batches,
        label_batches, strict=True,
    ):
        if ids.shape != valid.shape or ids.shape != weights.shape:
            raise ValueError("Token source metrics require aligned masks and IDs.")
        if len(labels) != ids.shape[0]:
            raise ValueError("Token source metrics require one domain per row.")
        for row, domain in enumerate(labels):
            if domain not in local:
                continue
            baseline = domain_weights.get(domain, 1.0)
            boosted = (valid[row] > 0) & (weights[row] > baseline + 1e-6)
            token_ids, counts = torch.unique(
                ids[row][boosted].detach(), return_counts=True
            )
            local[domain].update(dict(zip(
                token_ids.cpu().tolist(), counts.cpu().tolist(), strict=True
            )))

    gathered = [local]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, local)
    merged = {domain: Counter() for domain in domains}
    for rank_counts in gathered:
        for domain in domains:
            merged[domain].update(rank_counts[domain])
    global_counts = Counter()
    for counts in merged.values():
        global_counts.update(counts)
    merged["global"] = global_counts

    metrics: dict[str, float] = {}
    for domain, counts in merged.items():
        occurrence_counts = Counter()
        type_counts = Counter()
        for token_id, count in counts.items():
            kind = (
                "control" if token_id in CONTROL_TOKEN_IDS
                else "structure" if token_id in STRUCTURE_TOKEN_IDS
                else "other"
            )
            occurrence_counts[kind] += count
            type_counts[kind] += 1
        prefix = f"{domain}/token_weight/amplified_source"
        for unit, by_kind in (
            ("occurrence", occurrence_counts), ("unique_token", type_counts),
        ):
            total = sum(by_kind.values())
            # Every SP rank sees the same full response. Unique IDs already
            # deduplicate these replicas; occurrence sums must remove them.
            replicas = sequence_parallel_size if unit == "occurrence" else 1
            metrics[f"{prefix}_{unit}_count"] = float(total) / replicas
            for kind in ("control", "structure", "other"):
                metrics[f"{prefix}_{kind}_{unit}_count"] = float(by_kind[kind]) / replicas
                metrics[f"{prefix}_{kind}_{unit}_fraction"] = (
                    by_kind[kind] / total if total else 0.0
                )
    return metrics
