"""Pairwise cosine helpers for domain-aligned audit vectors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any


def tensor_cosine(left: Any, right: Any) -> float | None:
    """Return cosine for equal-length nonzero tensors, otherwise ``None``."""

    import torch

    left_vector = left.detach().float().flatten()
    right_vector = right.detach().float().flatten()
    if left_vector.numel() != right_vector.numel():
        return None
    denominator = torch.linalg.vector_norm(left_vector) * torch.linalg.vector_norm(right_vector)
    if float(denominator.detach().cpu().item()) <= 0.0:
        return None
    return float((torch.dot(left_vector, right_vector) / denominator).detach().cpu().item())


def iter_pairwise_domain_cosines(
    vectors_by_domain: Mapping[str, Mapping[str, Any]],
    domains: Sequence[str],
    vector_specs: Sequence[tuple[str, str]],
) -> Iterator[tuple[str, str, str, float]]:
    """Yield available cosines as ``(left, right, metric, value)`` tuples."""

    active_domains = [domain for domain in domains if domain in vectors_by_domain]
    for left_index, left_domain in enumerate(active_domains):
        for right_domain in active_domains[left_index + 1 :]:
            left_vectors = vectors_by_domain[left_domain]
            right_vectors = vectors_by_domain[right_domain]
            for metric_name, vector_key in vector_specs:
                if vector_key not in left_vectors or vector_key not in right_vectors:
                    continue
                cosine = tensor_cosine(left_vectors[vector_key], right_vectors[vector_key])
                if cosine is not None:
                    yield left_domain, right_domain, metric_name, cosine
