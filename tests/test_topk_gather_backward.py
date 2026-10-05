"""Exactness and backward-graph tests for the renormalized Student Top-K gather."""

from __future__ import annotations

import pytest
import torch

from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_SPARSE,
    topk_log_probs_from_logits,
)


def _chunked_reference(
    logits: torch.Tensor, gather_ids: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """The previous per-chunk gather, kept as the exactness oracle."""
    vocab_size = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab_size)
    flat_ids = gather_ids.reshape(-1, int(gather_ids.shape[-1]))
    chunks = [
        flat_logits[start : start + chunk_size].gather(
            dim=-1, index=flat_ids[start : start + chunk_size]
        )
        for start in range(0, int(flat_logits.shape[0]), chunk_size)
    ]
    return torch.cat(chunks, dim=0).reshape(
        *logits.shape[:-1], int(gather_ids.shape[-1])
    )


def _unique_support(
    generator: torch.Generator, rows: int, vocab_size: int, k: int
) -> torch.Tensor:
    return torch.stack(
        [torch.randperm(vocab_size, generator=generator)[:k] for _ in range(rows)]
    )


def _autograd_node_names(output: torch.Tensor) -> list[str]:
    names, pending, seen = [], [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(node.name())
        pending.extend(parent for parent, _ in node.next_functions)
    return names


@pytest.mark.parametrize(
    ("rows", "chunk_size"), [(1, 16), (37, 16), (64, 5), (130, 1024)]
)
def test_unnormalized_gather_matches_chunked_values_and_gradients(
    rows: int, chunk_size: int
) -> None:
    generator = torch.Generator().manual_seed(rows)
    vocab_size, k = 257, 8
    hidden = torch.randn(2, rows, 12, generator=generator)
    weight = torch.randn(12, vocab_size, generator=generator) * 0.1
    ids = _unique_support(generator, 2 * rows, vocab_size, k).reshape(2, rows, k)
    upstream = torch.randn(2, rows, k, generator=generator)

    results = []
    for use_reference in (True, False):
        leaf = weight.clone().requires_grad_(True)
        # The Student lm_head emits BF16 logits under autocast.
        logits = (hidden @ leaf).to(torch.bfloat16)
        logits.retain_grad()
        if use_reference:
            gathered = _chunked_reference(logits, ids, chunk_size)
        else:
            _, _, gathered = topk_log_probs_from_logits(
                logits,
                gather_topk_ids=ids,
                normalize_gathered=False,
                chunk_size=chunk_size,
                logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
            )
        (gathered.float() * upstream).sum().backward()
        results.append((gathered.detach(), logits.grad.detach(), leaf.grad.detach()))

    for expected, actual in zip(results[0], results[1], strict=True):
        assert expected.dtype == actual.dtype
        assert torch.equal(expected, actual)


def test_unnormalized_gather_backward_has_no_per_chunk_slices() -> None:
    logits = torch.randn(3, 40, 64, requires_grad=True)
    ids = torch.randint(0, 64, (3, 40, 4))

    _, _, gathered = topk_log_probs_from_logits(
        logits,
        gather_topk_ids=ids,
        normalize_gathered=False,
        chunk_size=16,
        logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
    )

    names = _autograd_node_names(gathered)
    assert names.count("GatherBackward0") == 1
    assert "SliceBackward0" not in names
    assert "CatBackward0" not in names


def test_normalized_gather_keeps_chunked_vocabulary_pass() -> None:
    logits = torch.randn(2, 9, 33, requires_grad=True)
    ids = torch.randint(0, 33, (2, 9, 3))

    _, _, gathered = topk_log_probs_from_logits(
        logits,
        gather_topk_ids=ids,
        normalize_gathered=True,
        chunk_size=4,
        logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
    )

    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids)
    torch.testing.assert_close(gathered, expected)
    assert "CatBackward0" in _autograd_node_names(gathered)
