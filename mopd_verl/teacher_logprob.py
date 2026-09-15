"""Fused teacher statistics over a shared full-vocabulary normalization."""

from __future__ import annotations

from collections.abc import Callable

import torch

from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_FULL_VOCAB,
    TOPK_LOGPROB_MODES,
)


def response_prediction_mask(
    flat_indices: torch.Tensor,
    *,
    sequence_length: int,
    response_length: int,
) -> torch.Tensor:
    """Select unpadded logits that predict tokens in the response window."""

    positions = flat_indices.remainder(sequence_length)
    return (positions >= sequence_length - response_length - 1) & (
        positions < sequence_length - 1
    )


def fused_logprob_entropy_topk_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    topk: int | None = None,
    gather_topk_ids: torch.Tensor | None = None,
    normalize_gathered: bool = True,
    chunk_size: int,
    logprob_mode: str,
    entropy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute chosen log-prob, entropy, and sparse support in one token pass.

    The teacher-only path has no gradients. Chosen log-prob and sparse support
    share one FP32 ``logsumexp``. Entropy deliberately retains the legacy input
    dtype and formula because EOPD applies a discrete entropy threshold. The
    response-only chunking still avoids a full-sequence entropy workspace.
    """

    if logits.dim() < 2:
        raise ValueError(
            f"logits must have at least 2 dims, got shape {tuple(logits.shape)}"
        )
    prefix_shape = tuple(logits.shape[:-1])
    if tuple(labels.shape) != prefix_shape:
        raise ValueError(
            f"labels must have shape {prefix_shape}, got {tuple(labels.shape)}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    mode = str(logprob_mode).lower()
    if mode not in TOPK_LOGPROB_MODES:
        raise ValueError(
            f"logprob_mode must be one of {sorted(TOPK_LOGPROB_MODES)}, "
            f"got {logprob_mode!r}."
        )
    if topk is None and gather_topk_ids is None:
        raise ValueError("topk or gather_topk_ids is required for the fused path")

    vocab_size = int(logits.shape[-1])
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1).to(device=logits.device, dtype=torch.long)
    topk_count = min(max(1, int(topk)), vocab_size) if topk is not None else None

    flat_gather_ids = None
    gather_shape = None
    if gather_topk_ids is not None:
        gather_ids = gather_topk_ids.to(device=logits.device, dtype=torch.long)
        if tuple(gather_ids.shape[:-1]) != prefix_shape:
            raise ValueError(
                "gather_topk_ids must match logits prefix shape, "
                f"got {tuple(gather_ids.shape)} for logits {tuple(logits.shape)}."
            )
        gather_shape = tuple(gather_ids.shape[-1:])
        flat_gather_ids = gather_ids.reshape(-1, int(gather_ids.shape[-1]))

    chosen_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    topk_id_chunks: list[torch.Tensor] = []
    topk_log_prob_chunks: list[torch.Tensor] = []
    gathered_log_prob_chunks: list[torch.Tensor] = []
    for start in range(0, int(flat_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(flat_logits.shape[0]))
        logits_chunk = flat_logits[start:end].float()
        log_norm = torch.logsumexp(logits_chunk, dim=-1, keepdim=True)

        chosen_logits = logits_chunk.gather(
            dim=-1,
            index=flat_labels[start:end, None],
        ).squeeze(-1)
        chosen_chunks.append(chosen_logits - log_norm.squeeze(-1))

        legacy_logits_chunk = flat_logits[start:end]
        if entropy_fn is None:
            probabilities = torch.softmax(legacy_logits_chunk, dim=-1)
            entropy_chunks.append(
                torch.logsumexp(legacy_logits_chunk, dim=-1)
                - torch.sum(probabilities * legacy_logits_chunk, dim=-1)
            )
            del probabilities
        else:
            entropy_chunks.append(entropy_fn(legacy_logits_chunk))

        if topk_count is not None:
            top_logits, top_ids = torch.topk(logits_chunk, topk_count, dim=-1)
            topk_id_chunks.append(top_ids)
            topk_log_prob_chunks.append(top_logits - log_norm)

        if flat_gather_ids is not None:
            gathered = logits_chunk.gather(
                dim=-1,
                index=flat_gather_ids[start:end],
            )
            if normalize_gathered or mode == TOPK_LOGPROB_MODE_FULL_VOCAB:
                gathered = gathered - log_norm
            gathered_log_prob_chunks.append(gathered)

    chosen_log_probs = torch.cat(chosen_chunks, dim=0).reshape(*prefix_shape)
    entropy = torch.cat(entropy_chunks, dim=0).reshape(*prefix_shape)
    topk_ids = None
    topk_log_probs = None
    gathered_log_probs = None
    if topk_count is not None:
        topk_ids = torch.cat(topk_id_chunks, dim=0).reshape(
            *prefix_shape,
            topk_count,
        )
        topk_log_probs = torch.cat(topk_log_prob_chunks, dim=0).reshape(
            *prefix_shape,
            topk_count,
        )
    if gather_shape is not None:
        gathered_log_probs = torch.cat(gathered_log_prob_chunks, dim=0).reshape(
            *prefix_shape,
            *gather_shape,
        )
    return chosen_log_probs, entropy, topk_ids, topk_log_probs, gathered_log_probs
