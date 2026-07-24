"""Vocab-vector logging for loss-ranked token-gradient selections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from mopd_verl.domain_gradient.token_selection import RankedToken


@dataclass(frozen=True)
class LocalTokenCandidate:
    """One locally owned token considered by loss-ranked selection."""

    micro_batch_index: int
    sample_index: int
    token_index: int
    token_id: int | None
    configured_loss: float
    loss_abs: float


def response_token_ids(
    model_inputs: Mapping[str, Any],
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Return response-aligned token IDs on CPU when present."""

    token_ids = None
    for key in ("responses", "response_ids", "input_ids"):
        value = model_inputs.get(key)
        if value is not None:
            token_ids = value
            break
    if (
        token_ids is None
        or not hasattr(token_ids, "detach")
        or token_ids.ndim != 2
    ):
        return None
    response_length = int(reference.shape[-1])
    if tuple(token_ids.shape) == tuple(reference.shape):
        return token_ids.detach().long().cpu()
    if (
        int(token_ids.shape[0]) == int(reference.shape[0])
        and int(token_ids.shape[-1]) >= response_length
    ):
        return token_ids[:, -response_length:].detach().long().cpu()
    return None


def _distributed_context() -> tuple[bool, int]:
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not distributed:
        return False, 0
    return True, torch.distributed.get_rank()


def _collective_device(distributed: bool) -> torch.device:
    if not distributed:
        return torch.device("cpu")
    backend = str(torch.distributed.get_backend()).lower()
    if "nccl" in backend:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _resolve_vocab_size(
    *,
    configured_vocab_size: int | None,
    distributed: bool,
    collective_device: torch.device,
    candidates_by_domain: Mapping[str, Sequence[LocalTokenCandidate]],
) -> int:
    if configured_vocab_size is not None and configured_vocab_size > 0:
        return int(configured_vocab_size)
    local_max = max(
        (
            candidate.token_id
            for candidates in candidates_by_domain.values()
            for candidate in candidates
            if candidate.token_id is not None and candidate.token_id >= 0
        ),
        default=-1,
    )
    global_max = torch.tensor(
        [local_max],
        dtype=torch.int64,
        device=collective_device,
    )
    if distributed:
        torch.distributed.all_reduce(
            global_max,
            op=torch.distributed.ReduceOp.MAX,
        )
    return int(global_max.item()) + 1


def _local_vocab_vectors(
    *,
    owner_rank: int,
    vocab_size: int,
    local_candidates: Sequence[LocalTokenCandidate],
    selected_tokens: Sequence[RankedToken],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    token_id_values: list[int] = []
    configured_loss_values: list[float] = []
    loss_abs_values: list[float] = []
    dropped_count = 0
    for token in selected_tokens:
        if token.owner_rank != owner_rank:
            continue
        candidate = local_candidates[token.owner_index]
        if (
            candidate.token_id is None
            or candidate.token_id < 0
            or candidate.token_id >= vocab_size
        ):
            dropped_count += 1
            continue
        token_id_values.append(candidate.token_id)
        configured_loss_values.append(candidate.configured_loss)
        loss_abs_values.append(candidate.loss_abs)
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    loss_sum = torch.zeros(vocab_size, dtype=torch.float64)
    loss_abs_sum = torch.zeros(vocab_size, dtype=torch.float64)
    if not token_id_values:
        return counts, loss_sum, loss_abs_sum, dropped_count
    token_ids = torch.tensor(
        token_id_values,
        dtype=torch.int64,
    )
    configured_losses = torch.tensor(
        configured_loss_values,
        dtype=torch.float64,
    )
    loss_abs_tensor = torch.tensor(
        loss_abs_values,
        dtype=torch.float64,
    )
    counts = torch.bincount(token_ids, minlength=vocab_size)
    loss_sum.index_add_(0, token_ids, configured_losses)
    loss_abs_sum.index_add_(0, token_ids, loss_abs_tensor)
    return counts, loss_sum, loss_abs_sum, dropped_count


def _global_vocab_vectors(
    *,
    distributed: bool,
    collective_device: torch.device,
    counts: torch.Tensor,
    loss_sum: torch.Tensor,
    loss_abs_sum: torch.Tensor,
    dropped_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not distributed:
        return counts, loss_sum, loss_abs_sum, dropped_count
    global_counts = counts.to(collective_device)
    global_loss_sums = torch.stack((loss_sum, loss_abs_sum)).to(
        collective_device
    )
    global_dropped = torch.tensor(
        [dropped_count],
        dtype=torch.int64,
        device=collective_device,
    )
    torch.distributed.all_reduce(global_counts)
    torch.distributed.all_reduce(global_loss_sums)
    torch.distributed.all_reduce(global_dropped)
    return (
        global_counts.cpu(),
        global_loss_sums[0].cpu(),
        global_loss_sums[1].cpu(),
        int(global_dropped.item()),
    )


def _vocab_vector_row(
    *,
    step: int,
    domain: str,
    selection_name: str,
    counts: torch.Tensor,
    loss_sum: torch.Tensor,
    loss_abs_sum: torch.Tensor,
    dropped_count: int,
) -> dict[str, Any]:
    nonzero_token_ids = torch.nonzero(
        counts > 0,
        as_tuple=False,
    ).flatten()
    observed_token_count = int(counts.sum().item())
    return {
        "step": step,
        "domain": domain,
        "selection": selection_name,
        "vector_value": "cumulative_occurrence_count",
        "vocab_size": int(counts.numel()),
        "selected_token_count": observed_token_count + dropped_count,
        "observed_token_count": observed_token_count,
        "dropped_token_count": dropped_count,
        "nonzero_token_id_count": int(nonzero_token_ids.numel()),
        "nonzero_token_ids": nonzero_token_ids.tolist(),
        "token_count_vector_vocab": counts.tolist(),
        "configured_token_loss_sum_vector_vocab": (
            loss_sum.float().tolist()
        ),
        "configured_token_loss_abs_sum_vector_vocab": (
            loss_abs_sum.float().tolist()
        ),
    }


def append_token_vocab_vectors_jsonl(
    *,
    output_dir: str | Path,
    step: int,
    configured_vocab_size: int | None,
    candidates_by_domain: Mapping[str, Sequence[LocalTokenCandidate]],
    selections_by_domain: Mapping[
        str,
        Mapping[str, Sequence[RankedToken]],
    ],
) -> None:
    """Append one deduplicated cumulative vocab-vector row per selection."""

    distributed, rank = _distributed_context()
    collective_device = _collective_device(distributed)
    vocab_size = _resolve_vocab_size(
        configured_vocab_size=configured_vocab_size,
        distributed=distributed,
        collective_device=collective_device,
        candidates_by_domain=candidates_by_domain,
    )
    rows: list[dict[str, Any]] = []
    for domain, named_selections in selections_by_domain.items():
        local_candidates = candidates_by_domain.get(domain, ())
        for selection_name, selected_tokens in named_selections.items():
            local_vectors = _local_vocab_vectors(
                owner_rank=rank,
                vocab_size=vocab_size,
                local_candidates=local_candidates,
                selected_tokens=selected_tokens,
            )
            global_vectors = _global_vocab_vectors(
                distributed=distributed,
                collective_device=collective_device,
                counts=local_vectors[0],
                loss_sum=local_vectors[1],
                loss_abs_sum=local_vectors[2],
                dropped_count=local_vectors[3],
            )
            if rank == 0:
                rows.append(
                    _vocab_vector_row(
                        step=step,
                        domain=domain,
                        selection_name=selection_name,
                        counts=global_vectors[0],
                        loss_sum=global_vectors[1],
                        loss_abs_sum=global_vectors[2],
                        dropped_count=global_vectors[3],
                    )
                )
    if rank != 0 or not rows:
        return
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "token_gradient_vocab_vectors.jsonl"
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
