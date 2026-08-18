"""Torch aggregation and logging for online Control-token selection."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mopd_verl.audit_io import step_jsonl_dir
from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionOutcome,
    OnlineControlSelectionState,
)


def global_candidate_loss_statistics(
    token_id_batches: Sequence[torch.Tensor],
    loss_batches: Sequence[torch.Tensor],
    mask_batches: Sequence[torch.Tensor],
    label_batches: Sequence[Sequence[str]],
    *,
    domains: Sequence[str],
    candidate_token_ids: Sequence[int] = (),
    domain_candidate_token_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, dict[int, tuple[float, int]]]:
    """Aggregate absolute configured loss and count in one dense all-reduce."""

    lengths = {
        len(token_id_batches),
        len(loss_batches),
        len(mask_batches),
        len(label_batches),
    }
    if lengths != {len(token_id_batches)} or not token_id_batches:
        raise ValueError(
            "Online Control observer batches must be non-empty and aligned."
        )
    normalized_domains = tuple(str(domain) for domain in domains)
    if domain_candidate_token_ids is not None:
        domain_candidates = {
            str(domain): tuple(sorted({int(item) for item in token_ids}))
            for domain, token_ids in domain_candidate_token_ids.items()
        }
    else:
        candidates = tuple(sorted({int(item) for item in candidate_token_ids}))
        domain_candidates = {domain: candidates for domain in normalized_domains}
    if set(domain_candidates) != set(normalized_domains):
        raise ValueError("Online Control domain candidates must exactly match domains.")
    candidates = tuple(
        sorted(
            {
                token_id
                for token_ids in domain_candidates.values()
                for token_id in token_ids
            }
        )
    )
    if not candidates:
        raise ValueError("Online Control candidates must be non-empty.")
    device = loss_batches[0].device
    packed = torch.zeros(
        (len(normalized_domains), 2, len(candidates)),
        device=device,
        dtype=torch.float64,
    )
    candidate_tensor = torch.tensor(
        candidates,
        device=device,
        dtype=torch.long,
    )
    allowed = torch.zeros(
        (len(normalized_domains), len(candidates)),
        device=device,
        dtype=torch.bool,
    )
    candidate_indices = {token_id: index for index, token_id in enumerate(candidates)}
    for domain_index, domain in enumerate(normalized_domains):
        allowed[
            domain_index,
            [candidate_indices[item] for item in domain_candidates[domain]],
        ] = True
    for token_ids, losses, mask, labels in zip(
        token_id_batches,
        loss_batches,
        mask_batches,
        label_batches,
        strict=True,
    ):
        if token_ids.shape != losses.shape or losses.shape != mask.shape:
            raise ValueError(
                "Online Control token IDs, configured loss, and mask must align."
            )
        if len(labels) != int(losses.shape[0]):
            raise ValueError("Online Control domain labels must align with batch rows.")
        valid = mask.to(device=device, dtype=torch.bool)
        loss_values = losses.to(device=device, dtype=torch.float64)
        if not torch.isfinite(loss_values[valid]).all():
            raise ValueError("Online Control configured loss must be finite.")
        ids = token_ids.to(device=device, dtype=torch.long)
        for domain_index, domain in enumerate(normalized_domains):
            rows = torch.tensor(
                [str(label) == domain for label in labels],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(-1)
            selected_mask = valid & rows
            selected_ids = ids[selected_mask]
            if selected_ids.numel() == 0:
                continue
            positions = torch.searchsorted(candidate_tensor, selected_ids)
            in_range = positions < candidate_tensor.numel()
            safe_positions = positions.clamp(max=candidate_tensor.numel() - 1)
            matched = (
                in_range
                & candidate_tensor[safe_positions].eq(selected_ids)
                & allowed[domain_index, safe_positions]
            )
            matched_positions = positions[matched]
            packed[domain_index, 0].scatter_add_(
                0,
                matched_positions,
                loss_values[selected_mask][matched].abs(),
            )
            packed[domain_index, 1].scatter_add_(
                0,
                matched_positions,
                torch.ones_like(matched_positions, dtype=torch.float64),
            )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            packed,
            op=torch.distributed.ReduceOp.SUM,
        )
    packed_cpu = packed.cpu()
    return {
        domain: {
            token_id: (
                float(packed_cpu[domain_index, 0, candidate_index]),
                int(packed_cpu[domain_index, 1, candidate_index]),
            )
            for candidate_index, token_id in enumerate(candidates)
            if token_id in set(domain_candidates[domain])
            if packed_cpu[domain_index, 1, candidate_index] > 0
        }
        for domain_index, domain in enumerate(normalized_domains)
    }


def append_online_control_selection_jsonl(
    *,
    output_dir: str,
    state: OnlineControlSelectionState,
    outcome: OnlineControlSelectionOutcome,
    control_weight: float,
) -> None:
    """Persist online selector observations and exact next-step membership."""

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    destination = step_jsonl_dir(output_dir, outcome.observed_step, create=True)
    record: dict[str, Any] = {
        "observed_step": outcome.observed_step,
        "applies_from_step": outcome.observed_step + 1,
        "audit_triggered": outcome.audit_triggered,
        "duplicate_step": outcome.duplicate_step,
        "history_reset": outcome.history_reset,
        "window_fill_steps": outcome.window_fill_steps,
        "window_steps": state.window_steps,
        "audit_interval_steps": state.audit_interval_steps,
        "min_mean_occurrences_per_step": (state.min_mean_occurrences_per_step),
        "top_k": state.top_k,
        "candidate_token_count": len(state.candidate_token_ids),
        "candidate_union_count": len(state.candidate_token_ids),
        "domain_candidate_token_counts": {
            domain: len(token_ids)
            for domain, token_ids in state.domain_candidate_token_ids
        },
        "control_weight": float(control_weight),
        "next_active_token_ids": state.active_map(),
        "domains": {
            result.domain: {
                "eligible_token_count": result.eligible_token_count,
                "selected_tokens": [
                    {
                        "token_id": item.token_id,
                        "occurrence_count": item.occurrence_count,
                        "mean_occurrences_per_step": (item.mean_occurrences_per_step),
                        "mean_abs_loss": item.mean_abs_loss,
                    }
                    for item in result.selected_tokens
                ],
            }
            for result in outcome.domain_results
        },
    }
    with (destination / "online_control_selection.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
