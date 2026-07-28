"""Pure helpers for configurable token-level gradient weighting."""

from __future__ import annotations

import heapq
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from verl.utils.device import get_device_id

from mopd_verl.domain_gradient.token_logging import LocalTokenCandidate
from mopd_verl.domain_gradient.token_weighting_state import (
    CUMULATIVE_ABS_LOSS_SELECTION,
    PER_STEP_MEAN_ABS_LOSS_SELECTION,
    CumulativeTokenLossState,
)

MAX_PACKED_TOKEN_ID = 1_000_000


@dataclass(frozen=True)
class SharedTokenSelection:
    """All-domain intersection of per-domain high-loss token types."""

    token_ids: tuple[int, ...]
    domain_top_token_ids: tuple[tuple[str, tuple[int, ...]], ...]

    def domain_top_map(self) -> dict[str, tuple[int, ...]]:
        return dict(self.domain_top_token_ids)


def append_shared_token_selection_jsonl(
    *,
    output_dir: str,
    step: int,
    top_k: int | None,
    selection: SharedTokenSelection,
    selection_mode: str = PER_STEP_MEAN_ABS_LOSS_SELECTION,
    cumulative_state: CumulativeTokenLossState | None = None,
) -> None:
    """Persist the exact token IDs boosted by the current actor update."""

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        "step": int(step),
        "selection_mode": selection_mode,
        "ranking": (
            "cumulative_abs_configured_loss"
            if selection_mode == CUMULATIVE_ABS_LOSS_SELECTION
            else "per_occurrence_mean_abs_configured_loss"
        ),
        "top_k": top_k,
        "domain_top_token_ids": selection.domain_top_map(),
        "all_domain_shared_token_ids": selection.token_ids,
        "all_domain_shared_token_count": len(selection.token_ids),
    }
    if cumulative_state is not None:
        record["domain_cumulative_summaries"] = (
            cumulative_state.domain_summaries()
        )
    with (destination / "shared_token_weighting.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def aligned_response_token_ids(
    model_inputs: Mapping[str, Any],
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Return response-aligned token IDs without moving them off device."""

    token_ids = next(
        (
            value
            for key in ("responses", "response_ids", "input_ids")
            if (value := model_inputs.get(key)) is not None
        ),
        None,
    )
    if (
        token_ids is None
        or not hasattr(token_ids, "detach")
        or token_ids.ndim != 2
    ):
        return None
    response_length = int(reference.shape[-1])
    if tuple(token_ids.shape) == tuple(reference.shape):
        return token_ids.detach().long()
    if (
        int(token_ids.shape[0]) == int(reference.shape[0])
        and int(token_ids.shape[-1]) >= response_length
    ):
        return token_ids[:, -response_length:].detach().long()
    return None


def _local_token_loss_statistics(
    candidates: Sequence[LocalTokenCandidate],
) -> dict[int, tuple[float, int]]:
    statistics: dict[int, tuple[float, int]] = {}
    for candidate in candidates:
        if candidate.token_id is None or candidate.token_id < 0:
            continue
        loss_sum, count = statistics.get(candidate.token_id, (0.0, 0))
        statistics[candidate.token_id] = (
            loss_sum + float(candidate.loss_abs),
            count + 1,
        )
    return statistics


def _global_token_loss_statistics(
    candidates_by_domain: Mapping[
        str,
        Sequence[LocalTokenCandidate],
    ],
    domains: Sequence[str],
) -> dict[str, dict[int, tuple[float, int]]]:
    """Merge per-rank absolute-loss sums and occurrence counts."""

    local_statistics = {
        domain: _local_token_loss_statistics(
            candidates_by_domain.get(domain, ())
        )
        for domain in domains
    }
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    gathered: list[dict[str, dict[int, tuple[float, int]]] | None]
    if distributed:
        local_max_token_id = max(
            (
                token_id
                for statistics in local_statistics.values()
                for token_id in statistics
            ),
            default=-1,
        )
        max_token_id_tensor = torch.tensor(
            local_max_token_id,
            device=get_device_id(),
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            max_token_id_tensor,
            op=torch.distributed.ReduceOp.MAX,
        )
        max_token_id = int(max_token_id_tensor.item())
        if max_token_id <= MAX_PACKED_TOKEN_ID:
            if max_token_id < 0:
                return {domain: {} for domain in domains}
            packed = torch.zeros(
                (len(domains), 2, max_token_id + 1),
                device=get_device_id(),
                dtype=torch.float64,
            )
            for domain_index, domain in enumerate(domains):
                for token_id, (loss_abs_sum, count) in (
                    local_statistics[domain].items()
                ):
                    packed[domain_index, 0, token_id] = loss_abs_sum
                    packed[domain_index, 1, token_id] = count
            torch.distributed.all_reduce(
                packed,
                op=torch.distributed.ReduceOp.SUM,
            )
            packed_cpu = packed.cpu()
            return {
                domain: {
                    int(token_id): (
                        float(packed_cpu[domain_index, 0, token_id]),
                        int(packed_cpu[domain_index, 1, token_id]),
                    )
                    for token_id in torch.nonzero(
                        packed_cpu[domain_index, 1] > 0,
                        as_tuple=False,
                    ).flatten().tolist()
                }
                for domain_index, domain in enumerate(domains)
            }

        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, local_statistics)
    else:
        gathered = [local_statistics]

    merged: dict[str, dict[int, tuple[float, int]]] = {
        domain: {} for domain in domains
    }
    for rank_statistics in gathered:
        for domain in domains:
            for token_id, (loss_sum, count) in (
                rank_statistics or {}
            ).get(domain, {}).items():
                prior_sum, prior_count = merged[domain].get(
                    token_id,
                    (0.0, 0),
                )
                merged[domain][token_id] = (
                    prior_sum + float(loss_sum),
                    prior_count + int(count),
                )
    return merged


def _selection_from_statistics(
    statistics: Mapping[str, Mapping[int, tuple[float, int]]],
    domains: Sequence[str],
    *,
    top_k: int | None,
    use_mean: bool,
) -> SharedTokenSelection:
    domain_top: list[tuple[str, tuple[int, ...]]] = []
    for domain in domains:
        domain_statistics = statistics.get(domain, {})
        scored_token_ids = (
            (
                -(
                    domain_statistics[token_id][0]
                    / max(domain_statistics[token_id][1], 1)
                    if use_mean
                    else domain_statistics[token_id][0]
                ),
                token_id,
            )
            for token_id in domain_statistics
        )
        ranked = (
            sorted(scored_token_ids)
            if top_k is None
            else heapq.nsmallest(
                top_k,
                scored_token_ids,
            )
        )
        selected = [token_id for _, token_id in ranked]
        domain_top.append((domain, tuple(selected)))

    shared = (
        set.intersection(*(set(token_ids) for _, token_ids in domain_top))
        if domain_top
        else set()
    )
    return SharedTokenSelection(
        token_ids=tuple(sorted(shared)),
        domain_top_token_ids=tuple(domain_top),
    )


def select_all_domain_shared_tokens(
    candidates_by_domain: Mapping[
        str,
        Sequence[LocalTokenCandidate],
    ],
    domains: Sequence[str],
    *,
    top_k: int | None,
) -> SharedTokenSelection:
    """Rank token types by mean absolute loss and intersect domain Top-Ks."""

    return _selection_from_statistics(
        _global_token_loss_statistics(candidates_by_domain, domains),
        domains,
        top_k=top_k,
        use_mean=True,
    )


def update_cumulative_shared_token_selection(
    candidates_by_domain: Mapping[
        str,
        Sequence[LocalTokenCandidate],
    ],
    domains: Sequence[str],
    *,
    top_k: int | None,
    step: int,
    state: CumulativeTokenLossState,
) -> tuple[SharedTokenSelection, CumulativeTokenLossState]:
    """Update cumulative absolute-loss mass once and select domain Top-Ks."""

    normalized_domains = tuple(str(domain) for domain in domains)
    if state.domains != normalized_domains:
        raise ValueError(
            "Cumulative token-loss state domains do not match configuration."
        )
    if (
        state.last_updated_step is not None
        and step < state.last_updated_step
    ):
        raise ValueError(
            "Cumulative token-loss state cannot move backward in global step."
        )

    if state.last_updated_step != step:
        merged = state.statistics_map()
        current = _global_token_loss_statistics(
            candidates_by_domain,
            normalized_domains,
        )
        for domain in normalized_domains:
            for token_id, (loss_abs_sum, count) in current[domain].items():
                prior_sum, prior_count = merged[domain].get(
                    token_id,
                    (0.0, 0),
                )
                merged[domain][token_id] = (
                    prior_sum + float(loss_abs_sum),
                    prior_count + int(count),
                )
        state = CumulativeTokenLossState(
            domains=normalized_domains,
            statistics=tuple(
                (
                    domain,
                    tuple(
                        (
                            token_id,
                            float(loss_abs_sum),
                            int(count),
                        )
                        for token_id, (loss_abs_sum, count) in sorted(
                            merged[domain].items()
                        )
                    ),
                )
                for domain in normalized_domains
            ),
            last_updated_step=int(step),
            selection_mode=state.selection_mode,
        )

    selection = _selection_from_statistics(
        state.statistics_map(),
        normalized_domains,
        top_k=top_k,
        use_mean=False,
    )
    return selection, state


def token_gradient_weights(
    token_ids: torch.Tensor,
    *,
    control_token_ids: Sequence[int] | torch.Tensor,
    control_token_weight: float,
    shared_token_ids: Sequence[int] | torch.Tensor,
    shared_token_weight: float,
) -> torch.Tensor:
    """Build multiplicative per-token backward weights."""

    weights = torch.ones_like(token_ids, dtype=torch.float32)
    for selected_ids, factor in (
        (control_token_ids, control_token_weight),
        (shared_token_ids, shared_token_weight),
    ):
        if factor == 1.0:
            continue
        if isinstance(selected_ids, torch.Tensor):
            if selected_ids.numel() == 0:
                continue
            selected = selected_ids.to(
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
        else:
            if not selected_ids:
                continue
            selected = torch.tensor(
                tuple(selected_ids),
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
        weights = weights * torch.where(
            torch.isin(token_ids, selected),
            float(factor),
            1.0,
        )
    return weights
