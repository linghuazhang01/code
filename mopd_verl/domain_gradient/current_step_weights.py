"""Apply disjoint head/tail masks with legacy microbatch/domain mean-one scope."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.current_step_selection import CurrentStepSelection
from mopd_verl.domain_gradient.occurrence_config import (
    uses_ungated_current_step_code_structure,
)
from mopd_verl.domain_gradient.structure_positions import (
    validated_control_position_mask,
)
from mopd_verl.domain_gradient.token_weighting import aligned_response_token_ids


def _mark(
    selected: frozenset[tuple[int, ...]],
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    unit: str,
    rank: int,
    batch_index: int,
) -> torch.Tensor:
    if unit == "token_id":
        tokens = torch.tensor([key[0] for key in selected], dtype=ids.dtype)
        return torch.isin(ids, tokens) & mask
    marked = torch.zeros_like(mask)
    for owner, batch, row, column in selected:
        if owner == rank and batch == batch_index:
            marked[row, column] = True
    return marked & mask


def install_current_step_masks(
    audit: Any,
    templates: Sequence[tuple[Any, torch.Tensor, list[str]]],
    selection: CurrentStepSelection,
    *,
    rank: int,
) -> dict[str, float]:
    """Keep zero-total-weight domain blocks at zero, with explicit diagnostics."""
    config = audit.config
    unit = config.control_token_online_selection_unit
    minima = {domain: float("inf") for domain in config.domains}
    maxima = {domain: 0.0 for domain in config.domains}
    zeros = {domain: 0 for domain in config.domains}
    structure_candidates = {domain: 0 for domain in config.domains}
    structure_applied = {domain: 0 for domain in config.domains}
    structure_ids = dict(getattr(config, "domain_structure_token_ids", ()))
    structure_enabled = bool(
        getattr(config, "structure_token_loss_weighting_enabled", False)
    )
    structure_positions_ungated = (
        getattr(config, "structure_token_position_profile", "none") == "none"
    )
    structure_only_domains = {
        domain
        for domain, mode in config.versioned_cs_selection_mode_map().items()
        if mode == "structure_only"
    }
    ungated_structure_only_domains = (
        {"code"} if uses_ungated_current_step_code_structure(config) else set()
    )
    for index, (batch, mask, labels) in enumerate(templates):
        inputs = {**batch.batch, **batch.non_tensor_batch}
        ids = aligned_response_token_ids(inputs, inputs["response_mask"]).cpu()
        control_mask = validated_control_position_mask(batch, config, mask)
        positioned_structure = None
        if structure_enabled:
            if structure_positions_ungated:
                positioned_structure = mask
            elif "mopd_structure_position_mask" not in batch.batch:
                raise ValueError(
                    "Fixed Structure weighting requires "
                    "mopd_structure_position_mask."
                )
            else:
                positioned_structure = batch.batch["mopd_structure_position_mask"]
        if positioned_structure is not None:
            positioned_structure = positioned_structure.detach().bool().cpu()
            if positioned_structure.shape != mask.shape:
                raise ValueError(
                    "Structure position mask must align with response mask."
                )
            if (positioned_structure & ~mask).any():
                raise ValueError(
                    "Structure positions must be valid response positions."
                )
        weights = mask.float()
        for domain in config.domains:
            rows = torch.tensor([label == domain for label in labels]).unsqueeze(1)
            valid = rows & mask
            if not valid.any():
                continue
            selection_position_mask = (
                mask
                if domain in ungated_structure_only_domains
                else (
                    positioned_structure
                    if domain in structure_only_domains
                    else control_mask
                )
            )
            head = _mark(
                selection.head[domain],
                ids,
                valid & selection_position_mask,
                unit=unit,
                rank=rank,
                batch_index=index,
            )
            tail = _mark(
                selection.tail[domain],
                ids,
                valid & selection_position_mask,
                unit=unit,
                rank=rank,
                batch_index=index,
            )
            if (head & tail).any():
                raise RuntimeError("Current-step head and tail overlap")
            structure = torch.zeros_like(valid)
            if positioned_structure is not None:
                domain_structure_ids = torch.tensor(
                    structure_ids[domain], dtype=ids.dtype
                )
                structure_occurrences = valid & torch.isin(ids, domain_structure_ids)
                structure_position_mask = (
                    mask
                    if domain in ungated_structure_only_domains
                    else positioned_structure
                )
                structure = structure_occurrences & structure_position_mask
                structure_candidates[domain] += int(structure_occurrences.sum())
                structure_applied[domain] += int(structure.sum())
                if domain in structure_only_domains:
                    if ((head | tail) & ~structure).any():
                        raise RuntimeError(
                            "Current-step Structure-only selection escaped its "
                            "eligible positions"
                        )
                elif ((head | tail) & structure).any():
                    raise RuntimeError("Current-step Control and Structure overlap")
                if domain not in structure_only_domains:
                    weights[structure] = config.structure_token_loss_weight
            weights[head] = config.control_token_weight
            weights[tail] = config.control_token_tail_weight
            denominator = weights[valid].mean()
            minima[domain] = min(minima[domain], float(denominator))
            if denominator == 0:
                zeros[domain] += 1
                weights[valid] = 0
            else:
                weights[valid] /= denominator
            maxima[domain] = max(maxima[domain], float(weights[valid].max()))
        audit._occurrence_masks[id(batch)] = (
            batch,
            weights.to(batch.batch["response_mask"].device),
        )
    device = templates[0][0].batch["response_mask"].device if templates else "cpu"
    extrema = torch.tensor(
        [[minima[d], -maxima[d]] for d in config.domains],
        device=device,
    )
    zero_counts = torch.tensor([zeros[d] for d in config.domains], device=device)
    structure_counts = torch.tensor(
        [[structure_candidates[d], structure_applied[d]] for d in config.domains],
        dtype=torch.float64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(zero_counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            structure_counts, op=torch.distributed.ReduceOp.SUM
        )
    metrics = {}
    for domain, (denominator, negative_max), zero in zip(
        config.domains,
        extrema.tolist(),
        zero_counts.tolist(),
        strict=True,
    ):
        prefix = f"{domain}/current_step/"
        metrics[prefix + "normalizer_min"] = (
            denominator if denominator != float("inf") else 0.0
        )
        metrics[prefix + "effective_weight_max"] = -negative_max
        metrics[prefix + "zero_weight_microbatch_count"] = float(zero)
    for domain, (candidate_count, applied_count) in zip(
        config.domains,
        structure_counts.tolist(),
        strict=True,
    ):
        prefix = f"{domain}/current_step/structure/"
        metrics[prefix + "candidate_occurrence_count"] = candidate_count
        metrics[prefix + "applied_occurrence_count"] = applied_count
        metrics[prefix + "position_acceptance_fraction"] = (
            applied_count / candidate_count if candidate_count else 0.0
        )
    return metrics
