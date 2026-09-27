"""Lagged V4/V5 selection from production losses, without a scoring forward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.control_top_loss_runtime import (
    GlobalCandidateLossStatistics,
    global_candidate_loss_statistics_with_valid_counts,
)
from mopd_verl.domain_gradient.structure_positions import (
    validated_control_position_mask,
)


def _inputs(batch: Any) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    from mopd_verl.domain_gradient.token_weighting import aligned_response_token_ids
    from mopd_verl.full_gradient.labels import _labels_from_mapping

    inputs = {**batch.batch, **batch.non_tensor_batch}
    valid = inputs["response_mask"].detach().bool()
    ids = aligned_response_token_ids(inputs, valid)
    if ids is None or ids.shape != valid.shape:
        raise ValueError("Token V4/V5 requires response-aligned token IDs.")
    labels = _labels_from_mapping(inputs, int(valid.shape[0]))
    return ids, valid, labels


@torch.no_grad()
def next_step_gradient_mask(audit: Any, batch: Any) -> torch.Tensor:
    """Apply previous-step Control IDs and current-position Structure once."""
    config = audit.config
    ids, valid, labels = _inputs(batch)
    control_positions = validated_control_position_mask(batch, config, valid)
    positions = batch.batch.get("mopd_structure_position_mask")
    if positions is None:
        raise ValueError(
            "Fixed Structure weighting requires mopd_structure_position_mask."
        )
    positions = positions.detach().bool().to(valid.device)
    if positions.shape != valid.shape or (positions & ~valid).any():
        raise ValueError(
            "Structure position mask must align with valid response positions."
        )
    if set(labels) - set(config.domains):
        raise ValueError("Token V4/V5 received an unsupported domain label.")
    structures = dict(config.domain_structure_token_ids)
    weights = valid.float()
    for domain in config.domains:
        rows = torch.tensor(
            [label == domain for label in labels], device=valid.device
        ).unsqueeze(-1)
        domain_valid = rows & valid
        if not domain_valid.any():
            continue
        selected = torch.tensor(
            audit._applied_online_control_token_ids.get(domain, ()),
            device=ids.device, dtype=ids.dtype,
        )
        structure_ids = torch.tensor(
            structures[domain], device=ids.device, dtype=ids.dtype,
        )
        control = domain_valid & control_positions & torch.isin(ids, selected)
        structure = domain_valid & positions & torch.isin(ids, structure_ids)
        if (control & structure).any():
            raise ValueError("Token V4/V5 Control and Structure must be disjoint.")
        weights[control] = config.control_token_weight
        weights[structure] = config.structure_token_loss_weight
        weights[domain_valid] /= weights[domain_valid].mean()
    return weights


@torch.no_grad()
def next_step_candidate_statistics(
    config: Any,
    micro_batches: Sequence[Any],
    raw_loss_batches: Sequence[torch.Tensor] | None,
    raw_mask_batches: Sequence[torch.Tensor] | None,
) -> GlobalCandidateLossStatistics:
    """Reduce detached pre-update RKL already computed by production training."""
    if (
        raw_loss_batches is None
        or raw_mask_batches is None
        or len(raw_loss_batches) != len(micro_batches)
        or len(raw_mask_batches) != len(micro_batches)
        or not micro_batches
    ):
        raise ValueError("Token V4/V5 next_step requires production raw RKL batches.")
    token_ids = []
    labels = []
    losses = []
    masks = []
    for batch, raw_loss, raw_mask in zip(
        micro_batches, raw_loss_batches, raw_mask_batches, strict=True
    ):
        ids, valid, domains = _inputs(batch)
        raw = raw_loss.detach().float()
        mask = raw_mask.detach().bool().to(valid.device)
        if (
            raw.shape != valid.shape or mask.shape != valid.shape
            or not torch.equal(mask, valid)
        ):
            raise ValueError(
                "Token V4/V5 raw RKL validity must equal the response mask."
            )
        positions = validated_control_position_mask(batch, config, valid)
        # -1 cannot match a frozen vocabulary ID. Mask only candidate identity,
        # preserving every valid token in the domain's top-p denominator.
        token_ids.append(ids.masked_fill(~positions, -1))
        labels.append(domains)
        losses.append(raw)
        masks.append(mask)
    return global_candidate_loss_statistics_with_valid_counts(
        token_ids, losses, masks, labels,
        domains=config.domains,
        domain_candidate_token_ids=config.effective_domain_candidate_map(),
        selection_mode="top_loss",
    )
