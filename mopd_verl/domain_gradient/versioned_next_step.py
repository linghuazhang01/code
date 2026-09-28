"""Lagged V4/V5 selection from production losses, without a scoring forward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.control_top_loss_runtime import (
    GlobalCandidateLossStatistics,
    global_candidate_loss_statistics_with_valid_counts,
)
from mopd_verl.domain_gradient.versioned_position_eligibility import (
    versioned_position_eligibility,
)


def versioned_selector_candidate_map(config: Any) -> dict[str, tuple[int, ...]]:
    """Return C-only or shared C+S pools for each versioned domain."""
    controls = config.effective_domain_candidate_map()
    structures = config.effective_domain_structure_map()
    modes = config.versioned_cs_selection_mode_map()
    return {
        domain: tuple(sorted(set(controls[domain]) | (
            set(structures[domain]) if modes[domain] != "position_fixed" else set()
        )))
        for domain in config.domains
    }


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


def _raise_synchronized_source_error(error: Exception | None) -> None:
    """Fail every rank before candidate-statistics collectives if input is bad."""
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        if error is not None:
            raise error
        return
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if distributed.get_backend() == "nccl" else torch.device("cpu")
    )
    failures = torch.tensor(int(error is not None), device=device)
    distributed.all_reduce(failures, op=distributed.ReduceOp.SUM)
    if failures.item():
        detail = f": {error}" if error is not None else "."
        raise ValueError(
            "Token V4/V5 source inputs are invalid on at least one rank"
            f"{detail}"
        ) from error


@torch.no_grad()
def next_step_gradient_mask(audit: Any, batch: Any) -> torch.Tensor:
    """Apply one lagged shared selector or positioned Structure once."""
    config = audit.config
    ids, valid, labels = _inputs(batch)
    modes = config.versioned_cs_selection_mode_map()
    control_eligible, structure_eligible = versioned_position_eligibility(
        batch, config, valid, labels,
    )
    structures = config.effective_domain_structure_map()
    controls = config.effective_domain_candidate_map()
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
        control_ids = torch.tensor(
            controls[domain], device=ids.device, dtype=ids.dtype,
        )
        structure_ids = torch.tensor(
            structures[domain], device=ids.device, dtype=ids.dtype,
        )
        control = (
            domain_valid & control_eligible & torch.isin(ids, control_ids)
            & torch.isin(ids, selected)
        )
        structure_candidates = (
            domain_valid & structure_eligible & torch.isin(ids, structure_ids)
        )
        if modes[domain] == "position_fixed":
            structure = structure_candidates
        else:
            structure = structure_candidates & torch.isin(ids, selected)
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
    """Reduce production RKL for domains using Control/C+S TopLoss."""
    source_error: Exception | None = None
    if (
        raw_loss_batches is None
        or raw_mask_batches is None
        or len(raw_loss_batches) != len(micro_batches)
        or len(raw_mask_batches) != len(micro_batches)
        or not micro_batches
    ):
        source_error = ValueError(
            "Token V4/V5 next_step requires production raw RKL batches."
        )
    token_ids = []
    labels = []
    losses = []
    masks = []
    controls = config.effective_domain_candidate_map()
    structures = config.effective_domain_structure_map()
    modes = config.versioned_cs_selection_mode_map()
    candidate_map = versioned_selector_candidate_map(config)
    if source_error is None:
        for batch, raw_loss, raw_mask in zip(
            micro_batches, raw_loss_batches, raw_mask_batches, strict=True
        ):
            try:
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
                if not torch.isfinite(raw[valid]).all():
                    raise ValueError(
                        "Token V4/V5 raw RKL must be finite on valid positions."
                    )
                control_eligible, structure_eligible = versioned_position_eligibility(
                    batch, config, valid, domains,
                    active_domains=tuple(
                        domain for domain in config.domains
                        if modes[domain] != "top_teacher_confidence"
                    ),
                    include_fixed_structure=False,
                )
                # The shared top-p denominator remains all valid response tokens.
                masked_ids = ids.clone()
                for domain in config.domains:
                    rows = torch.tensor(
                        [label == domain for label in domains], device=ids.device,
                    ).unsqueeze(-1)
                    control_ids = torch.tensor(
                        controls[domain], device=ids.device, dtype=ids.dtype,
                    )
                    structure_ids = torch.tensor(
                        structures[domain], device=ids.device, dtype=ids.dtype,
                    )
                    masked_ids.masked_fill_(
                        rows & (
                            (torch.isin(ids, control_ids) & ~control_eligible)
                            | (torch.isin(ids, structure_ids) & ~structure_eligible)
                        ),
                        -1,
                    )
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                source_error = exc
                break
            token_ids.append(masked_ids)
            labels.append(domains)
            losses.append(raw)
            masks.append(mask)
    _raise_synchronized_source_error(source_error)
    return global_candidate_loss_statistics_with_valid_counts(
        token_ids, losses, masks, labels,
        domains=config.domains,
        domain_candidate_token_ids={
            domain: (
                candidate_map[domain]
                if modes[domain] != "top_teacher_confidence" else ()
            )
            for domain in config.domains
        },
        selection_mode="top_loss",
    )


def merge_versioned_candidate_statistics(
    config: Any,
    loss_statistics: GlobalCandidateLossStatistics | None,
    confidence_statistics: GlobalCandidateLossStatistics | None,
) -> GlobalCandidateLossStatistics:
    """Use exactly one ranking signal and one budget per domain."""
    modes = config.versioned_cs_selection_mode_map()
    by_domain: dict[str, dict[int, tuple[float, int]]] = {}
    valid_counts: dict[str, int] = {}
    score_sums: dict[str, float] = {}
    for domain in config.domains:
        source = (
            confidence_statistics
            if modes[domain] == "top_teacher_confidence"
            else loss_statistics
        )
        if source is None:
            raise ValueError(f"Missing {modes[domain]} statistics for {domain}.")
        by_domain[domain] = source.by_domain[domain]
        valid_counts[domain] = source.valid_token_counts[domain]
        score_sums[domain] = source.valid_score_sums[domain]
    return GlobalCandidateLossStatistics(by_domain, valid_counts, score_sums)
