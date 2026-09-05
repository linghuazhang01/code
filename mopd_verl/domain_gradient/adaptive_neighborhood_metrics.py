"""Step-level metrics for per-token adaptive neighborhoods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mopd_verl.domain_gradient.adaptive_neighborhood import (
    PerTokenAdaptiveNeighborhoodResult,
)
from mopd_verl.tensorboard_tags import safe_name


@dataclass(frozen=True)
class AdaptiveNeighborhoodMetricComponents:
    """Packed device-side additive statistics for one actor micro-batch."""

    values: torch.Tensor
    threshold: float
    domains: tuple[str, ...]
    strict_threshold: bool = False


def adaptive_neighborhood_metric_components(
    result: PerTokenAdaptiveNeighborhoodResult,
    valid_loss_mask: torch.Tensor,
    *,
    threshold: float,
    labels: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    strict_threshold: bool = False,
) -> AdaptiveNeighborhoodMetricComponents:
    """Build additive metrics without synchronizing the accelerator."""

    valid = valid_loss_mask.to(
        device=result.center_mask.device,
        dtype=torch.bool,
    )
    if labels is None:
        domain_names: tuple[str, ...] = ()
        row_masks: tuple[torch.Tensor, ...] = ()
    else:
        if len(labels) != int(valid.shape[0]):
            raise ValueError("Adaptive metric labels must align with responses.")
        domain_names = tuple(
            str(domain) for domain in (domains or tuple(dict.fromkeys(labels)))
        )
        label_values = tuple(str(label) for label in labels)
        row_masks = tuple(
            torch.tensor(
                [label == domain for label in label_values],
                device=valid.device,
                dtype=torch.bool,
            )
            for domain in domain_names
        )

    def packed_values(row_mask: torch.Tensor | None) -> torch.Tensor:
        token_rows = (
            torch.ones_like(valid, dtype=torch.bool)
            if row_mask is None
            else row_mask.unsqueeze(-1).expand_as(valid)
        )
        response_rows = (
            torch.ones(
                valid.shape[0],
                device=valid.device,
                dtype=torch.bool,
            )
            if row_mask is None
            else row_mask
        )
        selected_mask = result.selected_neighbor_mask & token_rows
        valid_denominators = result.valid_denominator_mask & token_rows
        selected_scores = torch.where(
            selected_mask,
            result.relative_scores,
            torch.zeros_like(result.relative_scores),
        )
        return torch.stack(
            (
                (valid & token_rows).sum(),
                (result.center_mask & valid & token_rows).sum(),
                (result.candidate_neighbor_mask & token_rows).sum(),
                (result.eligible_neighbor_mask & token_rows).sum(),
                selected_mask.sum(),
                selected_scores.sum(),
                (result.center_denominators.lt(0) & valid_denominators).sum(),
                valid_denominators.sum(),
                (torch.isfinite(result.far_baselines) & response_rows).sum(),
                response_rows.sum(),
            )
        ).to(dtype=torch.float64)

    values = torch.stack(
        (packed_values(None), *(packed_values(mask) for mask in row_masks)),
        dim=0,
    )
    return AdaptiveNeighborhoodMetricComponents(
        values=values.detach(),
        threshold=float(threshold),
        domains=domain_names,
        strict_threshold=bool(strict_threshold),
    )


def _float_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def aggregate_adaptive_neighborhood_metrics(
    components: Sequence[AdaptiveNeighborhoodMetricComponents],
    *,
    reduce_distributed: bool,
) -> dict[str, float]:
    """Produce exact step totals with one packed device-to-host transfer."""

    if not components:
        return {}
    threshold = components[0].threshold
    if any(component.threshold != threshold for component in components[1:]):
        raise ValueError("Adaptive metric thresholds must match within one step.")
    domains = components[0].domains
    if any(component.domains != domains for component in components[1:]):
        raise ValueError("Adaptive metric domains must match within one step.")
    strict_threshold = components[0].strict_threshold
    if any(
        component.strict_threshold != strict_threshold
        for component in components[1:]
    ):
        raise ValueError(
            "Adaptive metric threshold operators must match within one step."
        )
    total = components[0].values.clone()
    for component in components[1:]:
        total.add_(component.values.to(device=total.device))
    if (
        reduce_distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(total)
    reduced_rows = total.detach().cpu().tolist()
    (
        valid_count,
        centers,
        candidates,
        eligible,
        selected,
        kernel_mass,
        negative_denominators,
        valid_denominator_count,
        far_valid_responses,
        response_count,
    ) = (float(value) for value in reduced_rows[0])
    filtered = eligible - selected
    total_weighted = centers + selected
    metrics = {
        "actor/adaptive_relative_loss_threshold": threshold,
        "actor/adaptive_threshold_is_strict": float(strict_threshold),
        "actor/adaptive_valid_token_count": valid_count,
        "actor/adaptive_fixed_d0_token_count": centers,
        "actor/adaptive_candidate_neighbor_token_count": candidates,
        "actor/adaptive_eligible_neighbor_token_count": eligible,
        "actor/adaptive_filtered_neighbor_token_count": filtered,
        "actor/adaptive_extra_weighted_token_count": selected,
        "actor/adaptive_total_weighted_token_count": total_weighted,
        "actor/adaptive_threshold_eligible_token_count": eligible,
        "actor/adaptive_threshold_pass_token_count": selected,
        "actor/adaptive_threshold_pass_token_fraction": _float_ratio(
            selected,
            eligible,
        ),
        "actor/adaptive_threshold_pass_valid_token_fraction": _float_ratio(
            selected,
            valid_count,
        ),
        "actor/adaptive_filtered_neighbor_fraction": _float_ratio(
            filtered,
            eligible,
        ),
        "actor/adaptive_extra_weighted_token_fraction": _float_ratio(
            selected,
            valid_count,
        ),
        "actor/adaptive_total_weighted_token_fraction": _float_ratio(
            total_weighted,
            valid_count,
        ),
        "actor/adaptive_extra_to_d0_ratio": _float_ratio(selected, centers),
        "actor/adaptive_extra_kernel_mass_to_d0": _float_ratio(
            kernel_mass,
            centers,
        ),
        "actor/adaptive_negative_denominator_fraction": _float_ratio(
            negative_denominators,
            valid_denominator_count,
        ),
        "actor/adaptive_far_baseline_valid_response_fraction": _float_ratio(
            far_valid_responses,
            response_count,
        ),
    }
    for domain, row in zip(domains, reduced_rows[1:], strict=True):
        (
            domain_valid_count,
            domain_centers,
            _domain_candidates,
            domain_eligible,
            domain_selected,
            _domain_kernel_mass,
            _domain_negative_denominators,
            _domain_valid_denominator_count,
            _domain_far_valid_responses,
            _domain_response_count,
        ) = (float(value) for value in row)
        prefix = f"actor/adaptive_domain/{safe_name(domain)}"
        metrics[f"{prefix}/valid_token_count"] = domain_valid_count
        metrics[f"{prefix}/fixed_d0_token_count"] = domain_centers
        metrics[f"{prefix}/threshold_eligible_token_count"] = domain_eligible
        metrics[f"{prefix}/threshold_pass_token_count"] = domain_selected
        metrics[f"{prefix}/threshold_pass_token_fraction"] = _float_ratio(
            domain_selected,
            domain_eligible,
        )
        metrics[f"{prefix}/threshold_pass_valid_token_fraction"] = (
            _float_ratio(domain_selected, domain_valid_count)
        )
        metrics[f"{prefix}/total_weighted_token_count"] = (
            domain_centers + domain_selected
        )
    return metrics


def adaptive_neighborhood_metrics(
    result: PerTokenAdaptiveNeighborhoodResult,
    valid_loss_mask: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    """Summarize one local batch; production aggregates once per step."""

    component = adaptive_neighborhood_metric_components(
        result,
        valid_loss_mask,
        threshold=threshold,
    )
    return aggregate_adaptive_neighborhood_metrics(
        (component,),
        reduce_distributed=False,
    )
