"""Position eligibility shared by versioned Next-Step scoring and weighting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.structure_positions import (
    validated_control_position_mask,
)


def versioned_position_eligibility(
    batch: Any,
    config: Any,
    valid: torch.Tensor,
    labels: Sequence[str],
    *,
    active_domains: Sequence[str] | None = None,
    include_fixed_structure: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eligible occurrences for Control and Structure separately.

    Dynamic Code uses both masks only when its position gate is enabled.
    Fixed Structure and Math retain their existing position rules.
    """
    if valid.ndim != 2 or len(labels) != valid.shape[0]:
        raise ValueError("Token V4/V5 position eligibility requires aligned labels.")
    modes = config.versioned_cs_selection_mode_map()
    unsupported = set(labels) - set(modes)
    if unsupported:
        raise ValueError("Token V4/V5 received an unsupported domain label.")

    active = set(modes) if active_domains is None else set(active_domains)
    code_gated = getattr(config, "code_cs_position_gate_enabled", None) is True
    needs_control = any(
        domain in active and (
            domain == "math"
            or modes[domain] == "position_fixed"
            or (domain == "code" and code_gated)
        )
        for domain in labels
    )
    needs_structure = any(
        domain in active and (
            (modes[domain] == "position_fixed" and include_fixed_structure)
            or (domain == "code" and code_gated)
        )
        for domain in labels
    )
    control_positions = (
        validated_control_position_mask(batch, config, valid)
        if needs_control else valid
    )
    structure_positions = valid
    if needs_structure:
        positions = batch.batch.get("mopd_structure_position_mask")
        if positions is None:
            raise ValueError(
                "Token V4/V5 requires mopd_structure_position_mask."
            )
        structure_positions = positions.detach().bool().to(valid.device)
        if (
            structure_positions.shape != valid.shape
            or (structure_positions & ~valid).any()
        ):
            raise ValueError(
                "Structure position mask must align with valid response positions."
            )

    control_eligible = valid.clone()
    structure_eligible = valid.clone()
    for row, domain in enumerate(labels):
        if domain not in active:
            continue
        if (
            domain == "math"
            or modes[domain] == "position_fixed"
            or (domain == "code" and code_gated)
        ):
            control_eligible[row] &= control_positions[row]
        if (
            modes[domain] == "position_fixed" and include_fixed_structure
        ) or (domain == "code" and code_gated):
            structure_eligible[row] &= structure_positions[row]
    return control_eligible, structure_eligible
