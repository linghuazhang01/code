"""Position-wise KL routing using the active frozen global token taxonomy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mopd_verl.domain_gradient.frozen_taxonomy import (
    CONTROL_TOKEN_IDS,
    STRUCTURE_TOKEN_IDS,
)
from mopd_verl.topk_distill import (
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    cfg_get,
    resolved_topk_distill_mode,
    topk_distill_include_tail,
    topk_distill_loss_matrix,
    topk_distill_temperature,
    uses_topk_distill_loss,
)


def validate_token_category_loss(config: Any) -> None:
    """Reject unsupported combinations instead of silently ignoring routing."""
    routes = cfg_get(config, "topk_distill_loss_by_domain", {})
    if not isinstance(routes, Mapping):
        raise ValueError("topk_distill_loss_by_domain must be a mapping.")
    if not routes:
        return
    for domain, categories in routes.items():
        if domain not in {"default", "math", "code", "science"}:
            raise ValueError(f"Unknown loss routing domain: {domain!r}")
        if not isinstance(categories, Mapping):
            raise ValueError(f"Loss routing for {domain} must be a mapping.")
        for category, direction in categories.items():
            if category not in {"control", "structure", "other"}:
                raise ValueError(f"Unknown loss routing category: {category!r}")
            if direction not in {"forward", "reverse"}:
                raise ValueError(f"Unknown KL direction: {direction!r}")
    if not uses_topk_distill_loss(config):
        raise ValueError("Token category loss routing requires the Top-K KL builder.")
    if int(cfg_get(config, "topk_distill_k", 0)) != 32:
        raise ValueError("Token category loss routing requires topk_distill_k=32.")
    if cfg_get(config, "topk_distill_support_source", "teacher") != "teacher":
        raise ValueError("Token category loss routing requires teacher support.")
    if topk_distill_include_tail(config) or resolved_topk_distill_mode(config) not in {
        TOPK_RENORMALIZED_FORWARD_KL, TOPK_RENORMALIZED_REVERSE_KL,
    }:
        raise ValueError("Token category loss routing requires renormalized KL without tail.")
    if cfg_get(config, "teacher_prefix_enabled", False):
        raise ValueError("Token category loss routing does not support teacher prefixes.")


def routed_topk_loss(
    *,
    student: torch.Tensor,
    teacher: torch.Tensor,
    config: Any,
    token_ids: torch.Tensor | None,
    domains: Sequence[str],
) -> torch.Tensor:
    """Replace KL direction per position, before selector/IS/token weighting.

    Domain overrides inherit category defaults; omitted categories inherit the
    existing global KL direction. Types are global, not candidate-pool labels.
    """
    validate_token_category_loss(config)
    mode = resolved_topk_distill_mode(config)
    routes = cfg_get(config, "topk_distill_loss_by_domain", {})
    kwargs = dict(
        student_topk_log_probs=student,
        teacher_topk_log_probs=teacher.detach(),
        include_tail=topk_distill_include_tail(config),
        temperature=topk_distill_temperature(config),
    )
    if not routes:
        return topk_distill_loss_matrix(mode=mode, **kwargs)
    if student.shape != teacher.shape or student.ndim != 3 or student.shape[-1] != 32:
        raise ValueError("Routed KL requires matching [batch, response, 32] tensors.")
    if token_ids is None or token_ids.shape != student.shape[:-1]:
        raise ValueError("Routed KL requires response-aligned token IDs.")
    if len(domains) != student.shape[0] or any(
        domain not in {"math", "code", "science"} for domain in domains
    ):
        raise ValueError("Routed KL requires a known domain for every response.")
    token_ids = token_ids.to(device=student.device)
    control = torch.isin(token_ids, token_ids.new_tensor(sorted(CONTROL_TOKEN_IDS)))
    structure = torch.isin(token_ids, token_ids.new_tensor(sorted(STRUCTURE_TOKEN_IDS)))
    masks = {"control": control, "structure": structure, "other": ~(control | structure)}
    forward = torch.full_like(control, mode == TOPK_RENORMALIZED_FORWARD_KL)
    for row, domain in enumerate(domains):
        directions = {**routes.get("default", {}), **routes.get(domain, {})}
        for category, direction in directions.items():
            forward[row] = torch.where(masks[category][row], direction == "forward", forward[row])
    forward_loss = topk_distill_loss_matrix(mode=TOPK_RENORMALIZED_FORWARD_KL, **kwargs)
    reverse_loss = topk_distill_loss_matrix(mode=TOPK_RENORMALIZED_REVERSE_KL, **kwargs)
    return torch.where(forward, forward_loss, reverse_loss)
