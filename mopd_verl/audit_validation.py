"""Validation-anchor scheduling helpers for MOPD full-gradient audit."""

from __future__ import annotations

from typing import Any


def should_update_validation_anchor(logger: Any, step: int) -> bool:
    if not logger.enabled or not logger.validation_anchor_enabled or not logger.full_gradient_enabled:
        return False
    if logger._validation_anchor_step is None:
        return True
    if step == logger._validation_anchor_step:
        return True
    if logger.validation_anchor_refresh_steps <= 0:
        return False
    return step - logger._validation_anchor_step >= logger.validation_anchor_refresh_steps


def log_validation_anchor_batch(logger: Any, batch: Any, step: int) -> dict[str, float]:
    del batch
    if not should_update_validation_anchor(logger, step):
        return {}
    logger._validation_anchor_step = step
    return {}
