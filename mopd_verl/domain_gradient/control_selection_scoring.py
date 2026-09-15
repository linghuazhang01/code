"""Ranking signals for online Control-token selection."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


TOP_LOSS_SELECTION_MODE = "top_loss"
TOP_LOGP_DIFF_SELECTION_MODE = "top_logp_diff"
TOP_SPEED_SELECTION_MODE = "top_speed"
TOP_Q_LOSS_ENTROPY_SELECTION_MODE = "top_q_loss_entropy"
TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE = "top_loss_teacher_confidence"
TOP_KL_STUDENT_ENTROPY_SELECTION_MODE = "top_kl_student_entropy"
TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE = (
    "top_teacher_confidence_student_entropy"
)
PAIRED_SIGNAL_SELECTION_MODES = frozenset(
    {
        TOP_KL_STUDENT_ENTROPY_SELECTION_MODE,
        TOP_TEACHER_CONFIDENCE_STUDENT_ENTROPY_SELECTION_MODE,
    }
)
ONLINE_CONTROL_SELECTION_MODES = frozenset(
    {
        TOP_LOSS_SELECTION_MODE,
        TOP_LOGP_DIFF_SELECTION_MODE,
        TOP_SPEED_SELECTION_MODE,
        TOP_Q_LOSS_ENTROPY_SELECTION_MODE,
        TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE,
        *PAIRED_SIGNAL_SELECTION_MODES,
    }
)

TOP_K_BUDGET_MODE = "top_k"
TOP_P_BUDGET_MODE = "top_p"
ONLINE_CONTROL_BUDGET_MODES = frozenset({TOP_K_BUDGET_MODE, TOP_P_BUDGET_MODE})

FIXED_ONLINE_WEIGHT_MODE = "fixed"
PAIRED_ONLINE_WEIGHT_MODE = "paired"
LOSS_RATIO_ONLINE_WEIGHT_MODE = "loss_ratio"
ONLINE_CONTROL_WEIGHT_MODES = frozenset(
    {
        FIXED_ONLINE_WEIGHT_MODE,
        PAIRED_ONLINE_WEIGHT_MODE,
        LOSS_RATIO_ONLINE_WEIGHT_MODE,
    }
)

LOSS_RATIO_EPSILON = 1e-12


def validate_q_selection_contract(
    selection_mode: str, weight_mode: str, interval: int, window: int
) -> None:
    """Freeze the initial Q implementation to step-global selection, fixed weights."""
    if selection_mode == TOP_Q_LOSS_ENTROPY_SELECTION_MODE and (
        weight_mode != FIXED_ONLINE_WEIGHT_MODE or interval != 1 or window != 1
    ):
        raise ValueError("Q selection requires fixed weighting and interval=window=1.")


def validate_loss_teacher_confidence_selection_contract(
    selection_mode: str,
    weight_mode: str,
    interval: int,
    window: int,
) -> None:
    """Require one-step, fixed-weight selection for robust batch normalization."""

    if selection_mode == TOP_LOSS_TEACHER_CONFIDENCE_SELECTION_MODE and (
        weight_mode != FIXED_ONLINE_WEIGHT_MODE or interval != 1 or window != 1
    ):
        raise ValueError(
            "Loss + teacher-confidence selection requires fixed weighting and "
            "interval=window=1."
        )


@dataclass(frozen=True)
class OptimizationSpeed:
    """Occurrence-weighted loss slope over distinct optimizer steps."""

    value: float
    observed_step_count: int


@dataclass(frozen=True)
class LossRatioWeight:
    """One selected-versus-other occurrence-level loss-ratio estimate."""

    selected_occurrence_count: int
    other_occurrence_count: int
    selected_mean_abs_loss: float | None
    other_mean_abs_loss: float | None
    raw_ratio: float | None
    clipped_weight: float
    scaled_weight: float


def validate_loss_ratio_alpha(
    alpha: float,
    *,
    weight_mode: str = LOSS_RATIO_ONLINE_WEIGHT_MODE,
) -> None:
    """Reject invalid or inactive loss-ratio strength overrides."""

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("control_token_loss_ratio_alpha must be finite and positive.")
    if alpha != 1.0 and weight_mode != LOSS_RATIO_ONLINE_WEIGHT_MODE:
        raise ValueError(
            "Non-default control_token_loss_ratio_alpha requires loss_ratio mode."
        )


def selected_to_other_loss_ratio_weight(
    *,
    selected_loss_abs_sum: float,
    selected_occurrence_count: int,
    valid_loss_abs_sum: float,
    valid_occurrence_count: int,
    max_weight: float,
    epsilon: float = LOSS_RATIO_EPSILON,
    alpha: float = 1.0,
) -> LossRatioWeight:
    """Scale the legacy bounded ratio without clipping the scaled weight."""

    validate_loss_ratio_alpha(alpha)
    values = (
        selected_loss_abs_sum,
        valid_loss_abs_sum,
        max_weight,
        epsilon,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Loss-ratio inputs must be finite.")
    if selected_loss_abs_sum < 0.0 or valid_loss_abs_sum < 0.0:
        raise ValueError("Loss-ratio loss sums must be non-negative.")
    if selected_occurrence_count < 0 or valid_occurrence_count < 0:
        raise ValueError("Loss-ratio occurrence counts must be non-negative.")
    if selected_occurrence_count > valid_occurrence_count:
        raise ValueError(
            "Selected occurrence count cannot exceed valid occurrence count."
        )
    if max_weight < 1.0:
        raise ValueError("Loss-ratio maximum weight must be at least 1.0.")
    if epsilon <= 0.0:
        raise ValueError("Loss-ratio epsilon must be positive.")

    other_occurrence_count = valid_occurrence_count - selected_occurrence_count
    if selected_occurrence_count == 0 or other_occurrence_count == 0:
        return LossRatioWeight(
            selected_occurrence_count=selected_occurrence_count,
            other_occurrence_count=other_occurrence_count,
            selected_mean_abs_loss=None,
            other_mean_abs_loss=None,
            raw_ratio=None,
            clipped_weight=1.0,
            scaled_weight=1.0,
        )

    subtraction_tolerance = 1e-9 * max(
        1.0,
        selected_loss_abs_sum,
        valid_loss_abs_sum,
    )
    other_loss_abs_sum = valid_loss_abs_sum - selected_loss_abs_sum
    if other_loss_abs_sum < -subtraction_tolerance:
        raise ValueError("Selected loss mass cannot exceed all-valid loss mass.")
    other_loss_abs_sum = max(other_loss_abs_sum, 0.0)
    selected_mean = selected_loss_abs_sum / selected_occurrence_count
    other_mean = other_loss_abs_sum / other_occurrence_count
    raw_ratio = selected_mean / max(other_mean, epsilon)
    if not math.isfinite(raw_ratio):
        raw_ratio = max_weight
    clipped_weight = min(max(raw_ratio, 1.0), max_weight)
    scaled_weight = alpha * clipped_weight
    validate_scaled_loss_ratio_weight(scaled_weight)
    return LossRatioWeight(
        selected_occurrence_count=selected_occurrence_count,
        other_occurrence_count=other_occurrence_count,
        selected_mean_abs_loss=float(selected_mean),
        other_mean_abs_loss=float(other_mean),
        raw_ratio=float(raw_ratio),
        clipped_weight=float(clipped_weight),
        scaled_weight=float(scaled_weight),
    )


def validate_scaled_loss_ratio_weight(weight: float) -> None:
    """Require positive finite weights in the runtime float32 buffer."""

    # Torch checks the original scalar range before conversion, unlike struct.
    float32_max = float.fromhex("0x1.fffffep+127")
    if weight > float32_max:
        raise ValueError("Scaled loss-ratio weight must be finite in float32.")
    try:
        represented = struct.unpack("f", struct.pack("f", weight))[0]
    except OverflowError as exc:
        raise ValueError("Scaled loss-ratio weight must be finite in float32.") from exc
    if not math.isfinite(represented) or represented <= 0.0:
        raise ValueError(
            "Scaled loss-ratio weight must be finite and positive in float32."
        )


def normalize_selection_mode(value: object) -> str:
    """Return one supported online Control-token selection mode."""

    mode = str(value).strip().lower()
    if mode not in ONLINE_CONTROL_SELECTION_MODES:
        allowed = ", ".join(sorted(ONLINE_CONTROL_SELECTION_MODES))
        raise ValueError("Online Control selection mode must be one of: " f"{allowed}.")
    return mode


def normalize_online_budget_mode(value: object) -> str:
    """Return one supported online Control-token selection budget mode."""

    mode = str(value).strip().lower()
    if mode not in ONLINE_CONTROL_BUDGET_MODES:
        allowed = ", ".join(sorted(ONLINE_CONTROL_BUDGET_MODES))
        raise ValueError("Online Control budget mode must be one of: " f"{allowed}.")
    return mode


def normalize_online_weight_mode(value: object) -> str:
    """Return one supported online Control-token weighting mode."""

    mode = str(value).strip().lower()
    if mode not in ONLINE_CONTROL_WEIGHT_MODES:
        allowed = ", ".join(sorted(ONLINE_CONTROL_WEIGHT_MODES))
        raise ValueError("Online Control weight mode must be one of: " f"{allowed}.")
    return mode


def _normalize_online_mode_by_domain(
    domains: Sequence[str],
    value: object,
    *,
    fallback: object,
    normalizer: Callable[[object], str],
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    """Normalize an optional complete per-domain mode map.

    An empty map is intentionally treated as the scalar fallback.  Once a
    map is populated, however, it must name every configured domain so a
    mixed-domain run cannot silently inherit an unintended mode.
    """

    normalized_domains = tuple(dict.fromkeys(str(domain) for domain in domains))
    normalized_fallback = normalizer(fallback)
    if value is None:
        raw_items: tuple[tuple[object, object], ...] = ()
    elif isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            raw_items = tuple(dict(value).items())
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} must be a mapping.") from exc
    else:
        raise TypeError(f"{field_name} must be a mapping.")
    if not raw_items:
        if not normalized_domains:
            return ()
        return tuple((domain, normalized_fallback) for domain in normalized_domains)
    raw_map = {str(domain): mode for domain, mode in raw_items}
    if len(raw_map) != len(raw_items) or set(raw_map) != set(normalized_domains):
        raise ValueError(f"{field_name} keys must exactly match domains.")
    return tuple((domain, normalizer(raw_map[domain])) for domain in normalized_domains)


def normalize_online_selection_mode_by_domain(
    domains: Sequence[str],
    value: object,
    fallback: object = TOP_LOSS_SELECTION_MODE,
) -> tuple[tuple[str, str], ...]:
    """Return effective online selection modes for every configured domain."""

    return _normalize_online_mode_by_domain(
        domains,
        value,
        fallback=fallback,
        normalizer=normalize_selection_mode,
        field_name="control_token_online_selection_mode_by_domain",
    )


def normalize_online_weight_mode_by_domain(
    domains: Sequence[str],
    value: object,
    fallback: object = FIXED_ONLINE_WEIGHT_MODE,
) -> tuple[tuple[str, str], ...]:
    """Return effective online weight modes for every configured domain."""

    return _normalize_online_mode_by_domain(
        domains,
        value,
        fallback=fallback,
        normalizer=normalize_online_weight_mode,
        field_name="control_token_online_weight_mode_by_domain",
    )


def validate_online_control_mode_contracts(
    domains: Sequence[str],
    *,
    selection_mode: object,
    weight_mode: object,
    selection_mode_by_domain: object = None,
    weight_mode_by_domain: object = None,
    interval: int,
    window: int,
    loss_ratio_alpha: float = 1.0,
    control_token_weight: float | None = None,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Normalize and validate all per-domain online mode contracts."""

    normalized_selection = normalize_selection_mode(selection_mode)
    normalized_weight = normalize_online_weight_mode(weight_mode)
    effective_selection = normalize_online_selection_mode_by_domain(
        domains,
        selection_mode_by_domain,
        normalized_selection,
    )
    effective_weight = normalize_online_weight_mode_by_domain(
        domains,
        weight_mode_by_domain,
        normalized_weight,
    )
    if not effective_selection:
        validate_q_selection_contract(
            normalized_selection,
            normalized_weight,
            interval,
            window,
        )
        validate_loss_teacher_confidence_selection_contract(
            normalized_selection,
            normalized_weight,
            interval,
            window,
        )
        if (
            normalized_weight == PAIRED_ONLINE_WEIGHT_MODE
            and normalized_selection not in PAIRED_SIGNAL_SELECTION_MODES
        ):
            raise ValueError(
                "Online paired weight mode requires a paired-signal " "selection mode."
            )
        if (
            normalized_weight == LOSS_RATIO_ONLINE_WEIGHT_MODE
            and normalized_selection != TOP_LOSS_SELECTION_MODE
        ):
            raise ValueError(
                "Online loss-ratio weight mode requires top-loss selection mode."
            )
        if normalized_selection == TOP_SPEED_SELECTION_MODE and window < 2:
            raise ValueError(
                "Online top-speed selection requires window_steps to be at least 2."
            )
        validate_loss_ratio_alpha(loss_ratio_alpha, weight_mode=normalized_weight)
        if (
            normalized_weight == LOSS_RATIO_ONLINE_WEIGHT_MODE
            and control_token_weight is not None
            and (not math.isfinite(control_token_weight) or control_token_weight < 1.0)
        ):
            raise ValueError(
                "Loss-ratio weighting uses control_token_weight as its maximum "
                "and requires it to be finite and at least 1."
            )
        return effective_selection, effective_weight
    weight_map = dict(effective_weight)
    for domain, mode in effective_selection:
        domain_weight = weight_map[domain]
        validate_q_selection_contract(mode, domain_weight, interval, window)
        validate_loss_teacher_confidence_selection_contract(
            mode,
            domain_weight,
            interval,
            window,
        )
        if (
            domain_weight == PAIRED_ONLINE_WEIGHT_MODE
            and mode not in PAIRED_SIGNAL_SELECTION_MODES
        ):
            raise ValueError(
                "Online paired weight mode requires a paired-signal " "selection mode."
            )
        if (
            domain_weight == LOSS_RATIO_ONLINE_WEIGHT_MODE
            and mode != TOP_LOSS_SELECTION_MODE
        ):
            raise ValueError(
                "Online loss-ratio weight mode requires top-loss selection mode."
            )
        if mode == TOP_SPEED_SELECTION_MODE and window < 2:
            raise ValueError(
                "Online top-speed selection requires window_steps to be at least 2."
            )
    any_loss_ratio = any(
        mode == LOSS_RATIO_ONLINE_WEIGHT_MODE for mode in weight_map.values()
    )
    validate_loss_ratio_alpha(
        loss_ratio_alpha,
        weight_mode=(
            LOSS_RATIO_ONLINE_WEIGHT_MODE
            if any_loss_ratio
            else FIXED_ONLINE_WEIGHT_MODE
        ),
    )
    if any_loss_ratio and control_token_weight is not None:
        if not math.isfinite(control_token_weight) or control_token_weight < 1.0:
            raise ValueError(
                "Loss-ratio weighting uses control_token_weight as its maximum "
                "and requires it to be finite and at least 1."
            )
    return effective_selection, effective_weight


def _masked_sequence_max_normalize(
    values: "torch.Tensor",
    mask: "torch.Tensor",
    *,
    name: str,
    absolute: bool,
) -> "torch.Tensor":
    """Normalize each response by its valid-position maximum."""

    import torch

    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError(
            f"{name} must be a rank-2 matrix matching response_mask, got "
            f"{tuple(values.shape)} and {tuple(mask.shape)}."
        )
    valid = mask.detach().to(device=values.device, dtype=torch.bool)
    detached = values.detach().float()
    if not torch.isfinite(detached[valid]).all():
        raise ValueError(f"{name} must be finite on valid response positions.")
    non_negative = detached.abs() if absolute else detached
    if not absolute and bool((non_negative[valid] < 0.0).any()):
        raise ValueError(f"{name} must be non-negative on valid positions.")
    masked = torch.where(valid, non_negative, torch.zeros_like(non_negative))
    maximum = masked.amax(dim=-1, keepdim=True)
    normalized = masked / maximum.clamp(min=1e-12)
    return torch.where(valid, normalized, torch.zeros_like(normalized))


def paired_selection_bonus(
    *,
    selection_mode: str,
    student_entropy: "torch.Tensor",
    response_mask: "torch.Tensor",
    configured_loss: "torch.Tensor | None" = None,
    teacher_entropy: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Return detached ``A + B + A*B`` scores for paired selectors."""

    import torch

    mode = normalize_selection_mode(selection_mode)
    if mode not in PAIRED_SIGNAL_SELECTION_MODES:
        raise ValueError(
            "Paired selection bonus requires a paired-signal selection mode."
        )
    valid = response_mask.detach().to(
        device=student_entropy.device,
        dtype=torch.bool,
    )
    student = _masked_sequence_max_normalize(
        student_entropy,
        valid,
        name="student_entropy",
        absolute=False,
    )
    if mode == TOP_KL_STUDENT_ENTROPY_SELECTION_MODE:
        if configured_loss is None:
            raise ValueError("KL + entropy selection requires configured_loss.")
        primary = _masked_sequence_max_normalize(
            configured_loss,
            valid,
            name="configured_loss",
            absolute=True,
        )
    else:
        if teacher_entropy is None:
            raise ValueError(
                "Teacher-confidence + entropy selection requires teacher_entropy."
            )
        normalized_teacher_entropy = _masked_sequence_max_normalize(
            teacher_entropy,
            valid,
            name="teacher_entropy",
            absolute=False,
        )
        primary = torch.where(
            valid,
            1.0 - normalized_teacher_entropy,
            torch.zeros_like(normalized_teacher_entropy),
        )
    return ((primary + student + primary * student) * valid).detach()


def occurrence_weighted_optimization_speed(
    observations: Sequence[tuple[int, float, int]],
) -> OptimizationSpeed | None:
    """Return negative weighted loss slope, or ``None`` when undefined."""

    valid = tuple(
        (int(step), float(mean_abs_loss), int(count))
        for step, mean_abs_loss, count in observations
        if count > 0 and math.isfinite(mean_abs_loss)
    )
    if len(valid) < 2 or len({step for step, _, _ in valid}) < 2:
        return None

    total_weight = float(sum(count for _, _, count in valid))
    mean_step = sum(step * count for step, _, count in valid) / total_weight
    mean_loss = (
        sum(mean_abs_loss * count for _, mean_abs_loss, count in valid) / total_weight
    )
    time_variance = sum(count * (step - mean_step) ** 2 for step, _, count in valid)
    if time_variance <= 1e-12:
        return None
    covariance = sum(
        count * (step - mean_step) * (mean_abs_loss - mean_loss)
        for step, mean_abs_loss, count in valid
    )
    speed = -covariance / time_variance
    if not math.isfinite(speed):
        return None
    return OptimizationSpeed(
        value=float(speed),
        observed_step_count=len(valid),
    )
