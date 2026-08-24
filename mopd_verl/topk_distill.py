"""Top-k distillation helpers shared by verl workers and tests."""

from __future__ import annotations

import math
from typing import Any

import torch


_TOPK_LOGPROB_CHUNK_SIZE = 16

CHOSEN_TOKEN_REVERSE_KL = "chosen_token_reverse_kl"
CHOSEN_TOKEN_POLICY_GRADIENT = "chosen_token_policy_gradient"
DISTILL_LOSS_BUILDER_AUTO = "auto"
DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL = "chosen_token_reverse_kl"
DISTILL_LOSS_BUILDER_EOPD = "eopd"
DISTILL_LOSS_BUILDER_EXOPD = "exopd"
DISTILL_LOSS_BUILDER_GOPD = "gopd"
DISTILL_LOSS_BUILDER_POLICY_GRADIENT = "policy_gradient"
DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB = "tip_full_vocab"
DISTILL_LOSS_BUILDER_TOPK_KL = "topk_kl"
DISTILL_LOSS_BUILDERS = {
    DISTILL_LOSS_BUILDER_AUTO,
    DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL,
    DISTILL_LOSS_BUILDER_EOPD,
    DISTILL_LOSS_BUILDER_EXOPD,
    DISTILL_LOSS_BUILDER_GOPD,
    DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB,
    DISTILL_LOSS_BUILDER_TOPK_KL,
}
DISTILL_LOSS_BUILDER_ALIASES = {
    "entropy_aware": DISTILL_LOSS_BUILDER_EOPD,
    "entropy_aware_opd": DISTILL_LOSS_BUILDER_EOPD,
    "extrapolated_opd": DISTILL_LOSS_BUILDER_EXOPD,
    "generalized_opd": DISTILL_LOSS_BUILDER_GOPD,
    "pg": DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    "chosen_token_pg": DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    CHOSEN_TOKEN_POLICY_GRADIENT: DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    "tip": DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB,
    "tip_native": DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB,
    "topk": DISTILL_LOSS_BUILDER_TOPK_KL,
    "topk_distill": DISTILL_LOSS_BUILDER_TOPK_KL,
    "topk_distillation": DISTILL_LOSS_BUILDER_TOPK_KL,
}
TOPK_LOGPROB_MODE_SPARSE = "sparse"
TOPK_LOGPROB_MODE_FULL_VOCAB = "full_vocab"
TOPK_LOGPROB_MODES = {TOPK_LOGPROB_MODE_SPARSE, TOPK_LOGPROB_MODE_FULL_VOCAB}
TOPK_SUPPORT_SOURCE_TEACHER = "teacher"
TOPK_SUPPORT_SOURCE_STUDENT = "student"
TOPK_SUPPORT_SOURCES = {TOPK_SUPPORT_SOURCE_TEACHER, TOPK_SUPPORT_SOURCE_STUDENT}
TOPK_FORWARD_KL_WITH_TAIL = "topk_forward_kl_with_tail"
TOPK_REVERSE_KL_WITH_TAIL = "topk_reverse_kl_with_tail"
TOPK_RENORMALIZED_FORWARD_KL = "topk_renormalized_forward_kl"
TOPK_RENORMALIZED_REVERSE_KL = "topk_renormalized_reverse_kl"
NAIVE_RENORMALIZED_TOPK_KL = "naive_renormalized_topk_kl"

TOPK_RENORMALIZED_MODES = {
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TOPK_DISTILL_MODES = {
    TOPK_FORWARD_KL_WITH_TAIL,
    TOPK_REVERSE_KL_WITH_TAIL,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TOPK_REVERSE_KL_MODES = {
    TOPK_REVERSE_KL_WITH_TAIL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TEACHER_PREFIX_PREFIX_AND_SUFFIX = "prefix_and_suffix"
TEACHER_PREFIX_SUFFIX_ONLY = "suffix_only"
TEACHER_PREFIX_PREFIX_ONLY = "prefix_only"
TEACHER_PREFIX_LOSS_REGIONS = {
    TEACHER_PREFIX_PREFIX_AND_SUFFIX,
    TEACHER_PREFIX_SUFFIX_ONLY,
    TEACHER_PREFIX_PREFIX_ONLY,
}


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def distill_mode(policy_loss_config: Any) -> str:
    return str(cfg_get(policy_loss_config, "distill_mode", CHOSEN_TOKEN_REVERSE_KL))


def distill_loss_builder(policy_loss_config: Any) -> str:
    builder = str(
        cfg_get(policy_loss_config, "distill_loss_builder", DISTILL_LOSS_BUILDER_AUTO)
        or DISTILL_LOSS_BUILDER_AUTO
    ).lower()
    builder = DISTILL_LOSS_BUILDER_ALIASES.get(builder, builder)
    if builder != DISTILL_LOSS_BUILDER_AUTO:
        if builder not in DISTILL_LOSS_BUILDERS:
            raise ValueError(
                "distill_loss_builder must be one of "
                f"{sorted(DISTILL_LOSS_BUILDERS)} or aliases "
                f"{sorted(DISTILL_LOSS_BUILDER_ALIASES)}, got {builder!r}."
            )
        return builder

    mode = distill_mode(policy_loss_config)
    if mode in {CHOSEN_TOKEN_POLICY_GRADIENT, DISTILL_LOSS_BUILDER_POLICY_GRADIENT}:
        return DISTILL_LOSS_BUILDER_POLICY_GRADIENT
    if mode in TOPK_DISTILL_MODES or bool(cfg_get(policy_loss_config, "topk_distill_enabled", False)):
        return DISTILL_LOSS_BUILDER_TOPK_KL
    return DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL


def uses_topk_distill_loss(policy_loss_config: Any) -> bool:
    return distill_loss_builder(policy_loss_config) == DISTILL_LOSS_BUILDER_TOPK_KL


def uses_eopd_loss(policy_loss_config: Any) -> bool:
    return distill_loss_builder(policy_loss_config) == DISTILL_LOSS_BUILDER_EOPD


def uses_tip_full_vocab_loss(policy_loss_config: Any) -> bool:
    """Return whether the dedicated paper-native TIP path is configured."""

    return (
        distill_loss_builder(policy_loss_config)
        == DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB
    )


def uses_teacher_topk_support(policy_loss_config: Any) -> bool:
    """Return whether training needs teacher-selected top-k tensors."""

    return uses_eopd_loss(policy_loss_config) or (
        uses_topk_distill_loss(policy_loss_config)
        and topk_distill_support_source(policy_loss_config)
        == TOPK_SUPPORT_SOURCE_TEACHER
    )


def configured_distill_loss_name(policy_loss_config: Any) -> str:
    """Return the per-token distillation loss represented by audit metrics."""

    builder = distill_loss_builder(policy_loss_config)
    name = (
        resolved_topk_distill_mode(policy_loss_config)
        if builder == DISTILL_LOSS_BUILDER_TOPK_KL
        else builder
    )
    if builder == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
        name = "policy_gradient_distillation_signal"
    elif builder == DISTILL_LOSS_BUILDER_EOPD:
        name = "policy_gradient+entropy_gated_topk_forward_kl"
    elif builder == DISTILL_LOSS_BUILDER_EXOPD:
        name = "extrapolated_policy_gradient_distillation_signal"
    elif builder == DISTILL_LOSS_BUILDER_GOPD:
        name = "generalized_policy_gradient_distillation_signal"
    elif builder == DISTILL_LOSS_BUILDER_TIP_FULL_VOCAB:
        name = "full_vocab_reverse_kl+tip_soft_or_toprho"
    if (
        bool(cfg_get(policy_loss_config, "teacher_prefix_enabled", False))
        and teacher_prefix_loss_region(policy_loss_config)
        != TEACHER_PREFIX_SUFFIX_ONLY
    ):
        return f"{name}+teacher_prefix_forward_kl"
    return name


def resolved_topk_distill_mode(policy_loss_config: Any) -> str:
    mode = distill_mode(policy_loss_config)
    if mode == CHOSEN_TOKEN_REVERSE_KL and bool(cfg_get(policy_loss_config, "topk_distill_enabled", False)):
        direction = str(cfg_get(policy_loss_config, "topk_distill_kl_direction", "reverse")).lower()
        if direction == "forward":
            return TOPK_RENORMALIZED_FORWARD_KL
        return TOPK_RENORMALIZED_REVERSE_KL
    return mode


def topk_distill_k(policy_loss_config: Any) -> int:
    return max(1, int(cfg_get(policy_loss_config, "topk_distill_k", 8) or 8))


def eopd_topk_k(policy_loss_config: Any) -> int:
    raw_value = cfg_get(policy_loss_config, "eopd_topk_k", 16)
    value = int(16 if raw_value is None else raw_value)
    if value <= 0:
        raise ValueError(f"eopd_topk_k must be positive, got {value}.")
    return value


def eopd_entropy_threshold(policy_loss_config: Any) -> float:
    value = float(cfg_get(policy_loss_config, "eopd_entropy_threshold", 0.8))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "eopd_entropy_threshold must be finite and non-negative, "
            f"got {value}."
        )
    return value


def eopd_forward_kl_weight(policy_loss_config: Any) -> float:
    value = float(cfg_get(policy_loss_config, "eopd_forward_kl_weight", 1.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "eopd_forward_kl_weight must be finite and non-negative, "
            f"got {value}."
        )
    return value


def topk_distill_support_source(policy_loss_config: Any) -> str:
    source = str(
        cfg_get(policy_loss_config, "topk_distill_support_source", TOPK_SUPPORT_SOURCE_TEACHER)
    ).lower()
    if source not in TOPK_SUPPORT_SOURCES:
        raise ValueError(
            "topk_distill_support_source must be one of "
            f"{sorted(TOPK_SUPPORT_SOURCES)}, got {source!r}."
        )
    return source


def topk_distill_weight(policy_loss_config: Any) -> float:
    return float(cfg_get(policy_loss_config, "topk_distill_loss_weight", 1.0) or 0.0)


def topk_distill_temperature(policy_loss_config: Any) -> float:
    value = float(cfg_get(policy_loss_config, "topk_distill_temperature", 1.0) or 1.0)
    return max(value, 1e-6)


def topk_distill_logprob_chunk_size(policy_loss_config: Any) -> int:
    return max(1, int(cfg_get(policy_loss_config, "topk_distill_logprob_chunk_size", _TOPK_LOGPROB_CHUNK_SIZE) or 1))


def topk_distill_logprob_mode(policy_loss_config: Any) -> str:
    mode = str(cfg_get(policy_loss_config, "topk_distill_logprob_mode", TOPK_LOGPROB_MODE_SPARSE)).lower()
    if mode not in TOPK_LOGPROB_MODES:
        raise ValueError(f"topk_distill_logprob_mode must be one of {sorted(TOPK_LOGPROB_MODES)}, got {mode!r}.")
    return mode


def topk_distill_include_tail(policy_loss_config: Any) -> bool:
    mode = resolved_topk_distill_mode(policy_loss_config)
    if mode in TOPK_RENORMALIZED_MODES:
        return False
    return bool(cfg_get(policy_loss_config, "topk_distill_tail_bucket", True))


def topk_distill_uses_renormalized_support(policy_loss_config: Any) -> bool:
    return resolved_topk_distill_mode(policy_loss_config) in TOPK_RENORMALIZED_MODES


def is_teacher_prefix_enabled(policy_loss_config: Any) -> bool:
    return bool(cfg_get(policy_loss_config, "teacher_prefix_enabled", False))


def teacher_prefix_loss_region(policy_loss_config: Any) -> str:
    region = str(
        cfg_get(policy_loss_config, "teacher_prefix_loss_region", TEACHER_PREFIX_SUFFIX_ONLY)
    ).lower()
    if region == "all":
        return TEACHER_PREFIX_PREFIX_AND_SUFFIX
    if region not in TEACHER_PREFIX_LOSS_REGIONS:
        raise ValueError(
            "teacher_prefix_loss_region must be one of "
            f"{sorted(TEACHER_PREFIX_LOSS_REGIONS)} or 'all', got {region!r}."
        )
    return region


def teacher_prefix_forward_weight(policy_loss_config: Any) -> float:
    return float(cfg_get(policy_loss_config, "teacher_prefix_forward_kl_weight", 1.0) or 0.0)


def teacher_type_at(opd_teacher: object, index: int) -> object:
    if hasattr(opd_teacher, "ndim"):
        if opd_teacher.ndim == 0:
            return opd_teacher.item()
        return opd_teacher[index]
    if isinstance(opd_teacher, (list, tuple)):
        return opd_teacher[index]
    return opd_teacher


def teacher_tensor_prefix(domain: object) -> str:
    text = str(domain or "math").strip().lower().replace("-", "_")
    safe = "".join(char if (char.isalnum() or char == "_") else "_" for char in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "math"


def teacher_tensor_key(domain: object, suffix: str) -> str:
    normalized_suffix = suffix.strip("_")
    return f"{teacher_tensor_prefix(domain)}_teacher_{normalized_suffix}"


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors are the same view of the same storage."""

    return left is right or (
        left.shape == right.shape
        and left.stride() == right.stride()
        and left.storage_offset() == right.storage_offset()
        and left.device == right.device
        and left.dtype == right.dtype
        and left.data_ptr() == right.data_ptr()
    )


def select_teacher_tensor_by_domain(
    model_inputs: dict[str, Any],
    policy_loss_config: Any,
    *,
    suffix: str,
    math_key: str | None = None,
    code_key: str | None = None,
) -> torch.Tensor:
    normalized_suffix = suffix.strip("_")
    fallback_math_key = math_key or teacher_tensor_key("math", normalized_suffix)
    fallback_code_key = code_key or teacher_tensor_key("code", normalized_suffix)
    if fallback_math_key not in model_inputs:
        raise ValueError(f"Teacher tensor selection requires {fallback_math_key!r} in model_inputs.")

    math_tensor = model_inputs[fallback_math_key]
    code_tensor = model_inputs.get(fallback_code_key, math_tensor)
    if not bool(cfg_get(policy_loss_config, "multi_teacher_distill", False)) or "opd_teacher" not in model_inputs:
        return math_tensor

    opd_teacher = model_inputs["opd_teacher"]
    selected_sources: list[torch.Tensor] = []
    for idx in range(int(math_tensor.shape[0])):
        teacher_type = teacher_type_at(opd_teacher, idx)
        dynamic_key = teacher_tensor_key(teacher_type, normalized_suffix)
        if dynamic_key in model_inputs:
            selected_sources.append(model_inputs[dynamic_key])
        elif (
            teacher_tensor_prefix(teacher_type) == "code"
            and fallback_code_key in model_inputs
        ):
            selected_sources.append(code_tensor)
        else:
            selected_sources.append(math_tensor)
    if all(_same_tensor_view(source, math_tensor) for source in selected_sources):
        return math_tensor

    selected = torch.empty_like(math_tensor)
    for idx, source in enumerate(selected_sources):
        selected[idx] = source[idx]
    return selected


def select_teacher_log_prob_tensor(
    model_inputs: dict[str, Any],
    policy_loss_config: Any,
    *,
    math_key: str = "math_teacher_log_prob",
    code_key: str = "code_teacher_log_prob",
) -> torch.Tensor:
    return select_teacher_tensor_by_domain(
        model_inputs,
        policy_loss_config,
        suffix="log_prob",
        math_key=math_key,
        code_key=code_key,
    )


def teacher_prefix_masks(
    model_inputs: dict[str, Any],
    response_mask: torch.Tensor,
    policy_loss_config: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return prefix-forward-KL and suffix-distillation masks.

    The prefix mask is expected to mark tokens sampled from the teacher. Suffix
    tokens remain student-sampled and keep the existing OPD/top-k objective.
    """

    if not is_teacher_prefix_enabled(policy_loss_config) or "teacher_prefix_mask" not in model_inputs:
        empty = torch.zeros_like(response_mask)
        return empty, response_mask, False

    prefix_mask = model_inputs["teacher_prefix_mask"].to(device=response_mask.device, dtype=response_mask.dtype)
    prefix_mask = prefix_mask * response_mask
    suffix_mask = (response_mask - prefix_mask).clamp(min=0.0, max=1.0)
    region = teacher_prefix_loss_region(policy_loss_config)
    if region == TEACHER_PREFIX_SUFFIX_ONLY:
        prefix_loss_mask = torch.zeros_like(response_mask)
    else:
        prefix_loss_mask = prefix_mask
    if "student_suffix_mask" in model_inputs:
        suffix_mask = model_inputs["student_suffix_mask"].to(device=response_mask.device, dtype=response_mask.dtype)
        suffix_mask = suffix_mask * response_mask
    if region == TEACHER_PREFIX_PREFIX_ONLY:
        suffix_mask = torch.zeros_like(response_mask)
    return prefix_loss_mask, suffix_mask, True


def chosen_token_forward_kl_matrix(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return teacher_log_probs.float() - student_log_probs.float()


def chosen_token_policy_gradient_reward_matrix(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return teacher_log_probs.float() - student_log_probs.float()


def topk_log_probs_from_logits(
    logits: torch.Tensor,
    *,
    topk: int | None = None,
    gather_topk_ids: torch.Tensor | None = None,
    normalize_gathered: bool = True,
    chunk_size: int = _TOPK_LOGPROB_CHUNK_SIZE,
    logprob_mode: str = TOPK_LOGPROB_MODE_SPARSE,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compute top-k log-probs with sparse or full-vocab normalization."""

    if topk is None and gather_topk_ids is None:
        return None, None, None
    if logits.dim() < 2:
        raise ValueError(f"logits must have at least 2 dims, got shape {tuple(logits.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    logprob_mode = str(logprob_mode).lower()
    if logprob_mode not in TOPK_LOGPROB_MODES:
        raise ValueError(f"logprob_mode must be one of {sorted(TOPK_LOGPROB_MODES)}, got {logprob_mode!r}.")

    vocab_size = int(logits.shape[-1])
    prefix_shape = tuple(logits.shape[:-1])
    flat_logits = logits.reshape(-1, vocab_size)
    topk_count = min(max(1, int(topk)), vocab_size) if topk is not None else None

    flat_gather_ids = None
    gather_shape = None
    if gather_topk_ids is not None:
        gather_ids = gather_topk_ids.to(device=logits.device, dtype=torch.long)
        if tuple(gather_ids.shape[:-1]) != prefix_shape:
            raise ValueError(
                "gather_topk_ids must match logits prefix shape, "
                f"got {tuple(gather_ids.shape)} for logits {tuple(logits.shape)}."
            )
        gather_shape = tuple(gather_ids.shape[-1:])
        flat_gather_ids = gather_ids.reshape(-1, int(gather_ids.shape[-1]))

    topk_id_chunks: list[torch.Tensor] = []
    topk_log_prob_chunks: list[torch.Tensor] = []
    gathered_log_prob_chunks: list[torch.Tensor] = []
    for start in range(0, int(flat_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(flat_logits.shape[0]))
        needs_vocab_normalizer = (
            topk_count is not None
            or logprob_mode == TOPK_LOGPROB_MODE_FULL_VOCAB
            or (flat_gather_ids is not None and normalize_gathered)
        )
        raw_logits_chunk = flat_logits[start:end]
        logits_chunk = raw_logits_chunk.float() if needs_vocab_normalizer else raw_logits_chunk
        log_norm = torch.logsumexp(logits_chunk, dim=-1, keepdim=True) if needs_vocab_normalizer else None
        if topk_count is not None:
            top_logits, top_ids = torch.topk(logits_chunk, topk_count, dim=-1)
            topk_id_chunks.append(top_ids)
            if log_norm is None:
                raise RuntimeError("top-k log-prob computation requires a vocabulary normalizer.")
            topk_log_prob_chunks.append(top_logits - log_norm)
        if flat_gather_ids is not None:
            ids_chunk = flat_gather_ids[start:end]
            gathered_logits = logits_chunk.gather(dim=-1, index=ids_chunk)
            if normalize_gathered or logprob_mode == TOPK_LOGPROB_MODE_FULL_VOCAB:
                if log_norm is None:
                    raise RuntimeError("gathered log-prob computation requires a vocabulary normalizer.")
                gathered_logits = gathered_logits - log_norm
            gathered_log_prob_chunks.append(gathered_logits)

    topk_ids = None
    topk_log_probs = None
    gathered_log_probs = None
    if topk_count is not None:
        topk_ids = torch.cat(topk_id_chunks, dim=0).reshape(*prefix_shape, topk_count)
        topk_log_probs = torch.cat(topk_log_prob_chunks, dim=0).reshape(*prefix_shape, topk_count)
    if gather_shape is not None:
        gathered_log_probs = torch.cat(gathered_log_prob_chunks, dim=0).reshape(*prefix_shape, *gather_shape)
    return topk_ids, topk_log_probs, gathered_log_probs


def _log_tail_prob(top_log_probs: torch.Tensor) -> torch.Tensor:
    top_mass = torch.exp(top_log_probs).sum(dim=-1).clamp(min=0.0, max=1.0)
    tail_mass = (1.0 - top_mass).clamp(min=1e-12)
    return torch.log(tail_mass)


def _bucket_log_probs(top_log_probs: torch.Tensor, include_tail: bool) -> torch.Tensor:
    if not include_tail:
        return torch.log_softmax(top_log_probs, dim=-1)
    tail = _log_tail_prob(top_log_probs).unsqueeze(-1)
    return torch.cat([top_log_probs, tail], dim=-1)


def _temperature_bucket_log_probs(bucket_log_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if abs(temperature - 1.0) <= 1e-6:
        return bucket_log_probs
    return torch.log_softmax(bucket_log_probs / temperature, dim=-1)


def topk_distill_loss_matrix(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    mode: str,
    include_tail: bool,
    temperature: float,
) -> torch.Tensor:
    """Return per-token KL loss on a shared top-k support.

    ``student_topk_log_probs`` and ``teacher_topk_log_probs`` must refer to
    the same selected token ids and have shape ``[batch, response, k]``.
    For renormalized support modes, these tensors only need to be log-scores
    on the selected support because ``log_softmax`` removes any global
    normalization constant.
    """

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "student_topk_log_probs and teacher_topk_log_probs must have identical shapes, "
            f"got {tuple(student_topk_log_probs.shape)} and {tuple(teacher_topk_log_probs.shape)}."
        )
    normalized_mode = str(mode)
    if normalized_mode not in TOPK_DISTILL_MODES:
        raise ValueError(f"Unsupported top-k distillation mode: {normalized_mode}")

    use_tail = include_tail and normalized_mode not in TOPK_RENORMALIZED_MODES
    teacher_log_q = _bucket_log_probs(teacher_topk_log_probs.float(), use_tail)
    student_log_q = _bucket_log_probs(student_topk_log_probs.float(), use_tail)
    teacher_log_q = _temperature_bucket_log_probs(teacher_log_q, temperature)
    student_log_q = _temperature_bucket_log_probs(student_log_q, temperature)

    teacher_q = torch.exp(teacher_log_q)
    student_q = torch.exp(student_log_q)
    if normalized_mode in TOPK_REVERSE_KL_MODES:
        return (student_q * (student_log_q - teacher_log_q)).sum(dim=-1)
    return (teacher_q * (teacher_log_q - student_log_q)).sum(dim=-1)


def eopd_forward_kl_matrix(
    *,
    student_full_vocab_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return the truncated forward-KL term from EOPD Eq. (10).

    The teacher distribution is renormalized over its selected top-k support.
    Student values must remain full-vocabulary-normalized log-probabilities
    gathered at the same teacher token ids.
    """

    if student_full_vocab_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "EOPD student and teacher top-k tensors must have identical shapes, "
            f"got {tuple(student_full_vocab_log_probs.shape)} and "
            f"{tuple(teacher_topk_log_probs.shape)}."
        )
    teacher_log_q = torch.log_softmax(
        teacher_topk_log_probs.float(),
        dim=-1,
    )
    teacher_q = torch.exp(teacher_log_q)
    student_log_p = student_full_vocab_log_probs.float()
    return (teacher_q * (teacher_log_q - student_log_p)).sum(dim=-1)


def eopd_teacher_student_cross_entropy_matrix(
    *,
    student_full_vocab_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return EOPD's teacher-top-k cross entropy against the full student."""

    if student_full_vocab_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "EOPD student and teacher top-k tensors must have identical shapes, "
            f"got {tuple(student_full_vocab_log_probs.shape)} and "
            f"{tuple(teacher_topk_log_probs.shape)}."
        )
    teacher_q = torch.softmax(teacher_topk_log_probs.float(), dim=-1)
    return -(teacher_q * student_full_vocab_log_probs.float()).sum(dim=-1)


def topk_teacher_student_cross_entropy_matrix(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    include_tail: bool,
    temperature: float,
) -> torch.Tensor:
    """Return per-token cross entropy H(p_teacher, p_student) on top-k buckets."""

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "student_topk_log_probs and teacher_topk_log_probs must have identical shapes, "
            f"got {tuple(student_topk_log_probs.shape)} and {tuple(teacher_topk_log_probs.shape)}."
        )
    teacher_log_q = _bucket_log_probs(teacher_topk_log_probs.float(), include_tail)
    student_log_q = _bucket_log_probs(student_topk_log_probs.float(), include_tail)
    teacher_log_q = _temperature_bucket_log_probs(teacher_log_q, temperature)
    student_log_q = _temperature_bucket_log_probs(student_log_q, temperature)
    teacher_q = torch.exp(teacher_log_q)
    return -(teacher_q * student_log_q).sum(dim=-1)


def topk_distill_bucket_metrics(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    student_values_are_log_probs: bool = True,
    support_source: str = TOPK_SUPPORT_SOURCE_TEACHER,
) -> dict[str, float]:
    mask = response_mask.detach().float()
    denom = float(mask.sum().detach().cpu().item())
    if denom <= 0.0:
        return {}

    teacher_mass = torch.exp(teacher_topk_log_probs.detach().float()).sum(dim=-1).clamp(min=0.0, max=1.0)

    def masked_mean(value: torch.Tensor) -> float:
        return float(((value * mask).sum() / mask.sum().clamp(min=1.0)).detach().cpu().item())

    metrics = {
        "support_teacher_mass": masked_mean(teacher_mass),
        "tail_teacher_mass_off_support": masked_mean(1.0 - teacher_mass),
        "topk_teacher_mass": masked_mean(teacher_mass),
        "tail_teacher_mass": masked_mean(1.0 - teacher_mass),
    }
    if student_values_are_log_probs:
        student_mass = torch.exp(student_topk_log_probs.detach().float()).sum(dim=-1).clamp(min=0.0, max=1.0)
        metrics["support_student_mass"] = masked_mean(student_mass)
        metrics["tail_student_mass_off_support"] = masked_mean(1.0 - student_mass)
        if support_source == TOPK_SUPPORT_SOURCE_TEACHER:
            metrics["topk_student_mass_on_teacher_ids"] = metrics["support_student_mass"]
            metrics["tail_student_mass_on_teacher_ids"] = metrics["tail_student_mass_off_support"]
    return metrics
