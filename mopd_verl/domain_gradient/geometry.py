"""FSDP1-aware geometry for saved CPU gradient vectors."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from mopd_verl.domain_gradient.config import _get
from verl.utils.device import get_device_id

GradientVector = tuple[torch.Tensor, ...]
_DOT_CHUNK_SIZE = 1_048_576
_STORAGE_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def gradient_replica_count(actor: Any) -> int:
    """Number of identical local gradient shards in the current FSDP1 mesh."""

    world_size = _world_size()
    fsdp_config = _get(getattr(actor, "config", {}), "fsdp_config", {})
    try:
        fsdp_size = int(_get(fsdp_config, "fsdp_size", -1))
    except (TypeError, ValueError):
        return 1
    if fsdp_size <= 0 or fsdp_size >= world_size or world_size % fsdp_size:
        return 1
    return world_size // fsdp_size


def _reduce(actor: Any, values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    divisor = float(gradient_replica_count(actor))
    return [float(value) / divisor for value in tensor.tolist()]


def actor_group_sum(actor: Any, value: float) -> float:
    """Sum a per-rank quantity without removing replicated gradient copies."""

    del actor  # Reserved for a future actor-specific process group.
    tensor = torch.tensor(float(value), device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _scale(actor: Any) -> float:
    scaler = getattr(actor, "scaler", None)
    return float(scaler.get_scale()) if scaler is not None else 1.0


def _require_local_tensor(value: torch.Tensor, *, context: str) -> torch.Tensor:
    if isinstance(value, DTensor):
        raise NotImplementedError(
            f"Domain-gradient {context} currently supports FSDP1 only; "
            "FSDP2 DTensor gradients require mesh-placement-aware reduction."
        )
    return value


def _trainable_parameters(actor: Any) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for parameter in actor.actor_module.parameters()
        if parameter.requires_grad
    )


def snapshot_gradients(
    actor: Any,
    storage_dtype: str = "float32",
) -> GradientVector:
    """Copy local FSDP1 gradient shards to CPU in the configured storage dtype."""

    scale = _scale(actor)
    dtype = _STORAGE_DTYPES.get(storage_dtype.lower())
    if dtype is None:
        raise ValueError(f"Unsupported gradient storage dtype: {storage_dtype!r}")
    chunks: list[torch.Tensor] = []
    for parameter in _trainable_parameters(actor):
        _require_local_tensor(parameter, context="snapshot")
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=dtype))
            continue
        gradient = _require_local_tensor(parameter.grad.detach(), context="snapshot")
        gradient = gradient.reshape(-1)
        if scale != 1.0:
            gradient = gradient / scale
        chunks.append(gradient.to(device="cpu", dtype=dtype, copy=True))
    return tuple(chunks)


def _double_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError("Gradient chunks have different sizes.")
    left = left.reshape(-1)
    right = right.reshape(-1)
    total = 0.0
    for start in range(0, int(left.numel()), _DOT_CHUNK_SIZE):
        end = min(start + _DOT_CHUNK_SIZE, int(left.numel()))
        total += float(
            torch.dot(left[start:end].double(), right[start:end].double()).item()
        )
    return total


def _local_dot(left: GradientVector, right: GradientVector) -> float:
    if len(left) != len(right):
        raise ValueError("Gradient vectors have different parameter counts.")
    return sum(
        _double_dot(left_chunk, right_chunk)
        for left_chunk, right_chunk in zip(left, right, strict=True)
    )


def vector_squared_norm(actor: Any, vector: GradientVector) -> float:
    return _reduce(actor, [_local_dot(vector, vector)])[0]


def vector_dot(
    actor: Any,
    left: GradientVector,
    right: GradientVector,
) -> float:
    """Return a global dot product, removing duplicated HSDP replicas."""

    return _reduce(actor, [_local_dot(left, right)])[0]


def vector_nbytes(vector: GradientVector) -> int:
    return sum(int(chunk.numel()) * int(chunk.element_size()) for chunk in vector)


def current_vs_vector_stats(
    actor: Any,
    reference: GradientVector,
) -> tuple[float, float]:
    """Return global ``(||current||², current·reference)`` without a second snapshot."""

    if len(reference) != len(_trainable_parameters(actor)):
        raise ValueError("Reference gradient has a different parameter count.")
    scale = _scale(actor)
    current_sq = 0.0
    dot = 0.0
    for parameter, saved in zip(
        _trainable_parameters(actor),
        reference,
        strict=True,
    ):
        _require_local_tensor(parameter, context="streaming statistics")
        if parameter.grad is None:
            continue
        gradient = _require_local_tensor(
            parameter.grad.detach(),
            context="streaming statistics",
        ).reshape(-1)
        if gradient.numel() != saved.numel():
            raise ValueError("Reference gradient has a different local shard size.")
        for start in range(0, int(gradient.numel()), _DOT_CHUNK_SIZE):
            end = min(start + _DOT_CHUNK_SIZE, int(gradient.numel()))
            current = gradient[start:end]
            if scale != 1.0:
                current = current / scale
            current = current.to(device="cpu", dtype=torch.float32, copy=True)
            current_sq += _double_dot(current, current)
            dot += _double_dot(current, saved[start:end])
    reduced = _reduce(actor, [current_sq, dot])
    return reduced[0], reduced[1]


def _comparison(left_sq: float, right_sq: float, dot: float) -> dict[str, float]:
    diff_sq = max(left_sq + right_sq - 2.0 * dot, 0.0)
    left_norm = math.sqrt(max(left_sq, 0.0))
    right_norm = math.sqrt(max(right_sq, 0.0))
    diff_norm = math.sqrt(diff_sq)
    return {
        "cosine": dot / max(left_norm * right_norm, 1e-30),
        "diff_norm": diff_norm,
        "rel_l2": diff_norm / max(right_norm, 1e-30),
        "norm_ratio": left_norm / max(right_norm, 1e-30),
        "projection_share": dot / max(right_sq, 1e-30),
    }


def _storage_roundoff_unit(storage_dtype: str) -> float:
    dtype = _STORAGE_DTYPES.get(storage_dtype.lower())
    if dtype is None:
        raise ValueError(f"Unsupported gradient storage dtype: {storage_dtype!r}")
    return 0.5 * float(torch.finfo(dtype).eps)


def domain_metrics_from_gram(
    actor: Any,
    domains: tuple[str, ...],
    *,
    total_sq: float,
    domain_sq: dict[str, float],
    domain_total_dot: dict[str, float],
    pair_dot: dict[tuple[str, str], float],
    closure_threshold: float,
    all_vectors_fp32: bool = True,
    storage_dtype: str = "float32",
) -> dict[str, float]:
    """Build all domain metrics from exact globally reduced Gram scalars."""

    metrics: dict[str, float] = {
        "global/audit/full_gradient_replica_count": float(
            gradient_replica_count(actor)
        ),
        "global/audit/gradient_correctness_storage_fp32": float(
            all_vectors_fp32
        ),
        "global/full_grad/total_grad_norm": math.sqrt(max(total_sq, 0.0)),
    }
    for domain in domains:
        safe_domain = _safe_name(domain)
        metrics[f"{safe_domain}/full_grad/grad_norm"] = math.sqrt(
            max(domain_sq[domain], 0.0)
        )
        comparison = _comparison(
            domain_sq[domain],
            total_sq,
            domain_total_dot[domain],
        )
        metrics[
            f"global/full_grad_alignment/{safe_domain}_vs_total/"
            "full_grad_cosine_domain_total"
        ] = comparison["cosine"]
        metrics[
            f"global/full_grad_contribution/{safe_domain}_to_total/"
            "signed_projection_share"
        ] = comparison["projection_share"]

    for left_index, left_domain in enumerate(domains):
        for right_domain in domains[left_index + 1 :]:
            dot = pair_dot[(left_domain, right_domain)]
            comparison = _comparison(
                domain_sq[left_domain],
                domain_sq[right_domain],
                dot,
            )
            pair = f"{_safe_name(left_domain)}_vs_{_safe_name(right_domain)}"
            metrics[
                f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"
            ] = comparison["cosine"]

    domain_sum_sq = sum(domain_sq.values()) + 2.0 * sum(pair_dot.values())
    domain_sum_total_dot = sum(domain_total_dot.values())
    closure = _comparison(domain_sum_sq, total_sq, domain_sum_total_dot)
    total_norm = math.sqrt(max(total_sq, 0.0))
    domain_vector_norm_sum = sum(
        math.sqrt(max(domain_sq[domain], 0.0)) for domain in domains
    )
    cancellation_condition = (
        0.0
        if total_norm == 0.0 and domain_vector_norm_sum == 0.0
        else domain_vector_norm_sum / max(total_norm, 1e-30)
    )
    closure_payload = {
        **closure,
        "domain_vector_norm_sum": domain_vector_norm_sum,
        "domain_sum_vector_norm": math.sqrt(max(domain_sum_sq, 0.0)),
        "domain_norm_sum_over_total_norm": cancellation_condition,
        "diff_norm_over_domain_vector_norm_sum": (
            0.0
            if closure["diff_norm"] == 0.0 and domain_vector_norm_sum == 0.0
            else closure["diff_norm"] / max(domain_vector_norm_sum, 1e-30)
        ),
    }
    # The domain sum and audit total are stored independently.  A conservative
    # first-order bound therefore includes quantization of both operands.
    estimated_roundoff = _storage_roundoff_unit(storage_dtype) * (
        cancellation_condition + 1.0
    )
    closure_payload["estimated_storage_roundoff_rel_l2"] = estimated_roundoff
    closure_payload["storage_roundoff_may_exceed_threshold"] = float(
        not all_vectors_fp32 and estimated_roundoff > closure_threshold
    )
    closure_payload["passed"] = float(closure["rel_l2"] <= closure_threshold)
    for prefix in (
        "global/full_grad_closure/domain_sum_vs_audit_total",
        # Compatibility alias: this legacy group has always used audit replay
        # total rather than the production training gradient.
        "global/full_grad_closure/domain_sum_vs_training",
    ):
        metrics.update(
            {f"{prefix}/{name}": value for name, value in closure_payload.items()}
        )
    return metrics


def training_parity_metrics(
    actor: Any,
    audit_total: GradientVector,
    threshold: float,
) -> dict[str, float]:
    total_sq = vector_squared_norm(actor, audit_total)
    training_sq, dot = current_vs_vector_stats(actor, audit_total)
    comparison = _comparison(total_sq, training_sq, dot)
    prefix = "global/full_grad_training_parity/audit_total_vs_training_total"
    metrics = {f"{prefix}/{name}": value for name, value in comparison.items()}
    metrics[f"{prefix}/candidate_norm"] = math.sqrt(max(total_sq, 0.0))
    metrics[f"{prefix}/reference_norm"] = math.sqrt(max(training_sq, 0.0))
    metrics[f"{prefix}/passed"] = float(comparison["rel_l2"] <= threshold)
    return metrics
