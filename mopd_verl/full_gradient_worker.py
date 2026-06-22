"""Full-parameter gradient audit helpers for patched verl FSDP workers."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_SPARSE,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_SUPPORT_SOURCE_STUDENT,
    TOPK_SUPPORT_SOURCE_TEACHER,
    chosen_token_forward_kl_matrix,
    is_topk_distill_enabled,
    resolved_topk_distill_mode,
    select_teacher_log_prob_tensor,
    teacher_prefix_forward_weight,
    teacher_prefix_masks,
    topk_distill_include_tail,
    topk_distill_logprob_chunk_size,
    topk_distill_logprob_mode,
    topk_distill_loss_matrix,
    topk_distill_support_source,
    topk_distill_temperature,
    topk_distill_uses_renormalized_support,
    topk_distill_weight,
)
from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device


_VECTOR_REDUCTION_CHUNK_SIZE = 1 << 26
_DOMAIN_LABEL_KEYS = ("opd_teacher", "domain", "source_domain", "ability")


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
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


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _non_tensor_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(length)]


def _label_from_extra_info(extra_info: Any) -> Any:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return None
    if not isinstance(extra_info, dict):
        return None
    for key in _DOMAIN_LABEL_KEYS:
        value = extra_info.get(key)
        if value is not None:
            return value
    return None


def _labels_from_mapping(mapping: dict[str, Any], batch_size: int) -> list[str]:
    for key in _DOMAIN_LABEL_KEYS:
        labels = _non_tensor_list(mapping.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    extra_infos = _non_tensor_list(mapping.get("extra_info"), batch_size)
    labels = [_label_from_extra_info(extra_info) for extra_info in extra_infos]
    if not all(label is None for label in labels):
        return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]


def _teacher_labels(data: DataProto) -> list[str]:
    return _labels_from_mapping(data.non_tensor_batch, len(data))


def _sample_ids(data: DataProto, step: int, fallback_prefix: str | None = None) -> list[str]:
    batch_size = len(data)
    sample_ids = _non_tensor_list(data.non_tensor_batch.get("sample_id"), batch_size)
    fallback_ids = _non_tensor_list(data.non_tensor_batch.get("id"), batch_size)
    extra_infos = _non_tensor_list(data.non_tensor_batch.get("extra_info"), batch_size)
    resolved: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id is not None:
            resolved.append(str(sample_id))
        elif fallback_ids[idx] is not None:
            resolved.append(str(fallback_ids[idx]))
        elif isinstance(extra_infos[idx], dict) and extra_infos[idx].get("sample_id") is not None:
            resolved.append(str(extra_infos[idx]["sample_id"]))
        elif isinstance(extra_infos[idx], dict) and extra_infos[idx].get("id") is not None:
            resolved.append(str(extra_infos[idx]["id"]))
        else:
            prefix = fallback_prefix or f"step{step}"
            resolved.append(f"{prefix}:row{idx}")
    return resolved


def _indices_by_label(
    labels: list[str],
    domains: list[str],
    max_samples_per_domain: int | None,
) -> dict[str, list[int]]:
    configured = list(dict.fromkeys(list(domains) + sorted(set(labels))))
    indices = {domain: [idx for idx, label in enumerate(labels) if label == domain] for domain in configured}
    if max_samples_per_domain is None:
        return indices
    return {domain: domain_indices[:max_samples_per_domain] for domain, domain_indices in indices.items()}


def _max_samples_from_cfg(cfg: dict[str, Any]) -> int | None:
    value = cfg.get("max_samples_per_domain")
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return max(1, int(value))


def _response_token_count(data: DataProto) -> float:
    if data.batch is None or len(data) == 0:
        return 0.0
    if "response_mask" in data.batch:
        return float(data.batch["response_mask"].sum().item())
    if "responses" in data.batch:
        return float(data.batch["responses"].numel())
    return float(len(data))


def _all_reduce_sum(value: float) -> float:
    tensor = torch.tensor(float(value), device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_values_sum(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return [float(value) for value in tensor.tolist()]


def _all_reduce_values_max(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return [float(value) for value in tensor.tolist()]


def _all_reduce_vector_sum(vector: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(vector, op=torch.distributed.ReduceOp.SUM)
    return vector


def _actor_no_sync_context(actor: Any) -> Any:
    actor_module = getattr(actor, "actor_module", None)
    no_sync = getattr(actor_module, "no_sync", None)
    if callable(no_sync):
        try:
            return no_sync()
        except Exception:
            return nullcontext()
    return nullcontext()


def _all_gather_list(values: list[Any]) -> list[Any]:
    if not torch.distributed.is_initialized():
        return list(values)
    gathered: list[list[Any] | None] = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, list(values))
    flattened: list[Any] = []
    for part in gathered:
        if part:
            flattened.extend(part)
    return flattened


def _distributed_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (RuntimeError, ValueError):
        return 0
    return 0


def _distributed_world_size() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except (RuntimeError, ValueError):
        return 1
    return 1


def _all_ranks_true(value: bool) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor(int(value), device=get_device_id(), dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def _all_ranks_equal_ints(values: list[int]) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    local = torch.tensor(values, device=get_device_id(), dtype=torch.int64)
    minimum = local.clone()
    maximum = local.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return bool(torch.equal(minimum, maximum))


def _actor_fsdp_size(actor: Any) -> int | None:
    actor_config = getattr(actor, "config", None)
    fsdp_config = _cfg_get(actor_config, "fsdp_config", {})
    raw_fsdp_size = _cfg_get(fsdp_config, "fsdp_size", -1)
    try:
        return int(raw_fsdp_size)
    except (TypeError, ValueError):
        return None


def _gradient_replica_count(actor: Any) -> int:
    world_size = _distributed_world_size()
    if world_size <= 1:
        return 1

    fsdp_size = _actor_fsdp_size(actor)
    if fsdp_size is None:
        return 1

    if fsdp_size <= 0 or fsdp_size >= world_size or world_size % fsdp_size != 0:
        return 1
    return world_size // fsdp_size


def _reduce_gradient_scalars(actor: Any, values: list[float]) -> list[float]:
    return _all_reduce_values_sum(values)


def _actor_has_full_local_params_for_sample_gradient(actor: Any) -> bool:
    return _actor_fsdp_size(actor) == 1


def _chunks_local_sumsq(chunks: tuple[torch.Tensor, ...]) -> float:
    total = 0.0
    for chunk in chunks:
        chunk_sumsq = _chunked_vector_dot(chunk.float(), chunk.float())
        if chunk_sumsq is not None:
            total += chunk_sumsq
    return total


def _chunked_vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.numel() == 0 or left_flat.numel() != right_flat.numel():
        return None

    total = 0.0
    for start in range(0, left_flat.numel(), _VECTOR_REDUCTION_CHUNK_SIZE):
        end = min(start + _VECTOR_REDUCTION_CHUNK_SIZE, left_flat.numel())
        left_chunk = left_flat[start:end].float()
        right_chunk = right_flat[start:end].float()
        total += float(torch.dot(left_chunk, right_chunk).item())
    return total


def _local_vector_norm(vector: torch.Tensor) -> float:
    sumsq = _chunked_vector_dot(vector, vector)
    if sumsq is None:
        return 0.0
    return float(max(sumsq, 0.0) ** 0.5)


def _local_vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    return _chunked_vector_dot(left, right)


def _vector_norm(vector: torch.Tensor) -> float:
    local_sumsq = _chunked_vector_dot(vector, vector)
    if local_sumsq is None:
        return 0.0
    return float(max(_all_reduce_sum(local_sumsq), 0.0) ** 0.5)


def _vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    local_dot = _chunked_vector_dot(left, right)
    if local_dot is None:
        return None
    return _all_reduce_sum(local_dot)


def _safe_cosine(dot: float | None, left_norm: float, right_norm: float) -> float | None:
    if dot is None or left_norm <= 0 or right_norm <= 0:
        return None
    return dot / (left_norm * right_norm)


def _storage_dtype(storage_dtype: str) -> torch.dtype:
    normalized = str(storage_dtype).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _max_memory_allocated_gb() -> float:
    max_memory_allocated = getattr(get_torch_device(), "max_memory_allocated", None)
    if not callable(max_memory_allocated):
        return 0.0
    try:
        return float(max_memory_allocated()) / (1024**3)
    except (RuntimeError, TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class _GradDifferenceSnapshot:
    first_norm_sq: float
    total_norm_sq: float
    first_total_dot: float
    second_norm_sq: float
    first_second_dot: float
    second_total_dot: float
    second_chunks: tuple[torch.Tensor, ...] | None
    second_target_norm_sq: float | None


class _HookState:
    """Mutable state shared by backward hook closures for incremental computation."""

    def __init__(self) -> None:
        self.norm_parts: list[torch.Tensor] | None = None
        self.dot_targets: dict[int, torch.Tensor] | None = None
        self.dot_accumulator: list[torch.Tensor] | None = None


def _current_grad_scale(actor: Any) -> float:
    scaler = getattr(actor, "scaler", None)
    if scaler is None or not hasattr(scaler, "get_scale"):
        return 1.0
    try:
        scale = float(scaler.get_scale())
    except (TypeError, ValueError):
        return 1.0
    return scale if scale > 0 else 1.0


def _current_grad_cpu_float(parameter: torch.nn.Parameter, scale: float) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    gradient = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    if scale != 1.0:
        gradient = gradient / scale
    return gradient


def _snapshot_current_grad_chunks(actor: Any, storage_dtype: str) -> tuple[torch.Tensor, ...]:
    dtype = _storage_dtype(storage_dtype)
    scale = _current_grad_scale(actor)
    pieces: list[torch.Tensor] = []
    for parameter in _trainable_parameters(actor):
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
            continue
        gradient = parameter.grad.detach()
        if scale != 1.0:
            gradient = gradient / scale
        pieces.append(gradient.reshape(-1).to(device="cpu", dtype=dtype, copy=True))
    return tuple(pieces)


def _current_grad_difference_snapshot(
    actor: Any,
    reference_chunks: tuple[torch.Tensor, ...],
    storage_dtype: str | None = None,
) -> _GradDifferenceSnapshot | None:
    parameters = _trainable_parameters(actor)
    if len(parameters) != len(reference_chunks):
        return None

    dtype = _storage_dtype(storage_dtype) if storage_dtype is not None else None
    second_pieces: list[torch.Tensor] | None = [] if dtype is not None else None
    scale = _current_grad_scale(actor)
    first_sumsq = 0.0
    total_sumsq = 0.0
    first_total_dot = 0.0
    second_sumsq = 0.0
    first_second_dot = 0.0
    second_total_dot = 0.0
    second_target_sumsq = 0.0

    for parameter, first in zip(parameters, reference_chunks):
        if first.numel() != parameter.numel():
            return None
        first_float = first.float()
        total_float = _current_grad_cpu_float(parameter, scale)
        if total_float is None:
            total_float = torch.zeros_like(first_float)
        second_float = total_float - first_float

        first_sumsq += _chunked_vector_dot(first_float, first_float) or 0.0
        total_sumsq += _chunked_vector_dot(total_float, total_float) or 0.0
        first_total_dot += _chunked_vector_dot(first_float, total_float) or 0.0
        second_sumsq += _chunked_vector_dot(second_float, second_float) or 0.0
        first_second_dot += _chunked_vector_dot(first_float, second_float) or 0.0
        second_total_dot += _chunked_vector_dot(second_float, total_float) or 0.0

        if second_pieces is not None and dtype is not None:
            second_piece = second_float.to(dtype=dtype)
            second_pieces.append(second_piece)
            second_piece_float = second_piece.float()
            second_target_sumsq += (
                _chunked_vector_dot(second_piece_float, second_piece_float) or 0.0
            )
            del second_piece_float
        del first_float, total_float, second_float

    scalar_values = [
        first_sumsq,
        total_sumsq,
        first_total_dot,
        second_sumsq,
        first_second_dot,
        second_total_dot,
    ]
    if second_pieces is not None:
        scalar_values.append(second_target_sumsq)
    reduced_values = _reduce_gradient_scalars(actor, scalar_values)

    return _GradDifferenceSnapshot(
        first_norm_sq=reduced_values[0],
        total_norm_sq=reduced_values[1],
        first_total_dot=reduced_values[2],
        second_norm_sq=reduced_values[3],
        first_second_dot=reduced_values[4],
        second_total_dot=reduced_values[5],
        second_chunks=tuple(second_pieces) if second_pieces is not None else None,
        second_target_norm_sq=reduced_values[6] if second_pieces is not None else None,
    )


def _collect_autograd_vector(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
    storage_dtype: str,
) -> torch.Tensor:
    dtype = _storage_dtype(storage_dtype)
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
        else:
            pieces.append(gradient.detach().reshape(-1).to(device="cpu", dtype=dtype))
    if not pieces:
        return torch.zeros(0, dtype=dtype, device="cpu")
    return torch.cat(pieces)


def _gpu_concat_grad_vector(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> torch.Tensor:
    """Concatenate autograd gradients into a flat GPU fp32 vector.

    Like ``_collect_autograd_vector`` but the result stays on GPU for
    efficient in-place accumulation.
    """
    device = get_device_id()
    pieces: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32, device=device))
        else:
            pieces.append(gradient.detach().reshape(-1).to(device=device, dtype=torch.float32))
    if not pieces:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(pieces)


def _vector_to_param_chunks(
    vector: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    storage_dtype: str,
) -> tuple[torch.Tensor, ...]:
    """Split a flat 1-D GPU vector back into per-parameter chunks.

    Each chunk has the same numel as the corresponding parameter.
    Chunks are cast to *storage_dtype* for compact caching.
    """
    dtype = _storage_dtype(storage_dtype)
    chunks: list[torch.Tensor] = []
    offset = 0
    for param in parameters:
        numel = param.numel()
        chunk = vector[offset : offset + numel].to(dtype=dtype)
        chunks.append(chunk)
        offset += numel
    return tuple(chunks)


def _parameter_grad_norm(
    parameters: tuple[torch.nn.Parameter, ...],
) -> float:
    local_sumsq = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float().reshape(-1)
            gradient_sumsq = _chunked_vector_dot(gradient, gradient)
            if gradient_sumsq is not None:
                local_sumsq += gradient_sumsq
    return float(max(_all_reduce_sum(local_sumsq), 0.0) ** 0.5)


def _target_map_matches_parameters(
    parameters: tuple[torch.nn.Parameter, ...],
    target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
) -> bool:
    if not target_map:
        return False
    for target_chunks, _target_norm_sq in target_map.values():
        if len(target_chunks) != len(parameters):
            return False
        for parameter, target in zip(parameters, target_chunks):
            if int(parameter.numel()) != int(target.numel()):
                return False
    return True


def _clear_parameter_grads(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _parameter_grad_dtypes(parameters: tuple[torch.nn.Parameter, ...]) -> tuple[torch.dtype | None, ...]:
    return tuple(parameter.grad.dtype if parameter.grad is not None else None for parameter in parameters)


def _snapshot_parameter_grads(
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
        if parameter.grad is not None
        else None
        for parameter in parameters
    )


def _restore_parameter_grads_from_snapshot(
    parameters: tuple[torch.nn.Parameter, ...],
    grad_snapshot: tuple[torch.Tensor | None, ...],
    grad_dtypes: tuple[torch.dtype | None, ...] | None = None,
) -> None:
    for param_idx, parameter in enumerate(parameters):
        if param_idx >= len(grad_snapshot) or grad_snapshot[param_idx] is None:
            parameter.grad = None
            continue
        snapshot = grad_snapshot[param_idx]
        grad_dtype = (
            grad_dtypes[param_idx]
            if grad_dtypes is not None and param_idx < len(grad_dtypes)
            else snapshot.dtype
        )
        try:
            parameter.grad = snapshot.to(device=parameter.device, dtype=grad_dtype).clone()
        except RuntimeError:
            parameter.grad = snapshot.to(device=parameter.device, dtype=parameter.dtype).clone()


def _parameter_grad_snapshot_diff_stats(
    parameters: tuple[torch.nn.Parameter, ...],
    grad_snapshot: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    local_diff_sq = 0.0
    local_snapshot_sq = 0.0
    local_max_abs = 0.0
    for param_idx, parameter in enumerate(parameters):
        snapshot = grad_snapshot[param_idx] if param_idx < len(grad_snapshot) else None
        if snapshot is None and parameter.grad is None:
            continue
        if snapshot is None:
            current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            reference = torch.zeros_like(current)
        else:
            reference = snapshot.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            if parameter.grad is None:
                current = torch.zeros_like(reference)
            else:
                current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if current.numel() != reference.numel():
            return {
                "rel_l2": float("inf"),
                "max_abs": float("inf"),
                "snapshot_norm": float("inf"),
            }
        diff = current - reference
        diff_sq = _chunked_vector_dot(diff, diff)
        snapshot_sq = _chunked_vector_dot(reference, reference)
        if diff_sq is not None:
            local_diff_sq += diff_sq
        if snapshot_sq is not None:
            local_snapshot_sq += snapshot_sq
        if diff.numel() > 0:
            local_max_abs = max(local_max_abs, float(diff.abs().max().item()))
        del current, reference, diff

    diff_sq = max(local_diff_sq, 0.0)
    snapshot_sq = max(local_snapshot_sq, 0.0)
    snapshot_norm = snapshot_sq**0.5
    return {
        "rel_l2": (diff_sq**0.5) / (snapshot_norm + 1e-12),
        "max_abs": local_max_abs,
        "snapshot_norm": snapshot_norm,
    }


def _parameter_grad_target_diff_stats(
    parameters: tuple[torch.nn.Parameter, ...],
    target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
) -> dict[str, float]:
    local_diff_sq = 0.0
    local_target_sq = 0.0
    local_max_abs = 0.0
    target_items = list(target_map.values())
    for param_idx, parameter in enumerate(parameters):
        target_total: torch.Tensor | None = None
        for target_chunks, _target_norm_sq in target_items:
            chunk = target_chunks[param_idx].detach().reshape(-1).float()
            target_total = chunk.clone() if target_total is None else target_total.add(chunk)
        if target_total is None:
            continue
        if parameter.grad is None:
            current = torch.zeros_like(target_total)
        else:
            current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if current.numel() != target_total.numel():
            return {
                "rel_l2": float("inf"),
                "max_abs": float("inf"),
                "target_norm": float("inf"),
            }
        diff = current - target_total
        diff_sq = _chunked_vector_dot(diff, diff)
        target_sq = _chunked_vector_dot(target_total, target_total)
        if diff_sq is not None:
            local_diff_sq += diff_sq
        if target_sq is not None:
            local_target_sq += target_sq
        if diff.numel() > 0:
            local_max_abs = max(local_max_abs, float(diff.abs().max().item()))
        del target_total, current, diff

    diff_sq = max(local_diff_sq, 0.0)
    target_sq = max(local_target_sq, 0.0)
    target_norm = target_sq**0.5
    return {
        "rel_l2": (diff_sq**0.5) / (target_norm + 1e-12),
        "max_abs": local_max_abs,
        "target_norm": target_norm,
    }


def _restore_parameter_grads_from_targets(
    parameters: tuple[torch.nn.Parameter, ...],
    target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    grad_dtypes: tuple[torch.dtype | None, ...] | None = None,
) -> None:
    target_items = list(target_map.values())
    for param_idx, parameter in enumerate(parameters):
        total_chunk: torch.Tensor | None = None
        for target_chunks, _target_norm_sq in target_items:
            chunk = target_chunks[param_idx].detach().reshape(-1).float()
            total_chunk = chunk.clone() if total_chunk is None else total_chunk.add(chunk)
        if total_chunk is None:
            parameter.grad = None
            continue
        restore_dtype = parameter.dtype
        if grad_dtypes is not None and param_idx < len(grad_dtypes):
            original_grad_dtype = grad_dtypes[param_idx]
            if original_grad_dtype is not None:
                restore_dtype = original_grad_dtype
        restored = total_chunk.reshape(parameter.shape).to(device=parameter.device, dtype=restore_dtype)
        try:
            parameter.grad = restored.clone()
        except RuntimeError:
            parameter.grad = total_chunk.reshape(parameter.shape).to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl_rows(output_dir: str | None, filename: str, rows: list[dict[str, Any]]) -> None:
    if not output_dir or not rows:
        return
    path = Path(str(output_dir)) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _trainable_parameters(actor: Any) -> tuple[torch.nn.Parameter, ...]:
    optimizer = getattr(actor, "actor_optimizer", None)
    if optimizer is not None:
        params = []
        seen: set[int] = set()
        for group in getattr(optimizer, "param_groups", []):
            for parameter in group.get("params", []):
                if parameter.requires_grad and id(parameter) not in seen:
                    params.append(parameter)
                    seen.add(id(parameter))
        if params:
            return tuple(params)
    return tuple(parameter for parameter in actor.actor_module.parameters() if parameter.requires_grad)


def _labels_from_inputs(model_inputs: dict[str, Any], batch_size: int) -> list[str]:
    return _labels_from_mapping(model_inputs, batch_size)


def _selected_teacher_topk_from_inputs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "math_teacher_topk_ids" not in model_inputs or "math_teacher_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Top-k distillation requires math_teacher_topk_ids and math_teacher_topk_logprobs in the batch."
        )
    math_ids = model_inputs["math_teacher_topk_ids"]
    math_log_probs = model_inputs["math_teacher_topk_logprobs"]
    code_ids = model_inputs.get("code_teacher_topk_ids", math_ids)
    code_log_probs = model_inputs.get("code_teacher_topk_logprobs", math_log_probs)
    if not bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False)):
        return math_ids, math_log_probs
    labels = _labels_from_inputs(model_inputs, int(math_ids.shape[0]))
    selected_ids = torch.empty_like(math_ids)
    selected_log_probs = torch.empty_like(math_log_probs)
    for idx, label in enumerate(labels):
        if label == "code" and "code_teacher_topk_ids" in model_inputs:
            selected_ids[idx] = code_ids[idx]
            selected_log_probs[idx] = code_log_probs[idx]
        else:
            selected_ids[idx] = math_ids[idx]
            selected_log_probs[idx] = math_log_probs[idx]
    return selected_ids, selected_log_probs


def _selected_student_topk_teacher_log_probs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> torch.Tensor:
    if "math_teacher_student_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Student top-k distillation requires math_teacher_student_topk_logprobs in the batch."
        )
    math_log_probs = model_inputs["math_teacher_student_topk_logprobs"]
    code_log_probs = model_inputs.get("code_teacher_student_topk_logprobs", math_log_probs)
    if not bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False)):
        return math_log_probs
    labels = _labels_from_inputs(model_inputs, int(math_log_probs.shape[0]))
    selected_log_probs = torch.empty_like(math_log_probs)
    for idx, label in enumerate(labels):
        if label == "code" and "code_teacher_student_topk_logprobs" in model_inputs:
            selected_log_probs[idx] = code_log_probs[idx]
        else:
            selected_log_probs[idx] = math_log_probs[idx]
    return selected_log_probs


def _selected_topk_support_from_inputs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_source = topk_distill_support_source(policy_loss_cfg)
    if support_source == TOPK_SUPPORT_SOURCE_STUDENT:
        if "student_topk_ids" not in model_inputs:
            raise ValueError("Student top-k distillation requires student_topk_ids in the batch.")
        return model_inputs["student_topk_ids"], _selected_student_topk_teacher_log_probs(
            model_inputs,
            policy_loss_cfg,
        )
    if support_source != TOPK_SUPPORT_SOURCE_TEACHER:
        raise ValueError(f"Unsupported top-k support source: {support_source!r}.")
    return _selected_teacher_topk_from_inputs(model_inputs, policy_loss_cfg)


def _response_token_count_from_mask(mask: torch.Tensor) -> float:
    return float(mask.detach().sum().item())


def _active_sequence_count(mask: torch.Tensor) -> float:
    return float((mask.detach().sum(dim=-1) > 0).float().sum().item())


def _domain_loss_weight(response_mask: torch.Tensor, domain_mask: torch.Tensor, loss_agg_mode: str) -> float:
    if loss_agg_mode == "token-mean":
        total = _response_token_count_from_mask(response_mask)
        selected = _response_token_count_from_mask(domain_mask)
        return 0.0 if total <= 0 else selected / total
    if loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-mean"}:
        total = _active_sequence_count(response_mask)
        selected = _active_sequence_count(domain_mask)
        return 0.0 if total <= 0 else selected / total
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return 1.0 if _response_token_count_from_mask(domain_mask) > 0 else 0.0
    total = _response_token_count_from_mask(response_mask)
    selected = _response_token_count_from_mask(domain_mask)
    return 0.0 if total <= 0 else selected / total


def _actor_reverse_kl_advantages(actor: Any, model_inputs: dict[str, Any], old_log_prob: torch.Tensor) -> torch.Tensor:
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    if not bool(_cfg_get(policy_loss_cfg, "only_reverse_kl_advantages", False)):
        return model_inputs["advantages"]
    if is_topk_distill_enabled(policy_loss_cfg):
        return torch.zeros_like(old_log_prob)

    if "base_log_prob" in model_inputs and "code_teacher_log_prob" in model_inputs:
        lambda_vals = float(_cfg_get(policy_loss_cfg, "lambda_vals", 1.0))
        if bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False)) and "opd_teacher" in model_inputs:
            opd_teacher = model_inputs["opd_teacher"]
            reverse_kl = torch.zeros_like(old_log_prob)
            for idx in range(old_log_prob.shape[0]):
                teacher_type = opd_teacher[idx] if isinstance(opd_teacher, (list, tuple, np.ndarray)) else opd_teacher
                if teacher_type == "math":
                    teacher_log_prob = model_inputs["math_teacher_log_prob"][idx]
                elif teacher_type == "code":
                    teacher_log_prob = model_inputs["code_teacher_log_prob"][idx]
                else:
                    teacher_log_prob = model_inputs["math_teacher_log_prob"][idx]
                if lambda_vals == 1.0:
                    reverse_kl[idx] = old_log_prob[idx] - teacher_log_prob
                else:
                    base_log_prob = model_inputs["base_log_prob"][idx]
                    reverse_kl[idx] = old_log_prob[idx] - base_log_prob - (teacher_log_prob - base_log_prob) * lambda_vals
            return -reverse_kl

        reverse_kl = old_log_prob - model_inputs["base_log_prob"]
        reward_correction = model_inputs["math_teacher_log_prob"] - model_inputs["base_log_prob"]
        if lambda_vals == 1.0:
            reverse_kl = old_log_prob - model_inputs["math_teacher_log_prob"]
        else:
            reverse_kl = reverse_kl - reward_correction * lambda_vals
        return -reverse_kl

    if (
        "code_teacher_log_prob" in model_inputs
        and bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False))
        and "opd_teacher" in model_inputs
    ):
        opd_teacher = model_inputs["opd_teacher"]
        reverse_kl = torch.zeros_like(old_log_prob)
        for idx in range(old_log_prob.shape[0]):
            teacher_type = opd_teacher[idx] if isinstance(opd_teacher, (list, tuple, np.ndarray)) else opd_teacher
            teacher_log_prob = model_inputs["code_teacher_log_prob"][idx] if teacher_type == "code" else model_inputs["math_teacher_log_prob"][idx]
            reverse_kl[idx] = old_log_prob[idx] - teacher_log_prob
        return -reverse_kl

    reverse_kl = old_log_prob - model_inputs["math_teacher_log_prob"]
    return -reverse_kl


def _response_token_id_matrix_from_inputs(model_inputs: dict[str, Any], response_mask: torch.Tensor) -> torch.Tensor | None:
    token_ids = None
    for key in ("responses", "response_ids", "input_ids"):
        if key in model_inputs:
            token_ids = model_inputs[key]
            break
    if token_ids is None or not hasattr(token_ids, "detach") or len(token_ids.shape) != 2:
        return None
    response_len = int(response_mask.shape[-1])
    if tuple(token_ids.shape) == tuple(response_mask.shape):
        return token_ids.detach().long()
    if int(token_ids.shape[0]) == int(response_mask.shape[0]) and int(token_ids.shape[-1]) >= response_len:
        return token_ids[:, -response_len:].detach().long()
    return None


def _data_proto_tensor_device(data: DataProto) -> torch.device | None:
    if data.batch is None:
        return None
    for tensor in data.batch.values():
        if hasattr(tensor, "device"):
            return tensor.device
    return None


def _copy_data_proto_rows_to_cpu(data: DataProto, indices: list[int]) -> DataProto | None:
    if not indices:
        return None
    try:
        device = _data_proto_tensor_device(data)
        if device is None:
            idxs = torch.tensor(indices, dtype=torch.long)
        else:
            idxs = torch.tensor(indices, dtype=torch.long, device=device)
        return data.select_idxs(idxs).to("cpu")
    except Exception:
        return None


def _token_contribution_scale(response_mask: torch.Tensor, sample_idx: int, loss_agg_mode: str) -> float:
    active_tokens_total = float(response_mask.detach().sum().item())
    active_tokens_sample = float(response_mask.detach()[sample_idx].sum().item())
    active_sequences = float((response_mask.detach().sum(dim=-1) > 0).float().sum().item())
    if active_tokens_sample <= 0.0:
        return 0.0
    if loss_agg_mode == "token-mean":
        return 0.0 if active_tokens_total <= 0.0 else 1.0 / active_tokens_total
    if loss_agg_mode == "seq-mean-token-sum":
        return 0.0 if active_sequences <= 0.0 else 1.0 / active_sequences
    if loss_agg_mode == "seq-mean-token-mean":
        return 0.0 if active_sequences <= 0.0 else 1.0 / (active_sequences * active_tokens_sample)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return 1.0 / max(int(response_mask.shape[-1]), 1)
    return 0.0 if active_tokens_total <= 0.0 else 1.0 / active_tokens_total


def _token_mask_contribution_scale(
    response_mask: torch.Tensor,
    token_mask: torch.Tensor,
    loss_agg_mode: str,
) -> float:
    response_mask = response_mask.detach().float()
    token_mask = token_mask.detach().float() * response_mask
    selected_tokens = float(token_mask.sum().item())
    if selected_tokens <= 0.0:
        return 0.0
    active_tokens_total = float(response_mask.sum().item())
    if loss_agg_mode == "token-mean":
        return 0.0 if active_tokens_total <= 0.0 else selected_tokens / active_tokens_total
    active_sequences = float((response_mask.sum(dim=-1) > 0).float().sum().item())
    selected_sequences = float((token_mask.sum(dim=-1) > 0).float().sum().item())
    if loss_agg_mode == "seq-mean-token-sum":
        return 0.0 if active_sequences <= 0.0 else selected_sequences / active_sequences
    if loss_agg_mode == "seq-mean-token-mean":
        total = 0.0
        for sample_idx in range(int(response_mask.shape[0])):
            sample_selected = float(token_mask[sample_idx].sum().item())
            sample_tokens = float(response_mask[sample_idx].sum().item())
            if sample_selected > 0.0 and sample_tokens > 0.0:
                total += sample_selected / sample_tokens
        return 0.0 if active_sequences <= 0.0 else total / active_sequences
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return selected_tokens / max(int(response_mask.shape[-1]), 1)
    return 0.0 if active_tokens_total <= 0.0 else selected_tokens / active_tokens_total


def _actor_micro_batch_loss(
    actor: Any,
    micro_batch: DataProto,
    *,
    loss_scale_factor: float,
    on_policy: bool,
    safe_logprob_backward: bool = False,
    response_mask_override: torch.Tensor | None = None,
) -> torch.Tensor:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    response_mask = model_inputs["response_mask"]
    entropy_coeff = float(_cfg_get(actor.config, "entropy_coeff", 0.0) or 0.0)
    forward_kwargs = {
        "temperature": float(micro_batch.meta_info.get("temperature", 1.0)),
        "calculate_entropy": entropy_coeff != 0,
    }
    if safe_logprob_backward:
        forward_kwargs["inplace_backward"] = False
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    topk_distill_active = is_topk_distill_enabled(policy_loss_cfg)
    use_renormalized_support = (
        topk_distill_active
        and topk_distill_uses_renormalized_support(policy_loss_cfg)
    )
    effective_topk_logprob_mode = topk_distill_logprob_mode(policy_loss_cfg)
    if use_renormalized_support:
        effective_topk_logprob_mode = TOPK_LOGPROB_MODE_SPARSE
    kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
    needs_log_probs = not topk_distill_active or (
        bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0
    )
    forward_kwargs["calculate_log_probs"] = needs_log_probs
    topk_support_ids = None
    teacher_support_log_probs = None
    if topk_distill_active:
        topk_support_ids, teacher_support_log_probs = _selected_topk_support_from_inputs(
            model_inputs,
            policy_loss_cfg,
        )
        forward_kwargs["gather_topk_ids"] = topk_support_ids
        forward_kwargs["normalize_gathered_topk"] = not use_renormalized_support
        forward_kwargs["topk_logprob_chunk_size"] = topk_distill_logprob_chunk_size(policy_loss_cfg)
        forward_kwargs["topk_logprob_mode"] = effective_topk_logprob_mode
        forward_kwargs["return_extra"] = True
    forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
    if topk_distill_active:
        entropy, log_prob, _topk_ids, _topk_log_probs, student_topk_log_probs = forward_output
    else:
        entropy, log_prob = forward_output
    if response_mask_override is not None:
        response_mask = response_mask_override.to(device=log_prob.device, dtype=response_mask.dtype)
    prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
        model_inputs,
        response_mask,
        policy_loss_cfg,
    )
    distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask
    loss_token_mask = (
        (prefix_loss_mask + suffix_loss_mask).clamp(max=1.0)
        if teacher_prefix_active
        else response_mask
    )
    if bool(_cfg_get(actor.config, "use_rollout_log_probs", False)):
        old_log_prob = model_inputs["old_log_probs"]
    elif on_policy:
        old_log_prob = log_prob.detach()
    else:
        old_log_prob = model_inputs["old_log_probs"]

    if topk_distill_active:
        policy_loss = log_prob.new_zeros(())
    else:
        advantages = _actor_reverse_kl_advantages(actor, model_inputs, old_log_prob)
        loss_mode = str(_cfg_get(_cfg_get(actor.config, "policy_loss", {}), "loss_mode", "vanilla"))
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            config=actor.config,
            rollout_is_weights=model_inputs.get("rollout_is_weights", None),
        )
        policy_loss = pg_loss
    if entropy_coeff != 0 and entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=loss_token_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss - entropy_loss * entropy_coeff
    if topk_distill_active:
        topk_loss_mat = topk_distill_loss_matrix(
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_support_log_probs,
            mode=resolved_topk_distill_mode(policy_loss_cfg),
            include_tail=topk_distill_include_tail(policy_loss_cfg),
            temperature=topk_distill_temperature(policy_loss_cfg),
        )
        topk_loss = agg_loss(
            loss_mat=topk_loss_mat,
            loss_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss + topk_loss * topk_distill_weight(policy_loss_cfg)
    if teacher_prefix_active:
        if topk_distill_active:
            prefix_loss_mat = topk_distill_loss_matrix(
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_support_log_probs,
                mode=TOPK_RENORMALIZED_FORWARD_KL,
                include_tail=False,
                temperature=topk_distill_temperature(policy_loss_cfg),
            )
        else:
            teacher_log_prob = select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)
            prefix_loss_mat = chosen_token_forward_kl_matrix(
                student_log_probs=log_prob,
                teacher_log_probs=teacher_log_prob,
            )
        prefix_loss = agg_loss(
            loss_mat=prefix_loss_mat,
            loss_mask=prefix_loss_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss + prefix_loss * teacher_prefix_forward_weight(policy_loss_cfg)
    if bool(_cfg_get(actor.config, "use_kl_loss", False)) and "math_teacher_log_prob" in model_inputs:
        if kl_coef != 0:
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["math_teacher_log_prob"],
                kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=distill_response_mask,
                loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            )
            policy_loss = policy_loss + kl_loss * kl_coef
    return policy_loss * float(loss_scale_factor)


def _actor_micro_batch_token_loss_scores(
    actor: Any,
    micro_batch: DataProto,
    *,
    on_policy: bool,
) -> tuple[torch.Tensor | None, str]:
    """Return a detached per-response-token loss score matrix for token selection."""

    from verl.trainer.ppo.core_algos import kl_penalty

    try:
        with torch.no_grad():
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = model_inputs["response_mask"]
            policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
            topk_distill_active = is_topk_distill_enabled(policy_loss_cfg)
            use_renormalized_support = (
                topk_distill_active
                and topk_distill_uses_renormalized_support(policy_loss_cfg)
            )
            effective_topk_logprob_mode = topk_distill_logprob_mode(policy_loss_cfg)
            if use_renormalized_support:
                effective_topk_logprob_mode = TOPK_LOGPROB_MODE_SPARSE
            kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
            needs_log_probs = not topk_distill_active or (
                bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0
            )
            forward_kwargs = {
                "temperature": float(micro_batch.meta_info.get("temperature", 1.0)),
                "calculate_entropy": False,
                "calculate_log_probs": needs_log_probs,
            }
            topk_support_ids = None
            teacher_support_log_probs = None
            if topk_distill_active:
                topk_support_ids, teacher_support_log_probs = _selected_topk_support_from_inputs(
                    model_inputs,
                    policy_loss_cfg,
                )
                forward_kwargs["gather_topk_ids"] = topk_support_ids
                forward_kwargs["normalize_gathered_topk"] = not use_renormalized_support
                forward_kwargs["topk_logprob_chunk_size"] = topk_distill_logprob_chunk_size(policy_loss_cfg)
                forward_kwargs["topk_logprob_mode"] = effective_topk_logprob_mode
                forward_kwargs["return_extra"] = True

            forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
            if topk_distill_active:
                _entropy, log_prob, _topk_ids, _topk_log_probs, student_topk_log_probs = forward_output
            else:
                _entropy, log_prob = forward_output

            prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
                model_inputs,
                response_mask,
                policy_loss_cfg,
            )
            distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask

            if topk_distill_active:
                loss_mat = topk_distill_loss_matrix(
                    student_topk_log_probs=student_topk_log_probs,
                    teacher_topk_log_probs=teacher_support_log_probs,
                    mode=resolved_topk_distill_mode(policy_loss_cfg),
                    include_tail=topk_distill_include_tail(policy_loss_cfg),
                    temperature=topk_distill_temperature(policy_loss_cfg),
                )
                loss_mat = loss_mat * topk_distill_weight(policy_loss_cfg)
                source = "topk_distill_loss"
            else:
                if bool(_cfg_get(actor.config, "use_rollout_log_probs", False)):
                    old_log_prob = model_inputs["old_log_probs"]
                elif on_policy:
                    old_log_prob = log_prob.detach()
                else:
                    old_log_prob = model_inputs["old_log_probs"]
                teacher_log_prob = select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)
                loss_mat = old_log_prob.float() - teacher_log_prob.float()
                source = "chosen_token_reverse_kl_proxy"

            if teacher_prefix_active and prefix_loss_mask.detach().sum().item() > 0:
                if topk_distill_active:
                    prefix_loss_mat = topk_distill_loss_matrix(
                        student_topk_log_probs=student_topk_log_probs,
                        teacher_topk_log_probs=teacher_support_log_probs,
                        mode=TOPK_RENORMALIZED_FORWARD_KL,
                        include_tail=False,
                        temperature=topk_distill_temperature(policy_loss_cfg),
                    )
                else:
                    teacher_log_prob = select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)
                    prefix_loss_mat = chosen_token_forward_kl_matrix(
                        student_log_probs=log_prob,
                        teacher_log_probs=teacher_log_prob,
                    )
                loss_mat = loss_mat + prefix_loss_mat * prefix_loss_mask * teacher_prefix_forward_weight(policy_loss_cfg)

            if bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0 and "math_teacher_log_prob" in model_inputs:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=model_inputs["math_teacher_log_prob"],
                    kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
                )
                loss_mat = loss_mat + kld * kl_coef

            return (loss_mat.detach().float() * distill_response_mask.detach().float()).cpu(), source
    except Exception as exc:
        return None, f"unavailable_{type(exc).__name__}"


class SequentialBackwardDomainGradientTracker:
    """Track domain and sample gradient geometry from the real actor backward pass."""

    def __init__(self, actor: Any, cfg: dict[str, Any]):
        self.actor = actor
        self.cfg = cfg
        self.domains = list(cfg.get("domains", []))
        self.storage_dtype = str(cfg.get("storage_dtype", "float32"))
        self.step = int(cfg.get("step", 0) or 0)
        self.domain_gradient_enabled = bool(cfg.get("domain_gradient_enabled", cfg.get("enabled", False)))
        self.sample_gradient_enabled = bool(cfg.get("sample_gradient_enabled", False))
        self._distributed_world_size = _distributed_world_size()
        requested_sample_norm = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_norm_enabled", True))
        requested_sample_cos = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_cos_enabled", False))
        requested_token_gradient = bool(cfg.get("token_gradient_enabled", False))
        self._sample_gradient_uses_full_local_params = _actor_has_full_local_params_for_sample_gradient(actor)
        self._sample_gradient_norm_distributed_unsupported = (
            requested_sample_norm and not self._sample_gradient_uses_full_local_params
        )
        self._sample_gradient_cos_distributed_unsupported = (
            requested_sample_cos and not self._sample_gradient_uses_full_local_params
        )
        self._token_gradient_distributed_unsupported = (
            requested_token_gradient and not self._sample_gradient_uses_full_local_params
        )
        self.sample_norm_enabled = requested_sample_norm and not self._sample_gradient_norm_distributed_unsupported
        self.sample_cos_enabled = requested_sample_cos and not self._sample_gradient_cos_distributed_unsupported
        self.token_gradient_enabled = requested_token_gradient and not self._token_gradient_distributed_unsupported
        self.token_gradient_top_k = max(
            1,
            int(cfg.get("token_gradient_top_k", 100) or 100),
        )
        self.token_gradient_gap_selection_enabled = bool(
            cfg.get("token_gradient_gap_selection_enabled", True)
        )
        self.token_gradient_gap_abs_selection_enabled = bool(
            cfg.get("token_gradient_gap_abs_selection_enabled", True)
        )
        self.token_gradient_loss_abs_selection_enabled = bool(
            cfg.get("token_gradient_loss_abs_selection_enabled", True)
        )
        token_gradient_top_p = cfg.get("token_gradient_top_p", 0.10)
        self.token_gradient_top_p = min(
            1.0,
            max(0.0, float(0.10 if token_gradient_top_p is None else token_gradient_top_p)),
        )
        self.token_gradient_strict_grad_restore = self.token_gradient_enabled and bool(
            cfg.get("token_gradient_strict_grad_restore", False)
        )
        self._sample_gradient_distributed_unsupported = not (
            self.sample_norm_enabled
            or self.sample_cos_enabled
            or self.token_gradient_enabled
            or not (requested_sample_norm or requested_sample_cos or requested_token_gradient)
        )
        self.sample_log_sample_level = (
            bool(cfg.get("sample_gradient_log_sample_level", True))
            and self.sample_norm_enabled
        )
        self.output_dir = str(cfg.get("output_dir", ""))
        self._sample_counts: dict[str, int] = {}
        self._first_domain_chunks: tuple[torch.Tensor, ...] | None = None
        self._expected_first_domain_samples: int | None = None
        self._started_at = 0.0
        self._domain_partition_meta = cfg.get("domain_partition", {})
        self._prepared_supported = len(self.domains) in (1, 2)
        self._hook_handles: list[Any] = []
        self._active_norm_parts: list[torch.Tensor] | None = None
        self._active_norm_context: dict[str, Any] | None = None
        self._sample_records: list[dict[str, Any]] = []
        self._sample_candidates: dict[str, list[dict[str, Any]]] = {}
        self._token_gradient_candidates: dict[str, list[dict[str, Any]]] = {}
        self._token_gradient_selected_sample_ids: dict[str, set[str]] = {}
        self._micro_batch_index = 0
        self._sample_zero_norm_count = 0

    def _domain_target_storage_dtype(self) -> str:
        return self.storage_dtype

    def prepare_micro_batches(
        self,
        micro_batches: list[Any],
        *,
        batch_idx_list: list[list[int]] | None = None,
    ) -> list[tuple[str | None, Any]]:
        original_micro_batches = list(micro_batches)
        self._expected_first_domain_samples = None
        partition_meta = self._domain_partition_meta if isinstance(self._domain_partition_meta, dict) else {}
        self._inject_partition_labels(
            original_micro_batches,
            partition_meta,
            batch_idx_list=batch_idx_list,
        )
        partition_aligned = (
            bool(partition_meta.get("aligned", False))
            if partition_meta
            else self._distributed_world_size <= 1
        )
        locally_supported = len(self.domains) in (1, 2) and partition_aligned
        buckets: dict[str, list[tuple[str, Any]]] = {domain: [] for domain in self.domains}
        if locally_supported:
            for micro_batch in original_micro_batches:
                labels = _teacher_labels(micro_batch)
                unique_labels = set(labels)
                if len(unique_labels) != 1:
                    locally_supported = False
                    break
                domain = next(iter(unique_labels))
                if domain not in buckets:
                    locally_supported = False
                    break
                buckets[domain].append((domain, micro_batch))
        if locally_supported:
            locally_supported = all(buckets[domain] for domain in self.domains)

        globally_supported = _all_ranks_true(locally_supported)
        domain_sample_counts = [
            sum(len(micro_batch) for _, micro_batch in buckets[domain]) if locally_supported else 0
            for domain in self.domains
        ]
        domain_micro_batch_counts = [
            len(buckets[domain]) if locally_supported else 0 for domain in self.domains
        ]
        counts_aligned = False
        if globally_supported:
            micro_batch_counts_aligned = _all_ranks_equal_ints(domain_micro_batch_counts)
            sample_counts_aligned = _all_ranks_equal_ints(domain_sample_counts)
            meta_sample_counts = partition_meta.get("domain_block_sample_counts", {})
            meta_counts_match = all(
                int(meta_sample_counts.get(domain, -1)) == domain_sample_counts[idx]
                for idx, domain in enumerate(self.domains)
            ) if meta_sample_counts else self._distributed_world_size <= 1
            counts_aligned = micro_batch_counts_aligned and sample_counts_aligned and meta_counts_match
        self._prepared_supported = globally_supported and counts_aligned
        if self.domain_gradient_enabled and not self._prepared_supported:
            self.domain_gradient_enabled = False
        if not self._prepared_supported:
            self._expected_first_domain_samples = None
            self._first_domain_chunks = None
            return [(None, item) for item in original_micro_batches]

        if partition_meta.get("domain_order") not in (None, list(self.domains)):
            self.domain_gradient_enabled = False
            self._prepared_supported = False
            return [(None, item) for item in original_micro_batches]

        self._expected_first_domain_samples = domain_sample_counts[0]
        ordered: list[tuple[str | None, Any]] = []
        for domain in self.domains:
            ordered.extend(buckets[domain])
        return ordered

    def _inject_partition_labels(
        self,
        micro_batches: list[Any],
        partition_meta: dict[str, Any],
        *,
        batch_idx_list: list[list[int]] | None,
    ) -> None:
        if not partition_meta or not bool(partition_meta.get("aligned", False)):
            return
        if not batch_idx_list or len(batch_idx_list) != len(micro_batches):
            return
        domains = list(partition_meta.get("domain_order") or self.domains)
        counts_by_domain = partition_meta.get("domain_block_sample_counts", {})
        if not domains or not isinstance(counts_by_domain, dict):
            return
        boundaries: list[tuple[int, int, str]] = []
        start = 0
        for domain in domains:
            count = int(counts_by_domain.get(domain, 0) or 0)
            if count <= 0:
                continue
            end = start + count
            boundaries.append((start, end, str(domain)))
            start = end
        if not boundaries:
            return
        for micro_batch, indices in zip(micro_batches, batch_idx_list):
            labels: list[str] = []
            for index in indices:
                label = None
                idx = int(index)
                for start, end, domain in boundaries:
                    if start <= idx < end:
                        label = domain
                        break
                labels.append(label or "unknown")
            if not labels or all(label == "unknown" for label in labels):
                continue
            current_labels = _teacher_labels(micro_batch)
            if any(label != "unknown" for label in current_labels):
                continue
            label_array = np.array(labels, dtype=object)
            micro_batch.non_tensor_batch["opd_teacher"] = label_array
            micro_batch.non_tensor_batch["domain"] = label_array.copy()

    def start_mini_batch(self) -> None:
        self._sample_counts = {}
        self._first_domain_chunks = None
        self._started_at = time.perf_counter()
        self._sample_records = []
        self._sample_candidates = {}
        self._token_gradient_candidates = {}
        self._token_gradient_selected_sample_ids = {}
        self._micro_batch_index = 0
        self._sample_zero_norm_count = 0
        self._install_sample_norm_hooks()

    def before_backward(
        self,
        domain: str | None,
        micro_batch: Any,
        *,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> None:
        if not (self.sample_norm_enabled or self.token_gradient_enabled):
            return
        labels = _teacher_labels(micro_batch)
        resolved_domain = domain
        if resolved_domain is None and len(set(labels)) == 1:
            resolved_domain = labels[0]
        sample_ids = _sample_ids(
            micro_batch,
            self.step,
            fallback_prefix=f"step{self.step}:micro{self._micro_batch_index}",
        )
        sample_id = sample_ids[0] if sample_ids else f"step{self.step}:micro{self._micro_batch_index}"
        self._active_norm_parts = [] if self.sample_norm_enabled else None
        self._active_norm_context = {
            "step": self.step,
            "domain": resolved_domain or "unknown",
            "sample_id": sample_id,
            "sample_count": len(micro_batch),
            "micro_batch_index": self._micro_batch_index,
            "effective_tokens": _response_token_count(micro_batch),
            "loss_scale_factor": float(loss_scale_factor),
            "on_policy": bool(on_policy),
        }

    def after_backward(self, domain: str | None, sample_count: int, micro_batch: Any | None = None) -> None:
        if self.sample_norm_enabled or self.token_gradient_enabled:
            self._finish_active_sample(micro_batch)
        self._micro_batch_index += 1

        if not self._prepared_supported or domain is None:
            return
        self._sample_counts[domain] = self._sample_counts.get(domain, 0) + sample_count
        if not self.domain_gradient_enabled:
            return
        if domain != self.domains[0] or self._first_domain_chunks is not None:
            return
        expected_count = self._expected_first_domain_samples
        if expected_count is None or self._sample_counts[domain] >= expected_count:
            self._first_domain_chunks = _snapshot_current_grad_chunks(
                self.actor,
                self._domain_target_storage_dtype(),
            )

    def finish_mini_batch(self) -> dict[str, float]:
        finish_started_at = time.perf_counter()
        metrics: dict[str, float] = {
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/audit/sample_gradient_zero_norm_count": 0.0,
            "global/audit/full_gradient_domain_sequential_available": 0.0,
            "global/audit/full_gradient_domain_sequential_unsupported": float(not self._prepared_supported),
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": _max_memory_allocated_gb(),
        }
        replica_count = _gradient_replica_count(self.actor)
        if replica_count > 1:
            metrics["global/audit/full_gradient_replicated_all_reduce"] = 1.0
            metrics["global/audit/full_gradient_replica_count"] = float(replica_count)
        if self._sample_gradient_distributed_unsupported:
            metrics["global/audit/sample_gradient_distributed_unsupported"] = 1.0
            metrics["global/audit/sample_gradient_distributed_world_size"] = float(self._distributed_world_size)
        if self._sample_gradient_norm_distributed_unsupported:
            metrics["global/audit/sample_gradient_norm_distributed_unsupported"] = 1.0
        if self._sample_gradient_cos_distributed_unsupported:
            metrics["global/audit/sample_gradient_cos_distributed_unsupported"] = 1.0
        if self._token_gradient_distributed_unsupported:
            metrics["global/audit/token_gradient_distributed_unsupported"] = 1.0
        first_chunks = self._first_domain_chunks
        self._first_domain_chunks = None
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}

        if self.domain_gradient_enabled and self._prepared_supported and len(self.domains) == 1 and first_chunks is not None:
            domain_metrics, domain_targets = self._finish_single_domain_gradient_metrics(first_chunks)
            metrics.update(domain_metrics)
        elif self.domain_gradient_enabled and self._prepared_supported and len(self.domains) == 2 and first_chunks is not None:
            domain_summary_started_at = time.perf_counter()
            domain_metrics, domain_targets = self._finish_domain_gradient_metrics(first_chunks)
            metrics["global/full_grad_cost/domain_summary_seconds"] = (
                time.perf_counter() - domain_summary_started_at
            )
            metrics.update(domain_metrics)

        metrics.update(self._sample_norm_metrics())
        if self.sample_cos_enabled and domain_targets:
            metrics.update(self._sample_cos_metrics(domain_targets))
        if self.token_gradient_enabled:
            metrics.update(self._token_gradient_metrics(domain_targets))
        metrics["global/audit/sample_gradient_zero_norm_count"] = _all_reduce_sum(
            self._sample_zero_norm_count
        )
        if self.sample_log_sample_level:
            _write_jsonl_rows(self.output_dir, "sample_grad_metrics.jsonl", self._sample_records)
        self._remove_sample_norm_hooks()
        self._active_norm_parts = None
        self._active_norm_context = None
        self._sample_candidates = {}
        self._token_gradient_candidates = {}
        self._token_gradient_selected_sample_ids = {}
        metrics["global/full_grad_cost/finish_mini_batch_seconds"] = time.perf_counter() - finish_started_at
        return metrics

    def _install_sample_norm_hooks(self) -> None:
        if not self.sample_norm_enabled or self._hook_handles:
            return

        def hook(gradient: torch.Tensor) -> torch.Tensor:
            parts = self._active_norm_parts
            if parts is not None:
                parts.append(gradient.detach().float().square().sum())
            return gradient

        for parameter in _trainable_parameters(self.actor):
            self._hook_handles.append(parameter.register_hook(hook))

    def _remove_sample_norm_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _finish_active_sample(self, micro_batch: Any | None) -> None:
        context = self._active_norm_context
        parts = self._active_norm_parts
        self._active_norm_context = None
        self._active_norm_parts = None
        if context is None:
            return
        row: dict[str, Any] = {**context}
        if self.sample_norm_enabled and parts is not None:
            local_sumsq = torch.stack(parts).sum().item() if parts else 0.0
            scale = _current_grad_scale(self.actor)
            if scale != 1.0:
                local_sumsq /= scale * scale
            if self._sample_gradient_uses_full_local_params:
                norm_sumsq = local_sumsq
            else:
                norm_sumsq = _all_reduce_sum(local_sumsq)
            grad_norm = float(max(norm_sumsq, 0.0) ** 0.5)
            row.update(
                {
                    "sample_grad_norm": grad_norm,
                    "computed_for_cos": False,
                    "sample_to_domain_cos": None,
                    "sample_projection_share": None,
                }
            )
            self._sample_records.append(row)
        domain = str(context["domain"])
        stored_micro_batch = None
        if self.sample_cos_enabled and self.sample_norm_enabled and micro_batch is not None:
            stored_micro_batch = _copy_data_proto_rows_to_cpu(
                micro_batch,
                list(range(len(micro_batch))),
            )
            if stored_micro_batch is not None:
                self._sample_candidates.setdefault(domain, []).append(
                    {"row": row, "micro_batch": stored_micro_batch}
                )
        if self.token_gradient_enabled and micro_batch is not None:
            token_domains = [domain] if domain in self.domains else []
            if not token_domains:
                labels = set(_teacher_labels(micro_batch))
                token_domains = [candidate for candidate in self.domains if candidate in labels]
            for token_domain in token_domains:
                self._store_token_gradient_candidates(
                    token_domain,
                    micro_batch,
                    row,
                    stored_micro_batch=stored_micro_batch,
                )

    def _store_token_gradient_candidates(
        self,
        domain: str,
        micro_batch: DataProto,
        context: dict[str, Any],
        *,
        stored_micro_batch: DataProto | None = None,
    ) -> None:
        token_candidates = self._select_token_gradient_candidates(
            micro_batch,
            domain=domain,
            fallback_prefix=str(context.get("sample_id", f"step{self.step}")),
            on_policy=bool(context.get("on_policy", False)),
        )
        if not token_candidates:
            return

        if stored_micro_batch is not None:
            normalized_rows = []
            for row in token_candidates:
                normalized_row = dict(row)
                normalized_row["original_sample_index"] = int(row["sample_index"])
                normalized_row["source_micro_batch_index"] = int(context.get("micro_batch_index", 0) or 0)
                normalized_rows.append(normalized_row)
            self._token_gradient_candidates.setdefault(domain, []).append(
                {
                    "context": dict(context),
                    "micro_batch": stored_micro_batch,
                    "tokens": normalized_rows,
                }
            )
            return

        sample_indices = sorted({int(row["sample_index"]) for row in token_candidates})
        if not sample_indices:
            return
        sample_micro_batch = _copy_data_proto_rows_to_cpu(micro_batch, sample_indices)
        if sample_micro_batch is None:
            return
        index_map = {sample_idx: new_idx for new_idx, sample_idx in enumerate(sample_indices)}
        normalized_rows = []
        for row in token_candidates:
            original_sample_index = int(row["sample_index"])
            normalized_row = dict(row)
            normalized_row["original_sample_index"] = original_sample_index
            normalized_row["sample_index"] = index_map[original_sample_index]
            normalized_row["source_micro_batch_index"] = int(context.get("micro_batch_index", 0) or 0)
            normalized_rows.append(normalized_row)
        self._token_gradient_candidates.setdefault(domain, []).append(
            {
                "context": dict(context),
                "micro_batch": sample_micro_batch,
                "tokens": normalized_rows,
            }
        )

    def _finish_domain_gradient_metrics(
        self,
        first_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        needs_domain_target_chunks = self.sample_cos_enabled or self.token_gradient_enabled
        snapshot = _current_grad_difference_snapshot(
            self.actor,
            first_chunks,
            self._domain_target_storage_dtype() if needs_domain_target_chunks else None,
        )
        if snapshot is None:
            return metrics, domain_targets
        first_norm_sq = snapshot.first_norm_sq
        total_norm_sq = snapshot.total_norm_sq
        first_total_dot = snapshot.first_total_dot
        if first_norm_sq <= 0.0 or total_norm_sq <= 0.0:
            return metrics, domain_targets

        first_domain, second_domain = self.domains[0], self.domains[1]
        first_norm = first_norm_sq**0.5
        total_norm = total_norm_sq**0.5
        second_norm_sq = max(snapshot.second_norm_sq, 0.0)
        second_norm = second_norm_sq**0.5
        first_second_dot = snapshot.first_second_dot
        second_total_dot = snapshot.second_total_dot

        first_safe = _safe_name(first_domain)
        second_safe = _safe_name(second_domain)
        metrics["global/audit/full_gradient_domain_sequential_available"] = 1.0
        metrics[f"{first_safe}/full_grad/grad_norm"] = first_norm
        metrics[f"{first_safe}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(first_domain, 0))
        metrics[f"{second_safe}/full_grad/grad_norm"] = second_norm
        metrics[f"{second_safe}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(second_domain, 0))
        pair = f"{first_safe}_vs_{second_safe}"
        domain_cosine = _safe_cosine(first_second_dot, first_norm, second_norm)
        if domain_cosine is not None:
            metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = domain_cosine
            metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -domain_cosine)

        first_total_cosine = _safe_cosine(first_total_dot, first_norm, total_norm)
        if first_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{first_safe}_vs_total/full_grad_cosine_domain_total"] = first_total_cosine
        second_total_cosine = _safe_cosine(second_total_dot, second_norm, total_norm)
        if second_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{second_safe}_vs_total/full_grad_cosine_domain_total"] = second_total_cosine

        if total_norm_sq > 0:
            metrics[f"global/full_grad_contribution/{first_safe}_to_total/signed_projection_share"] = first_total_dot / total_norm_sq
            metrics[f"global/full_grad_contribution/{second_safe}_to_total/signed_projection_share"] = second_total_dot / total_norm_sq

        if needs_domain_target_chunks and snapshot.second_chunks is not None:
            domain_targets[first_domain] = (first_chunks, first_norm_sq)
            second_target_norm_sq = snapshot.second_target_norm_sq
            if second_target_norm_sq is not None and second_target_norm_sq > 0.0:
                domain_targets[second_domain] = (snapshot.second_chunks, second_target_norm_sq)

        return metrics, domain_targets

    def _finish_single_domain_gradient_metrics(
        self,
        domain_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        if not self.domains:
            return metrics, domain_targets
        norm_sq = 0.0
        for chunk in domain_chunks:
            chunk_float = chunk.float()
            norm_sq += _chunked_vector_dot(chunk_float, chunk_float) or 0.0
            del chunk_float
        norm_sq = _reduce_gradient_scalars(self.actor, [norm_sq])[0]
        if norm_sq <= 0.0:
            return metrics, domain_targets
        domain = self.domains[0]
        safe_domain = _safe_name(domain)
        metrics["global/audit/full_gradient_domain_sequential_available"] = 1.0
        metrics["global/audit/full_gradient_single_domain_target"] = 1.0
        metrics[f"{safe_domain}/full_grad/grad_norm"] = norm_sq**0.5
        metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(
            self._sample_counts.get(domain, 0)
        )
        domain_targets[domain] = (domain_chunks, norm_sq)
        return metrics, domain_targets

    def _sample_norm_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        by_domain: dict[str, list[float]] = {}
        for row in self._sample_records:
            by_domain.setdefault(str(row["domain"]), []).append(float(row["sample_grad_norm"]))
        domain_names = sorted(set(self.domains) | set(_all_gather_list(list(by_domain))))
        for domain in domain_names:
            finite = _finite_values(_all_gather_list(by_domain.get(domain, [])))
            if not finite:
                continue
            safe_domain = _safe_name(domain)
            std = _std(finite) or 0.0
            mean = _mean(finite) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_mean"] = mean
            metrics[f"{safe_domain}/sample_grad/norm_p50"] = _percentile(finite, 50.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_p95"] = _percentile(finite, 95.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_max"] = max(finite)
            metrics[f"{safe_domain}/sample_grad/norm_cv"] = std / (abs(mean) + 1e-12)
            metrics[f"{safe_domain}/sample_grad/sample_count"] = float(len(finite))
        return metrics

    def _sample_cos_metrics(self, domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        structural_unavailable_reason: str | None = None
        for domain, candidates in sorted(self._sample_candidates.items()):
            if domain not in domain_targets:
                continue
            if not candidates:
                continue
            target_chunks, target_norm_sq = domain_targets[domain]
            target_norm = target_norm_sq**0.5
            cos_values: list[float] = []
            share_values: list[float] = []
            availability_values: list[float] = []
            disconnected_values: list[float] = []
            for candidate in candidates:
                row = candidate["row"]
                if structural_unavailable_reason is None:
                    stats = self._recompute_sample_to_domain_stats(
                        candidate["micro_batch"],
                        target_chunks=target_chunks,
                        target_norm=target_norm,
                        target_norm_sq=target_norm_sq,
                        restore_target_map=domain_targets,
                        loss_scale_factor=float(row.get("loss_scale_factor", 1.0)),
                        on_policy=bool(row.get("on_policy", False)),
                    )
                    if stats.get("sample_recompute_autograd_error") == "all_parameters_disconnected":
                        structural_unavailable_reason = "all_parameters_disconnected"
                else:
                    stats = {
                        "sample_to_domain_cos": None,
                        "sample_projection_share": None,
                        "sample_recompute_grad_norm": 0.0,
                        "sample_recompute_non_none_grad_count": 0.0,
                        "sample_recompute_available": 0.0,
                        "sample_recompute_autograd_error": structural_unavailable_reason,
                    }
                row["computed_for_cos"] = True
                row.update(stats)
                cos_value = stats.get("sample_to_domain_cos")
                share_value = stats.get("sample_projection_share")
                disconnected_values.append(
                    float(stats.get("sample_recompute_autograd_error") == "all_parameters_disconnected")
                )
                available = float(
                    stats.get(
                        "sample_recompute_available",
                        cos_value is not None or share_value is not None,
                    )
                )
                availability_values.append(available)
                if cos_value is not None:
                    cos_values.append(float(cos_value))
                if share_value is not None:
                    share_values.append(float(share_value))
            safe_domain = _safe_name(domain)
            global_cos_values = _finite_values(_all_gather_list(cos_values))
            global_share_values = _finite_values(_all_gather_list(share_values))
            global_availability_values = _finite_values(_all_gather_list(availability_values))
            global_disconnected_values = _finite_values(_all_gather_list(disconnected_values))
            if global_availability_values:
                attempted_count = len(global_availability_values)
                available_count = sum(value > 0.5 for value in global_availability_values)
                metrics[f"{safe_domain}/sample_grad_cos/attempted_count"] = float(attempted_count)
                metrics[f"{safe_domain}/sample_grad_cos/unavailable_count"] = float(
                    attempted_count - available_count
                )
                metrics[f"{safe_domain}/sample_grad_cos/valid_frac"] = available_count / attempted_count
            if global_disconnected_values:
                metrics[f"{safe_domain}/sample_grad_cos/all_parameters_disconnected_count"] = float(
                    sum(value > 0.5 for value in global_disconnected_values)
                )
            if global_cos_values:
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_mean"] = _mean(global_cos_values) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_p05"] = _percentile(global_cos_values, 5.0) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_negative_frac"] = float(
                    np.mean([value < 0.0 for value in global_cos_values])
                )
                metrics[f"{safe_domain}/sample_grad_cos/sample_count"] = float(len(global_cos_values))
            if global_share_values:
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_mean"] = _mean(global_share_values) or 0.0
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_min"] = min(global_share_values)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_max"] = max(global_share_values)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_negative_frac"] = float(
                    np.mean([value < 0.0 for value in global_share_values])
                )
                metrics[f"{safe_domain}/sample_grad_contribution/top1_abs_share"] = max(
                    abs(value) for value in global_share_values
                )
                projection_share_sum_across_replicas = sum(global_share_values)
                replica_count = (
                    _gradient_replica_count(self.actor)
                    if self._sample_gradient_uses_full_local_params
                    else 1
                )
                projection_share_sum = projection_share_sum_across_replicas
                metrics[
                    f"{safe_domain}/sample_grad_contribution/projection_share_sum_across_replicas"
                ] = projection_share_sum_across_replicas
                metrics[
                    f"{safe_domain}/sample_grad_contribution/projection_share_replica_count"
                ] = float(replica_count)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum"] = projection_share_sum
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum_error"] = abs(
                    projection_share_sum - 1.0
                )
        return metrics

    def _token_gradient_metrics(
        self,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> dict[str, float]:
        started_at = time.perf_counter()
        metrics: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        local_rank = _distributed_rank()
        world_size = self._distributed_world_size
        parameters_for_final_restore = _trainable_parameters(self.actor)
        final_grad_dtypes = _parameter_grad_dtypes(parameters_for_final_restore)
        token_recompute_attempted = False
        local_records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        local_metadata: list[dict[str, Any]] = []
        local_supported_targets = [str(domain) for domain in domain_targets]
        supported_target_counts: dict[str, int] = {}
        for domain in _all_gather_list(local_supported_targets):
            supported_target_counts[str(domain)] = supported_target_counts.get(str(domain), 0) + 1
        globally_supported_targets = {
            domain for domain, count in supported_target_counts.items() if count == max(world_size, 1)
        }

        for domain, candidates in sorted(self._token_gradient_candidates.items()):
            if domain not in globally_supported_targets:
                continue
            for candidate_index, candidate in enumerate(candidates):
                micro_batch = candidate["micro_batch"]
                for token_candidate in candidate.get("tokens", []):
                    token_candidate = dict(token_candidate)
                    token_candidate_id = len(local_metadata)
                    token_candidate["candidate_index"] = candidate_index
                    token_candidate["micro_batch"] = micro_batch
                    token_candidate["loss_scale_factor"] = float(candidate["context"].get("loss_scale_factor", 1.0))
                    token_candidate["on_policy"] = bool(candidate["context"].get("on_policy", False))
                    token_candidate["owner_rank"] = local_rank
                    token_candidate["token_candidate_id"] = token_candidate_id
                    key = (local_rank, token_candidate_id)
                    local_records_by_key[key] = token_candidate
                    local_metadata.append(self._token_gradient_metadata(domain, token_candidate))

        global_metadata = _all_gather_list(local_metadata)
        domains = sorted(
            {
                str(row.get("domain"))
                for row in global_metadata
                if str(row.get("domain")) in globally_supported_targets
            }
        )
        for domain in domains:
            domain_rows: list[dict[str, Any]] = []
            token_metadata = [
                row for row in global_metadata if str(row.get("domain")) == domain
            ]
            selected_samples_by_domain = len(
                self._token_sample_keys(token_metadata)
            )
            total_gap_mass = sum(
                self._token_score_mass_value(row, "gap") for row in token_metadata
            )
            total_gap_abs_mass = sum(float(row.get("gap_abs", 0.0)) for row in token_metadata)
            total_loss_abs_mass = sum(float(row.get("loss_abs", 0.0) or 0.0) for row in token_metadata)
            for selection_name, selection_score_key, selected_metadata in self._token_score_selections(token_metadata):
                if not selected_metadata:
                    continue
                local_selected_tokens = self._local_tokens_for_global_selection(
                    selected_metadata,
                    local_records_by_key,
                    local_rank=local_rank,
                )
                stats = self._recompute_token_selection_gradient_stats(
                    local_selected_tokens,
                    target_map=domain_targets,
                    restore_grads=self.token_gradient_strict_grad_restore,
                )
                token_recompute_attempted = True
                other_domain = self._other_domain(domain, domain_targets)
                own_cos = stats.get(f"{_safe_name(domain)}_cos")
                other_cos = stats.get(f"{_safe_name(other_domain)}_cos") if other_domain is not None else None
                own_projection = stats.get(f"{_safe_name(domain)}_projection_share")
                other_projection = (
                    stats.get(f"{_safe_name(other_domain)}_projection_share")
                    if other_domain is not None
                    else None
                )
                selected_gap_mass = sum(
                    self._token_score_mass_value(row, "gap") for row in selected_metadata
                )
                selected_gap_abs_mass = sum(float(row.get("gap_abs", 0.0)) for row in selected_metadata)
                selected_loss_abs_mass = sum(float(row.get("loss_abs", 0.0) or 0.0) for row in selected_metadata)
                selected_score_mass = sum(
                    self._token_score_mass_value(row, selection_score_key) for row in selected_metadata
                )
                total_score_mass = sum(
                    self._token_score_mass_value(row, selection_score_key) for row in token_metadata
                )
                rank_selected_token_counts = self._rank_token_counts(selected_metadata)
                row = {
                    "step": self.step,
                    "domain": domain,
                    "selection": selection_name,
                    "selection_score": selection_score_key,
                    "selection_scope": "global",
                    "rank": "global",
                    "world_size": world_size,
                    "other_domain": other_domain,
                    "own_domain_cos": own_cos,
                    "other_domain_cos": other_cos,
                    "conflict_to_other": max(0.0, -float(other_cos)) if other_cos is not None else None,
                    "own_projection_share": own_projection,
                    "other_projection_share": other_projection,
                    "selected_token_count": float(len(selected_metadata)),
                    "selected_sample_count": float(len(self._token_sample_keys(selected_metadata))),
                    "selected_rank_count": float(len(rank_selected_token_counts)),
                    "rank_selected_token_counts": rank_selected_token_counts,
                    "local_selected_token_count": float(len(local_selected_tokens)),
                    "global_candidate_token_count": float(len(token_metadata)),
                    "global_candidate_sample_count": float(selected_samples_by_domain),
                    "global_candidate_gap_mass": total_gap_mass,
                    "global_candidate_gap_abs_mass": total_gap_abs_mass,
                    "global_candidate_loss_abs_mass": total_loss_abs_mass,
                    "global_candidate_score_mass": total_score_mass,
                    "global_candidate_scope": "all_valid_response_tokens",
                    "selected_gap_mass": selected_gap_mass,
                    "selected_gap_mass_frac": selected_gap_mass / total_gap_mass
                    if total_gap_mass > 0.0
                    else 0.0,
                    "selected_gap_mean": selected_gap_mass / max(len(selected_metadata), 1),
                    "selected_gap_abs_mass": selected_gap_abs_mass,
                    "selected_gap_abs_mass_frac": selected_gap_abs_mass / total_gap_abs_mass
                    if total_gap_abs_mass > 0.0
                    else 0.0,
                    "selected_gap_abs_mean": selected_gap_abs_mass / max(len(selected_metadata), 1),
                    "selected_loss_abs_mass": selected_loss_abs_mass,
                    "selected_loss_abs_mass_frac": selected_loss_abs_mass / total_loss_abs_mass
                    if total_loss_abs_mass > 0.0
                    else 0.0,
                    "selected_loss_abs_mean": selected_loss_abs_mass / max(len(selected_metadata), 1),
                    "selected_score_mass": selected_score_mass,
                    "selected_score_mass_frac": selected_score_mass / total_score_mass
                    if total_score_mass > 0.0
                    else 0.0,
                    "selected_score_mean": selected_score_mass / max(len(selected_metadata), 1),
                    **stats,
                }
                domain_rows.append(row)
            rows.extend(domain_rows)
            metrics.update(self._summarize_token_gradient_rows(domain, domain_rows))
        if token_recompute_attempted and not self.token_gradient_strict_grad_restore:
            _restore_parameter_grads_from_targets(
                parameters_for_final_restore,
                domain_targets,
                grad_dtypes=final_grad_dtypes,
            )
        if rows and local_rank == 0:
            _write_jsonl_rows(self.output_dir, "token_grad_metrics.jsonl", rows)
        total_seconds = _all_reduce_values_max([time.perf_counter() - started_at])[0]
        metrics.update(self._summarize_global_token_gradient_cost(rows, total_seconds))
        return metrics

    def _token_gradient_metadata(self, domain: str, token: dict[str, Any]) -> dict[str, Any]:
        metadata_keys = (
            "sample_id",
            "sample_index",
            "position",
            "rank_in_sample",
            "gap",
            "gap_signed",
            "gap_abs",
            "teacher_logp",
            "student_logp",
            "loss_signed",
            "loss_abs",
            "loss_score_source",
            "effective_tokens",
            "token_id",
            "original_sample_index",
            "source_micro_batch_index",
            "owner_rank",
            "token_candidate_id",
        )
        metadata = {"domain": domain}
        for key in metadata_keys:
            if key in token:
                metadata[key] = token[key]
        return _json_safe(metadata)

    def _local_tokens_for_global_selection(
        self,
        selected_metadata: list[dict[str, Any]],
        local_records_by_key: dict[tuple[int, int], dict[str, Any]],
        *,
        local_rank: int,
    ) -> list[dict[str, Any]]:
        local_tokens: list[dict[str, Any]] = []
        for row in selected_metadata:
            owner_rank = int(row.get("owner_rank", -1))
            if owner_rank != local_rank:
                continue
            key = (owner_rank, int(row.get("token_candidate_id", -1)))
            token = local_records_by_key.get(key)
            if token is not None:
                local_tokens.append(token)
        return local_tokens

    def _rank_token_counts(self, selected_metadata: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in selected_metadata:
            rank = str(int(row.get("owner_rank", -1)))
            counts[rank] = counts.get(rank, 0) + 1
        return counts

    def _token_sample_keys(self, token_metadata: list[dict[str, Any]]) -> set[tuple[str, ...]]:
        keys: set[tuple[str, ...]] = set()
        for row in token_metadata:
            owner_rank = str(row.get("owner_rank", -1))
            sample_id = str(row.get("sample_id", ""))
            source_micro_batch_index = row.get("source_micro_batch_index")
            original_sample_index = row.get("original_sample_index", row.get("sample_index"))
            if source_micro_batch_index is not None and original_sample_index is not None:
                keys.add((owner_rank, str(source_micro_batch_index), str(original_sample_index), sample_id))
            elif sample_id:
                keys.add((owner_rank, sample_id))
            else:
                keys.add((owner_rank, str(row.get("token_candidate_id", -1))))
        return keys

    def _token_score_selections(
        self,
        token_records: list[dict[str, Any]],
    ) -> list[tuple[str, str, list[dict[str, Any]]]]:
        if not token_records:
            return []
        selections: list[tuple[str, str, list[dict[str, Any]]]] = []
        score_keys: list[str] = []
        if self.token_gradient_gap_selection_enabled:
            score_keys.append("gap")
        if self.token_gradient_gap_abs_selection_enabled:
            score_keys.append("gap_abs")
        if self.token_gradient_loss_abs_selection_enabled:
            score_keys.append("loss_abs")
        for score_key in score_keys:
            scored_records = [
                row
                for row in token_records
                if row.get(score_key) is not None and math.isfinite(float(row.get(score_key, 0.0)))
            ]
            if not scored_records:
                continue
            sorted_records = sorted(
                scored_records,
                key=lambda row: (
                    -float(row.get(score_key, 0.0)),
                    int(row.get("owner_rank", 0)),
                    int(row.get("token_candidate_id", 0)),
                ),
            )
            top_k = sorted_records[: min(self.token_gradient_top_k, len(sorted_records))]
            total_mass = sum(self._token_score_mass_value(row, score_key) for row in sorted_records)
            top_p_selection: list[dict[str, Any]] = []
            top_p_threshold = total_mass * self.token_gradient_top_p
            if total_mass > 0.0:
                running = 0.0
                for row in sorted_records:
                    top_p_selection.append(row)
                    running += self._token_score_mass_value(row, score_key)
                    if running >= top_p_threshold:
                        break
            else:
                top_p_selection = sorted_records[:1]
            selections.append((f"top{self.token_gradient_top_k}_{score_key}", score_key, top_k))
            selections.append((self._top_p_selection_name(score_key), score_key, top_p_selection))
        return selections

    def _token_score_mass_value(self, row: dict[str, Any], score_key: str) -> float:
        value = float(row.get(score_key, 0.0) or 0.0)
        if score_key == "gap":
            return max(0.0, value)
        return value

    def _gap_abs_token_selections(
        self,
        token_records: list[dict[str, Any]],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        return [
            (selection, rows)
            for selection, score_key, rows in self._token_score_selections(token_records)
            if score_key == "gap_abs"
        ]

    def _top_p_selection_name(self, score_key: str) -> str:
        percent = self.token_gradient_top_p * 100.0
        rounded = round(percent)
        if abs(percent - rounded) < 1e-6:
            percent_label = str(int(rounded))
        else:
            percent_label = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"topp{percent_label}_{score_key}_mass"

    def _select_token_gradient_candidates(
        self,
        micro_batch: DataProto,
        *,
        domain: str,
        fallback_prefix: str | None = None,
        on_policy: bool = False,
    ) -> list[dict[str, Any]]:
        if micro_batch.batch is None:
            return []
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        if "old_log_probs" not in model_inputs or "math_teacher_log_prob" not in model_inputs:
            return []
        old_log_probs = model_inputs["old_log_probs"].detach().float().cpu()
        response_mask = model_inputs["response_mask"].detach().float().cpu()
        math_teacher = model_inputs["math_teacher_log_prob"].detach().float().cpu()
        code_teacher = model_inputs.get("code_teacher_log_prob", math_teacher)
        if hasattr(code_teacher, "detach"):
            code_teacher = code_teacher.detach().float().cpu()
        else:
            code_teacher = math_teacher
        labels = _labels_from_inputs(model_inputs, int(old_log_probs.shape[0]))
        sample_ids = _sample_ids(micro_batch, self.step, fallback_prefix=fallback_prefix)
        token_ids = _response_token_id_matrix_from_inputs(model_inputs, response_mask)
        loss_scores = None
        loss_score_source = "disabled"
        if self.token_gradient_loss_abs_selection_enabled:
            loss_scores, loss_score_source = _actor_micro_batch_token_loss_scores(
                self.actor,
                micro_batch,
                on_policy=on_policy,
            )
        if loss_scores is not None:
            loss_scores = loss_scores.to(dtype=torch.float32, device=response_mask.device)
        rows: list[dict[str, Any]] = []
        for sample_idx, label in enumerate(labels):
            if label != domain:
                continue
            selected_teacher = code_teacher[sample_idx] if label == "code" else math_teacher[sample_idx]
            gap_signed = (selected_teacher - old_log_probs[sample_idx]) * response_mask[sample_idx]
            gap_abs = gap_signed.abs()
            valid_positions = torch.nonzero(response_mask[sample_idx] > 0, as_tuple=False).reshape(-1)
            if valid_positions.numel() == 0:
                continue
            valid_scores = gap_abs[valid_positions]
            sorted_offsets = torch.argsort(valid_scores, descending=True)
            for rank, offset in enumerate(sorted_offsets.tolist(), start=1):
                position = int(valid_positions[offset].item())
                gap_abs_value = float(valid_scores[offset].detach().cpu().item())
                gap_signed_value = float(gap_signed[position].detach().cpu().item())
                row: dict[str, Any] = {
                    "sample_id": sample_ids[sample_idx],
                    "sample_index": int(sample_idx),
                    "position": position,
                    "rank_in_sample": rank,
                    "gap": gap_signed_value,
                    "gap_signed": gap_signed_value,
                    "gap_abs": gap_abs_value,
                    "teacher_logp": float(selected_teacher[position].detach().cpu().item()),
                    "student_logp": float(old_log_probs[sample_idx, position].detach().cpu().item()),
                    "effective_tokens": float(response_mask[sample_idx].sum().detach().cpu().item()),
                }
                if loss_scores is not None and tuple(loss_scores.shape) == tuple(response_mask.shape):
                    loss_signed_value = float(loss_scores[sample_idx, position].detach().cpu().item())
                    row["loss_signed"] = loss_signed_value
                    row["loss_abs"] = abs(loss_signed_value)
                    row["loss_score_source"] = loss_score_source
                if token_ids is not None:
                    row["token_id"] = int(token_ids[sample_idx, position].detach().cpu().item())
                rows.append(row)
        return rows

    def _other_domain(
        self,
        domain: str,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> str | None:
        for candidate in self.domains:
            if candidate != domain and candidate in domain_targets:
                return candidate
        for candidate in sorted(domain_targets):
            if candidate != domain:
                return candidate
        return None

    def _recompute_token_gradient_stats(
        self,
        micro_batch: DataProto,
        *,
        token_mask: torch.Tensor,
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        loss_scale_factor: float,
        on_policy: bool,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        loss_agg_mode = str(_cfg_get(self.actor.config, "loss_agg_mode", "token-mean"))
        active_position = torch.nonzero(token_mask.detach() > 0, as_tuple=False)
        if active_position.numel() == 0:
            return {
                "token_grad_available": 0.0,
                "token_grad_norm": None,
                "token_grad_autograd_error": "empty_token_mask",
                "token_grad_seconds": time.perf_counter() - started_at,
            }
        contribution_scale = _token_mask_contribution_scale(
            micro_batch.batch["response_mask"],
            token_mask,
            loss_agg_mode,
        )
        if contribution_scale <= 0.0:
            return {
                "token_grad_available": 0.0,
                "token_grad_norm": None,
                "token_grad_autograd_error": "zero_contribution_scale",
                "token_grad_seconds": time.perf_counter() - started_at,
            }
        autograd_started_at = time.perf_counter()
        try:
            loss = _actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=loss_scale_factor * contribution_scale,
                on_policy=on_policy,
                safe_logprob_backward=True,
                response_mask_override=token_mask,
            )
            gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
            autograd_error: str | None = None
        except Exception as exc:
            gradients = tuple(None for _ in parameters)
            autograd_error = type(exc).__name__
        autograd_seconds = time.perf_counter() - autograd_started_at
        non_none_grad_count = sum(gradient is not None for gradient in gradients)
        if non_none_grad_count == 0 and autograd_error is None:
            autograd_error = "all_parameters_disconnected"
        fallback_stats: tuple[float | None, dict[str, float] | None] | None = None
        fallback_seconds = 0.0
        fallback_used = 0.0
        restore_diff_stats: dict[str, float] = {}
        if non_none_grad_count == 0 and _target_map_matches_parameters(parameters, target_map):
            fallback_started_at = time.perf_counter()
            (
                fallback_local_norm_sq,
                fallback_local_dots,
                fallback_non_none_count,
                fallback_error,
                restore_diff_stats,
            ) = self._recompute_token_gradient_stats_with_backward(
                micro_batch,
                token_mask=token_mask,
                target_map=target_map,
                loss_scale_factor=loss_scale_factor * contribution_scale,
                on_policy=on_policy,
            )
            fallback_seconds = time.perf_counter() - fallback_started_at
            if fallback_non_none_count > 0:
                non_none_grad_count = fallback_non_none_count
                autograd_error = None
                fallback_stats = (fallback_local_norm_sq, fallback_local_dots)
                fallback_used = 1.0
            elif fallback_error is not None:
                autograd_error = fallback_error
        if fallback_stats is None:
            local_norm_sq, local_dots = self._grad_multi_target_stats_from_tensors(gradients, target_map)
        else:
            local_norm_sq, local_dots = fallback_stats
        if local_norm_sq is None or local_dots is None:
            return {
                "token_grad_available": 0.0,
                "token_grad_norm": None,
                "token_grad_non_none_grad_count": float(non_none_grad_count),
                "token_grad_autograd_error": autograd_error or "parameter_target_mismatch",
                "token_loss_contribution_scale": contribution_scale,
                "token_grad_seconds": time.perf_counter() - started_at,
                "token_grad_autograd_seconds": autograd_seconds,
                "token_grad_backward_fallback_seconds": fallback_seconds,
                "token_grad_backward_fallback_used": fallback_used,
                **restore_diff_stats,
            }
        if self._sample_gradient_uses_full_local_params:
            norm_sq = max(local_norm_sq, 0.0)
            dots = local_dots
        else:
            norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
            dots = {target_domain: _all_reduce_sum(dot) for target_domain, dot in local_dots.items()}
        token_norm = norm_sq**0.5
        available = non_none_grad_count > 0 and token_norm > 0.0
        stats: dict[str, Any] = {
            "token_grad_available": float(available),
            "token_grad_norm": token_norm if available else None,
            "token_grad_non_none_grad_count": float(non_none_grad_count),
            "token_grad_autograd_error": autograd_error,
            "token_loss_contribution_scale": contribution_scale,
            "token_grad_seconds": time.perf_counter() - started_at,
            "token_grad_autograd_seconds": autograd_seconds,
            "token_grad_backward_fallback_seconds": fallback_seconds,
            "token_grad_backward_fallback_used": fallback_used,
            **restore_diff_stats,
        }
        for target_domain, dot in dots.items():
            _chunks, target_norm_sq = target_map[target_domain]
            target_norm = target_norm_sq**0.5
            safe_domain = _safe_name(target_domain)
            cosine = dot / (token_norm * target_norm) if available and target_norm > 0.0 else None
            projection_share = dot / target_norm_sq if available and target_norm_sq > 0.0 else None
            stats[f"{safe_domain}_cos"] = cosine
            stats[f"{safe_domain}_projection_share"] = projection_share
        return stats

    def _recompute_token_gradient_stats_with_backward(
        self,
        micro_batch: DataProto,
        *,
        token_mask: torch.Tensor,
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        loss_scale_factor: float,
        on_policy: bool,
    ) -> tuple[float | None, dict[str, float] | None, int, str | None, dict[str, float]]:
        parameters = _trainable_parameters(self.actor)
        grad_dtypes = _parameter_grad_dtypes(parameters)
        debug_restore_diff = self.token_gradient_strict_grad_restore
        grad_snapshot = (
            _snapshot_parameter_grads(parameters)
            if debug_restore_diff
            else None
        )
        pre_diff = _parameter_grad_target_diff_stats(parameters, target_map) if debug_restore_diff else {}
        post_diff: dict[str, float] = {}
        original_diff: dict[str, float] | None = None
        local_norm_sq: float | None = None
        local_dots: dict[str, float] | None = None
        non_none_grad_count = 0
        error: str | None = None
        _clear_parameter_grads(parameters)
        try:
            with _actor_no_sync_context(self.actor):
                loss = _actor_micro_batch_loss(
                    self.actor,
                    micro_batch,
                    loss_scale_factor=loss_scale_factor,
                    on_policy=on_policy,
                    safe_logprob_backward=False,
                    response_mask_override=token_mask,
                )
                loss.backward()
            gradients = tuple(parameter.grad for parameter in parameters)
            non_none_grad_count = sum(gradient is not None for gradient in gradients)
            local_norm_sq, local_dots = self._grad_multi_target_stats_from_tensors(gradients, target_map)
            del gradients
        except Exception as exc:
            error = f"backward_{type(exc).__name__}"
        finally:
            if grad_snapshot is not None:
                _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)
                if debug_restore_diff:
                    original_diff = _parameter_grad_snapshot_diff_stats(parameters, grad_snapshot)
            else:
                _restore_parameter_grads_from_targets(parameters, target_map, grad_dtypes=grad_dtypes)
            if debug_restore_diff:
                post_diff = _parameter_grad_target_diff_stats(parameters, target_map)
        restore_stats = {
            "token_grad_restore_pre_target_rel_l2": pre_diff["rel_l2"],
            "token_grad_restore_pre_target_max_abs": pre_diff["max_abs"],
            "token_grad_restore_post_target_rel_l2": post_diff["rel_l2"],
            "token_grad_restore_post_target_max_abs": post_diff["max_abs"],
            "token_grad_restore_target_norm": pre_diff["target_norm"],
        }
        if original_diff is not None:
            restore_stats.update(
                {
                    "token_grad_restore_original_rel_l2": original_diff["rel_l2"],
                    "token_grad_restore_original_max_abs": original_diff["max_abs"],
                    "token_grad_restore_original_norm": original_diff["snapshot_norm"],
                }
            )
        return local_norm_sq, local_dots, non_none_grad_count, error, restore_stats

    def _recompute_token_selection_gradient_stats(
        self,
        selected_tokens: list[dict[str, Any]],
        *,
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        restore_grads: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        grad_dtypes = _parameter_grad_dtypes(parameters) if selected_tokens else tuple()
        grad_snapshot = (
            _snapshot_parameter_grads(parameters)
            if selected_tokens and self.token_gradient_strict_grad_restore and restore_grads
            else None
        )
        pre_diff = (
            _parameter_grad_target_diff_stats(parameters, target_map)
            if selected_tokens and restore_grads
            else {}
        )
        post_diff: dict[str, float] = {}
        original_diff: dict[str, float] | None = None
        local_norm_sq: float | None = 0.0
        local_dots: dict[str, float] | None = {domain: 0.0 for domain in target_map}
        non_none_grad_count = 0
        autograd_error: str | None = None
        autograd_started_at = time.perf_counter()
        loss_agg_mode = str(_cfg_get(self.actor.config, "loss_agg_mode", "token-mean"))

        grouped: dict[int, dict[str, Any]] = {}
        for token in selected_tokens:
            group_key = int(token["candidate_index"])
            group = grouped.setdefault(
                group_key,
                {
                    "micro_batch": token["micro_batch"],
                    "loss_scale_factor": float(token.get("loss_scale_factor", 1.0)),
                    "on_policy": bool(token.get("on_policy", False)),
                    "positions": [],
                },
                )
            group["positions"].append((int(token["sample_index"]), int(token["position"])))

        used_micro_batches: list[DataProto] = []
        if selected_tokens:
            _clear_parameter_grads(parameters)
            try:
                with _actor_no_sync_context(self.actor):
                    for group in grouped.values():
                        micro_batch = group["micro_batch"]
                        used_micro_batches.append(micro_batch)
                        token_mask = torch.zeros_like(micro_batch.batch["response_mask"], dtype=torch.float32)
                        for sample_idx, position in group["positions"]:
                            token_mask[sample_idx, position] = 1.0
                        contribution_scale = _token_mask_contribution_scale(
                            micro_batch.batch["response_mask"],
                            token_mask,
                            loss_agg_mode,
                        )
                        if contribution_scale <= 0.0:
                            continue
                        loss = _actor_micro_batch_loss(
                            self.actor,
                            micro_batch,
                            loss_scale_factor=float(group["loss_scale_factor"]) * contribution_scale,
                            on_policy=bool(group["on_policy"]),
                            safe_logprob_backward=False,
                            response_mask_override=token_mask,
                        )
                        loss.backward()
                gradients = tuple(parameter.grad for parameter in parameters)
                non_none_grad_count = sum(gradient is not None for gradient in gradients)
                local_norm_sq, local_dots = self._grad_multi_target_stats_from_tensors(gradients, target_map)
                del gradients
            except Exception as exc:
                autograd_error = f"backward_{type(exc).__name__}"
            finally:
                if restore_grads:
                    if grad_snapshot is not None:
                        _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)
                        original_diff = _parameter_grad_snapshot_diff_stats(parameters, grad_snapshot)
                    else:
                        _restore_parameter_grads_from_targets(parameters, target_map, grad_dtypes=grad_dtypes)
                    post_diff = _parameter_grad_target_diff_stats(parameters, target_map)
                for micro_batch in used_micro_batches:
                    try:
                        micro_batch.to("cpu")
                    except Exception:
                        pass

        autograd_seconds = time.perf_counter() - autograd_started_at
        restore_stats = {
            "token_grad_restore_pre_target_rel_l2": pre_diff.get("rel_l2", 0.0),
            "token_grad_restore_pre_target_max_abs": pre_diff.get("max_abs", 0.0),
            "token_grad_restore_post_target_rel_l2": post_diff.get("rel_l2", 0.0),
            "token_grad_restore_post_target_max_abs": post_diff.get("max_abs", 0.0),
            "token_grad_restore_target_norm": pre_diff.get("target_norm", 0.0),
        }
        if original_diff is not None:
            restore_stats.update(
                {
                    "token_grad_restore_original_rel_l2": original_diff["rel_l2"],
                    "token_grad_restore_original_max_abs": original_diff["max_abs"],
                    "token_grad_restore_original_norm": original_diff["snapshot_norm"],
                }
            )
        else:
            restore_stats.update(
                {
                    "token_grad_restore_original_rel_l2": 0.0,
                    "token_grad_restore_original_max_abs": 0.0,
                    "token_grad_restore_original_norm": 0.0,
                }
            )
        local_error = autograd_error
        if local_norm_sq is None or local_dots is None:
            local_error = autograd_error or "parameter_target_mismatch"
            local_norm_sq = 0.0
            local_dots = {domain: 0.0 for domain in target_map}

        target_domains = sorted(target_map)
        reduced_values = _all_reduce_values_sum(
            [float(max(local_norm_sq, 0.0)), float(non_none_grad_count)]
            + [float(local_dots.get(domain, 0.0)) for domain in target_domains]
        )
        norm_sq = max(reduced_values[0], 0.0)
        global_non_none_grad_count = reduced_values[1]
        dots = {
            domain: reduced_values[2 + idx]
            for idx, domain in enumerate(target_domains)
        }
        max_keys = [
            "token_grad_seconds",
            "token_grad_autograd_seconds",
            "token_grad_restore_pre_target_rel_l2",
            "token_grad_restore_pre_target_max_abs",
            "token_grad_restore_post_target_rel_l2",
            "token_grad_restore_post_target_max_abs",
            "token_grad_restore_target_norm",
            "token_grad_restore_original_rel_l2",
            "token_grad_restore_original_max_abs",
            "token_grad_restore_original_norm",
        ]
        local_timing_and_restore = {
            "token_grad_seconds": time.perf_counter() - started_at,
            "token_grad_autograd_seconds": autograd_seconds,
            **restore_stats,
        }
        max_values = _all_reduce_values_max([float(local_timing_and_restore.get(key, 0.0)) for key in max_keys])
        reduced_max = {key: max_values[idx] for idx, key in enumerate(max_keys)}
        global_errors = sorted(set(str(error) for error in _all_gather_list([local_error] if local_error else [])))

        token_norm = norm_sq**0.5
        available = not global_errors and global_non_none_grad_count > 0 and token_norm > 0.0
        stats: dict[str, Any] = {
            "token_grad_available": float(available),
            "token_grad_norm": token_norm if available else None,
            "token_grad_non_none_grad_count": float(global_non_none_grad_count),
            "token_grad_autograd_error": ";".join(global_errors) if global_errors else None,
            "token_grad_backward_fallback_seconds": 0.0,
            "token_grad_backward_fallback_used": 0.0,
            **reduced_max,
        }
        for target_domain, dot in dots.items():
            _chunks, target_norm_sq = target_map[target_domain]
            target_norm = target_norm_sq**0.5
            safe_domain = _safe_name(target_domain)
            stats[f"{safe_domain}_cos"] = dot / (token_norm * target_norm) if available and target_norm > 0.0 else None
            stats[f"{safe_domain}_projection_share"] = (
                dot / target_norm_sq if available and target_norm_sq > 0.0 else None
            )
        return stats

    def _grad_multi_target_stats_from_tensors(
        self,
        gradients: tuple[torch.Tensor | None, ...],
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> tuple[float | None, dict[str, float] | None]:
        local_norm_sq = 0.0
        local_dots = {domain: 0.0 for domain in target_map}
        for param_idx, gradient in enumerate(gradients):
            if gradient is None:
                continue
            gradient_gpu = gradient.detach().reshape(-1).float()
            gradient_norm_sq = _chunked_vector_dot(gradient_gpu, gradient_gpu)
            if gradient_norm_sq is None:
                return None, None
            local_norm_sq += gradient_norm_sq
            for domain, (target_chunks, _target_norm_sq) in target_map.items():
                if param_idx >= len(target_chunks):
                    return None, None
                target = target_chunks[param_idx]
                if gradient_gpu.numel() != target.numel():
                    return None, None
                target_gpu = target.reshape(-1).to(device=gradient_gpu.device, dtype=torch.float32)
                gradient_target_dot = _chunked_vector_dot(gradient_gpu, target_gpu)
                if gradient_target_dot is None:
                    return None, None
                local_dots[domain] += gradient_target_dot
        return local_norm_sq, local_dots

    def _summarize_token_gradient_rows(
        self,
        domain: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        safe_domain = _safe_name(domain)
        finite_norms = _finite_values([row.get("token_grad_norm") for row in rows])
        other_cos: list[float] = []
        own_projection: list[float] = []
        other_projection: list[float] = []
        seconds = _finite_values([row.get("token_grad_seconds") for row in rows])
        autograd_seconds = _finite_values([row.get("token_grad_autograd_seconds") for row in rows])
        fallback_seconds = _finite_values([row.get("token_grad_backward_fallback_seconds") for row in rows])
        fallback_used = _finite_values([row.get("token_grad_backward_fallback_used") for row in rows])
        availability = _finite_values([row.get("token_grad_available") for row in rows])
        pre_restore_rel_l2 = _finite_values([row.get("token_grad_restore_pre_target_rel_l2") for row in rows])
        post_restore_rel_l2 = _finite_values([row.get("token_grad_restore_post_target_rel_l2") for row in rows])
        pre_restore_max_abs = _finite_values([row.get("token_grad_restore_pre_target_max_abs") for row in rows])
        post_restore_max_abs = _finite_values([row.get("token_grad_restore_post_target_max_abs") for row in rows])
        original_restore_rel_l2 = _finite_values([row.get("token_grad_restore_original_rel_l2") for row in rows])
        original_restore_max_abs = _finite_values([row.get("token_grad_restore_original_max_abs") for row in rows])
        for row in rows:
            other_domain = row.get("other_domain")
            if other_domain is not None:
                other_cos.extend(_finite_values([row.get(f"{_safe_name(other_domain)}_cos")]))
                other_projection.extend(_finite_values([row.get(f"{_safe_name(other_domain)}_projection_share")]))
            own_projection.extend(_finite_values([row.get(f"{safe_domain}_projection_share")]))
        selected_token_total = sum(
            float(row.get("selected_token_count", 1.0) or 0.0) for row in rows
        )
        selected_sample_total = sum(
            float(row.get("selected_sample_count", 0.0) or 0.0) for row in rows
        )
        global_candidate_token_count = max(
            _finite_values([row.get("global_candidate_token_count") for row in rows]) or [0.0]
        )
        global_candidate_sample_count = max(
            _finite_values([row.get("global_candidate_sample_count") for row in rows]) or [0.0]
        )
        global_candidate_gap_mass = max(
            _finite_values([row.get("global_candidate_gap_mass") for row in rows]) or [0.0]
        )
        global_candidate_gap_abs_mass = max(
            _finite_values([row.get("global_candidate_gap_abs_mass") for row in rows]) or [0.0]
        )
        global_candidate_loss_abs_mass = max(
            _finite_values([row.get("global_candidate_loss_abs_mass") for row in rows]) or [0.0]
        )
        metrics: dict[str, float] = {
            f"{safe_domain}/token_grad/selected_sample_count": float(selected_sample_total),
            f"{safe_domain}/token_grad/selected_token_count": float(selected_token_total),
            f"{safe_domain}/token_grad/global_candidate_sample_count": float(global_candidate_sample_count),
            f"{safe_domain}/token_grad/global_candidate_token_count": float(global_candidate_token_count),
            f"{safe_domain}/token_grad/global_candidate_gap_mass": float(global_candidate_gap_mass),
            f"{safe_domain}/token_grad/global_candidate_gap_abs_mass": float(global_candidate_gap_abs_mass),
            f"{safe_domain}/token_grad/global_candidate_loss_abs_mass": float(global_candidate_loss_abs_mass),
        }
        for row in rows:
            selection = row.get("selection")
            if selection is None:
                continue
            selection_key = _safe_name(selection)
            own_cos = row.get(f"{safe_domain}_cos")
            own_projection_value = row.get(f"{safe_domain}_projection_share")
            if own_cos is not None:
                metrics[f"{safe_domain}/token_grad/{selection_key}_cos_to_domain"] = float(own_cos)
            if own_projection_value is not None:
                metrics[
                    f"{safe_domain}/token_grad_contribution/{selection_key}_projection_share"
                ] = float(own_projection_value)
            metrics[f"{safe_domain}/token_grad/{selection_key}_selected_token_count"] = float(
                row.get("selected_token_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_selected_sample_count"] = float(
                row.get("selected_sample_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_mass"] = float(
                row.get("selected_gap_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_mass_frac"] = float(
                row.get("selected_gap_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_abs_mass"] = float(
                row.get("selected_gap_abs_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_abs_mass_frac"] = float(
                row.get("selected_gap_abs_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_loss_abs_mass"] = float(
                row.get("selected_loss_abs_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_loss_abs_mass_frac"] = float(
                row.get("selected_loss_abs_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_score_mass"] = float(
                row.get("selected_score_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_score_mass_frac"] = float(
                row.get("selected_score_mass_frac", 0.0) or 0.0
            )
        if availability:
            available_count = sum(value > 0.5 for value in availability)
            metrics[f"{safe_domain}/token_grad_cost/available_token_count"] = float(available_count)
            metrics[f"{safe_domain}/token_grad_cost/unavailable_token_count"] = float(
                len(availability) - available_count
            )
            metrics[f"{safe_domain}/token_grad_cost/valid_frac"] = available_count / len(availability)
        if seconds:
            seconds_sum = sum(seconds)
            metrics[f"{safe_domain}/token_grad_cost/seconds_sum"] = seconds_sum
            metrics[f"{safe_domain}/token_grad_cost/seconds_mean"] = _mean(seconds) or 0.0
            metrics[f"{safe_domain}/token_grad_cost/seconds_per_selected_token"] = seconds_sum / max(
                selected_token_total,
                1.0,
            )
        if autograd_seconds:
            metrics[f"{safe_domain}/token_grad_cost/autograd_seconds_sum"] = sum(autograd_seconds)
        if fallback_seconds:
            metrics[f"{safe_domain}/token_grad_cost/backward_fallback_seconds_sum"] = sum(fallback_seconds)
        if fallback_used:
            metrics[f"{safe_domain}/token_grad_cost/backward_fallback_count"] = float(
                sum(value > 0.5 for value in fallback_used)
            )
        if pre_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_pre_target_rel_l2_max"] = max(pre_restore_rel_l2)
        if post_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_post_target_rel_l2_max"] = max(post_restore_rel_l2)
        if pre_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_pre_target_max_abs_max"] = max(pre_restore_max_abs)
        if post_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_post_target_max_abs_max"] = max(post_restore_max_abs)
        if original_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_original_rel_l2_max"] = max(original_restore_rel_l2)
        if original_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_original_max_abs_max"] = max(original_restore_max_abs)
        if finite_norms:
            metrics[f"{safe_domain}/token_grad/norm_mean"] = _mean(finite_norms) or 0.0
            metrics[f"{safe_domain}/token_grad/norm_p95"] = _percentile(finite_norms, 95.0) or 0.0
            metrics[f"{safe_domain}/token_grad/norm_max"] = max(finite_norms)
        if other_cos:
            conflicts = [max(0.0, -value) for value in other_cos]
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_mean"] = _mean(other_cos) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_p05"] = _percentile(other_cos, 5.0) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_negative_frac"] = float(
                np.mean([value < 0.0 for value in other_cos])
            )
            metrics[f"{safe_domain}/token_grad_conflict/conflict_to_other_mean"] = _mean(conflicts) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/conflict_to_other_max"] = max(conflicts)
        if own_projection:
            metrics[f"{safe_domain}/token_grad_contribution/own_projection_share_mean"] = (
                _mean(own_projection) or 0.0
            )
            metrics[f"{safe_domain}/token_grad_contribution/own_projection_share_sum"] = sum(own_projection)
        if other_projection:
            metrics[f"{safe_domain}/token_grad_contribution/other_projection_share_mean"] = (
                _mean(other_projection) or 0.0
            )
            metrics[f"{safe_domain}/token_grad_contribution/negative_other_projection_share_sum"] = sum(
                max(0.0, -value) for value in other_projection
            )
        return metrics

    def _summarize_global_token_gradient_cost(
        self,
        rows: list[dict[str, Any]],
        total_seconds: float,
    ) -> dict[str, float]:
        selected_token_total = sum(
            float(row.get("selected_token_count", 1.0) or 0.0) for row in rows
        )
        metrics: dict[str, float] = {
            "global/token_grad_cost/seconds": float(total_seconds),
            "global/token_grad_cost/selected_token_count": float(selected_token_total),
            "global/token_grad_cost/max_memory_allocated_gb": _max_memory_allocated_gb(),
        }
        candidate_token_counts: dict[str, float] = {}
        candidate_sample_counts: dict[str, float] = {}
        candidate_gap_mass: dict[str, float] = {}
        candidate_gap_abs_mass: dict[str, float] = {}
        candidate_loss_abs_mass: dict[str, float] = {}
        for row in rows:
            domain = str(row.get("domain", "unknown"))
            candidate_token_counts[domain] = max(
                candidate_token_counts.get(domain, 0.0),
                float(row.get("global_candidate_token_count", 0.0) or 0.0),
            )
            candidate_sample_counts[domain] = max(
                candidate_sample_counts.get(domain, 0.0),
                float(row.get("global_candidate_sample_count", 0.0) or 0.0),
            )
            candidate_gap_mass[domain] = max(
                candidate_gap_mass.get(domain, 0.0),
                float(row.get("global_candidate_gap_mass", 0.0) or 0.0),
            )
            candidate_gap_abs_mass[domain] = max(
                candidate_gap_abs_mass.get(domain, 0.0),
                float(row.get("global_candidate_gap_abs_mass", 0.0) or 0.0),
            )
            candidate_loss_abs_mass[domain] = max(
                candidate_loss_abs_mass.get(domain, 0.0),
                float(row.get("global_candidate_loss_abs_mass", 0.0) or 0.0),
            )
        if candidate_token_counts:
            metrics["global/token_grad_cost/global_candidate_token_count"] = float(
                sum(candidate_token_counts.values())
            )
            metrics["global/token_grad_cost/global_candidate_sample_count"] = float(
                sum(candidate_sample_counts.values())
            )
            metrics["global/token_grad_cost/global_candidate_gap_mass"] = float(
                sum(candidate_gap_mass.values())
            )
            metrics["global/token_grad_cost/global_candidate_gap_abs_mass"] = float(
                sum(candidate_gap_abs_mass.values())
            )
            metrics["global/token_grad_cost/global_candidate_loss_abs_mass"] = float(
                sum(candidate_loss_abs_mass.values())
            )
        metrics["global/token_grad_cost/selected_sample_count"] = float(
            sum(float(row.get("selected_sample_count", 0.0) or 0.0) for row in rows)
        )
        if selected_token_total > 0.0:
            metrics["global/token_grad_cost/seconds_per_selected_token"] = float(total_seconds) / selected_token_total

        availability = _finite_values([row.get("token_grad_available") for row in rows])
        if availability:
            available_count = sum(value > 0.5 for value in availability)
            metrics["global/token_grad_cost/available_token_count"] = float(available_count)
            metrics["global/token_grad_cost/unavailable_token_count"] = float(
                len(availability) - available_count
            )
            metrics["global/token_grad_cost/valid_frac"] = available_count / len(availability)

        autograd_seconds = _finite_values([row.get("token_grad_autograd_seconds") for row in rows])
        if autograd_seconds:
            metrics["global/token_grad_cost/autograd_seconds_sum"] = sum(autograd_seconds)

        fallback_seconds = _finite_values([row.get("token_grad_backward_fallback_seconds") for row in rows])
        if fallback_seconds:
            metrics["global/token_grad_cost/backward_fallback_seconds_sum"] = sum(fallback_seconds)

        fallback_used = _finite_values([row.get("token_grad_backward_fallback_used") for row in rows])
        if fallback_used:
            metrics["global/token_grad_cost/backward_fallback_count"] = float(
                sum(value > 0.5 for value in fallback_used)
            )
        pre_restore_rel_l2 = _finite_values([row.get("token_grad_restore_pre_target_rel_l2") for row in rows])
        if pre_restore_rel_l2:
            metrics["global/token_grad_cost/restore_pre_target_rel_l2_max"] = max(pre_restore_rel_l2)
        post_restore_rel_l2 = _finite_values([row.get("token_grad_restore_post_target_rel_l2") for row in rows])
        if post_restore_rel_l2:
            metrics["global/token_grad_cost/restore_post_target_rel_l2_max"] = max(post_restore_rel_l2)
        pre_restore_max_abs = _finite_values([row.get("token_grad_restore_pre_target_max_abs") for row in rows])
        if pre_restore_max_abs:
            metrics["global/token_grad_cost/restore_pre_target_max_abs_max"] = max(pre_restore_max_abs)
        post_restore_max_abs = _finite_values([row.get("token_grad_restore_post_target_max_abs") for row in rows])
        if post_restore_max_abs:
            metrics["global/token_grad_cost/restore_post_target_max_abs_max"] = max(post_restore_max_abs)
        original_restore_rel_l2 = _finite_values([row.get("token_grad_restore_original_rel_l2") for row in rows])
        if original_restore_rel_l2:
            metrics["global/token_grad_cost/restore_original_rel_l2_max"] = max(original_restore_rel_l2)
        original_restore_max_abs = _finite_values([row.get("token_grad_restore_original_max_abs") for row in rows])
        if original_restore_max_abs:
            metrics["global/token_grad_cost/restore_original_max_abs_max"] = max(original_restore_max_abs)
        return metrics

    def _recompute_sample_to_domain_stats(
        self,
        micro_batch: DataProto,
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
        restore_target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        loss_scale_factor: float,
        on_policy: bool,
    ) -> dict[str, float | None]:
        parameters = _trainable_parameters(self.actor)
        if len(parameters) != len(target_chunks):
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": 0.0,
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": "parameter_target_mismatch",
            }
        try:
            grad_dtypes = _parameter_grad_dtypes(parameters)
            _clear_parameter_grads(parameters)
            with _actor_no_sync_context(self.actor):
                loss = _actor_micro_batch_loss(
                    self.actor,
                    micro_batch,
                    loss_scale_factor=loss_scale_factor,
                    on_policy=on_policy,
                    safe_logprob_backward=False,
                )
                loss.backward()
            gradients = tuple(parameter.grad for parameter in parameters)
            local_norm_sq, local_dot = self._grad_stats_from_tensors(gradients, target_chunks)
            non_none_grad_count = sum(gradient is not None for gradient in gradients)
            autograd_error: str | None = None
        except Exception as exc:
            gradients = tuple(None for _ in parameters)
            local_norm_sq, local_dot = None, None
            non_none_grad_count = 0
            autograd_error = f"backward_{type(exc).__name__}"
        finally:
            if "grad_dtypes" in locals():
                _restore_parameter_grads_from_targets(
                    parameters,
                    restore_target_map,
                    grad_dtypes=grad_dtypes,
                )
        if non_none_grad_count == 0 and autograd_error is None:
            autograd_error = "all_parameters_disconnected"
        if local_norm_sq is None or local_dot is None:
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": float(non_none_grad_count),
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": autograd_error,
            }
        if self._sample_gradient_uses_full_local_params:
            norm_sq = max(local_norm_sq, 0.0)
            dot = local_dot
        else:
            norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
            dot = _all_reduce_sum(local_dot)
        if non_none_grad_count > 0 and norm_sq <= 0.0:
            self._sample_zero_norm_count += 1
        sample_norm = norm_sq**0.5
        available = non_none_grad_count > 0 and sample_norm > 0.0 and target_norm > 0.0
        cosine = dot / (sample_norm * target_norm) if available else None
        projection_share = dot / target_norm_sq if available and target_norm_sq > 0.0 else None
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_recompute_grad_norm": sample_norm,
            "sample_recompute_non_none_grad_count": float(non_none_grad_count),
            "sample_recompute_available": float(available),
            "sample_recompute_autograd_error": autograd_error,
        }

    def _recompute_sample_to_domain_stats_autograd(
        self,
        micro_batch: DataProto,
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> dict[str, float | None]:
        parameters = _trainable_parameters(self.actor)
        if len(parameters) != len(target_chunks):
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": 0.0,
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": "parameter_target_mismatch",
            }
        try:
            loss = _actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=loss_scale_factor,
                on_policy=on_policy,
                safe_logprob_backward=True,
            )
            gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
            autograd_error: str | None = None
        except Exception as exc:
            gradients = tuple(None for _ in parameters)
            autograd_error = type(exc).__name__
        local_norm_sq, local_dot = self._grad_stats_from_tensors(gradients, target_chunks)
        non_none_grad_count = sum(gradient is not None for gradient in gradients)
        if non_none_grad_count == 0 and autograd_error is None:
            autograd_error = "all_parameters_disconnected"
        if local_norm_sq is None or local_dot is None:
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": float(non_none_grad_count),
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": autograd_error,
            }
        if self._sample_gradient_uses_full_local_params:
            norm_sq = max(local_norm_sq, 0.0)
            dot = local_dot
        else:
            norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
            dot = _all_reduce_sum(local_dot)
        if non_none_grad_count > 0 and norm_sq <= 0.0:
            self._sample_zero_norm_count += 1
        sample_norm = norm_sq**0.5
        available = non_none_grad_count > 0 and sample_norm > 0.0 and target_norm > 0.0
        cosine = dot / (sample_norm * target_norm) if available else None
        projection_share = dot / target_norm_sq if available and target_norm_sq > 0.0 else None
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_recompute_grad_norm": sample_norm,
            "sample_recompute_non_none_grad_count": float(non_none_grad_count),
            "sample_recompute_available": float(available),
            "sample_recompute_autograd_error": autograd_error,
        }

    def _grad_stats_from_tensors(
        self,
        gradients: tuple[torch.Tensor | None, ...],
        target_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[float | None, float | None]:
        local_norm_sq = 0.0
        local_dot = 0.0
        for gradient, target in zip(gradients, target_chunks):
            if gradient is None:
                continue
            gradient_gpu = gradient.detach().reshape(-1).float()
            if gradient_gpu.numel() != target.numel():
                return None, None
            target_gpu = target.reshape(-1).to(device=gradient_gpu.device, dtype=torch.float32)
            gradient_norm_sq = _chunked_vector_dot(gradient_gpu, gradient_gpu)
            gradient_target_dot = _chunked_vector_dot(gradient_gpu, target_gpu)
            if gradient_norm_sq is None or gradient_target_dot is None:
                return None, None
            local_norm_sq += gradient_norm_sq
            local_dot += gradient_target_dot
        return local_norm_sq, local_dot


class SameForwardDomainGradientProbe:
    """Capture per-domain gradients from the training forward graph.

    Domain gradients are accumulated on GPU as fp32 vectors.  When a domain
    block completes, its gradient may be offloaded to CPU (in *storage_dtype*)
    to free GPU memory.  At the end of the mini-batch all domain vectors are
    loaded back to GPU on-demand for pairwise cosine computation and then
    deleted.

    Per-sample gradient norms are computed incrementally via backward hooks
    during the training ``loss.backward()`` — no materialised gradient tensor
    needed.

    Sample-to-domain cosines use a **two-pass** recomputation:
    micro-batches are stored on CPU during training, then after all domain
    gradients are complete each stored micro-batch is re-forwarded through the
    model and ``autograd.grad`` is called to obtain the sample gradient.  The
    dot product with the domain target is computed chunk-by-chunk to bound
    peak GPU memory (~1.5 GB per chunk for a 4B model).
    """

    def __init__(self, actor: Any, cfg: dict[str, Any]):
        self.actor = actor
        self.cfg = cfg
        self.domains = list(cfg.get("domains", []))
        self.max_samples = _max_samples_from_cfg(cfg)
        self.storage_dtype = str(cfg.get("storage_dtype", "float32"))
        self.offload = bool(cfg.get("offload_domain_gradients", True))
        self.step = int(cfg.get("step", 0) or 0)
        self.sample_gradient_enabled = bool(cfg.get("sample_gradient_enabled", False))
        self._distributed_world_size = _distributed_world_size()
        requested_sample_norm = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_norm_enabled", True))
        requested_sample_cos = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_cos_enabled", False))
        requested_domain_gradient = bool(cfg.get("domain_gradient_enabled", cfg.get("enabled", False)))
        self.domain_gradient_enabled = requested_domain_gradient or requested_sample_cos
        self._sample_gradient_distributed_unsupported = False
        self.sample_norm_enabled = requested_sample_norm
        self.sample_cos_enabled = requested_sample_cos
        self.sample_log_sample_level = bool(cfg.get("sample_gradient_log_sample_level", True))
        self.output_dir = str(cfg.get("output_dir", ""))

        # Domain gradient state (GPU)
        self._gpu_vectors: dict[str, torch.Tensor] = {}
        self._cpu_vectors: dict[str, torch.Tensor] = {}
        self._sample_counts: dict[str, int] = {}
        self._domain_micro_batch_total: dict[str, int] = {}
        self._domain_micro_batch_done: dict[str, int] = {}
        # Per-domain target chunks for sample-to-domain cosine recomputation.
        # Set by _domain_block_finished.  Format: domain -> tuple of per-param chunks.
        self._domain_target_chunks: dict[str, tuple[torch.Tensor, ...]] = {}
        self._domain_norms: dict[str, float] = {}
        self._domain_norm_sqs: dict[str, float] = {}
        self._domain_vectors_are_global = False
        # Backward hook state
        self._hook_state = _HookState()
        self._hook_handles: list[Any] = []
        # Active sample context
        self._active_norm_context: dict[str, Any] | None = None
        self._sample_records: list[dict[str, Any]] = []
        self._sample_candidates: dict[str, list[dict[str, Any]]] = {}
        self._sample_zero_norm_count = 0
        self._micro_batch_index = 0
        self._started_at = 0.0

    # -- public API (called by dp_actor) -----------------------------------

    def prepare_micro_batches(
        self, micro_batches: list[Any]
    ) -> list[tuple[str | None, Any]]:
        """Reorder micro-batches by domain so each domain block is contiguous.

        When *every* domain bucket is non-empty and no micro-batch contains
        mixed-domain samples, the output is sorted by the configured domain
        order.  Otherwise a fallback ``[(None, mb), …]`` list is returned and
        domain-gradient capture is skipped.
        """
        self._domain_micro_batch_total = {}
        self._domain_micro_batch_done = {}
        if not self.domain_gradient_enabled:
            return [(None, mb) for mb in micro_batches]

        buckets: dict[str, list[Any]] = {domain: [] for domain in self.domains}
        for micro_batch in micro_batches:
            labels = _teacher_labels(micro_batch)
            unique_labels = set(labels)
            if len(unique_labels) != 1:
                return [(None, mb) for mb in micro_batches]
            domain = next(iter(unique_labels))
            if domain not in buckets:
                return [(None, mb) for mb in micro_batches]
            buckets[domain].append(micro_batch)

        if not all(buckets[domain] for domain in self.domains):
            return [(None, mb) for mb in micro_batches]

        self._domain_micro_batch_total = {d: len(buckets[d]) for d in self.domains}
        self._domain_micro_batch_done = {d: 0 for d in self.domains}
        ordered: list[tuple[str | None, Any]] = []
        for domain in self.domains:
            ordered.extend([(domain, mb) for mb in buckets[domain]])
        return ordered

    def start_mini_batch(self) -> None:
        self._gpu_vectors = {}
        self._cpu_vectors = {}
        self._sample_counts = {}
        self._domain_norms = {}
        self._domain_norm_sqs = {}
        self._domain_vectors_are_global = False
        self._sample_records = []
        self._micro_batch_index = 0
        self._sample_candidates = {}
        self._domain_target_chunks = {}
        self._sample_zero_norm_count = 0
        self._started_at = time.perf_counter()
        self._install_hooks()

    def capture_micro_batch(
        self,
        *,
        model_inputs: dict[str, Any],
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        entropy: torch.Tensor | None,
        policy_loss_fn: Any,
        loss_agg_mode: str,
        loss_scale_factor: float,
        rollout_is_weights: torch.Tensor | None,
    ) -> None:
        """Compute per-domain gradients via :func:`torch.autograd.grad`.

        Must be called **after** the training forward pass and **before**
        ``loss.backward()`` so the computation graph is still live.
        ``retain_graph=True`` keeps the graph intact for the subsequent
        training backward.
        """
        if not self.domain_gradient_enabled:
            return
        labels = _labels_from_inputs(model_inputs, int(log_prob.shape[0]))
        for domain, indices in _indices_by_label(labels, self.domains, None).items():
            if not indices:
                continue
            remaining = None if self.max_samples is None else self.max_samples - self._sample_counts.get(domain, 0)
            if remaining is not None:
                if remaining <= 0:
                    continue
                indices = indices[:remaining]
            domain_mask = torch.zeros_like(response_mask, dtype=response_mask.dtype)
            domain_mask[indices] = response_mask[indices]
            weight = _domain_loss_weight(response_mask, domain_mask, loss_agg_mode)
            if weight <= 0:
                continue
            domain_loss = self._domain_policy_loss(
                model_inputs=model_inputs,
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                domain_mask=domain_mask,
                entropy=entropy,
                policy_loss_fn=policy_loss_fn,
                loss_agg_mode=loss_agg_mode,
                rollout_is_weights=rollout_is_weights,
            )
            scaled_loss = domain_loss * weight * float(loss_scale_factor)
            vector = self._gradient_vector(scaled_loss)
            if domain in self._gpu_vectors:
                self._gpu_vectors[domain] = self._gpu_vectors[domain] + vector
            else:
                self._gpu_vectors[domain] = vector
            self._sample_counts[domain] = self._sample_counts.get(domain, 0) + len(indices)
            # Track per-micro_batch completion when prepare_micro_batches ordered batches
            if self._domain_micro_batch_total:
                self._domain_micro_batch_done[domain] = self._domain_micro_batch_done.get(domain, 0) + 1
                if self._domain_micro_batch_done[domain] >= self._domain_micro_batch_total.get(domain, 0):
                    self._domain_block_finished(domain)

    def before_backward(
        self,
        domain: str | None,
        micro_batch: Any,
        *,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> None:
        """Set up per-sample hook state before ``loss.backward()``."""
        if not self.sample_norm_enabled:
            return
        labels = _teacher_labels(micro_batch)
        resolved_domain = domain
        if resolved_domain is None and len(set(labels)) == 1:
            resolved_domain = labels[0]
        sample_ids = _sample_ids(
            micro_batch,
            self.step,
            fallback_prefix=f"step{self.step}:micro{self._micro_batch_index}",
        )
        sample_id = sample_ids[0] if sample_ids else f"step{self.step}:micro{self._micro_batch_index}"
        self._active_norm_context = {
            "step": self.step,
            "domain": resolved_domain or "unknown",
            "sample_id": sample_id,
            "sample_count": len(micro_batch),
            "micro_batch_index": self._micro_batch_index,
            "effective_tokens": _response_token_count(micro_batch),
            "loss_scale_factor": float(loss_scale_factor),
            "on_policy": bool(on_policy),
        }
        # Always enable norm accumulation
        self._hook_state.norm_parts = []
        self._hook_state.dot_targets = None
        self._hook_state.dot_accumulator = None

    def after_backward(
        self, domain: str | None, sample_count: int, micro_batch: Any | None = None
    ) -> None:
        if self.sample_norm_enabled:
            self._finish_active_sample(micro_batch)
        self._micro_batch_index += 1

    def finish_mini_batch(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/audit/sample_gradient_zero_norm_count": 0.0,
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": _max_memory_allocated_gb(),
        }
        if self._distributed_world_size > 1 and self.domain_gradient_enabled:
            metrics["global/audit/full_gradient_replicated_all_reduce"] = 1.0

        if self.domain_gradient_enabled:
            metrics.update(self._compute_domain_summary_metrics())

        # Sample-to-domain cosine (two-pass recomputation)
        if self.sample_cos_enabled:
            metrics.update(self._sample_cos_metrics())

        # Sample norm metrics
        metrics.update(self._sample_norm_metrics())

        # Sample-to-domain cosine metrics
        metrics["global/audit/sample_gradient_zero_norm_count"] = _all_reduce_sum(self._sample_zero_norm_count)
        if self._sample_gradient_distributed_unsupported:
            metrics["global/audit/sample_gradient_distributed_unsupported"] = 1.0
            metrics["global/audit/sample_gradient_distributed_world_size"] = float(self._distributed_world_size)

        if self.sample_log_sample_level:
            sample_records = _all_gather_list(self._sample_records)
            if _distributed_rank() == 0:
                _write_jsonl_rows(self.output_dir, "sample_grad_metrics.jsonl", sample_records)

        # Cleanup
        self._remove_hooks()
        self._gpu_vectors = {}
        self._cpu_vectors = {}
        self._domain_target_chunks = {}
        self._active_norm_context = None
        self._hook_state.norm_parts = None
        self._hook_state.dot_targets = None
        self._hook_state.dot_accumulator = None
        return metrics

    # -- backward hooks ---------------------------------------------------

    def _install_hooks(self) -> None:
        if self._hook_handles:
            return
        if not self.sample_norm_enabled:
            return
        state = self._hook_state

        for param in _trainable_parameters(self.actor):
            param_id = id(param)

            def _make_hook(pid: int):
                def hook(gradient: torch.Tensor) -> torch.Tensor:
                    if state.norm_parts is not None:
                        state.norm_parts.append(gradient.detach().float().square().sum())
                    if state.dot_targets is not None and pid in state.dot_targets:
                        target = state.dot_targets[pid]
                        flat = gradient.detach().float().reshape(-1)
                        if flat.numel() == target.numel():
                            state.dot_accumulator.append(torch.dot(flat, target))
                    return gradient

                return hook

            self._hook_handles.append(param.register_hook(_make_hook(param_id)))

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    # -- sample record ----------------------------------------------------

    def _finish_active_sample(self, micro_batch: Any | None = None) -> None:
        context = self._active_norm_context
        norm_parts = self._hook_state.norm_parts
        self._active_norm_context = None
        self._hook_state.norm_parts = None
        self._hook_state.dot_targets = None
        self._hook_state.dot_accumulator = None

        if context is None or norm_parts is None:
            return
        local_sumsq = torch.stack(norm_parts).sum().item() if norm_parts else 0.0
        scale = _current_grad_scale(self.actor)
        if scale != 1.0:
            local_sumsq /= scale * scale
        grad_norm = float(max(local_sumsq, 0.0) ** 0.5)

        if grad_norm <= 0.0:
            self._sample_zero_norm_count += 1

        row: dict[str, Any] = {
            **context,
            "sample_grad_norm": grad_norm,
            "sample_to_domain_cos": None,
            "sample_projection_share": None,
        }
        self._sample_records.append(row)
        if self.sample_cos_enabled and micro_batch is not None:
            try:
                stored_micro_batch = micro_batch.to("cpu")
            except Exception:
                stored_micro_batch = None
            if stored_micro_batch is not None:
                domain = str(context["domain"])
                self._sample_candidates.setdefault(domain, []).append({
                    "row": row,
                    "micro_batch": stored_micro_batch,
                })

    # -- domain gradient helpers -------------------------------------------

    def _domain_policy_loss(
        self,
        *,
        model_inputs: dict[str, Any],
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        domain_mask: torch.Tensor,
        entropy: torch.Tensor | None,
        policy_loss_fn: Any,
        loss_agg_mode: str,
        rollout_is_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        from verl.trainer.ppo.core_algos import agg_loss, kl_penalty

        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=domain_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.actor.config,
            rollout_is_weights=rollout_is_weights,
        )
        policy_loss = pg_loss
        entropy_coeff = float(_cfg_get(self.actor.config, "entropy_coeff", 0.0) or 0.0)
        if entropy_coeff != 0 and entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=domain_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss - entropy_loss * entropy_coeff
        if bool(_cfg_get(self.actor.config, "use_kl_loss", False)) and "math_teacher_log_prob" in model_inputs:
            kl_coef = float(_cfg_get(self.actor.config, "kl_loss_coef", 0.0) or 0.0)
            if kl_coef != 0:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=model_inputs["math_teacher_log_prob"],
                    kl_penalty=str(_cfg_get(self.actor.config, "kl_loss_type", "kl")),
                )
                kl_loss = agg_loss(loss_mat=kld, loss_mask=domain_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = policy_loss + kl_loss * kl_coef
        return policy_loss

    def _gradient_vector(self, loss: torch.Tensor) -> torch.Tensor:
        parameters = _trainable_parameters(self.actor)
        if not parameters:
            return torch.zeros(0, dtype=torch.float32, device=get_device_id())
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        if not any(g is not None for g in gradients):
            return torch.zeros(0, dtype=torch.float32, device=get_device_id())
        return _gpu_concat_grad_vector(parameters, gradients)

    def _domain_block_finished(self, domain: str) -> None:
        """Called when all micro-batches for *domain* have been accumulated."""
        if self._distributed_world_size > 1:
            return
        vector = self._gpu_vectors.get(domain)
        if vector is None or vector.numel() == 0:
            return
        norm = _vector_norm(vector)
        norm_sq = norm * norm
        self._domain_norms[domain] = norm
        self._domain_norm_sqs[domain] = norm_sq

        # Prepare per-param target chunks for sample-to-domain cosine
        # recomputation (done later in finish_mini_batch via two-pass).
        if self.sample_cos_enabled:
            parameters = _trainable_parameters(self.actor)
            if self.offload:
                cpu_vector = vector.to(device="cpu", dtype=_storage_dtype(self.storage_dtype))
                self._cpu_vectors[domain] = cpu_vector
                chunks = _vector_to_param_chunks(cpu_vector.float(), parameters, self.storage_dtype)
                del self._gpu_vectors[domain]
            else:
                chunks = _vector_to_param_chunks(vector.float(), parameters, self.storage_dtype)
            self._domain_target_chunks[domain] = chunks
        elif self.offload:
            cpu_vector = vector.to(device="cpu", dtype=_storage_dtype(self.storage_dtype))
            self._cpu_vectors[domain] = cpu_vector
            del self._gpu_vectors[domain]

    # -- metrics aggregation -----------------------------------------------

    def _domain_sync_device(self) -> torch.device | int:
        for vector in self._gpu_vectors.values():
            return vector.device
        try:
            parameters = _trainable_parameters(self.actor)
        except AttributeError:
            parameters = ()
        if parameters:
            return parameters[0].device
        for vector in self._cpu_vectors.values():
            return vector.device
        return get_device_id()

    def _domain_vector_numel(self) -> int:
        for vector in list(self._gpu_vectors.values()) + list(self._cpu_vectors.values()):
            return int(vector.numel())
        try:
            return sum(parameter.numel() for parameter in _trainable_parameters(self.actor))
        except AttributeError:
            return 0

    def _ensure_global_domain_vectors(self) -> None:
        """Synchronize replicated full-parameter domain vectors across ranks.

        This path assumes tensor/model parallel size is 1 and each rank holds a
        complete parameter vector.  The all-reduced vector is then identical on
        every rank, so later norm/dot computations must be local.
        """
        if self._domain_vectors_are_global:
            return
        if self._distributed_world_size <= 1:
            self._domain_vectors_are_global = True
            return

        local_names = self._domain_names()
        domain_names = list(dict.fromkeys(list(self.domains) + sorted(set(_all_gather_list(local_names)))))
        local_numel = self._domain_vector_numel()
        gathered_numels = _all_gather_list([local_numel])
        positive_numels = {int(numel) for numel in gathered_numels if int(numel) > 0}
        if not positive_numels:
            self._domain_vectors_are_global = True
            return
        if len(positive_numels) != 1:
            self._sample_gradient_distributed_unsupported = True
            self._domain_vectors_are_global = True
            return
        vector_numel = next(iter(positive_numels))

        device = self._domain_sync_device()
        self._domain_target_chunks = {}
        parameters = _trainable_parameters(self.actor)
        next_gpu_vectors: dict[str, torch.Tensor] = {}
        next_cpu_vectors: dict[str, torch.Tensor] = {}
        for domain in domain_names:
            local_vector = self._domain_vector(domain, prefer_cpu=False)
            if local_vector is None:
                sync_vector = torch.zeros(vector_numel, dtype=torch.float32, device=device)
            else:
                sync_vector = local_vector.detach().to(device=device, dtype=torch.float32, copy=True)
                if sync_vector.numel() != vector_numel:
                    sync_vector = torch.zeros(vector_numel, dtype=torch.float32, device=device)
            _all_reduce_vector_sum(sync_vector)
            if sync_vector.numel() == 0:
                continue
            norm = _local_vector_norm(sync_vector)
            if norm <= 0.0 and _all_reduce_sum(self._sample_counts.get(domain, 0)) <= 0.0:
                continue
            self._domain_norms[domain] = norm
            self._domain_norm_sqs[domain] = norm * norm

            if self.offload:
                cpu_vector = sync_vector.to(device="cpu", dtype=_storage_dtype(self.storage_dtype))
                next_cpu_vectors[domain] = cpu_vector
                if self.sample_cos_enabled and parameters:
                    self._domain_target_chunks[domain] = _vector_to_param_chunks(
                        cpu_vector.float(), parameters, self.storage_dtype
                    )
            else:
                next_gpu_vectors[domain] = sync_vector
                if self.sample_cos_enabled and parameters:
                    self._domain_target_chunks[domain] = _vector_to_param_chunks(
                        sync_vector.float(), parameters, self.storage_dtype
                    )

        self._gpu_vectors = next_gpu_vectors
        self._cpu_vectors = next_cpu_vectors
        self._domain_vectors_are_global = True

    def _domain_names(self) -> list[str]:
        return sorted(set(list(self._gpu_vectors.keys()) + list(self._cpu_vectors.keys())))

    def _domain_vector(self, domain: str, *, prefer_cpu: bool = False) -> torch.Tensor | None:
        if domain in self._cpu_vectors:
            vec = self._cpu_vectors[domain]
            return vec.float() if vec.dtype != torch.float32 else vec
        if domain in self._gpu_vectors:
            vec = self._gpu_vectors[domain]
            if prefer_cpu:
                return vec.detach().to(device="cpu", dtype=torch.float32)
            return vec.float() if vec.dtype != torch.float32 else vec
        return None

    def _compute_domain_summary_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        self._ensure_global_domain_vectors()
        domain_names = self._domain_names()
        if not domain_names:
            return metrics

        prefer_cpu = bool(self._cpu_vectors)
        domain_vectors: dict[str, torch.Tensor] = {}
        for domain in domain_names:
            vector = self._domain_vector(domain, prefer_cpu=prefer_cpu)
            if vector is None or vector.numel() == 0:
                continue
            domain_vectors[domain] = vector
            grad_norm = self._domain_norms.get(domain)
            if grad_norm is None:
                grad_norm = _local_vector_norm(vector) if self._domain_vectors_are_global else _vector_norm(vector)
                self._domain_norms[domain] = grad_norm
                self._domain_norm_sqs[domain] = grad_norm * grad_norm
            safe_domain = _safe_name(domain)
            metrics[f"{safe_domain}/full_grad/grad_norm"] = grad_norm
            metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(domain, 0))

        if not domain_vectors:
            return metrics

        total_vector = None
        for vector in domain_vectors.values():
            total_vector = vector if total_vector is None else total_vector + vector
        if total_vector is not None:
            total_norm = _local_vector_norm(total_vector) if self._domain_vectors_are_global else _vector_norm(total_vector)
            total_norm_sq = total_norm * total_norm
            if total_norm > 0.0:
                for domain, vector in domain_vectors.items():
                    domain_norm = self._domain_norms.get(domain, 0.0)
                    dot_total = (
                        _local_vector_dot(vector, total_vector)
                        if self._domain_vectors_are_global
                        else _vector_dot(vector, total_vector)
                    )
                    safe_domain = _safe_name(domain)
                    domain_total_cosine = _safe_cosine(dot_total, domain_norm, total_norm)
                    if domain_total_cosine is not None:
                        metrics[
                            f"global/full_grad_alignment/{safe_domain}_vs_total/full_grad_cosine_domain_total"
                        ] = domain_total_cosine
                    if dot_total is not None and total_norm_sq > 0.0:
                        metrics[
                            f"global/full_grad_contribution/{safe_domain}_to_total/signed_projection_share"
                        ] = dot_total / total_norm_sq

        metrics.update(self._compute_cross_domain_metrics(domain_vectors=domain_vectors))

        return metrics

    def _compute_cross_domain_metrics(
        self,
        domain_vectors: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Compute pairwise cosine between every pair of domain gradients."""
        metrics: dict[str, float] = {}
        owns_vectors = domain_vectors is None
        if domain_vectors is None:
            domain_vectors = {}
            prefer_cpu = bool(self._cpu_vectors)
            for domain in self._domain_names():
                vector = self._domain_vector(domain, prefer_cpu=prefer_cpu)
                if vector is not None:
                    domain_vectors[domain] = vector
        domain_names = sorted(domain_vectors)
        if len(domain_names) < 2:
            return metrics

        for left_idx, left in enumerate(domain_names):
            left_vec = domain_vectors[left]
            if left_vec is None or left_vec.numel() == 0:
                continue
            left_norm = self._domain_norms.get(
                left,
                _local_vector_norm(left_vec) if self._domain_vectors_are_global else _vector_norm(left_vec),
            )
            if left_norm <= 0:
                continue

            for right in domain_names[left_idx + 1 :]:
                right_vec = domain_vectors[right]
                if right_vec is None or right_vec.numel() == 0:
                    continue
                right_norm = self._domain_norms.get(
                    right,
                    _local_vector_norm(right_vec) if self._domain_vectors_are_global else _vector_norm(right_vec),
                )
                if right_norm <= 0:
                    continue

                grad_dot = (
                    _local_vector_dot(left_vec, right_vec)
                    if self._domain_vectors_are_global
                    else _vector_dot(left_vec, right_vec)
                )
                if grad_dot is not None:
                    denom = left_norm * right_norm
                    grad_cosine = None if denom <= 0.0 else grad_dot / denom
                    if grad_cosine is not None:
                        pair = f"{_safe_name(left)}_vs_{_safe_name(right)}"
                        metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = grad_cosine
                        metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -grad_cosine)
        if owns_vectors:
            for vector in domain_vectors.values():
                del vector
        return metrics

    def _sample_norm_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        by_domain: dict[str, list[float]] = {}
        for row in self._sample_records:
            by_domain.setdefault(str(row["domain"]), []).append(float(row["sample_grad_norm"]))
        domain_names = sorted(set(self.domains) | set(_all_gather_list(list(by_domain))))
        for domain in domain_names:
            finite = _finite_values(_all_gather_list(by_domain.get(domain, [])))
            if not finite:
                continue
            safe_domain = _safe_name(domain)
            std = _std(finite) or 0.0
            mean = _mean(finite) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_mean"] = mean
            metrics[f"{safe_domain}/sample_grad/norm_p50"] = _percentile(finite, 50.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_p95"] = _percentile(finite, 95.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_max"] = max(finite)
            metrics[f"{safe_domain}/sample_grad/norm_cv"] = std / (abs(mean) + 1e-12)
            metrics[f"{safe_domain}/sample_grad/sample_count"] = float(len(finite))
        return metrics

    def _sample_cos_metrics(self) -> dict[str, float]:
        """Compute per-sample cosine with its domain gradient via two-pass recomputation.

        Each stored micro-batch is re-forwarded through the model and
        ``autograd.grad`` is called to obtain the per-sample gradient vector.
        The cosine (and projection share) against the completed domain target
        are computed chunk-by-chunk to bound peak GPU memory.
        """
        metrics: dict[str, float] = {}
        self._ensure_global_domain_vectors()

        domain_names = sorted(set(self.domains) | set(_all_gather_list(list(self._sample_candidates))))
        for domain in domain_names:
            candidates = self._sample_candidates.get(domain, [])
            target_chunks = self._domain_target_chunks.get(domain)
            target_norm = self._domain_norms.get(domain, 0.0)
            target_norm_sq = self._domain_norm_sqs.get(domain, 0.0)

            cosines: list[float] = []
            proj_shares: list[float] = []
            if target_chunks is not None and target_norm > 0:
                for candidate in candidates:
                    row = candidate["row"]
                    stored_mb = candidate["micro_batch"]
                    result = self._recompute_sample_to_domain_stats(
                        stored_mb,
                        target_chunks=target_chunks,
                        target_norm=target_norm,
                        target_norm_sq=target_norm_sq,
                        loss_scale_factor=row["loss_scale_factor"],
                        on_policy=row["on_policy"],
                    )
                    cos = result["cosine"]
                    proj = result["projection_share"]
                    row["sample_to_domain_cos"] = cos
                    row["sample_projection_share"] = proj
                    if cos is not None:
                        cosines.append(cos)
                    if proj is not None:
                        proj_shares.append(proj)

            safe_domain = _safe_name(domain)
            finite_cos = _finite_values(_all_gather_list(cosines))
            if finite_cos:
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_mean"] = _mean(finite_cos) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_p05"] = _percentile(finite_cos, 5.0) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_negative_frac"] = float(
                    np.mean([value < 0.0 for value in finite_cos])
                )
                metrics[f"{safe_domain}/sample_grad_cos/sample_count"] = float(len(finite_cos))
            finite_proj = _finite_values(_all_gather_list(proj_shares))
            if finite_proj:
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_mean"] = (
                    _mean(finite_proj) or 0.0
                )
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_min"] = min(finite_proj)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_max"] = max(finite_proj)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_negative_frac"] = float(
                    np.mean([value < 0.0 for value in finite_proj])
                )
                metrics[f"{safe_domain}/sample_grad_contribution/top1_abs_share"] = max(
                    abs(value) for value in finite_proj
                )

        return metrics

    def _recompute_sample_to_domain_stats(
        self,
        micro_batch: Any,
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> dict[str, float | None]:
        """Re-forward *micro_batch* and compare its gradient to the domain target.

        Uses chunk-by-chunk dot products so that only one parameter's target
        is on GPU at a time (~1.5 GB peak for a 4B model).
        """
        from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

        device = get_device_id()
        parameters = _trainable_parameters(self.actor)
        if not parameters or target_norm <= 0:
            return {"cosine": None, "projection_share": None}

        # -- re-forward -------------------------------------------------
        try:
            mb_gpu = micro_batch.to(device)
        except Exception:
            return {"cosine": None, "projection_share": None}
        model_inputs = {**mb_gpu.batch, **mb_gpu.non_tensor_batch}

        temperature = float(_cfg_get(self.actor.config, "temperature", 1.0) or 1.0)
        entropy_coeff = float(_cfg_get(self.actor.config, "entropy_coeff", 0.0) or 0.0)
        calculate_entropy = entropy_coeff != 0

        try:
            entropy, log_prob = self.actor._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
            )
        except Exception:
            return {"cosine": None, "projection_share": None}

        # -- compute sample loss ----------------------------------------
        response_mask = model_inputs["response_mask"]
        if bool(_cfg_get(self.actor.config, "use_rollout_log_probs", False)):
            old_log_prob = model_inputs["old_log_probs"]
        elif on_policy:
            old_log_prob = log_prob.detach()
        else:
            old_log_prob = model_inputs["old_log_probs"]
        advantages = _actor_reverse_kl_advantages(self.actor, model_inputs, old_log_prob)
        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

        loss_mode = str(_cfg_get(self.actor.config.policy_loss, "loss_mode", "vanilla"))
        loss_agg_mode = str(_cfg_get(self.actor.config, "loss_agg_mode", "token-mean"))
        policy_loss_fn = get_policy_loss_fn(loss_mode)

        pg_loss, _pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.actor.config,
            rollout_is_weights=rollout_is_weights,
        )
        policy_loss = pg_loss
        if entropy_coeff != 0 and entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss - entropy_loss * entropy_coeff
        if bool(_cfg_get(self.actor.config, "use_kl_loss", False)) and "math_teacher_log_prob" in model_inputs:
            kl_coef = float(_cfg_get(self.actor.config, "kl_loss_coef", 0.0) or 0.0)
            if kl_coef != 0:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=model_inputs["math_teacher_log_prob"],
                    kl_penalty=str(_cfg_get(self.actor.config, "kl_loss_type", "kl")),
                )
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = policy_loss + kl_loss * kl_coef

        sample_loss = policy_loss * loss_scale_factor

        # -- autograd.grad → sample gradient vector --------------------
        try:
            grads = torch.autograd.grad(
                sample_loss, parameters, retain_graph=False, allow_unused=True
            )
        except Exception:
            del model_inputs, mb_gpu, log_prob, sample_loss
            return {"cosine": None, "projection_share": None}

        if not any(g is not None for g in grads):
            del model_inputs, mb_gpu, log_prob, sample_loss
            return {"cosine": None, "projection_share": None}

        sample_vector = _gpu_concat_grad_vector(parameters, grads)
        del grads

        if sample_vector.numel() == 0:
            del model_inputs, mb_gpu, log_prob, sample_loss
            return {"cosine": None, "projection_share": None}

        # -- chunk-by-chunk dot product --------------------------------
        dot_total = 0.0
        sample_norm_sq = 0.0
        offset = 0
        for target_chunk in target_chunks:
            chunk_len = target_chunk.numel()
            sample_chunk = sample_vector[offset : offset + chunk_len]
            sample_norm_sq += sample_chunk.float().square().sum().item()
            if target_chunk.device != sample_chunk.device:
                target_chunk_gpu = target_chunk.to(device=device, dtype=torch.float32)
            else:
                target_chunk_gpu = target_chunk.float()
            chunk_dot = _chunked_vector_dot(sample_chunk, target_chunk_gpu)
            if chunk_dot is None:
                del target_chunk_gpu
                return {"cosine": None, "projection_share": None}
            dot_total += chunk_dot
            del target_chunk_gpu
            offset += chunk_len

        sample_norm = max(sample_norm_sq, 0.0) ** 0.5

        del sample_vector, model_inputs, mb_gpu, log_prob, sample_loss

        if sample_norm <= 0 or target_norm <= 0:
            return {"cosine": None, "projection_share": None}

        cosine = dot_total / (sample_norm * target_norm)
        projection_share = dot_total / target_norm_sq if target_norm_sq > 0 else None

        return {"cosine": float(cosine), "projection_share": float(projection_share) if projection_share is not None else None}
