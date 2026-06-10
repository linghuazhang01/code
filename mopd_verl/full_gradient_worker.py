"""Full-parameter gradient audit helpers for patched verl FSDP workers."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device


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


def _teacher_labels(data: DataProto) -> list[str]:
    batch_size = len(data)
    for key in ("opd_teacher", "domain", "source_domain", "ability"):
        labels = _non_tensor_list(data.non_tensor_batch.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]


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


def _all_reduce_vector_sum(vector: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(vector, op=torch.distributed.ReduceOp.SUM)
    return vector


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


def _local_vector_norm(vector: torch.Tensor) -> float:
    if vector.numel() == 0:
        return 0.0
    return float(max(torch.dot(vector.float(), vector.float()).item(), 0.0) ** 0.5)


def _local_vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
        return None
    return float(torch.dot(left.float(), right.float()).item())


def _vector_norm(vector: torch.Tensor) -> float:
    if vector.numel() == 0:
        return 0.0
    local_sumsq = torch.dot(vector.float(), vector.float()).item()
    return float(max(_all_reduce_sum(local_sumsq), 0.0) ** 0.5)


def _vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
        return None
    return _all_reduce_sum(torch.dot(left.float(), right.float()).item())


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

        first_sumsq += torch.dot(first_float, first_float).item()
        total_sumsq += torch.dot(total_float, total_float).item()
        first_total_dot += torch.dot(first_float, total_float).item()
        second_sumsq += torch.dot(second_float, second_float).item()
        first_second_dot += torch.dot(first_float, second_float).item()
        second_total_dot += torch.dot(second_float, total_float).item()

        if second_pieces is not None and dtype is not None:
            second_piece = second_float.to(dtype=dtype)
            second_pieces.append(second_piece)
            second_piece_float = second_piece.float()
            second_target_sumsq += torch.dot(second_piece_float, second_piece_float).item()
            del second_piece_float
        del first_float, total_float, second_float

    return _GradDifferenceSnapshot(
        first_norm_sq=_all_reduce_sum(first_sumsq),
        total_norm_sq=_all_reduce_sum(total_sumsq),
        first_total_dot=_all_reduce_sum(first_total_dot),
        second_norm_sq=_all_reduce_sum(second_sumsq),
        first_second_dot=_all_reduce_sum(first_second_dot),
        second_total_dot=_all_reduce_sum(second_total_dot),
        second_chunks=tuple(second_pieces) if second_pieces is not None else None,
        second_target_norm_sq=(
            _all_reduce_sum(second_target_sumsq) if second_pieces is not None else None
        ),
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
            local_sumsq += torch.dot(gradient, gradient).item()
    return float(max(_all_reduce_sum(local_sumsq), 0.0) ** 0.5)


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
    for key in ("opd_teacher", "domain", "source_domain", "ability"):
        labels = _non_tensor_list(model_inputs.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]


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


def _actor_micro_batch_loss(
    actor: Any,
    micro_batch: DataProto,
    *,
    loss_scale_factor: float,
    on_policy: bool,
) -> torch.Tensor:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    response_mask = model_inputs["response_mask"]
    entropy_coeff = float(_cfg_get(actor.config, "entropy_coeff", 0.0) or 0.0)
    entropy, log_prob = actor._forward_micro_batch(
        model_inputs,
        temperature=float(micro_batch.meta_info.get("temperature", 1.0)),
        calculate_entropy=entropy_coeff != 0,
    )
    if hasattr(actor.config, "use_rollout_log_probs") and actor.config.use_rollout_log_probs:
        old_log_prob = model_inputs["old_log_probs"]
    elif on_policy:
        old_log_prob = log_prob.detach()
    else:
        old_log_prob = model_inputs["old_log_probs"]

    advantages = _actor_reverse_kl_advantages(actor, model_inputs, old_log_prob)
    loss_mode = str(_cfg_get(_cfg_get(actor.config, "policy_loss", {}), "loss_mode", "vanilla"))
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, _ = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        config=actor.config,
        rollout_is_weights=model_inputs.get("rollout_is_weights", None),
    )
    policy_loss = pg_loss
    if entropy_coeff != 0 and entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss - entropy_loss * entropy_coeff
    if bool(_cfg_get(actor.config, "use_kl_loss", False)) and "math_teacher_log_prob" in model_inputs:
        kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
        if kl_coef != 0:
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["math_teacher_log_prob"],
                kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            )
            policy_loss = policy_loss + kl_loss * kl_coef
    return policy_loss * float(loss_scale_factor)


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
        self._sample_gradient_distributed_unsupported = (
            self._distributed_world_size > 1 and (requested_sample_norm or requested_sample_cos)
        )
        self.sample_norm_enabled = requested_sample_norm and not self._sample_gradient_distributed_unsupported
        self.sample_cos_enabled = requested_sample_cos and not self._sample_gradient_distributed_unsupported
        self.sample_log_sample_level = (
            bool(cfg.get("sample_gradient_log_sample_level", True))
            and not self._sample_gradient_distributed_unsupported
        )
        self.output_dir = str(cfg.get("output_dir", ""))
        self._sample_counts: dict[str, int] = {}
        self._first_domain_chunks: tuple[torch.Tensor, ...] | None = None
        self._expected_first_domain_samples: int | None = None
        self._started_at = 0.0
        self._prepared_supported = len(self.domains) == 2
        self._hook_handles: list[Any] = []
        self._active_norm_parts: list[torch.Tensor] | None = None
        self._active_norm_context: dict[str, Any] | None = None
        self._sample_records: list[dict[str, Any]] = []
        self._sample_candidates: dict[str, list[dict[str, Any]]] = {}
        self._micro_batch_index = 0
        self._sample_zero_norm_count = 0

    def prepare_micro_batches(self, micro_batches: list[Any]) -> list[tuple[str | None, Any]]:
        self._expected_first_domain_samples = None
        if len(self.domains) != 2:
            self._prepared_supported = False
            return [(None, micro_batch) for micro_batch in micro_batches]

        buckets: dict[str, list[tuple[str, Any]]] = {domain: [] for domain in self.domains}
        for micro_batch in micro_batches:
            labels = _teacher_labels(micro_batch)
            unique_labels = set(labels)
            if len(unique_labels) != 1:
                self._prepared_supported = False
                return [(None, item) for item in micro_batches]
            domain = next(iter(unique_labels))
            if domain not in buckets:
                self._prepared_supported = False
                return [(None, item) for item in micro_batches]
            buckets[domain].append((domain, micro_batch))

        if not all(buckets[domain] for domain in self.domains):
            self._prepared_supported = False
            return [(None, item) for item in micro_batches]

        self._prepared_supported = True
        self._expected_first_domain_samples = sum(len(micro_batch) for _, micro_batch in buckets[self.domains[0]])
        ordered: list[tuple[str | None, Any]] = []
        for domain in self.domains:
            ordered.extend(buckets[domain])
        return ordered

    def start_mini_batch(self) -> None:
        self._sample_counts = {}
        self._first_domain_chunks = None
        self._started_at = time.perf_counter()
        self._sample_records = []
        self._sample_candidates = {}
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
        self._active_norm_parts = []
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
        if self.sample_norm_enabled:
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
            self._first_domain_chunks = _snapshot_current_grad_chunks(self.actor, self.storage_dtype)

    def finish_mini_batch(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/audit/sample_gradient_zero_norm_count": 0.0,
            "global/audit/full_gradient_domain_sequential_available": 0.0,
            "global/audit/full_gradient_domain_sequential_unsupported": float(not self._prepared_supported),
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / (1024**3),
        }
        if self._sample_gradient_distributed_unsupported:
            metrics["global/audit/sample_gradient_distributed_unsupported"] = 1.0
            metrics["global/audit/sample_gradient_distributed_world_size"] = float(self._distributed_world_size)
        first_chunks = self._first_domain_chunks
        self._first_domain_chunks = None
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}

        if self.domain_gradient_enabled and self._prepared_supported and len(self.domains) == 2 and first_chunks is not None:
            domain_metrics, domain_targets = self._finish_domain_gradient_metrics(first_chunks)
            metrics.update(domain_metrics)

        metrics.update(self._sample_norm_metrics())
        if self.sample_cos_enabled and domain_targets:
            metrics.update(self._sample_cos_metrics(domain_targets))
        metrics["global/audit/sample_gradient_zero_norm_count"] = float(self._sample_zero_norm_count)
        if self.sample_log_sample_level:
            _write_jsonl_rows(self.output_dir, "sample_grad_metrics.jsonl", self._sample_records)
        self._remove_sample_norm_hooks()
        self._active_norm_parts = None
        self._active_norm_context = None
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
        if context is None or parts is None:
            return
        local_sumsq = torch.stack(parts).sum().item() if parts else 0.0
        scale = _current_grad_scale(self.actor)
        if scale != 1.0:
            local_sumsq /= scale * scale
        grad_norm = float(max(_all_reduce_sum(local_sumsq), 0.0) ** 0.5)
        row = {
            **context,
            "sample_grad_norm": grad_norm,
            "computed_for_cos": False,
            "sample_to_domain_cos": None,
            "sample_projection_share": None,
        }
        self._sample_records.append(row)
        domain = str(context["domain"])
        if self.sample_cos_enabled and micro_batch is not None:
            try:
                stored_micro_batch = micro_batch.to("cpu")
            except Exception:
                stored_micro_batch = None
            if stored_micro_batch is not None:
                self._sample_candidates.setdefault(domain, []).append(
                    {"row": row, "micro_batch": stored_micro_batch}
                )

    def _finish_domain_gradient_metrics(
        self,
        first_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        snapshot = _current_grad_difference_snapshot(
            self.actor,
            first_chunks,
            self.storage_dtype if self.sample_cos_enabled else None,
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
        metrics["global/full_grad/total_grad_norm"] = total_norm

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

        if self.sample_cos_enabled and snapshot.second_chunks is not None:
            domain_targets[first_domain] = (first_chunks, first_norm_sq)
            second_target_norm_sq = snapshot.second_target_norm_sq
            if second_target_norm_sq is not None and second_target_norm_sq > 0.0:
                domain_targets[second_domain] = (snapshot.second_chunks, second_target_norm_sq)

        return metrics, domain_targets

    def _sample_norm_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        by_domain: dict[str, list[float]] = {}
        for row in self._sample_records:
            by_domain.setdefault(str(row["domain"]), []).append(float(row["sample_grad_norm"]))
        for domain, values in sorted(by_domain.items()):
            finite = _finite_values(values)
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
        for domain, candidates in sorted(self._sample_candidates.items()):
            if domain not in domain_targets:
                continue
            if not candidates:
                continue
            target_chunks, target_norm_sq = domain_targets[domain]
            target_norm = target_norm_sq**0.5
            cos_values: list[float] = []
            share_values: list[float] = []
            for candidate in candidates:
                row = candidate["row"]
                stats = self._recompute_sample_to_domain_stats(
                    candidate["micro_batch"],
                    target_chunks=target_chunks,
                    target_norm=target_norm,
                    target_norm_sq=target_norm_sq,
                    loss_scale_factor=float(row.get("loss_scale_factor", 1.0)),
                    on_policy=bool(row.get("on_policy", False)),
                )
                row["computed_for_cos"] = True
                row.update(stats)
                cos_value = stats.get("sample_to_domain_cos")
                share_value = stats.get("sample_projection_share")
                if cos_value is not None:
                    cos_values.append(float(cos_value))
                if share_value is not None:
                    share_values.append(float(share_value))
            safe_domain = _safe_name(domain)
            if cos_values:
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_mean"] = _mean(cos_values) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_p05"] = _percentile(cos_values, 5.0) or 0.0
                metrics[f"{safe_domain}/sample_grad_cos/domain_cos_negative_frac"] = float(
                    np.mean([value < 0.0 for value in cos_values])
                )
                metrics[f"{safe_domain}/sample_grad_cos/sample_count"] = float(len(cos_values))
            if share_values:
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_mean"] = _mean(share_values) or 0.0
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_min"] = min(share_values)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_max"] = max(share_values)
                metrics[f"{safe_domain}/sample_grad_contribution/projection_share_negative_frac"] = float(
                    np.mean([value < 0.0 for value in share_values])
                )
                metrics[f"{safe_domain}/sample_grad_contribution/top1_abs_share"] = max(abs(value) for value in share_values)
        return metrics

    def _recompute_sample_to_domain_stats(
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
            return {"sample_to_domain_cos": None, "sample_projection_share": None, "sample_recompute_grad_norm": None}
        loss = _actor_micro_batch_loss(
            self.actor,
            micro_batch,
            loss_scale_factor=loss_scale_factor,
            on_policy=on_policy,
        )
        gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
        local_norm_sq, local_dot = self._grad_stats_from_tensors(gradients, target_chunks)
        if local_norm_sq is None or local_dot is None:
            return {"sample_to_domain_cos": None, "sample_projection_share": None, "sample_recompute_grad_norm": None}
        norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
        dot = _all_reduce_sum(local_dot)
        if norm_sq <= 0.0:
            self._sample_zero_norm_count += 1
        sample_norm = norm_sq**0.5
        cosine = None if sample_norm <= 0.0 or target_norm <= 0.0 else dot / (sample_norm * target_norm)
        projection_share = None if target_norm_sq <= 0.0 else dot / target_norm_sq
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_recompute_grad_norm": sample_norm,
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
            local_norm_sq += torch.dot(gradient_gpu, gradient_gpu).item()
            local_dot += torch.dot(gradient_gpu, target_gpu).item()
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
            "global/full_grad_cost/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / (1024**3),
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
            metrics["global/full_grad/total_grad_norm"] = total_norm
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
        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]
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
            dot_total += torch.dot(sample_chunk.float(), target_chunk_gpu).item()
            del target_chunk_gpu
            offset += chunk_len

        sample_norm = max(sample_norm_sq, 0.0) ** 0.5

        del sample_vector, model_inputs, mb_gpu, log_prob, sample_loss

        if sample_norm <= 0 or target_norm <= 0:
            return {"cosine": None, "projection_share": None}

        cosine = dot_total / (sample_norm * target_norm)
        projection_share = dot_total / target_norm_sq if target_norm_sq > 0 else None

        return {"cosine": float(cosine), "projection_share": float(projection_share) if projection_share is not None else None}
