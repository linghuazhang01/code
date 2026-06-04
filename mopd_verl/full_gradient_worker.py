"""Full-parameter gradient audit helpers for patched verl FSDP workers."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu


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


def _cfg_get_path(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        current = _cfg_get(current, key, None)
        if current is None:
            return default
    return current


def _policy_loss_config(worker: Any, actor_cfg: Any | None = None) -> Any:
    for candidate in (
        _cfg_get(actor_cfg, "policy_loss", None),
        _cfg_get_path(worker.config, ("actor", "policy_loss"), None),
        _cfg_get_path(worker.config, ("actor_rollout_ref", "actor", "policy_loss"), None),
    ):
        if candidate is not None:
            return candidate
    return {}


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


def _validation_labels(data: DataProto) -> list[str]:
    batch_size = len(data)
    explicit = _non_tensor_list(data.non_tensor_batch.get("validation_dataset"), batch_size)
    data_sources = _non_tensor_list(data.non_tensor_batch.get("data_source"), batch_size)
    abilities = _non_tensor_list(data.non_tensor_batch.get("ability"), batch_size)
    labels: list[str] = []
    for idx in range(batch_size):
        if explicit[idx] is not None:
            labels.append(str(explicit[idx]))
            continue
        data_source = None if data_sources[idx] is None else str(data_sources[idx])
        ability = None if abilities[idx] is None else str(abilities[idx])
        if data_source:
            labels.append(data_source)
        elif ability in {"math", "code"}:
            labels.append(ability)
        else:
            labels.append("unknown")
    return labels


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


def _zero_grad(worker: Any) -> None:
    worker.actor.actor_module.zero_grad(set_to_none=True)


def _all_reduce_sum(value: float) -> float:
    tensor = torch.tensor(float(value), device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


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


def _collect_grad_vector(worker: Any, storage_dtype: str) -> torch.Tensor:
    dtype = torch.float16 if str(storage_dtype).lower() in {"float16", "fp16", "half"} else torch.float32
    pieces = []
    for parameter in worker.actor.actor_module.parameters():
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
        else:
            pieces.append(parameter.grad.detach().reshape(-1).to(device="cpu", dtype=dtype))
    if not pieces:
        return torch.zeros(0, dtype=dtype, device="cpu")
    return torch.cat(pieces)


def _current_grad_scale(actor: Any) -> float:
    scaler = getattr(actor, "scaler", None)
    if scaler is None or not hasattr(scaler, "get_scale"):
        return 1.0
    try:
        scale = float(scaler.get_scale())
    except (TypeError, ValueError):
        return 1.0
    return scale if scale > 0 else 1.0


def _collect_current_grad_vector(actor: Any, storage_dtype: str) -> torch.Tensor:
    dtype = torch.float16 if str(storage_dtype).lower() in {"float16", "fp16", "half"} else torch.float32
    scale = _current_grad_scale(actor)
    pieces = []
    for parameter in _trainable_parameters(actor):
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
            continue
        gradient = parameter.grad.detach()
        if scale != 1.0:
            gradient = gradient / scale
        pieces.append(gradient.reshape(-1).to(device="cpu", dtype=dtype))
    if not pieces:
        return torch.zeros(0, dtype=dtype, device="cpu")
    return torch.cat(pieces)


def _current_grad_cpu_float(parameter: torch.nn.Parameter, scale: float) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    gradient = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    if scale != 1.0:
        gradient = gradient / scale
    return gradient


def _snapshot_current_grad_chunks(actor: Any, storage_dtype: str) -> tuple[torch.Tensor, ...]:
    dtype = torch.float16 if str(storage_dtype).lower() in {"float16", "fp16", "half"} else torch.float32
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


def _restore_grad_chunks(actor: Any, chunks: tuple[torch.Tensor, ...]) -> bool:
    parameters = _trainable_parameters(actor)
    if len(parameters) != len(chunks):
        return False
    for parameter, chunk in zip(parameters, chunks):
        if parameter.numel() != chunk.numel():
            return False
    scale = _current_grad_scale(actor)
    for parameter, chunk in zip(parameters, chunks):
        restored = chunk.reshape_as(parameter).to(device=parameter.device, dtype=parameter.dtype)
        if scale != 1.0:
            restored = restored * scale
        if parameter.grad is None:
            parameter.grad = restored.clone()
        else:
            parameter.grad.detach().copy_(restored)
    return True


def _zero_actor_gradients(actor: Any) -> None:
    optimizer = getattr(actor, "actor_optimizer", None)
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
        return
    for parameter in _trainable_parameters(actor):
        parameter.grad = None


def _subtract_grad_chunks(
    left_chunks: tuple[torch.Tensor, ...],
    right_chunks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | None:
    if len(left_chunks) != len(right_chunks):
        return None
    pieces: list[torch.Tensor] = []
    for left, right in zip(left_chunks, right_chunks):
        if left.numel() != right.numel():
            return None
        pieces.append(left.float() - right.float())
    return tuple(pieces)


def _chunk_norm_sq(chunks: tuple[torch.Tensor, ...]) -> float:
    local_sumsq = 0.0
    for chunk in chunks:
        chunk_float = chunk.float()
        local_sumsq += torch.dot(chunk_float, chunk_float).item()
    return _all_reduce_sum(local_sumsq)


def _current_grad_streaming_stats(
    actor: Any,
    reference_chunks: tuple[torch.Tensor, ...],
) -> tuple[float, float, float] | None:
    parameters = _trainable_parameters(actor)
    if len(parameters) != len(reference_chunks):
        return None

    scale = _current_grad_scale(actor)
    reference_sumsq = 0.0
    current_sumsq = 0.0
    reference_current_dot = 0.0
    for parameter, reference in zip(parameters, reference_chunks):
        if reference.numel() != parameter.numel():
            return None
        reference_float = reference.float()
        reference_sumsq += torch.dot(reference_float, reference_float).item()

        current = _current_grad_cpu_float(parameter, scale)
        if current is None:
            continue
        current_sumsq += torch.dot(current, current).item()
        reference_current_dot += torch.dot(reference_float, current).item()

    return (
        _all_reduce_sum(reference_sumsq),
        _all_reduce_sum(current_sumsq),
        _all_reduce_sum(reference_current_dot),
    )


def _current_grad_anchor_dots(
    actor: Any,
    reference_chunks: tuple[torch.Tensor, ...],
    anchor_vector: torch.Tensor,
) -> tuple[float, float] | None:
    parameters = _trainable_parameters(actor)
    if len(parameters) != len(reference_chunks):
        return None

    scale = _current_grad_scale(actor)
    offset = 0
    reference_anchor_dot = 0.0
    current_anchor_dot = 0.0
    anchor = anchor_vector.detach().reshape(-1)
    for parameter, reference in zip(parameters, reference_chunks):
        if reference.numel() != parameter.numel():
            return None
        next_offset = offset + reference.numel()
        if next_offset > anchor.numel():
            return None
        anchor_chunk = anchor[offset:next_offset].float()
        reference_anchor_dot += torch.dot(reference.float(), anchor_chunk).item()

        current = _current_grad_cpu_float(parameter, scale)
        if current is not None:
            current_anchor_dot += torch.dot(current, anchor_chunk).item()
        offset = next_offset

    if offset != anchor.numel():
        return None
    return _all_reduce_sum(reference_anchor_dot), _all_reduce_sum(current_anchor_dot)


def _collect_autograd_vector(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
    storage_dtype: str,
) -> torch.Tensor:
    dtype = torch.float16 if str(storage_dtype).lower() in {"float16", "fp16", "half"} else torch.float32
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
        else:
            pieces.append(gradient.detach().reshape(-1).to(device="cpu", dtype=dtype))
    if not pieces:
        return torch.zeros(0, dtype=dtype, device="cpu")
    return torch.cat(pieces)


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


def _teacher_log_prob(
    worker: Any,
    model_inputs: dict[str, Any],
    labels: list[str],
    reference: torch.Tensor,
    actor_cfg: Any | None = None,
):
    old_log_prob = model_inputs.get("old_log_probs", reference).detach()
    ref_log_prob = model_inputs.get("ref_log_prob", old_log_prob).detach()
    base_log_prob = model_inputs.get("base_log_prob", old_log_prob).detach()
    base_ref_log_prob = model_inputs.get("base_ref_log_prob", ref_log_prob).detach()
    teacher_log_prob = torch.zeros_like(reference)
    for idx, label in enumerate(labels):
        teacher_log_prob[idx] = base_ref_log_prob[idx] if label == "code" else ref_log_prob[idx]
    policy_loss_cfg = _policy_loss_config(worker, actor_cfg)
    lambda_vals = float(_cfg_get(policy_loss_cfg, "lambda_vals", 1.0))
    if lambda_vals == 1.0:
        return old_log_prob, teacher_log_prob
    corrected_teacher = base_log_prob + (teacher_log_prob - base_log_prob) * lambda_vals
    return old_log_prob, corrected_teacher


def _actor_reverse_kl_advantages(actor: Any, model_inputs: dict[str, Any], old_log_prob: torch.Tensor) -> torch.Tensor:
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    if not bool(_cfg_get(policy_loss_cfg, "only_reverse_kl_advantages", False)):
        return model_inputs["advantages"]

    if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
        lambda_vals = float(_cfg_get(policy_loss_cfg, "lambda_vals", 1.0))
        if bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False)) and "opd_teacher" in model_inputs:
            opd_teacher = model_inputs["opd_teacher"]
            reverse_kl = torch.zeros_like(old_log_prob)
            for idx in range(old_log_prob.shape[0]):
                teacher_type = opd_teacher[idx] if isinstance(opd_teacher, (list, tuple, np.ndarray)) else opd_teacher
                if teacher_type == "math":
                    teacher_log_prob = model_inputs["ref_log_prob"][idx]
                elif teacher_type == "code":
                    teacher_log_prob = model_inputs["base_ref_log_prob"][idx]
                else:
                    teacher_log_prob = model_inputs["ref_log_prob"][idx]
                if lambda_vals == 1.0:
                    reverse_kl[idx] = old_log_prob[idx] - teacher_log_prob
                else:
                    base_log_prob = model_inputs["base_log_prob"][idx]
                    reverse_kl[idx] = old_log_prob[idx] - base_log_prob - (teacher_log_prob - base_log_prob) * lambda_vals
            return -reverse_kl

        reverse_kl = old_log_prob - model_inputs["base_log_prob"]
        reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]
        if lambda_vals == 1.0:
            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
        else:
            reverse_kl = reverse_kl - reward_correction * lambda_vals
        return -reverse_kl

    if (
        "base_ref_log_prob" in model_inputs
        and bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False))
        and "opd_teacher" in model_inputs
    ):
        opd_teacher = model_inputs["opd_teacher"]
        reverse_kl = torch.zeros_like(old_log_prob)
        for idx in range(old_log_prob.shape[0]):
            teacher_type = opd_teacher[idx] if isinstance(opd_teacher, (list, tuple, np.ndarray)) else opd_teacher
            teacher_log_prob = model_inputs["base_ref_log_prob"][idx] if teacher_type == "code" else model_inputs["ref_log_prob"][idx]
            reverse_kl[idx] = old_log_prob[idx] - teacher_log_prob
        return -reverse_kl

    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
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
    if bool(_cfg_get(actor.config, "use_kl_loss", False)) and "ref_log_prob" in model_inputs:
        kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
        if kl_coef != 0:
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["ref_log_prob"],
                kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            )
            policy_loss = policy_loss + kl_loss * kl_coef
    return policy_loss * float(loss_scale_factor)


def _backward_domain_loss(worker: Any, data: DataProto) -> None:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

    if len(data) == 0:
        return
    tensor_keys = [
        "responses",
        "response_mask",
        "input_ids",
        "attention_mask",
        "position_ids",
        "old_log_probs",
        "ref_log_prob",
        "base_log_prob",
        "base_ref_log_prob",
        "rollout_is_weights",
    ]
    tensor_keys = [key for key in tensor_keys if key in data.batch]
    non_tensor_keys = [
        key
        for key in ("multi_modal_inputs", "opd_teacher", "domain", "source_domain", "ability", "data_source", "extra_info")
        if key in data.non_tensor_batch
    ]
    data = data.select(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
    data.meta_info = dict(data.meta_info)
    audit_cfg = data.meta_info.get("mopd_full_gradient", {})
    temperature = float(data.meta_info.get("temperature", worker.config.rollout.temperature))
    micro_batch_size = max(1, int(audit_cfg.get("micro_batch_size_per_gpu", 1)))
    micro_batches = data.split(micro_batch_size)
    total_tokens = max(_response_token_count(data), 1.0)
    actor_cfg = getattr(worker.actor, "config", _cfg_get(worker.config, "actor", {}))
    policy_loss_cfg = _policy_loss_config(worker, actor_cfg)
    loss_mode = str(_cfg_get(policy_loss_cfg, "loss_mode", "vanilla"))
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    entropy_coeff = float(_cfg_get(actor_cfg, "entropy_coeff", 0.0) or 0.0)
    loss_agg_mode = str(_cfg_get(actor_cfg, "loss_agg_mode", "token-mean"))
    use_kl_loss = bool(_cfg_get(actor_cfg, "use_kl_loss", False))
    kl_loss_coef = float(_cfg_get(actor_cfg, "kl_loss_coef", 0.0) or 0.0)
    kl_loss_type = str(_cfg_get(actor_cfg, "kl_loss_type", "kl"))
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        response_mask = model_inputs.get("response_mask")
        entropy, log_prob = worker.actor._forward_micro_batch(
            model_inputs,
            temperature=temperature,
            calculate_entropy=entropy_coeff != 0,
        )
        if response_mask is None:
            response_mask = torch.ones_like(log_prob, dtype=torch.float32, device=log_prob.device)
        else:
            response_mask = response_mask.float()
        old_log_prob, teacher_log_prob = _teacher_log_prob(
            worker,
            model_inputs,
            _teacher_labels(micro_batch),
            log_prob,
            actor_cfg,
        )
        reverse_kl = old_log_prob - teacher_log_prob
        advantages = -reverse_kl.detach()
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=actor_cfg,
            rollout_is_weights=model_inputs.get("rollout_is_weights", None),
        )
        policy_loss = pg_loss
        if entropy_coeff != 0 and entropy is not None:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss - entropy_loss * entropy_coeff
        if use_kl_loss and kl_loss_coef != 0 and "ref_log_prob" in model_inputs:
            kld = kl_penalty(logprob=log_prob, ref_logprob=model_inputs["ref_log_prob"], kl_penalty=kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = policy_loss + kl_loss * kl_loss_coef
        loss = policy_loss * (_response_token_count(micro_batch) / total_tokens)
        loss.backward()


def _gradient_vector(worker: Any, data: DataProto, storage_dtype: str) -> torch.Tensor:
    _zero_grad(worker)
    _backward_domain_loss(worker, data)
    vector = _collect_grad_vector(worker, storage_dtype)
    _zero_grad(worker)
    return vector


def _init_anchor_state(worker: Any) -> None:
    if not hasattr(worker, "_mopd_full_gradient_anchor_vectors"):
        worker._mopd_full_gradient_anchor_vectors = {}
        worker._mopd_full_gradient_anchor_counts = {}
        worker._mopd_full_gradient_anchor_token_counts = {}
        worker._mopd_full_gradient_anchor_step = None


def _weighted_anchor_mean(
    old_vector: torch.Tensor | None,
    old_token_count: float,
    new_vector: torch.Tensor,
    new_token_count: float,
) -> torch.Tensor:
    if old_vector is None or old_token_count <= 0:
        return new_vector
    total_token_count = max(old_token_count + new_token_count, 1.0)
    return (old_vector * old_token_count + new_vector * new_token_count) / total_token_count


def _sync_anchor_state_to_actor(worker: Any) -> None:
    actor = getattr(worker, "actor", None)
    if actor is None:
        return
    actor._mopd_full_gradient_anchor_vectors = worker._mopd_full_gradient_anchor_vectors
    actor._mopd_full_gradient_anchor_counts = worker._mopd_full_gradient_anchor_counts
    actor._mopd_full_gradient_anchor_token_counts = worker._mopd_full_gradient_anchor_token_counts
    actor._mopd_full_gradient_anchor_step = worker._mopd_full_gradient_anchor_step


def _update_validation_anchors(worker: Any, data: DataProto, cfg: dict[str, Any]) -> dict[str, float]:
    _init_anchor_state(worker)
    step = int(cfg.get("step", 0))
    refresh_steps = int(cfg.get("validation_anchor_refresh_steps", 0))
    if worker._mopd_full_gradient_anchor_step is None:
        worker._mopd_full_gradient_anchor_step = step
    elif step != worker._mopd_full_gradient_anchor_step and (
        refresh_steps <= 0 or step - worker._mopd_full_gradient_anchor_step >= refresh_steps
    ):
        worker._mopd_full_gradient_anchor_vectors.clear()
        worker._mopd_full_gradient_anchor_counts.clear()
        worker._mopd_full_gradient_anchor_token_counts.clear()
        worker._mopd_full_gradient_anchor_step = step

    labels = _validation_labels(data)
    domains = list(cfg.get("domains", []))
    max_samples = _max_samples_from_cfg(cfg)
    storage_dtype = str(cfg.get("storage_dtype", "float32"))
    metrics: dict[str, float] = {}
    for domain, indices in _indices_by_label(labels, domains, max_samples).items():
        current_count = int(worker._mopd_full_gradient_anchor_counts.get(domain, 0))
        selected = indices if max_samples is None else indices[: max(0, max_samples - current_count)]
        if not selected:
            continue
        selected_data = data.select_idxs(selected)
        vector = _gradient_vector(worker, selected_data, storage_dtype)
        selected_token_count = _all_reduce_sum(_response_token_count(selected_data))
        current_token_count = float(worker._mopd_full_gradient_anchor_token_counts.get(domain, 0.0))
        worker._mopd_full_gradient_anchor_vectors[domain] = _weighted_anchor_mean(
            worker._mopd_full_gradient_anchor_vectors.get(domain),
            current_token_count,
            vector,
            selected_token_count,
        )
        worker._mopd_full_gradient_anchor_token_counts[domain] = current_token_count + selected_token_count
        worker._mopd_full_gradient_anchor_counts[domain] = current_count + len(selected)
        safe_domain = _safe_name(domain)
        anchor_vector = worker._mopd_full_gradient_anchor_vectors[domain]
        metrics[f"{safe_domain}/full_grad_anchor/validation_anchor_sample_count"] = _all_reduce_sum(
            worker._mopd_full_gradient_anchor_counts[domain]
        )
        metrics[f"{safe_domain}/full_grad_anchor/validation_anchor_token_count"] = (
            worker._mopd_full_gradient_anchor_token_counts[domain]
        )
        metrics[f"{safe_domain}/full_grad_anchor/validation_anchor_grad_norm"] = _vector_norm(anchor_vector)
    metrics["global/audit/full_gradient_anchor_available"] = float(
        bool(worker._mopd_full_gradient_anchor_vectors)
    )
    metrics["global/audit/full_gradient_anchor_count"] = _all_reduce_sum(
        sum(worker._mopd_full_gradient_anchor_counts.values())
    )
    metrics["global/audit/full_gradient_anchor_token_count"] = sum(
        worker._mopd_full_gradient_anchor_token_counts.values()
    )
    _sync_anchor_state_to_actor(worker)
    return metrics


class SequentialBackwardDomainGradientTracker:
    """Track domain and sample gradient geometry from the real actor backward pass."""

    def __init__(self, actor: Any, cfg: dict[str, Any]):
        self.actor = actor
        self.cfg = cfg
        self.domains = list(cfg.get("domains", []))
        self.storage_dtype = str(cfg.get("storage_dtype", "float32"))
        self.learning_rate = float(cfg.get("learning_rate", 0.0) or 0.0)
        self.step = int(cfg.get("step", 0) or 0)
        self.domain_gradient_enabled = bool(cfg.get("domain_gradient_enabled", cfg.get("enabled", False)))
        self.sample_gradient_enabled = bool(cfg.get("sample_gradient_enabled", False))
        self.sample_norm_enabled = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_norm_enabled", True))
        self.sample_cos_enabled = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_cos_enabled", False))
        self.sample_cos_max_samples_per_domain = _max_samples_from_cfg(
            {"max_samples_per_domain": cfg.get("sample_gradient_cos_max_samples_per_domain", 8)}
        )
        self.sample_cos_selection = str(cfg.get("sample_gradient_cos_selection", "top_norm_plus_random"))
        self.sample_log_sample_level = bool(cfg.get("sample_gradient_log_sample_level", True))
        self.sample_seed = int(cfg.get("sample_gradient_seed", 17) or 17)
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
        self._sample_restore_grad_chunks: tuple[torch.Tensor, ...] | None = None

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
        self._sample_restore_grad_chunks = None
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
        anchor_vectors = getattr(self.actor, "_mopd_full_gradient_anchor_vectors", {})
        metrics: dict[str, float] = {
            "global/audit/full_gradient_anchor_available": float(bool(anchor_vectors)),
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/audit/full_gradient_domain_sequential_available": 0.0,
            "global/audit/full_gradient_domain_sequential_unsupported": float(not self._prepared_supported),
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / (1024**3),
        }
        first_chunks = self._first_domain_chunks
        self._first_domain_chunks = None
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}

        if self.domain_gradient_enabled and self._prepared_supported and len(self.domains) == 2 and first_chunks is not None:
            domain_metrics, domain_targets = self._finish_domain_gradient_metrics(first_chunks, anchor_vectors)
            metrics.update(domain_metrics)

        metrics.update(self._sample_norm_metrics())
        if self.sample_cos_enabled and domain_targets:
            metrics.update(self._sample_cos_metrics(domain_targets))
        if self.sample_log_sample_level:
            _write_jsonl_rows(self.output_dir, "sample_grad_metrics.jsonl", self._sample_records)
        self._remove_sample_norm_hooks()
        self._active_norm_parts = None
        self._active_norm_context = None
        self._sample_restore_grad_chunks = None
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
            "selected_for_cos": False,
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
                self._sample_candidates.setdefault(domain, []).append({"row": row, "micro_batch": stored_micro_batch})

    def _finish_domain_gradient_metrics(
        self,
        first_chunks: tuple[torch.Tensor, ...],
        anchor_vectors: dict[str, torch.Tensor],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        stats = _current_grad_streaming_stats(self.actor, first_chunks)
        if stats is None:
            return metrics, domain_targets
        first_norm_sq, total_norm_sq, first_total_dot = stats
        if first_norm_sq <= 0.0 or total_norm_sq <= 0.0:
            return metrics, domain_targets

        first_domain, second_domain = self.domains[0], self.domains[1]
        first_norm = first_norm_sq**0.5
        total_norm = total_norm_sq**0.5
        second_norm_sq = max(total_norm_sq + first_norm_sq - 2.0 * first_total_dot, 0.0)
        second_norm = second_norm_sq**0.5
        first_second_dot = first_total_dot - first_norm_sq
        second_total_dot = total_norm_sq - first_total_dot

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

        if self.sample_cos_enabled:
            total_chunks = _snapshot_current_grad_chunks(self.actor, self.storage_dtype)
            self._sample_restore_grad_chunks = total_chunks
            second_chunks = _subtract_grad_chunks(total_chunks, first_chunks)
            domain_targets[first_domain] = (first_chunks, first_norm_sq)
            if second_chunks is not None:
                domain_targets[second_domain] = (second_chunks, second_norm_sq)

        for val_domain, anchor_vector in sorted(anchor_vectors.items()):
            safe_val_domain = _safe_name(val_domain)
            anchor_norm = _vector_norm(anchor_vector)
            anchor_dots = _current_grad_anchor_dots(self.actor, first_chunks, anchor_vector)
            if anchor_dots is None:
                continue
            first_anchor_dot, total_anchor_dot = anchor_dots
            second_anchor_dot = total_anchor_dot - first_anchor_dot
            metrics[f"{first_safe}/full_grad_anchor/{safe_val_domain}/predicted_val_opd_loss_delta_i_j"] = (
                -self.learning_rate * first_anchor_dot
            )
            first_anchor_cosine = _safe_cosine(first_anchor_dot, first_norm, anchor_norm)
            if first_anchor_cosine is not None:
                metrics[f"{first_safe}/full_grad_anchor/{safe_val_domain}/full_grad_cosine_i_j"] = first_anchor_cosine
            metrics[f"{second_safe}/full_grad_anchor/{safe_val_domain}/predicted_val_opd_loss_delta_i_j"] = (
                -self.learning_rate * second_anchor_dot
            )
            second_anchor_cosine = _safe_cosine(second_anchor_dot, second_norm, anchor_norm)
            if second_anchor_cosine is not None:
                metrics[f"{second_safe}/full_grad_anchor/{safe_val_domain}/full_grad_cosine_i_j"] = second_anchor_cosine

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
            selected = self._select_sample_candidates(domain, candidates)
            if not selected:
                continue
            target_chunks, target_norm_sq = domain_targets[domain]
            target_norm = target_norm_sq**0.5
            cos_values: list[float] = []
            share_values: list[float] = []
            for candidate in selected:
                row = candidate["row"]
                stats = self._recompute_sample_to_domain_stats(
                    candidate["micro_batch"],
                    target_chunks=target_chunks,
                    target_norm=target_norm,
                    target_norm_sq=target_norm_sq,
                    loss_scale_factor=float(row.get("loss_scale_factor", 1.0)),
                    on_policy=bool(row.get("on_policy", False)),
                )
                row["selected_for_cos"] = True
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

    def _select_sample_candidates(self, domain: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = self.sample_cos_max_samples_per_domain
        if limit is None or limit >= len(candidates):
            return list(candidates)
        if limit <= 0:
            return []
        ordered = sorted(candidates, key=lambda item: float(item["row"].get("sample_grad_norm", 0.0)), reverse=True)
        if self.sample_cos_selection == "top_norm":
            return ordered[:limit]
        rng = random.Random(f"{self.sample_seed}:{self.step}:{domain}")
        if self.sample_cos_selection == "random":
            return rng.sample(candidates, limit)
        top_count = max(1, limit // 2)
        selected = ordered[:top_count]
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in candidates if id(item) not in selected_ids]
        if remaining and len(selected) < limit:
            selected.extend(rng.sample(remaining, min(limit - len(selected), len(remaining))))
        return selected

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
        if local_norm_sq <= 0.0 and self._sample_restore_grad_chunks is not None:
            local_norm_sq, local_dot = self._recompute_sample_stats_with_backward(
                micro_batch,
                target_chunks=target_chunks,
                loss_scale_factor=loss_scale_factor,
                on_policy=on_policy,
            )
        if local_norm_sq is None or local_dot is None:
            return {"sample_to_domain_cos": None, "sample_projection_share": None, "sample_recompute_grad_norm": None}
        norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
        dot = _all_reduce_sum(local_dot)
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
            gradient_cpu = gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            target_cpu = target.reshape(-1).float()
            if gradient_cpu.numel() != target_cpu.numel():
                return None, None
            local_norm_sq += torch.dot(gradient_cpu, gradient_cpu).item()
            local_dot += torch.dot(gradient_cpu, target_cpu).item()
        return local_norm_sq, local_dot

    def _recompute_sample_stats_with_backward(
        self,
        micro_batch: DataProto,
        *,
        target_chunks: tuple[torch.Tensor, ...],
        loss_scale_factor: float,
        on_policy: bool,
    ) -> tuple[float | None, float | None]:
        restore_chunks = self._sample_restore_grad_chunks
        if restore_chunks is None:
            return None, None
        _zero_actor_gradients(self.actor)
        try:
            loss = _actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=loss_scale_factor,
                on_policy=on_policy,
            )
            loss.backward()
            sample_chunks = _snapshot_current_grad_chunks(self.actor, self.storage_dtype)
            return self._chunk_stats(sample_chunks, target_chunks)
        finally:
            _zero_actor_gradients(self.actor)
            _restore_grad_chunks(self.actor, restore_chunks)

    def _chunk_stats(
        self,
        sample_chunks: tuple[torch.Tensor, ...],
        target_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[float | None, float | None]:
        if len(sample_chunks) != len(target_chunks):
            return None, None
        local_norm_sq = 0.0
        local_dot = 0.0
        for sample, target in zip(sample_chunks, target_chunks):
            if sample.numel() != target.numel():
                return None, None
            sample_float = sample.reshape(-1).float()
            target_float = target.reshape(-1).float()
            local_norm_sq += torch.dot(sample_float, sample_float).item()
            local_dot += torch.dot(sample_float, target_float).item()
        return local_norm_sq, local_dot


class SameForwardDomainGradientProbe:
    """Collect per-domain gradients from the current actor forward graph."""

    def __init__(self, actor: Any, cfg: dict[str, Any]):
        self.actor = actor
        self.cfg = cfg
        self.domains = list(cfg.get("domains", []))
        self.max_samples = _max_samples_from_cfg(cfg)
        self.storage_dtype = str(cfg.get("storage_dtype", "float32"))
        self.learning_rate = float(cfg.get("learning_rate", 0.0) or 0.0)
        self._vectors: dict[str, torch.Tensor] = {}
        self._sample_counts: dict[str, int] = {}
        self._started_at = 0.0

    def start_mini_batch(self) -> None:
        self._vectors = {}
        self._sample_counts = {}
        self._started_at = time.perf_counter()

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
            if domain in self._vectors:
                self._vectors[domain] = self._vectors[domain] + vector
            else:
                self._vectors[domain] = vector
            self._sample_counts[domain] = self._sample_counts.get(domain, 0) + len(indices)

    def finish_mini_batch(self) -> dict[str, float]:
        anchor_vectors = getattr(self.actor, "_mopd_full_gradient_anchor_vectors", {})
        metrics: dict[str, float] = {
            "global/audit/full_gradient_anchor_available": float(bool(anchor_vectors)),
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": get_torch_device().max_memory_allocated() / (1024**3),
        }
        fallback_norms: dict[str, float] = {}
        autograd_norms = {domain: _vector_norm(vector) for domain, vector in self._vectors.items()}
        active_domains = [domain for domain, count in self._sample_counts.items() if count > 0]
        if active_domains and all(norm <= 0 for norm in autograd_norms.values()):
            true_backward_norm = _parameter_grad_norm(_trainable_parameters(self.actor))
            if len(active_domains) == 1 and true_backward_norm > 0:
                fallback_norms[active_domains[0]] = true_backward_norm
                self._vectors.setdefault(active_domains[0], torch.zeros(0, dtype=torch.float32, device="cpu"))
                metrics["global/audit/full_gradient_true_backward_fallback"] = 1.0
            else:
                metrics["global/audit/full_gradient_autograd_unavailable"] = 1.0
        norms: dict[str, float] = {}
        for domain, vector in sorted(self._vectors.items()):
            if vector.numel() == 0 and domain not in fallback_norms:
                continue
            safe_domain = _safe_name(domain)
            grad_norm = fallback_norms.get(domain, _vector_norm(vector))
            norms[domain] = grad_norm
            metrics[f"{safe_domain}/full_grad/grad_norm"] = grad_norm
            metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(domain, 0))
            if domain in fallback_norms:
                continue
            for val_domain, anchor_vector in sorted(anchor_vectors.items()):
                safe_val_domain = _safe_name(val_domain)
                anchor_norm = _vector_norm(anchor_vector)
                grad_dot = _vector_dot(vector, anchor_vector)
                if grad_dot is None:
                    continue
                metrics[f"{safe_domain}/full_grad_anchor/{safe_val_domain}/predicted_val_opd_loss_delta_i_j"] = (
                    -self.learning_rate * grad_dot
                )
                if grad_norm > 0 and anchor_norm > 0:
                    metrics[f"{safe_domain}/full_grad_anchor/{safe_val_domain}/full_grad_cosine_i_j"] = (
                        grad_dot / (grad_norm * anchor_norm)
                    )
        domain_names = sorted(self._vectors)
        for left_idx, left in enumerate(domain_names):
            for right in domain_names[left_idx + 1 :]:
                grad_dot = _vector_dot(self._vectors[left], self._vectors[right])
                if grad_dot is None:
                    continue
                denom = norms[left] * norms[right]
                grad_cosine = None if denom <= 0 else grad_dot / denom
                if grad_cosine is not None:
                    pair = f"{_safe_name(left)}_vs_{_safe_name(right)}"
                    metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = grad_cosine
                    metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -grad_cosine)
        return metrics

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
        if bool(_cfg_get(self.actor.config, "use_kl_loss", False)) and "ref_log_prob" in model_inputs:
            kl_coef = float(_cfg_get(self.actor.config, "kl_loss_coef", 0.0) or 0.0)
            if kl_coef != 0:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=model_inputs["ref_log_prob"],
                    kl_penalty=str(_cfg_get(self.actor.config, "kl_loss_type", "kl")),
                )
                kl_loss = agg_loss(loss_mat=kld, loss_mask=domain_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = policy_loss + kl_loss * kl_coef
        return policy_loss

    def _gradient_vector(self, loss: torch.Tensor) -> torch.Tensor:
        parameters = _trainable_parameters(self.actor)
        if not parameters:
            return torch.zeros(0, dtype=torch.float32, device="cpu")
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        if not any(gradient is not None for gradient in gradients):
            return torch.zeros(0, dtype=torch.float32, device="cpu")
        return _collect_autograd_vector(parameters, gradients, self.storage_dtype)


def compute_mopd_full_gradient_metrics(worker: Any, data: DataProto) -> DataProto:
    """Compute validation-anchor full-parameter gradient metrics."""
    cfg = data.meta_info.get("mopd_full_gradient", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return DataProto(meta_info={"metrics": {}})
    if str(cfg.get("mode", "")) != "validation_anchor":
        return DataProto(meta_info={"metrics": {}})

    started_at = time.perf_counter()
    was_training = worker.actor.actor_module.training
    if worker._is_offload_param:
        load_fsdp_model_to_gpu(worker.actor_module_fsdp)
    try:
        worker.actor.actor_module.train()
        with worker.ulysses_sharding_manager:
            data = data.to("cpu")
            metrics = _update_validation_anchors(worker, data, cfg)
        metrics["global/full_grad_cost/backward_seconds"] = time.perf_counter() - started_at
        metrics["global/full_grad_cost/max_memory_allocated_gb"] = (
            get_torch_device().max_memory_allocated() / (1024**3)
        )
    finally:
        _zero_grad(worker)
        worker.actor.actor_module.train(was_training)
        if worker._is_offload_param:
            offload_fsdp_model_to_cpu(worker.actor_module_fsdp)
    return DataProto(meta_info={"metrics": metrics}).to("cpu")
