"""Full-parameter gradient audit helpers for patched verl FSDP workers."""

from __future__ import annotations

import time
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
        for key in ("multi_modal_inputs", "opd_teacher", "domain", "source_domain", "ability", "data_source")
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
    return metrics


def _compute_train_metrics(worker: Any, data: DataProto, cfg: dict[str, Any]) -> dict[str, float]:
    _init_anchor_state(worker)
    labels = _teacher_labels(data)
    domains = list(cfg.get("domains", []))
    max_samples = _max_samples_from_cfg(cfg)
    storage_dtype = str(cfg.get("storage_dtype", "float32"))
    learning_rate = float(cfg.get("learning_rate", 0.0) or 0.0)
    metrics: dict[str, float] = {
        "global/audit/full_gradient_anchor_available": float(bool(worker._mopd_full_gradient_anchor_vectors))
    }
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    for domain, indices in _indices_by_label(labels, domains, max_samples).items():
        if not indices:
            continue
        vector = _gradient_vector(worker, data.select_idxs(indices), storage_dtype)
        vectors[domain] = vector
        grad_norm = _vector_norm(vector)
        norms[domain] = grad_norm
        safe_domain = _safe_name(domain)
        metrics[f"{safe_domain}/full_grad/grad_norm"] = grad_norm
        metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(len(indices))
        for val_domain, anchor_vector in sorted(worker._mopd_full_gradient_anchor_vectors.items()):
            safe_val_domain = _safe_name(val_domain)
            anchor_norm = _vector_norm(anchor_vector)
            grad_dot = _vector_dot(vector, anchor_vector)
            if grad_dot is None:
                continue
            metrics[f"{safe_domain}/full_grad_anchor/{safe_val_domain}/predicted_val_opd_loss_delta_i_j"] = (
                -learning_rate * grad_dot
            )
            if grad_norm > 0 and anchor_norm > 0:
                metrics[f"{safe_domain}/full_grad_anchor/{safe_val_domain}/full_grad_cosine_i_j"] = (
                    grad_dot / (grad_norm * anchor_norm)
                )

    domain_names = sorted(vectors)
    for left_idx, left in enumerate(domain_names):
        for right in domain_names[left_idx + 1 :]:
            grad_dot = _vector_dot(vectors[left], vectors[right])
            if grad_dot is None:
                continue
            pair = f"{_safe_name(left)}_vs_{_safe_name(right)}"
            denom = norms[left] * norms[right]
            grad_cosine = None if denom <= 0 else grad_dot / denom
            if grad_cosine is not None:
                metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = grad_cosine
                metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -grad_cosine)
    return metrics


def compute_mopd_full_gradient_metrics(worker: Any, data: DataProto) -> DataProto:
    """Compute exact full-parameter gradient dot/cosine metrics on the configured domain subset."""
    cfg = data.meta_info.get("mopd_full_gradient", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return DataProto(meta_info={"metrics": {}})

    started_at = time.perf_counter()
    was_training = worker.actor.actor_module.training
    if worker._is_offload_param:
        load_fsdp_model_to_gpu(worker.actor_module_fsdp)
    try:
        worker.actor.actor_module.train()
        with worker.ulysses_sharding_manager:
            data = data.to("cpu")
            if str(cfg.get("mode", "train")) == "validation_anchor":
                metrics = _update_validation_anchors(worker, data, cfg)
            else:
                metrics = _compute_train_metrics(worker, data, cfg)
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
