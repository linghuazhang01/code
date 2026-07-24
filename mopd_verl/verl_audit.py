"""Lightweight MOPD audit helpers injected into the G-OPD verl trainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from mopd_verl.audit_math import (
    ece,
    finite_float,
)
from mopd_verl.audit_proxy import extract_sample_ids, extract_teacher_domains, response_mask_from_batch
from mopd_verl.audit_scalar_logging import (
    log_training_cost as _log_training_cost,
    log_validation_metrics as _log_validation_metrics,
)
from mopd_verl.audit_vector_cosine import iter_pairwise_domain_cosines
from mopd_verl.tensorboard_filter import (
    filter_tensorboard_metrics as _filter_tensorboard_metrics,
    is_direct_audit_metric_key,
)
from mopd_verl.tensorboard_tags import domain_metric_category, safe_name
from mopd_verl.topk_distill import (
    select_teacher_log_prob_tensor,
    teacher_tensor_prefix,
)


_DOMAIN_PARTITION_META_KEY = "mopd_domain_gradient_partition"


def _to_builtin(value: Any) -> Any:
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
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


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


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _var(values: list[float]) -> float | None:
    return float(np.var(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def _optional_positive_int(value: Any) -> int | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return max(1, int(value))


def _optional_bool_with_fallback(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return fallback
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected an optional boolean value, got {value!r}.")


def _mask_mean(matrix: Any, mask: Any) -> Any:
    import torch

    denom = mask.sum(dim=-1).clamp(min=1)
    return (matrix * mask).sum(dim=-1) / denom


def _masked_token_stats(matrix: Any, mask: Any) -> dict[str, float | None]:
    import torch

    denom = mask.sum()
    if float(denom.detach().cpu().item()) <= 0:
        return {"mean": None, "std": None, "variance": None}
    mean = (matrix * mask).sum() / denom
    sq_mean = (matrix.square() * mask).sum() / denom
    variance = torch.clamp(sq_mean - mean.square(), min=0.0)
    std = torch.sqrt(variance)
    return {
        "mean": float(mean.detach().cpu().item()),
        "std": float(std.detach().cpu().item()),
        "variance": float(variance.detach().cpu().item()),
    }


def _token_distribution_stats(values: Any, prefix: str) -> dict[str, float | None]:
    import torch

    if values is None or int(values.numel()) == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_variance": None,
            f"{prefix}_p05": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
            f"{prefix}_sum": None,
        }
    values = values.detach().float()
    mean = values.mean()
    variance = values.var(unbiased=False)
    return {
        f"{prefix}_mean": float(mean.detach().cpu().item()),
        f"{prefix}_std": float(torch.sqrt(variance).detach().cpu().item()),
        f"{prefix}_variance": float(variance.detach().cpu().item()),
        f"{prefix}_p05": float(torch.quantile(values, 0.05).detach().cpu().item()),
        f"{prefix}_p50": float(torch.quantile(values, 0.50).detach().cpu().item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).detach().cpu().item()),
        f"{prefix}_max": float(values.max().detach().cpu().item()),
        f"{prefix}_sum": float(values.sum().detach().cpu().item()),
    }


def _sample_value_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "variance": _var(values),
    }


def _tensor_to_float_list(tensor: Any) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().tolist()]


def _tensor_to_int_list(tensor: Any) -> list[int]:
    return [int(x) for x in tensor.detach().long().cpu().tolist()]


def _project_jsonl_rows(
    rows: list[dict[str, Any]],
    *,
    required_field: str,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {field: row[field] for field in fields if field in row}
        for row in rows
        if required_field in row
    ]


def _select_domain_tensor(
    *,
    tensor_batch: Any,
    batch_keys: set[str],
    labels: list[str],
    reference: Any,
    suffix: str,
    generic_key: str | None = None,
    fallback_domain: str = "math",
) -> Any | None:
    import torch

    domain_tensors = {
        key[: -len(suffix)]: tensor_batch[key].detach().float()
        for key in batch_keys
        if key.endswith(suffix)
    }
    generic_tensor = (
        tensor_batch[generic_key].detach().float()
        if generic_key is not None and generic_key in batch_keys
        else None
    )
    if not domain_tensors:
        return generic_tensor

    fallback_tensor = generic_tensor
    if fallback_tensor is None:
        fallback_tensor = domain_tensors.get(fallback_domain)
    if fallback_tensor is None:
        fallback_tensor = next(iter(domain_tensors.values()))
    selected = torch.zeros_like(reference)
    for idx, label in enumerate(labels):
        domain_key = teacher_tensor_prefix(label)
        selected[idx] = domain_tensors.get(domain_key, fallback_tensor)[idx]
    return selected


def _infer_tokenizer_vocab_size(tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None
    try:
        return int(len(tokenizer))
    except TypeError:
        pass
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if vocab is not None:
            return int(len(vocab))
    return None


def _infer_model_config_vocab_size(config: Any) -> int | None:
    actor_rollout_ref = _cfg_get(config, "actor_rollout_ref", {})
    model_config = _cfg_get(actor_rollout_ref, "model", {})
    model_path = _cfg_get(model_config, "path", None)
    if model_path is None:
        return None

    model_path_text = str(model_path)
    config_path = Path(model_path_text) / "config.json"
    try:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as handle:
                raw_config = json.load(handle)
            vocab_size = _optional_positive_int(raw_config.get("vocab_size"))
            if vocab_size is not None:
                return vocab_size
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    try:
        from transformers import AutoConfig

        trust_remote_code = bool(_cfg_get(model_config, "trust_remote_code", False))
        hf_config = AutoConfig.from_pretrained(model_path_text, trust_remote_code=trust_remote_code)
        return _optional_positive_int(getattr(hf_config, "vocab_size", None))
    except Exception:
        return None


def _response_token_id_matrix(tensor_batch: Any, batch_keys: set[str], response_mask: Any) -> Any | None:
    if "responses" in batch_keys:
        token_ids = tensor_batch["responses"]
    elif "response_ids" in batch_keys:
        token_ids = tensor_batch["response_ids"]
    elif "input_ids" in batch_keys:
        token_ids = tensor_batch["input_ids"]
    else:
        return None

    if not hasattr(token_ids, "detach") or len(token_ids.shape) != 2:
        return None

    response_len = int(response_mask.shape[-1])
    if tuple(token_ids.shape) == tuple(response_mask.shape):
        return token_ids.detach().long().cpu()
    if int(token_ids.shape[0]) == int(response_mask.shape[0]) and int(token_ids.shape[-1]) >= response_len:
        return token_ids[:, -response_len:].detach().long().cpu()
    return None


def _token_gap_vocab_tensors(
    *,
    token_ids: Any,
    response_mask: Any,
    gap_signed: Any,
    gap_abs: Any,
    vocab_size: int,
) -> dict[str, Any] | None:
    import torch

    valid = response_mask.detach().bool().cpu()
    flat_ids = token_ids.detach().long().cpu()[valid]
    if int(flat_ids.numel()) == 0:
        return None

    flat_signed = gap_signed.detach().float().cpu()[valid]
    flat_abs = gap_abs.detach().float().cpu()[valid]
    in_vocab = (flat_ids >= 0) & (flat_ids < int(vocab_size))
    dropped = int((~in_vocab).sum().item())
    flat_ids = flat_ids[in_vocab]
    flat_signed = flat_signed[in_vocab]
    flat_abs = flat_abs[in_vocab]
    if int(flat_ids.numel()) == 0:
        return None

    counts_int = torch.bincount(flat_ids, minlength=int(vocab_size))
    counts = counts_int.to(dtype=torch.float32)
    signed_sum = torch.zeros(int(vocab_size), dtype=torch.float32)
    abs_sum = torch.zeros(int(vocab_size), dtype=torch.float32)
    signed_sum.index_add_(0, flat_ids, flat_signed)
    abs_sum.index_add_(0, flat_ids, flat_abs)
    denom = counts.clamp(min=1.0)
    signed_mean = signed_sum / denom
    abs_mean = abs_sum / denom
    nonzero_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()

    return {
        "vocab_size": int(vocab_size),
        "observed_token_count": int(flat_ids.numel()),
        "dropped_token_count": dropped,
        "nonzero_token_id_count": int(nonzero_ids.numel()),
        "nonzero_token_ids": nonzero_ids,
        "token_count_vector_vocab": counts_int,
        "gap_signed_sum_vector_vocab": signed_sum,
        "gap_abs_sum_vector_vocab": abs_sum,
        "gap_signed_mean_vector_vocab": signed_mean,
        "gap_abs_mean_vector_vocab": abs_mean,
    }


def _token_gap_vocab_json_fields(
    vectors: dict[str, Any],
    *,
    include_mean_vectors: bool = True,
) -> dict[str, Any]:
    fields = {
        "vocab_size": int(vectors["vocab_size"]),
        "observed_token_count": int(vectors["observed_token_count"]),
        "dropped_token_count": int(vectors["dropped_token_count"]),
        "nonzero_token_id_count": int(vectors["nonzero_token_id_count"]),
        "nonzero_token_ids": _tensor_to_int_list(vectors["nonzero_token_ids"]),
        "token_count_vector_vocab": _tensor_to_int_list(vectors["token_count_vector_vocab"]),
        "gap_signed_sum_vector_vocab": _tensor_to_float_list(vectors["gap_signed_sum_vector_vocab"]),
        "gap_abs_sum_vector_vocab": _tensor_to_float_list(vectors["gap_abs_sum_vector_vocab"]),
    }
    if include_mean_vectors:
        fields.update(
            {
                "gap_signed_mean_vector_vocab": _tensor_to_float_list(
                    vectors["gap_signed_mean_vector_vocab"]
                ),
                "gap_abs_mean_vector_vocab": _tensor_to_float_list(
                    vectors["gap_abs_mean_vector_vocab"]
                ),
            }
        )
    return fields


def _logp_vocab_json_fields(
    vectors: dict[str, Any],
    *,
    include_mean_vectors: bool = True,
) -> dict[str, Any]:
    fields = {
        "vocab_size": int(vectors["vocab_size"]),
        "observed_token_count": int(vectors["observed_token_count"]),
        "dropped_token_count": int(vectors["dropped_token_count"]),
        "nonzero_token_id_count": int(vectors["nonzero_token_id_count"]),
        "nonzero_token_ids": _tensor_to_int_list(vectors["nonzero_token_ids"]),
        "token_count_vector_vocab": _tensor_to_int_list(vectors["token_count_vector_vocab"]),
        "logp_sum_vector_vocab": _tensor_to_float_list(vectors["gap_signed_sum_vector_vocab"]),
    }
    if include_mean_vectors:
        fields["logp_mean_vector_vocab"] = _tensor_to_float_list(
            vectors["gap_signed_mean_vector_vocab"]
        )
    return fields


def _logp_abs_vocab_json_fields(
    vectors: dict[str, Any],
    *,
    include_mean_vectors: bool = True,
) -> dict[str, Any]:
    fields = {
        "vocab_size": int(vectors["vocab_size"]),
        "observed_token_count": int(vectors["observed_token_count"]),
        "dropped_token_count": int(vectors["dropped_token_count"]),
        "nonzero_token_id_count": int(vectors["nonzero_token_id_count"]),
        "nonzero_token_ids": _tensor_to_int_list(vectors["nonzero_token_ids"]),
        "token_count_vector_vocab": _tensor_to_int_list(vectors["token_count_vector_vocab"]),
        "logp_abs_sum_vector_vocab": _tensor_to_float_list(vectors["gap_abs_sum_vector_vocab"]),
    }
    if include_mean_vectors:
        fields["logp_abs_mean_vector_vocab"] = _tensor_to_float_list(
            vectors["gap_abs_mean_vector_vocab"]
        )
    return fields


def _entropy_vocab_tensors(
    *,
    token_ids: Any,
    response_mask: Any,
    student_entropy: Any | None,
    teacher_student_cross_entropy: Any | None,
    vocab_size: int,
) -> dict[str, Any] | None:
    import torch

    valid = response_mask.detach().bool().cpu()
    flat_ids = token_ids.detach().long().cpu()[valid]
    if int(flat_ids.numel()) == 0:
        return None

    signal_values = {
        "student_entropy": None if student_entropy is None else student_entropy.detach().float().cpu()[valid],
        "teacher_student_cross_entropy": None
        if teacher_student_cross_entropy is None
        else teacher_student_cross_entropy.detach().float().cpu()[valid],
    }
    in_vocab = (flat_ids >= 0) & (flat_ids < int(vocab_size))
    dropped = int((~in_vocab).sum().item())
    flat_ids = flat_ids[in_vocab]
    if int(flat_ids.numel()) == 0:
        return None

    counts_int = torch.bincount(flat_ids, minlength=int(vocab_size))
    counts = counts_int.to(dtype=torch.float32)
    nonzero_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    output: dict[str, Any] = {
        "vocab_size": int(vocab_size),
        "observed_token_count": int(flat_ids.numel()),
        "dropped_token_count": dropped,
        "nonzero_token_id_count": int(nonzero_ids.numel()),
        "nonzero_token_ids": nonzero_ids,
        "token_count_vector_vocab": counts_int,
    }
    for name, values in signal_values.items():
        if values is None:
            continue
        flat_values = values[in_vocab]
        value_sum = torch.zeros(int(vocab_size), dtype=torch.float32)
        value_sum.index_add_(0, flat_ids, flat_values)
        output[f"{name}_sum_vector_vocab"] = value_sum
        output[f"{name}_mean_vector_vocab"] = value_sum / counts.clamp(min=1.0)
    return output


def _entropy_vocab_json_fields(
    vectors: dict[str, Any],
    *,
    include_mean_vectors: bool = True,
) -> dict[str, Any]:
    fields = {
        "vocab_size": int(vectors["vocab_size"]),
        "observed_token_count": int(vectors["observed_token_count"]),
        "dropped_token_count": int(vectors["dropped_token_count"]),
        "nonzero_token_id_count": int(vectors["nonzero_token_id_count"]),
        "nonzero_token_ids": _tensor_to_int_list(vectors["nonzero_token_ids"]),
        "token_count_vector_vocab": _tensor_to_int_list(vectors["token_count_vector_vocab"]),
    }
    for name in ("student_entropy", "teacher_student_cross_entropy"):
        sum_key = f"{name}_sum_vector_vocab"
        mean_key = f"{name}_mean_vector_vocab"
        if sum_key in vectors:
            fields[sum_key] = _tensor_to_float_list(vectors[sum_key])
        if include_mean_vectors and mean_key in vectors:
            fields[mean_key] = _tensor_to_float_list(vectors[mean_key])
    return fields



def _scalar_float(value: Any) -> float | None:
    converted = _to_builtin(value)
    if isinstance(converted, dict):
        for key in ("lr", "learning_rate"):
            numeric = _scalar_float(converted.get(key))
            if numeric is not None:
                return numeric
        return None
    if isinstance(converted, (list, tuple)):
        return _scalar_float(converted[0]) if converted else None
    return finite_float(converted)


def _ensure_meta_info(batch: Any) -> dict[str, Any]:
    meta_info = getattr(batch, "meta_info", None)
    if not isinstance(meta_info, dict):
        meta_info = {}
        setattr(batch, "meta_info", meta_info)
    return meta_info


class MOPDAuditLogger:
    """Writes per-domain audit JSONL rows and TensorBoard-compatible scalars."""

    def __init__(self, config: Any, tokenizer: Any | None = None):
        self.config = config
        self.tokenizer = tokenizer
        audit_config = _cfg_get(config, "mopd_audit", {})
        self.enabled = bool(_cfg_get(audit_config, "enabled", False))
        self.output_dir = Path(str(_cfg_get(audit_config, "output_dir", "mopd_audit")))
        self.domains = list(_cfg_get(audit_config, "domains", ["math", "code"]))
        self.prefix = str(_cfg_get(audit_config, "tensorboard_prefix", "mopd"))
        self.tensorboard_layout = str(_cfg_get(audit_config, "tensorboard_layout", "domain_category"))
        self.tensorboard_prune_mode = str(_cfg_get(audit_config, "tensorboard_prune_mode", "none")).lower()
        self.max_samples_per_domain = _optional_positive_int(_cfg_get(audit_config, "max_samples_per_domain", None))
        self.high_variance_cv_threshold = float(_cfg_get(audit_config, "high_variance_cv_threshold", 1.0))
        self.log_sample_level = bool(_cfg_get(audit_config, "log_sample_level", True))
        self.log_sample_level_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "log_sample_level_freq_steps", 1)),
        )
        self.log_validation = bool(_cfg_get(audit_config, "log_validation_metrics", True))
        self.log_validation_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "log_validation_metrics_freq_steps", 1)),
        )
        self.tier2_window_size = max(2, int(_cfg_get(audit_config, "tier2_window_size", 20)))
        self.calibration_bins = max(1, int(_cfg_get(audit_config, "calibration_bins", 10)))
        self.full_gradient_enabled = bool(_cfg_get(audit_config, "full_gradient_enabled", False))
        self.full_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "full_gradient_freq_steps", 1)))
        full_grad_training_parity_freq_steps = _cfg_get(
            audit_config,
            "full_grad_training_parity_freq_steps",
            1,
        )
        self.full_grad_training_parity_freq_steps = int(
            1 if full_grad_training_parity_freq_steps is None else full_grad_training_parity_freq_steps
        )
        self.full_grad_training_parity_rel_l2_threshold = float(
            _cfg_get(audit_config, "full_grad_training_parity_rel_l2_threshold", 1e-5)
        )
        self.full_gradient_train_max_samples_per_domain = _optional_positive_int(
            _cfg_get(audit_config, "full_gradient_train_max_samples_per_domain", None)
        )
        self.full_gradient_micro_batch_size_per_gpu = max(
            1,
            int(_cfg_get(audit_config, "full_gradient_micro_batch_size_per_gpu", 1)),
        )
        self.full_gradient_storage_dtype = str(_cfg_get(audit_config, "full_gradient_storage_dtype", "float32"))
        self.execution_timing = str(_cfg_get(audit_config, "execution_timing", "pre_update")).lower()
        self.full_gradient_direct_recompute_enabled = bool(
            _cfg_get(audit_config, "full_gradient_direct_recompute_enabled", True)
        )
        self.sequence_masked_target_enabled = bool(
            _cfg_get(audit_config, "sequence_masked_target_enabled", False)
        )
        self.sequence_masked_target_use_as_primary = bool(
            _cfg_get(audit_config, "sequence_masked_target_use_as_primary", False)
        )
        self.sequence_replay_skip_non_target_domains = bool(
            _cfg_get(audit_config, "sequence_replay_skip_non_target_domains", False)
        )
        self.sequence_masked_target_closure_rel_l2_threshold = float(
            _cfg_get(audit_config, "sequence_masked_target_closure_rel_l2_threshold", 0.02)
        )
        self.sample_gradient_enabled = bool(_cfg_get(audit_config, "sample_gradient_enabled", False))
        self.sample_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "sample_gradient_freq_steps", 1)))
        self.sample_gradient_norm_enabled = bool(_cfg_get(audit_config, "sample_gradient_norm_enabled", True))
        self.sample_gradient_cos_enabled = bool(_cfg_get(audit_config, "sample_gradient_cos_enabled", False))
        self.sample_gradient_cos_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "sample_gradient_cos_freq_steps", 1)),
        )
        self.sample_gradient_backward_recompute_enabled = bool(
            _cfg_get(audit_config, "sample_gradient_backward_recompute_enabled", True)
        )
        self.sample_gradient_backward_sync_enabled = bool(
            _cfg_get(audit_config, "sample_gradient_backward_sync_enabled", True)
        )
        self.sample_gradient_log_sample_level = bool(
            _cfg_get(audit_config, "sample_gradient_log_sample_level", True)
        )
        self.sample_gradient_log_sample_level_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "sample_gradient_log_sample_level_freq_steps", 1)),
        )
        self.full_gradient_offload_domain_gradients = bool(
            _cfg_get(audit_config, "full_gradient_offload_domain_gradients", True)
        )
        self.token_gap_enabled = bool(_cfg_get(audit_config, "token_gap_enabled", True))
        self.token_gap_freq_steps = max(1, int(_cfg_get(audit_config, "token_gap_freq_steps", 1)))
        self.token_gap_vocab_vector_enabled = bool(
            _cfg_get(audit_config, "token_gap_vocab_vector_enabled", False)
        )
        self.token_gap_vocab_vector_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "token_gap_vocab_vector_freq_steps", 1)),
        )
        self.token_gap_vocab_size = _optional_positive_int(_cfg_get(audit_config, "token_gap_vocab_size", None))
        self.vocab_per_occurrence_mean_vector_enabled = bool(
            _cfg_get(audit_config, "vocab_per_occurrence_mean_vector_enabled", True)
        )
        self.logp_vocab_per_occurrence_mean_vector_enabled = _optional_bool_with_fallback(
            _cfg_get(audit_config, "logp_vocab_per_occurrence_mean_vector_enabled", None),
            self.vocab_per_occurrence_mean_vector_enabled,
        )
        self.logp_abs_vocab_per_occurrence_mean_vector_enabled = _optional_bool_with_fallback(
            _cfg_get(audit_config, "logp_abs_vocab_per_occurrence_mean_vector_enabled", None),
            self.vocab_per_occurrence_mean_vector_enabled,
        )
        self.entropy_vocab_per_occurrence_mean_vector_enabled = _optional_bool_with_fallback(
            _cfg_get(audit_config, "entropy_vocab_per_occurrence_mean_vector_enabled", None),
            self.vocab_per_occurrence_mean_vector_enabled,
        )
        self.token_gap_vocab_size_source = "config" if self.token_gap_vocab_size is not None else "unavailable"
        if self.token_gap_vocab_size is None:
            self.token_gap_vocab_size = _infer_model_config_vocab_size(config)
            if self.token_gap_vocab_size is not None:
                self.token_gap_vocab_size_source = "model_config"
        if self.token_gap_vocab_size is None:
            self.token_gap_vocab_size = _infer_tokenizer_vocab_size(tokenizer)
            if self.token_gap_vocab_size is not None:
                self.token_gap_vocab_size_source = "tokenizer"
        self.entropy_enabled = bool(_cfg_get(audit_config, "entropy_enabled", True))
        self.entropy_freq_steps = max(1, int(_cfg_get(audit_config, "entropy_freq_steps", 1)))
        self.entropy_vocab_vector_enabled = bool(
            _cfg_get(audit_config, "entropy_vocab_vector_enabled", False)
        )
        self.entropy_vocab_vector_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "entropy_vocab_vector_freq_steps", 1)),
        )
        self.topk_teacher_student_cross_entropy_vocab_enabled = bool(
            _cfg_get(audit_config, "topk_teacher_student_cross_entropy_vocab_enabled", False)
        )
        self.topk_teacher_student_cross_entropy_vocab_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "topk_teacher_student_cross_entropy_vocab_freq_steps", 1)),
        )
        self.topk_teacher_student_cross_entropy_k = max(
            1,
            int(_cfg_get(audit_config, "topk_teacher_student_cross_entropy_k", 32)),
        )
        self.topk_teacher_student_cross_entropy_include_tail = bool(
            _cfg_get(audit_config, "topk_teacher_student_cross_entropy_include_tail", False)
        )
        self.topk_teacher_student_cross_entropy_temperature = max(
            1e-6,
            float(_cfg_get(audit_config, "topk_teacher_student_cross_entropy_temperature", 1.0)),
        )
        self.logp_vector_enabled = bool(
            _cfg_get(audit_config, "logp_vector_enabled", False)
        )
        self.logp_vector_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "logp_vector_freq_steps", 1)),
        )
        self.logp_abs_vector_enabled = bool(
            _cfg_get(audit_config, "logp_abs_vector_enabled", False)
        )
        self.logp_abs_vector_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "logp_abs_vector_freq_steps", 1)),
        )
        self.token_gradient_enabled = bool(_cfg_get(audit_config, "token_gradient_enabled", False))
        self.token_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "token_gradient_freq_steps", 10)))
        self.token_gradient_tail_enabled = bool(
            _cfg_get(audit_config, "token_gradient_tail_enabled", True)
        )
        self.token_gradient_tail_fraction = float(
            _cfg_get(audit_config, "token_gradient_tail_fraction", 0.10)
        )
        self.token_gradient_tail_min_tokens = max(
            1,
            int(_cfg_get(audit_config, "token_gradient_tail_min_tokens", 1)),
        )
        self.token_gradient_gap_selection_enabled = bool(
            _cfg_get(audit_config, "token_gradient_gap_selection_enabled", True)
        )
        self.token_gradient_gap_abs_selection_enabled = bool(
            _cfg_get(audit_config, "token_gradient_gap_abs_selection_enabled", True)
        )
        self.token_gradient_loss_abs_selection_enabled = bool(
            _cfg_get(audit_config, "token_gradient_loss_abs_selection_enabled", True)
        )
        token_gradient_top_k = _cfg_get(
            audit_config,
            "token_gradient_top_k",
            100,
        )
        self.token_gradient_top_k = (
            None
            if token_gradient_top_k is None
            else max(1, int(token_gradient_top_k))
        )
        self.token_gradient_top_p_enabled = bool(
            _cfg_get(audit_config, "token_gradient_top_p_enabled", False)
        )
        token_gradient_top_p = _cfg_get(audit_config, "token_gradient_top_p", 0.10)
        self.token_gradient_top_p = min(
            1.0,
            max(0.0, float(0.10 if token_gradient_top_p is None else token_gradient_top_p)),
        )
        self.token_gradient_log_tokens_jsonl_enabled = bool(
            _cfg_get(
                audit_config,
                "token_gradient_log_tokens_jsonl_enabled",
                True,
            )
        )
        self.token_gradient_strict_grad_restore = bool(
            _cfg_get(audit_config, "token_gradient_strict_grad_restore", False)
        )
        self.token_gradient_backward_recompute_enabled = bool(
            _cfg_get(audit_config, "token_gradient_backward_recompute_enabled", True)
        )
        self.token_gradient_backward_sync_enabled = bool(
            _cfg_get(audit_config, "token_gradient_backward_sync_enabled", True)
        )
        self.dynamic_domain_loss_weighting_enabled = bool(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_enabled",
                False,
            )
        )
        self.dynamic_domain_loss_weighting_freq_steps = max(
            1,
            int(
                _cfg_get(
                    audit_config,
                    "dynamic_domain_loss_weighting_freq_steps",
                    10,
                )
            ),
        )
        self.dynamic_domain_loss_weighting_ema_beta = float(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_ema_beta",
                0.90,
            )
        )
        self.dynamic_domain_loss_weighting_weight_ema_beta = float(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_weight_ema_beta",
                0.90,
            )
        )
        self.dynamic_domain_loss_weighting_alpha = float(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_alpha",
                0.50,
            )
        )
        self.dynamic_domain_loss_weighting_min = float(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_min",
                1.0 / 3.0,
            )
        )
        self.dynamic_domain_loss_weighting_max = float(
            _cfg_get(
                audit_config,
                "dynamic_domain_loss_weighting_max",
                3.0,
            )
        )
        policy_loss = _cfg_get(_cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "actor", {}), "policy_loss", {})
        self.lambda_vals = float(_cfg_get(policy_loss, "lambda_vals", 1.0))
        self._last_validation_metrics: dict[str, float] = {}
        self._validation_gain_history: dict[str, list[float]] = {}
        self._seen_sample_ids: dict[str, set[str]] = {domain: set() for domain in self.domains}
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tag(self, root: str, category: str, metric: str, *segments: str) -> str:
        parts = [safe_name(root), safe_name(category)]
        parts.extend(safe_name(segment) for segment in segments if segment)
        parts.append(safe_name(metric))
        if self.tensorboard_layout == "prefixed_domain_category" and self.prefix:
            parts.insert(0, safe_name(self.prefix))
        return "/".join(part for part in parts if part)

    def _domain_tag(self, domain: str, category: str, metric: str, *segments: str) -> str:
        return self._tag(domain, category, metric, *segments)

    def _global_tag(self, category: str, metric: str, *segments: str) -> str:
        return self._tag("global", category, metric, *segments)

    def _validation_tag_parts(self, key: str) -> tuple[str, str]:
        safe_domains = {safe_name(domain): safe_name(domain) for domain in self.domains}
        slash_parts = [part for part in str(key).replace("\\", "/").split("/") if part]
        for idx, part in enumerate(slash_parts):
            safe_part = safe_name(part)
            if safe_part in safe_domains:
                tail = [safe_name(item) for item in slash_parts[idx + 1 :]]
                if tail:
                    return safe_domains[safe_part], "_".join(tail)
                prefix = [safe_name(item) for item in slash_parts[:idx] if item not in {"val", "validation"}]
                return safe_domains[safe_part], "_".join(prefix) or "value"

        safe_key = safe_name(key)
        for domain in self.domains:
            safe_domain = safe_name(domain)
            for prefix in (f"val_{safe_domain}_", f"validation_{safe_domain}_", f"{safe_domain}_"):
                if safe_key.startswith(prefix):
                    return safe_domain, safe_key[len(prefix) :] or "value"
        return "global", safe_key

    def _is_direct_audit_metric_key(self, key: str) -> bool:
        return is_direct_audit_metric_key(str(key))

    def filter_tensorboard_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Return the TensorBoard-facing metric subset for compact monitoring."""

        return _filter_tensorboard_metrics(metrics, self.tensorboard_prune_mode)

    def _freq_active(self, enabled: bool, freq_steps: int, step: int) -> bool:
        return self.enabled and enabled and step % max(1, int(freq_steps)) == 0

    def should_log_sample_level(self, step: int) -> bool:
        return self._freq_active(self.log_sample_level, self.log_sample_level_freq_steps, step)

    def should_log_validation_metrics(self, step: int) -> bool:
        return self._freq_active(self.log_validation, self.log_validation_freq_steps, step)

    def should_log_token_gap(self, step: int) -> bool:
        return self._freq_active(self.token_gap_enabled, self.token_gap_freq_steps, step)

    def should_log_token_gap_vocab_vector(self, step: int) -> bool:
        return self._freq_active(
            self.token_gap_vocab_vector_enabled,
            self.token_gap_vocab_vector_freq_steps,
            step,
        )

    def should_log_entropy(self, step: int) -> bool:
        return self._freq_active(self.entropy_enabled, self.entropy_freq_steps, step)

    def should_log_entropy_vocab_vector(self, step: int) -> bool:
        return self._freq_active(
            self.entropy_vocab_vector_enabled,
            self.entropy_vocab_vector_freq_steps,
            step,
        )

    def should_log_topk_teacher_student_cross_entropy_vocab(self, step: int) -> bool:
        return self._freq_active(
            self.topk_teacher_student_cross_entropy_vocab_enabled,
            self.topk_teacher_student_cross_entropy_vocab_freq_steps,
            step,
        )

    def should_log_logp_vector(self, step: int) -> bool:
        return self._freq_active(
            self.logp_vector_enabled,
            self.logp_vector_freq_steps,
            step,
        )

    def should_log_logp_abs_vector(self, step: int) -> bool:
        return self._freq_active(
            self.logp_abs_vector_enabled,
            self.logp_abs_vector_freq_steps,
            step,
        )

    def should_compute_sample_gradient(self, step: int) -> bool:
        return self._freq_active(self.sample_gradient_enabled, self.sample_gradient_freq_steps, step)

    def should_log_sample_gradient_level(self, step: int) -> bool:
        return self._freq_active(
            self.sample_gradient_log_sample_level,
            self.sample_gradient_log_sample_level_freq_steps,
            step,
        )

    def should_compute_full_gradient(self, step: int) -> bool:
        full_gradient_active = self.should_compute_domain_gradient(step)
        sample_gradient_active = self.should_compute_sample_gradient(step) and (
            self.sample_gradient_norm_enabled
            or (
                self.sample_gradient_cos_enabled
                and step % self.sample_gradient_cos_freq_steps == 0
            )
        )
        return self.enabled and (
            full_gradient_active
            or sample_gradient_active
            or self.should_compute_token_gradient(step)
            or self.dynamic_domain_loss_weighting_enabled
        )

    def should_compute_domain_gradient(self, step: int) -> bool:
        full_gradient_active = self.full_gradient_enabled and step % self.full_gradient_freq_steps == 0
        return self.enabled and (
            full_gradient_active
            or self.should_compute_token_gradient(step)
            or self.should_update_dynamic_domain_loss_weighting(step)
        )

    def should_compute_token_gradient(self, step: int) -> bool:
        subset_enabled = (
            self.token_gradient_tail_enabled
            or self.token_gradient_top_p_enabled
        )
        return self._freq_active(
            self.token_gradient_enabled and subset_enabled,
            self.token_gradient_freq_steps,
            step,
        )

    def should_update_dynamic_domain_loss_weighting(self, step: int) -> bool:
        return self._freq_active(
            self.dynamic_domain_loss_weighting_enabled,
            self.dynamic_domain_loss_weighting_freq_steps,
            step,
        )

    def should_log_full_grad_training_parity(self, step: int) -> bool:
        freq_steps = int(self.full_grad_training_parity_freq_steps)
        return self.enabled and freq_steps >= 0 and step % max(1, freq_steps) == 0

    def inspect_domain_gradient_batch_layout(
        self,
        batch: Any,
        *,
        step: int,
        world_size: int,
    ) -> dict[str, float]:
        """Inspect domain layout without changing the production batch payload."""

        if not self.should_compute_domain_gradient(step):
            return {}

        meta_info = _ensure_meta_info(batch)
        partition_meta: dict[str, Any] = {
            "aligned": False,
            "unsupported_reason": "not_checked",
            "step": int(step),
            "world_size": int(world_size),
            "domains": list(self.domains),
            "domain_order": list(self.domains),
            "micro_batch_size_per_gpu": int(self.full_gradient_micro_batch_size_per_gpu),
            "inspection_only": True,
            "production_batch_reordered": False,
            "layout_source": "post_standard_trainer_balance",
        }
        meta_info[_DOMAIN_PARTITION_META_KEY] = partition_meta
        metrics = {
            "global/audit/full_gradient_domain_partition_aligned": 0.0,
            "global/audit/full_gradient_domain_partition_unsupported": 1.0,
            "global/audit/full_gradient_domain_partition_inspection_only": 1.0,
            "global/audit/full_gradient_domain_partition_batch_reordered": 0.0,
        }
        if world_size <= 0:
            partition_meta["unsupported_reason"] = "invalid_world_size"
            return metrics
        if not self.domains or "attention_mask" not in batch.batch:
            partition_meta["unsupported_reason"] = "requires_domains_and_attention_mask"
            return metrics

        attention_mask = batch.batch["attention_mask"]
        batch_size = int(attention_mask.shape[0])
        if batch_size == 0 or batch_size % world_size != 0:
            partition_meta["unsupported_reason"] = "batch_size_not_divisible_by_world_size"
            return metrics

        labels = extract_teacher_domains(batch.non_tensor_batch, batch_size)
        if set(labels) != set(self.domains):
            partition_meta["unsupported_reason"] = "domains_do_not_match_batch_labels"
            return metrics

        micro_batch_size = self.full_gradient_micro_batch_size_per_gpu
        required_multiple = world_size * micro_batch_size
        domain_indices = {
            domain: [idx for idx, label in enumerate(labels) if label == domain] for domain in self.domains
        }
        if any(not indices or len(indices) % required_multiple != 0 for indices in domain_indices.values()):
            partition_meta["unsupported_reason"] = "domain_counts_not_divisible_by_rank_micro_batch"
            return metrics

        expected_rank_size = batch_size // world_size
        rank_label_chunks = [
            labels[rank * expected_rank_size : (rank + 1) * expected_rank_size]
            for rank in range(world_size)
        ]
        rank_domain_sample_counts = [
            {domain: rank_labels.count(domain) for domain in self.domains}
            for rank_labels in rank_label_chunks
        ]
        partition_meta.update(
            {
                "rank_sample_count": int(expected_rank_size),
                "rank_domain_sample_counts": rank_domain_sample_counts,
            }
        )

        first_rank_counts = rank_domain_sample_counts[0]
        if any(counts != first_rank_counts for counts in rank_domain_sample_counts[1:]):
            partition_meta["unsupported_reason"] = "rank_domain_counts_not_aligned"
            return metrics

        expected_rank_labels = [
            domain
            for domain in self.domains
            for _ in range(first_rank_counts[domain])
        ]
        if any(rank_labels != expected_rank_labels for rank_labels in rank_label_chunks):
            partition_meta["unsupported_reason"] = "rank_domain_blocks_not_aligned"
            return metrics

        has_mixed_micro_batch = any(
            len(set(rank_labels[start : start + micro_batch_size])) != 1
            for rank_labels in rank_label_chunks
            for start in range(0, expected_rank_size, micro_batch_size)
        )
        if has_mixed_micro_batch:
            partition_meta["unsupported_reason"] = "rank_micro_batches_contain_mixed_domains"
            return metrics

        partition_meta.update(
            {
                "aligned": True,
                "unsupported_reason": "",
                # Retained for tracker compatibility. These are observed per-rank
                # counts; this inspector does not create or reorder domain blocks.
                "domain_block_sample_counts": dict(first_rank_counts),
            }
        )
        metrics["global/audit/full_gradient_domain_partition_aligned"] = 1.0
        metrics["global/audit/full_gradient_domain_partition_unsupported"] = 0.0
        return metrics

    def balance_domain_gradient_batch(
        self,
        batch: Any,
        *,
        step: int,
        world_size: int,
    ) -> dict[str, float]:
        """Compatibility wrapper for trainers patched before read-only inspection."""

        return self.inspect_domain_gradient_batch_layout(
            batch,
            step=step,
            world_size=world_size,
        )

    def full_gradient_meta(
        self,
        mode: str,
        step: int,
        domain_partition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "mopd_full_gradient": {
                "enabled": self.should_compute_full_gradient(step),
                "domain_gradient_enabled": self.should_compute_domain_gradient(step),
                "mode": mode,
                "step": step,
                "domains": self.domains,
                "output_dir": str(self.output_dir),
                "max_samples_per_domain": self.full_gradient_train_max_samples_per_domain,
                "micro_batch_size_per_gpu": self.full_gradient_micro_batch_size_per_gpu,
                "storage_dtype": self.full_gradient_storage_dtype,
                "execution_timing": self.execution_timing,
                "full_gradient_direct_recompute_enabled": self.full_gradient_direct_recompute_enabled,
                "sequence_masked_target_enabled": self.sequence_masked_target_enabled,
                "sequence_masked_target_use_as_primary": self.sequence_masked_target_use_as_primary,
                "sequence_replay_skip_non_target_domains": self.sequence_replay_skip_non_target_domains,
                "sequence_masked_target_closure_rel_l2_threshold": (
                    self.sequence_masked_target_closure_rel_l2_threshold
                ),
                "full_grad_training_parity_freq_steps": self.full_grad_training_parity_freq_steps,
                "full_grad_training_parity_rel_l2_threshold": (
                    self.full_grad_training_parity_rel_l2_threshold
                ),
                "learning_rate": self._current_learning_rate_value(),
                "sample_gradient_enabled": self.should_compute_sample_gradient(step) and mode == "train",
                "sample_gradient_freq_steps": self.sample_gradient_freq_steps,
                "sample_gradient_norm_enabled": self.sample_gradient_norm_enabled,
                "sample_gradient_cos_enabled": self.sample_gradient_cos_enabled
                and mode == "train"
                and step % self.sample_gradient_cos_freq_steps == 0,
                "sample_gradient_cos_freq_steps": self.sample_gradient_cos_freq_steps,
                "sample_gradient_backward_recompute_enabled": self.sample_gradient_backward_recompute_enabled,
                "sample_gradient_backward_sync_enabled": self.sample_gradient_backward_sync_enabled,
                "sample_gradient_log_sample_level": self.should_log_sample_gradient_level(step),
                "sample_gradient_log_sample_level_freq_steps": self.sample_gradient_log_sample_level_freq_steps,
                "offload_domain_gradients": self.full_gradient_offload_domain_gradients,
                "token_gradient_enabled": self.should_compute_token_gradient(step) and mode == "train",
                "token_gradient_freq_steps": self.token_gradient_freq_steps,
                "token_gradient_tail_enabled": self.token_gradient_tail_enabled,
                "token_gradient_tail_fraction": self.token_gradient_tail_fraction,
                "token_gradient_tail_min_tokens": self.token_gradient_tail_min_tokens,
                "token_gradient_gap_selection_enabled": self.token_gradient_gap_selection_enabled,
                "token_gradient_gap_abs_selection_enabled": self.token_gradient_gap_abs_selection_enabled,
                "token_gradient_loss_abs_selection_enabled": self.token_gradient_loss_abs_selection_enabled,
                "token_gradient_top_k": self.token_gradient_top_k,
                "token_gradient_top_p_enabled": self.token_gradient_top_p_enabled,
                "token_gradient_top_p": self.token_gradient_top_p,
                "token_gradient_log_tokens_jsonl_enabled": (
                    self.token_gradient_log_tokens_jsonl_enabled
                ),
                "token_gradient_vocab_size": self.token_gap_vocab_size,
                "token_gradient_strict_grad_restore": self.token_gradient_strict_grad_restore,
                "token_gradient_backward_recompute_enabled": self.token_gradient_backward_recompute_enabled,
                "token_gradient_backward_sync_enabled": self.token_gradient_backward_sync_enabled,
                "dynamic_domain_loss_weighting_enabled": (
                    self.dynamic_domain_loss_weighting_enabled and mode == "train"
                ),
                "dynamic_domain_loss_weighting_update_enabled": (
                    self.should_update_dynamic_domain_loss_weighting(step)
                    and mode == "train"
                ),
                "dynamic_domain_loss_weighting_freq_steps": (
                    self.dynamic_domain_loss_weighting_freq_steps
                ),
                "dynamic_domain_loss_weighting_ema_beta": (
                    self.dynamic_domain_loss_weighting_ema_beta
                ),
                "dynamic_domain_loss_weighting_weight_ema_beta": (
                    self.dynamic_domain_loss_weighting_weight_ema_beta
                ),
                "dynamic_domain_loss_weighting_alpha": (
                    self.dynamic_domain_loss_weighting_alpha
                ),
                "dynamic_domain_loss_weighting_min": (
                    self.dynamic_domain_loss_weighting_min
                ),
                "dynamic_domain_loss_weighting_max": (
                    self.dynamic_domain_loss_weighting_max
                ),
                "domain_partition": domain_partition or {},
            }
        }

    def _current_learning_rate_value(self) -> float:
        policy_lr = None
        try:
            policy_lr = _cfg_get(
                _cfg_get(
                    _cfg_get(self.config, "actor_rollout_ref", {}),
                    "actor",
                    {},
                ),
                "optim",
                {},
            )
            policy_lr = _cfg_get(policy_lr, "lr", policy_lr)
        except Exception:
            policy_lr = None
        return _scalar_float(policy_lr) or 0.0

    def _learning_rate_value(self, lr: Any) -> float:
        numeric = _scalar_float(lr)
        return numeric if numeric is not None else self._current_learning_rate_value()

    def _write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> None:
        if not self.enabled or not rows:
            return
        path = self.output_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_to_builtin(row), sort_keys=True) + "\n")

    def log_training_step(self, batch: Any, step: int, lr: Any = None) -> dict[str, float]:
        if not self.enabled:
            return {}

        started_at = time.perf_counter()
        try:
            (
                metrics,
                domain_rows,
                variance_rows,
                sample_rows,
                token_gap_rows,
                token_gap_vocab_rows,
                entropy_distribution_rows,
                entropy_vocab_rows,
            ) = self._compute_training_rows(batch, step, lr)
        except Exception as exc:  # pragma: no cover - defensive remote logging
            self._write_jsonl("audit_errors.jsonl", [{"step": step, "stage": "training", "error": repr(exc)}])
            return {self._global_tag("audit", "error"): 1.0}

        metrics[self._global_tag("audit", "wall_time_step")] = time.perf_counter() - started_at
        self._write_jsonl("domain_step_metrics.jsonl", domain_rows)
        self._write_jsonl("loss_variance_domain_step.jsonl", variance_rows)
        self._write_jsonl("loss_variance_sample.jsonl", sample_rows)
        if token_gap_rows:
            occurrence_meta_fields = ("step", "domain", "learning_rate", "token_count")
            self._write_jsonl(
                "token_gap_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_rows,
                    required_field="gap_signed_vector_domain",
                    fields=occurrence_meta_fields
                    + (
                        "gap_signed_vector_domain",
                        "gap_abs_vector_domain",
                        "gap_vector_domain",
                    ),
                ),
            )
            self._write_jsonl(
                "logp_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_rows,
                    required_field="logp_vector_domain",
                    fields=occurrence_meta_fields + ("logp_vector_domain",),
                ),
            )
            self._write_jsonl(
                "logp_abs_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_rows,
                    required_field="logp_abs_vector_domain",
                    fields=occurrence_meta_fields + ("logp_abs_vector_domain",),
                ),
            )
        if token_gap_vocab_rows:
            vocab_meta_fields = (
                "step",
                "domain",
                "learning_rate",
                "vocab_size_source",
                "vocab_size",
                "observed_token_count",
                "dropped_token_count",
                "nonzero_token_id_count",
                "nonzero_token_ids",
                "token_count_vector_vocab",
            )
            self._write_jsonl(
                "token_gap_vocab_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_vocab_rows,
                    required_field="gap_signed_sum_vector_vocab",
                    fields=vocab_meta_fields
                    + (
                        "gap_signed_sum_vector_vocab",
                        "gap_abs_sum_vector_vocab",
                        "gap_signed_mean_vector_vocab",
                        "gap_abs_mean_vector_vocab",
                    ),
                ),
            )
            self._write_jsonl(
                "logp_vocab_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_vocab_rows,
                    required_field="logp_sum_vector_vocab",
                    fields=vocab_meta_fields
                    + ("logp_sum_vector_vocab", "logp_mean_vector_vocab"),
                ),
            )
            self._write_jsonl(
                "logp_abs_vocab_vectors.jsonl",
                _project_jsonl_rows(
                    token_gap_vocab_rows,
                    required_field="logp_abs_sum_vector_vocab",
                    fields=vocab_meta_fields
                    + ("logp_abs_sum_vector_vocab", "logp_abs_mean_vector_vocab"),
                ),
            )
        if entropy_distribution_rows:
            self._write_jsonl("entropy_distribution_vectors.jsonl", entropy_distribution_rows)
        if entropy_vocab_rows:
            self._write_jsonl("entropy_vocab_vectors.jsonl", entropy_vocab_rows)
            self._write_jsonl(
                "topk_teacher_student_cross_entropy_vocab_vectors.jsonl",
                _project_jsonl_rows(
                    entropy_vocab_rows,
                    required_field="teacher_student_cross_entropy_sum_vector_vocab",
                    fields=(
                        "step",
                        "domain",
                        "learning_rate",
                        "vocab_size_source",
                        "vocab_size",
                        "observed_token_count",
                        "dropped_token_count",
                        "nonzero_token_id_count",
                        "nonzero_token_ids",
                        "token_count_vector_vocab",
                        "teacher_student_cross_entropy_sum_vector_vocab",
                        "teacher_student_cross_entropy_mean_vector_vocab",
                        "topk_support_source",
                        "topk_k",
                        "topk_include_tail",
                        "topk_temperature",
                    ),
                ),
            )
        return metrics

    def log_validation_metrics(self, val_metrics: dict[str, Any], step: int) -> dict[str, float]:
        if not self.should_log_validation_metrics(step):
            return {}
        return _log_validation_metrics(self, val_metrics, step)

    def log_training_cost(self, metrics: dict[str, Any], step: int, n_gpus: int = 1) -> dict[str, float]:
        return _log_training_cost(self, metrics, step, n_gpus)

    def _compute_training_rows(
        self, batch: Any, step: int, lr: Any
    ) -> tuple[dict[str, float], list, list, list, list, list, list, list]:
        import torch

        tensor_batch = batch.batch
        non_tensor = batch.non_tensor_batch
        batch_meta_info = getattr(batch, "meta_info", {})
        if not isinstance(batch_meta_info, dict):
            batch_meta_info = {}
        old_log_probs = tensor_batch["old_log_probs"].detach().float()
        response_mask = response_mask_from_batch(tensor_batch, old_log_probs)
        batch_keys = set(tensor_batch.keys())
        base_log_prob = (tensor_batch["base_log_prob"] if "base_log_prob" in batch_keys else old_log_probs).detach().float()
        batch_size = int(old_log_probs.shape[0])

        labels = extract_teacher_domains(non_tensor, batch_size)
        sample_ids = extract_sample_ids(non_tensor, batch_size, step)

        model_inputs = {**tensor_batch, **non_tensor}
        try:
            teacher_log_probs = select_teacher_log_prob_tensor(
                model_inputs,
                {"multi_teacher_distill": True},
            ).detach().float()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "teacher_student_gap requires a valid per-domain teacher "
                "log-prob tensor; refusing to substitute synthetic zeros."
            ) from exc
        reverse_kl = (
            old_log_probs - teacher_log_probs
            if self.lambda_vals == 1.0
            else (
                old_log_probs - base_log_prob
                - (teacher_log_probs - base_log_prob) * self.lambda_vals
            )
        )
        gap_signed = teacher_log_probs - old_log_probs
        gap_abs = gap_signed.abs()
        configured_token_loss_available = "configured_token_loss" in batch_keys
        configured_token_loss = (
            tensor_batch["configured_token_loss"].detach().float()
            if configured_token_loss_available
            else reverse_kl
        )
        if tuple(configured_token_loss.shape) != tuple(old_log_probs.shape):
            raise ValueError(
                "configured_token_loss must match old_log_probs shape, got "
                f"{tuple(configured_token_loss.shape)} versus "
                f"{tuple(old_log_probs.shape)}."
            )
        configured_token_loss_mask = (
            tensor_batch["configured_token_loss_mask"].detach().float()
            if "configured_token_loss_mask" in batch_keys
            else response_mask
        )
        if tuple(configured_token_loss_mask.shape) != tuple(
            configured_token_loss.shape
        ):
            raise ValueError(
                "configured_token_loss_mask must match configured_token_loss "
                f"shape, got {tuple(configured_token_loss_mask.shape)} versus "
                f"{tuple(configured_token_loss.shape)}."
            )
        configured_token_loss_mask = (
            configured_token_loss_mask * response_mask
        )
        if configured_token_loss_available:
            configured_token_loss_name = str(
                batch_meta_info.get(
                    "mopd_configured_token_loss_name",
                    "chosen_token_reverse_kl",
                )
            )
            configured_token_loss_epoch_reduction = str(
                batch_meta_info.get(
                    "mopd_configured_token_loss_epoch_reduction",
                    "single_forward",
                )
            )
            configured_token_loss_epoch_count = int(
                batch_meta_info.get(
                    "mopd_configured_token_loss_epoch_count",
                    1,
                )
            )
        else:
            # The legacy fallback remains available for callers that only provide
            # chosen-token log-probs, but it must never inherit a configured loss
            # label or a multi-epoch reduction that did not actually run.
            configured_token_loss_name = "chosen_token_reverse_kl_fallback"
            configured_token_loss_epoch_reduction = "single_forward"
            configured_token_loss_epoch_count = 1

        student_entropy = (
            tensor_batch["student_entropy"].detach().float() if "student_entropy" in batch_keys else None
        )
        teacher_entropy = _select_domain_tensor(
            tensor_batch=tensor_batch,
            batch_keys=batch_keys,
            labels=labels,
            reference=old_log_probs,
            suffix="_teacher_entropy",
        )
        teacher_student_cross_entropy = _select_domain_tensor(
            tensor_batch=tensor_batch,
            batch_keys=batch_keys,
            labels=labels,
            reference=old_log_probs,
            suffix="_teacher_student_cross_entropy",
            generic_key="teacher_student_cross_entropy",
        )

        sample_token_opd_loss_mean = _mask_mean(
            configured_token_loss,
            configured_token_loss_mask,
        )
        sample_opd_loss = (
            configured_token_loss * configured_token_loss_mask
        ).sum(dim=-1)
        sample_loss_sq_mean = _mask_mean(
            configured_token_loss.square(),
            configured_token_loss_mask,
        )
        sample_loss_var = torch.clamp(sample_loss_sq_mean - sample_token_opd_loss_mean.square(), min=0.0)
        sample_loss_std = torch.sqrt(sample_loss_var)
        sample_loss_cv = sample_loss_std / (sample_token_opd_loss_mean.abs() + 1e-8)
        effective_tokens = response_mask.sum(dim=-1).detach().cpu().tolist()
        teacher_student_gap = _mask_mean(gap_signed, response_mask)
        teacher_logprob_mean = _mask_mean(teacher_log_probs, response_mask)
        advantages = (
            tensor_batch["advantages"].detach().float()
            if "advantages" in batch_keys
            else -configured_token_loss
        )
        sample_advantage_mean = _mask_mean(advantages, response_mask)

        token_scores = tensor_batch["token_level_scores"].detach().float() if "token_level_scores" in batch_keys else None
        sample_reward = None
        sample_correctness = None
        if token_scores is not None:
            sample_reward = (token_scores * response_mask).sum(dim=-1)
            sample_correctness = sample_reward.gt(0).detach().float()

        configured_domains = list(dict.fromkeys(self.domains + sorted(set(labels))))
        total_tokens = float(response_mask.sum().item())
        total_samples = float(batch_size)
        learning_rate = self._learning_rate_value(lr)
        metrics: dict[str, float] = {}
        metrics[self._global_tag("optimization", "learning_rate")] = learning_rate
        domain_rows: list[dict[str, Any]] = []
        variance_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        token_gap_rows: list[dict[str, Any]] = []
        token_gap_vocab_rows: list[dict[str, Any]] = []
        token_gap_vocab_vectors_by_domain: dict[str, dict[str, Any]] = {}
        entropy_distribution_rows: list[dict[str, Any]] = []
        entropy_vocab_rows: list[dict[str, Any]] = []
        entropy_vocab_vectors_by_domain: dict[str, dict[str, Any]] = {}
        token_gap_active = self.should_log_token_gap(step)
        logp_active = self.should_log_logp_vector(step)
        logp_abs_active = self.should_log_logp_abs_vector(step)
        logp_vector_active = token_gap_active or logp_active or logp_abs_active
        token_gap_vocab_active = token_gap_active and self.should_log_token_gap_vocab_vector(step)
        token_gap_vocab_compute_active = token_gap_vocab_active or logp_active or logp_abs_active
        topk_cross_entropy_vocab_active = (
            self.should_log_topk_teacher_student_cross_entropy_vocab(step)
        )
        entropy_distribution_active = self.should_log_entropy(step)
        entropy_vocab_vector_active = self.should_log_entropy_vocab_vector(step)
        entropy_compute_active = (
            entropy_distribution_active
            or entropy_vocab_vector_active
            or topk_cross_entropy_vocab_active
        )
        entropy_vocab_active = entropy_vocab_vector_active or topk_cross_entropy_vocab_active
        sample_level_active = self.should_log_sample_level(step)

        opd_losses = _tensor_to_float_list(sample_opd_loss)
        sample_token_opd_loss_means = _tensor_to_float_list(sample_token_opd_loss_mean)
        sample_loss_vars = _tensor_to_float_list(sample_loss_var)
        loss_cvs = _tensor_to_float_list(sample_loss_cv)
        token_counts = [float(x) for x in effective_tokens]
        gap_means = _tensor_to_float_list(teacher_student_gap)
        teacher_logprob_means = _tensor_to_float_list(teacher_logprob_mean)
        advantage_means = _tensor_to_float_list(sample_advantage_mean)
        reward_values = _tensor_to_float_list(sample_reward) if sample_reward is not None else None
        correctness_values = _tensor_to_float_list(sample_correctness) if sample_correctness is not None else None

        indices_by_domain = {
            domain: [idx for idx, label in enumerate(labels) if label == domain] for domain in configured_domains
        }
        token_count_by_domain = {
            domain: sum(token_counts[idx] for idx in indices) for domain, indices in indices_by_domain.items()
        }
        sample_count_by_domain = {domain: len(indices) for domain, indices in indices_by_domain.items()}
        token_ids = None
        if token_gap_vocab_compute_active or entropy_vocab_active:
            token_ids = _response_token_id_matrix(tensor_batch, batch_keys, response_mask)
        token_gap_vocab_size = self.token_gap_vocab_size
        token_gap_vocab_size_source = self.token_gap_vocab_size_source
        if (logp_vector_active or entropy_vocab_active) and token_ids is not None and token_gap_vocab_size is None:
            observed_valid_ids = token_ids[response_mask.detach().bool().cpu()]
            if int(observed_valid_ids.numel()) > 0:
                token_gap_vocab_size = int(observed_valid_ids.max().item()) + 1
                token_gap_vocab_size_source = "observed_max_token_id"
        for domain in configured_domains:
            indices = indices_by_domain[domain]
            safe_domain = safe_name(domain)
            domain_token_count = token_count_by_domain[domain]
            domain_sample_count = sample_count_by_domain[domain]
            domain_loss_vars = [sample_loss_vars[idx] for idx in indices]
            domain_cvs = [loss_cvs[idx] for idx in indices]
            domain_teacher_logprobs = [teacher_logprob_means[idx] for idx in indices]
            domain_advantages = [advantage_means[idx] for idx in indices]
            domain_rewards = [reward_values[idx] for idx in indices] if reward_values is not None else []
            domain_sample_ids = [sample_ids[idx] for idx in indices]
            domain_token_counts = [token_counts[idx] for idx in indices]
            domain_valid_mask = (
                response_mask[indices].detach().bool() if indices else None
            )
            domain_loss_valid_mask = (
                configured_token_loss_mask[indices].detach().bool()
                if indices
                else None
            )
            domain_loss_vector = (
                configured_token_loss[indices][domain_loss_valid_mask]
                if domain_loss_valid_mask is not None
                else None
            )
            domain_gap_vector = (
                gap_signed[indices][domain_valid_mask]
                if domain_valid_mask is not None
                else None
            )
            domain_gap_abs_vector = (
                gap_abs[indices][domain_valid_mask]
                if domain_valid_mask is not None
                else None
            )
            domain_token_stats = _token_distribution_stats(
                domain_loss_vector,
                "token_opd_loss",
            )
            teacher_student_gap_stats = _token_distribution_stats(
                domain_gap_vector,
                "teacher_student_gap",
            )
            signed_gap_stats: dict[str, float | None] = {}
            abs_gap_stats: dict[str, float | None] = {}
            if logp_vector_active:
                if token_gap_active:
                    signed_gap_stats = _token_distribution_stats(domain_gap_vector, "gap_signed")
                if token_gap_active or logp_abs_active:
                    abs_gap_stats = _token_distribution_stats(domain_gap_abs_vector, "gap_abs")
                if domain_gap_vector is not None and int(domain_gap_vector.numel()) > 0:
                    token_gap_row = {
                        "step": step,
                        "domain": domain,
                        "learning_rate": learning_rate,
                        "token_count": int(domain_gap_vector.numel()),
                    }
                    if token_gap_active:
                        token_gap_row.update(
                            {
                                "gap_signed_vector_domain": _tensor_to_float_list(domain_gap_vector),
                                "gap_abs_vector_domain": _tensor_to_float_list(domain_gap_abs_vector),
                                "gap_vector_domain": _tensor_to_float_list(domain_gap_vector),
                            }
                        )
                    if logp_active:
                        token_gap_row["logp_vector_domain"] = _tensor_to_float_list(
                            domain_gap_vector
                        )
                    if logp_abs_active:
                        token_gap_row["logp_abs_vector_domain"] = _tensor_to_float_list(
                            domain_gap_abs_vector
                        )
                    token_gap_rows.append(token_gap_row)
                if (
                    token_gap_vocab_compute_active
                    and token_ids is not None
                    and indices
                    and token_gap_vocab_size is not None
                ):
                    domain_token_ids = token_ids[indices]
                    vocab_vectors = _token_gap_vocab_tensors(
                        token_ids=domain_token_ids,
                        response_mask=response_mask[indices],
                        gap_signed=gap_signed[indices],
                        gap_abs=gap_abs[indices],
                        vocab_size=int(token_gap_vocab_size),
                    )
                    if vocab_vectors is not None:
                        token_gap_vocab_vectors_by_domain[domain] = vocab_vectors
                        if token_gap_vocab_active or logp_active or logp_abs_active:
                            vocab_row: dict[str, Any] = {
                                "step": step,
                                "domain": domain,
                                "learning_rate": learning_rate,
                                "vocab_size_source": token_gap_vocab_size_source,
                            }
                            if token_gap_vocab_active:
                                vocab_row.update(
                                    _token_gap_vocab_json_fields(
                                        vocab_vectors,
                                        include_mean_vectors=(
                                            self.vocab_per_occurrence_mean_vector_enabled
                                        ),
                                    )
                                )
                            if logp_active:
                                vocab_row.update(
                                    _logp_vocab_json_fields(
                                        vocab_vectors,
                                        include_mean_vectors=(
                                            self.logp_vocab_per_occurrence_mean_vector_enabled
                                        ),
                                    )
                                )
                            if logp_abs_active:
                                vocab_row.update(
                                    _logp_abs_vocab_json_fields(
                                        vocab_vectors,
                                        include_mean_vectors=(
                                            self.logp_abs_vocab_per_occurrence_mean_vector_enabled
                                        ),
                                    )
                                )
                            token_gap_vocab_rows.append(vocab_row)
            entropy_metrics: dict[str, float | None] = {}
            teacher_entropy_stats: dict[str, float | None] = {}
            student_entropy_stats: dict[str, float | None] = {}
            cross_entropy_stats: dict[str, float | None] = {}
            if entropy_compute_active:
                teacher_entropy_vector = None
                student_entropy_vector = None
                cross_entropy_vector = None
                if indices:
                    domain_response_mask = response_mask[indices]
                    domain_valid_mask = domain_response_mask.detach().bool()
                    if teacher_entropy is not None:
                        teacher_entropy_vector = teacher_entropy[indices][domain_valid_mask]
                    if student_entropy is not None:
                        student_entropy_vector = student_entropy[indices][domain_valid_mask]
                    if teacher_student_cross_entropy is not None:
                        cross_entropy_vector = teacher_student_cross_entropy[indices][domain_valid_mask]
                if entropy_distribution_active:
                    teacher_entropy_stats = _token_distribution_stats(
                        teacher_entropy_vector,
                        "teacher_entropy",
                    )
                    student_entropy_stats = _token_distribution_stats(
                        student_entropy_vector,
                        "student_entropy",
                    )
                    cross_entropy_stats = _token_distribution_stats(
                        cross_entropy_vector,
                        "teacher_student_cross_entropy",
                    )
                    teacher_entropy_sum = teacher_entropy_stats["teacher_entropy_sum"]
                    student_entropy_sum = student_entropy_stats["student_entropy_sum"]
                    cross_entropy_sum = cross_entropy_stats["teacher_student_cross_entropy_sum"]
                    entropy_metrics = {
                        "sum_teacher_entropy": teacher_entropy_sum,
                        "sum_student_entropy": student_entropy_sum,
                        "sum_teacher_student_cross_entropy": cross_entropy_sum,
                        "entropy_distribution_available": float(
                            teacher_entropy_sum is not None or student_entropy_sum is not None
                        ),
                        "cross_entropy_available": float(cross_entropy_sum is not None),
                    }
                    entropy_row: dict[str, Any] = {
                        "step": step,
                        "domain": domain,
                        "learning_rate": learning_rate,
                        "token_count": int(domain_token_count),
                    }
                    if teacher_entropy_vector is not None and int(teacher_entropy_vector.numel()) > 0:
                        entropy_row["teacher_entropy_vector_domain"] = _tensor_to_float_list(
                            teacher_entropy_vector
                        )
                    if student_entropy_vector is not None and int(student_entropy_vector.numel()) > 0:
                        entropy_row["student_entropy_vector_domain"] = _tensor_to_float_list(
                            student_entropy_vector
                        )
                    if cross_entropy_vector is not None and int(cross_entropy_vector.numel()) > 0:
                        entropy_row["teacher_student_cross_entropy_vector_domain"] = _tensor_to_float_list(
                            cross_entropy_vector
                        )
                    if len(entropy_row) > 4:
                        entropy_distribution_rows.append(entropy_row)
                if (
                    entropy_vocab_active
                    and token_ids is not None
                    and indices
                    and token_gap_vocab_size is not None
                    and (
                        (student_entropy_vector is not None and int(student_entropy_vector.numel()) > 0)
                        or (cross_entropy_vector is not None and int(cross_entropy_vector.numel()) > 0)
                    )
                ):
                    vocab_vectors = _entropy_vocab_tensors(
                        token_ids=token_ids[indices],
                        response_mask=response_mask[indices],
                        student_entropy=(
                            None
                            if not entropy_vocab_vector_active or student_entropy is None
                            else student_entropy[indices]
                        ),
                        teacher_student_cross_entropy=(
                            None
                            if teacher_student_cross_entropy is None
                            else teacher_student_cross_entropy[indices]
                        ),
                        vocab_size=int(token_gap_vocab_size),
                    )
                    if vocab_vectors is not None:
                        entropy_vocab_vectors_by_domain[domain] = vocab_vectors
                        entropy_vocab_row = {
                            "step": step,
                            "domain": domain,
                            "learning_rate": learning_rate,
                            "vocab_size_source": token_gap_vocab_size_source,
                            **_entropy_vocab_json_fields(
                                vocab_vectors,
                                include_mean_vectors=(
                                    self.entropy_vocab_per_occurrence_mean_vector_enabled
                                ),
                            ),
                        }
                        if "teacher_student_cross_entropy_sum_vector_vocab" in entropy_vocab_row:
                            entropy_vocab_row.update(
                                {
                                    "topk_support_source": str(
                                        batch_meta_info.get("topk_distill_support_source", "teacher")
                                    ),
                                    "topk_k": int(
                                        batch_meta_info.get(
                                            "teacher_topk_k",
                                            batch_meta_info.get(
                                                "student_topk_k",
                                                self.topk_teacher_student_cross_entropy_k,
                                            ),
                                        )
                                    ),
                                    "topk_include_tail": bool(
                                        batch_meta_info.get(
                                            "topk_distill_include_tail",
                                            self.topk_teacher_student_cross_entropy_include_tail,
                                        )
                                    ),
                                    "topk_temperature": float(
                                        batch_meta_info.get(
                                            "topk_distill_temperature",
                                            self.topk_teacher_student_cross_entropy_temperature,
                                        )
                                    ),
                                }
                            )
                        entropy_vocab_rows.append(entropy_vocab_row)
            domain_sample_losses = [opd_losses[idx] for idx in indices]
            domain_sample_stats = _sample_value_stats(domain_sample_losses)

            confidence_values = [float(np.clip(math.exp(value), 0.0, 1.0)) for value in domain_teacher_logprobs]
            correctness_for_domain = [correctness_values[idx] for idx in indices] if correctness_values is not None else []
            calibration_error = ece(confidence_values, correctness_for_domain, self.calibration_bins)

            old_seen = self._seen_sample_ids.setdefault(domain, set())
            duplicate_count = sum(1 for sample_id in domain_sample_ids if sample_id in old_seen)
            for sample_id in domain_sample_ids:
                old_seen.add(sample_id)
            duplicate_rate = None if not domain_sample_ids else duplicate_count / len(domain_sample_ids)

            row = {
                "step": step,
                "domain": domain,
                "learning_rate": learning_rate,
                "domain_sample_count": domain_sample_count,
                "domain_token_count": domain_token_count,
                "domain_token_frac": domain_token_count / total_tokens if total_tokens else 0.0,
                "sample_opd_loss_mean": domain_sample_stats["mean"],
                "sample_opd_loss_std": domain_sample_stats["std"],
                "sample_opd_loss_variance": domain_sample_stats["variance"],
                "high_variance_sample_rate": None
                if not domain_cvs
                else float(np.mean([cv > self.high_variance_cv_threshold for cv in domain_cvs])),
                "advantage_mean": _mean(domain_advantages),
                "positive_frac": None
                if not domain_advantages
                else float(np.mean([value > 0.0 for value in domain_advantages])),
                "response_mean": _mean(domain_token_counts),
                "response_p95": _percentile(domain_token_counts, 95.0),
                "response_clip_ratio": None
                if not domain_token_counts
                else float(np.mean([count >= response_mask.shape[-1] for count in domain_token_counts])),
                "training_reward_mean": _mean(domain_rewards),
                "training_accuracy": _mean(correctness_for_domain),
                "teacher_confidence_mean": _mean(confidence_values),
                "calibration_error": calibration_error,
                "duplicate_rate": duplicate_rate,
            }
            row.update(domain_token_stats)
            row.update(teacher_student_gap_stats)
            row.update(signed_gap_stats)
            row.update(abs_gap_stats)
            row.update(entropy_metrics)
            row.update(teacher_entropy_stats)
            row.update(student_entropy_stats)
            row.update(cross_entropy_stats)
            domain_rows.append(row)
            variance_rows.append(
                {
                    "step": step,
                    "domain": domain,
                    "learning_rate": learning_rate,
                    "metric_scope": "domain_step",
                    "loss_name": configured_token_loss_name,
                    "loss_epoch_reduction": (
                        configured_token_loss_epoch_reduction
                    ),
                    "loss_epoch_count": configured_token_loss_epoch_count,
                    "domain_sample_count": domain_sample_count,
                    "domain_token_count": domain_token_count,
                    "token_opd_loss_mean": row["token_opd_loss_mean"],
                    "token_opd_loss_std": row["token_opd_loss_std"],
                    "token_opd_loss_variance": row["token_opd_loss_variance"],
                    "token_opd_loss_p05": row["token_opd_loss_p05"],
                    "token_opd_loss_p50": row["token_opd_loss_p50"],
                    "token_opd_loss_p95": row["token_opd_loss_p95"],
                    "token_opd_loss_sum": row["token_opd_loss_sum"],
                    "teacher_student_gap_mean": row[
                        "teacher_student_gap_mean"
                    ],
                    "teacher_student_gap_p05": row[
                        "teacher_student_gap_p05"
                    ],
                    "teacher_student_gap_p50": row[
                        "teacher_student_gap_p50"
                    ],
                    "teacher_student_gap_p95": row[
                        "teacher_student_gap_p95"
                    ],
                    "teacher_student_gap_sum": row[
                        "teacher_student_gap_sum"
                    ],
                    "sample_opd_loss_mean": row["sample_opd_loss_mean"],
                    "sample_opd_loss_std": row["sample_opd_loss_std"],
                    "sample_opd_loss_variance": row["sample_opd_loss_variance"],
                    "high_variance_sample_rate": row["high_variance_sample_rate"],
                }
            )

            domain_metric_keys = {
                "domain_sample_count",
                "domain_token_count",
                "domain_token_frac",
                "token_opd_loss_mean",
                "token_opd_loss_std",
                "token_opd_loss_variance",
                "token_opd_loss_p05",
                "token_opd_loss_p50",
                "token_opd_loss_p95",
                "token_opd_loss_max",
                "token_opd_loss_sum",
                "sample_opd_loss_mean",
                "sample_opd_loss_std",
                "sample_opd_loss_variance",
                "high_variance_sample_rate",
                "advantage_mean",
                "positive_frac",
                "response_mean",
                "response_p95",
                "response_clip_ratio",
                "training_reward_mean",
                "training_accuracy",
                "teacher_student_gap_mean",
                "teacher_student_gap_std",
                "teacher_student_gap_variance",
                "teacher_student_gap_p05",
                "teacher_student_gap_p50",
                "teacher_student_gap_p95",
                "teacher_student_gap_max",
                "teacher_student_gap_sum",
                "teacher_confidence_mean",
                "calibration_error",
                "duplicate_rate",
                "gap_signed_mean",
                "gap_signed_std",
                "gap_signed_p05",
                "gap_signed_p50",
                "gap_signed_p95",
                "gap_signed_max",
                "gap_signed_sum",
                "gap_abs_mean",
                "gap_abs_std",
                "gap_abs_p05",
                "gap_abs_p50",
                "gap_abs_p95",
                "gap_abs_max",
                "gap_abs_sum",
                "sum_teacher_entropy",
                "sum_student_entropy",
                "sum_teacher_student_cross_entropy",
                "teacher_entropy_mean",
                "teacher_entropy_std",
                "teacher_entropy_p05",
                "teacher_entropy_p50",
                "teacher_entropy_p95",
                "teacher_entropy_max",
                "teacher_entropy_sum",
                "student_entropy_mean",
                "student_entropy_std",
                "student_entropy_p05",
                "student_entropy_p50",
                "student_entropy_p95",
                "student_entropy_max",
                "student_entropy_sum",
                "teacher_student_cross_entropy_mean",
                "teacher_student_cross_entropy_std",
                "teacher_student_cross_entropy_p05",
                "teacher_student_cross_entropy_p50",
                "teacher_student_cross_entropy_p95",
                "teacher_student_cross_entropy_max",
                "teacher_student_cross_entropy_sum",
                "entropy_distribution_available",
                "cross_entropy_available",
            }
            for key in domain_metric_keys:
                numeric = finite_float(row.get(key))
                if numeric is not None:
                    metrics[self._domain_tag(safe_domain, domain_metric_category(key), key)] = numeric

            if sample_level_active and indices:
                sample_indices = (
                    indices
                    if self.max_samples_per_domain is None
                    else indices[: self.max_samples_per_domain]
                )
                for idx in sample_indices:
                    sample_rows.append(
                        {
                            "step": step,
                            "domain": domain,
                            "sample_id": sample_ids[idx],
                            "learning_rate": learning_rate,
                            "metric_scope": "sample_token",
                            "loss_name": configured_token_loss_name,
                            "loss_epoch_reduction": (
                                configured_token_loss_epoch_reduction
                            ),
                            "loss_epoch_count": (
                                configured_token_loss_epoch_count
                            ),
                            "effective_tokens": token_counts[idx],
                            "opd_loss": opd_losses[idx],
                            "sample_token_opd_loss_mean": sample_token_opd_loss_means[idx],
                            "sample_token_opd_loss_variance": float(sample_loss_var[idx].detach().cpu().item()),
                            "training_reward": None if reward_values is None else reward_values[idx],
                            "training_correctness": None if correctness_values is None else correctness_values[idx],
                        }
                    )

        token_gap_vector_specs = [
            ("token_count_cosine", "token_count_vector_vocab"),
            ("gap_signed_sum_cosine", "gap_signed_sum_vector_vocab"),
            ("gap_abs_sum_cosine", "gap_abs_sum_vector_vocab"),
        ]
        logp_vector_specs = [
            ("token_count_cosine", "token_count_vector_vocab"),
            ("logp_sum_cosine", "gap_signed_sum_vector_vocab"),
        ]
        logp_abs_vector_specs = [
            ("token_count_cosine", "token_count_vector_vocab"),
            ("logp_abs_sum_cosine", "gap_abs_sum_vector_vocab"),
        ]
        entropy_vector_specs = [
            ("token_count_cosine", "token_count_vector_vocab"),
            ("student_entropy_sum_cosine", "student_entropy_sum_vector_vocab"),
            (
                "teacher_student_cross_entropy_sum_cosine",
                "teacher_student_cross_entropy_sum_vector_vocab",
            ),
        ]
        if self.vocab_per_occurrence_mean_vector_enabled:
            token_gap_vector_specs.extend(
                [
                    ("gap_signed_mean_cosine", "gap_signed_mean_vector_vocab"),
                    ("gap_abs_mean_cosine", "gap_abs_mean_vector_vocab"),
                ]
            )
        if self.logp_vocab_per_occurrence_mean_vector_enabled:
            logp_vector_specs.append(
                ("logp_mean_cosine", "gap_signed_mean_vector_vocab")
            )
        if self.logp_abs_vocab_per_occurrence_mean_vector_enabled:
            logp_abs_vector_specs.append(
                ("logp_abs_mean_cosine", "gap_abs_mean_vector_vocab")
            )
        if self.entropy_vocab_per_occurrence_mean_vector_enabled:
            entropy_vector_specs.extend(
                [
                    ("student_entropy_mean_cosine", "student_entropy_mean_vector_vocab"),
                    (
                        "teacher_student_cross_entropy_mean_cosine",
                        "teacher_student_cross_entropy_mean_vector_vocab",
                    ),
                ]
            )

        cosine_groups: list[
            tuple[bool, str, dict[str, dict[str, Any]], tuple[tuple[str, str], ...]]
        ] = [
            (
                token_gap_vocab_active,
                "token_gap_vocab_cosine",
                token_gap_vocab_vectors_by_domain,
                tuple(token_gap_vector_specs),
            ),
            (
                logp_active,
                "logp_vocab_cosine",
                token_gap_vocab_vectors_by_domain,
                tuple(logp_vector_specs),
            ),
            (
                logp_abs_active,
                "logp_abs_vocab_cosine",
                token_gap_vocab_vectors_by_domain,
                tuple(logp_abs_vector_specs),
            ),
            (
                entropy_vocab_active,
                "entropy_vocab_cosine",
                entropy_vocab_vectors_by_domain,
                tuple(entropy_vector_specs),
            ),
        ]
        for active, category, vectors_by_domain, vector_specs in cosine_groups:
            if not active:
                continue
            for left_domain, right_domain, metric_name, cosine in iter_pairwise_domain_cosines(
                vectors_by_domain,
                configured_domains,
                vector_specs,
            ):
                pair_name = f"{safe_name(left_domain)}_vs_{safe_name(right_domain)}"
                metrics[self._global_tag(category, metric_name, pair_name)] = cosine

        if total_tokens:
            global_valid_mask = response_mask.detach().bool()
            global_loss_valid_mask = (
                configured_token_loss_mask.detach().bool()
            )
            global_token_stats = _token_distribution_stats(
                configured_token_loss[global_loss_valid_mask],
                "token_opd_loss",
            )
            global_gap_stats = _token_distribution_stats(
                gap_signed[global_valid_mask],
                "teacher_student_gap",
            )
            global_sample_stats = _sample_value_stats(opd_losses)
            global_loss_metrics = {
                **global_token_stats,
                "sample_opd_loss_mean": global_sample_stats["mean"],
                "sample_opd_loss_std": global_sample_stats["std"],
                "sample_opd_loss_variance": global_sample_stats["variance"],
            }
            for key, value in global_loss_metrics.items():
                numeric = finite_float(value)
                if numeric is not None:
                    metrics[self._global_tag("loss", key)] = numeric
            for key, value in global_gap_stats.items():
                numeric = finite_float(value)
                if numeric is not None:
                    metrics[self._global_tag("teacher", key)] = numeric
            mix = [row["domain_token_frac"] for row in domain_rows if row["domain_token_frac"]]
            entropy = -sum(frac * math.log(frac) for frac in mix)
            metrics[self._global_tag("data", "domain_mix_entropy")] = entropy
            metrics[self._global_tag("data", "total_tokens")] = total_tokens
            metrics[self._global_tag("data", "total_samples")] = total_samples

        return (
            metrics,
            domain_rows,
            variance_rows,
            sample_rows,
            token_gap_rows,
            token_gap_vocab_rows,
            entropy_distribution_rows,
            entropy_vocab_rows,
        )
