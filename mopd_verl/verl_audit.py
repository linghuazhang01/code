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
from mopd_verl.tensorboard_filter import (
    filter_tensorboard_metrics as _filter_tensorboard_metrics,
    is_direct_audit_metric_key,
)
from mopd_verl.tensorboard_tags import domain_metric_category, safe_name


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
            f"{prefix}_p05": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
            f"{prefix}_sum": None,
        }
    values = values.detach().float()
    return {
        f"{prefix}_mean": float(values.mean().detach().cpu().item()),
        f"{prefix}_std": float(values.std(unbiased=False).detach().cpu().item()),
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


def _token_conflict_attribution(
    *,
    labels: list[str],
    domains: list[str],
    token_ids: Any | None,
    response_mask: Any,
    reverse_kl: Any,
    teacher_teacher_diff: Any,
    student_teacher_diff: Any,
    combined_diff: Any,
    top_k: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    import torch

    scores = combined_diff.detach().float() * response_mask
    abs_losses = reverse_kl.detach().float().abs() * response_mask
    teacher_diffs = teacher_teacher_diff.detach().float() * response_mask
    student_diffs = student_teacher_diff.detach().float() * response_mask
    total_proxy = float(scores.sum().detach().cpu().item())
    total_teacher_diff = float(teacher_diffs.sum().detach().cpu().item())
    summaries: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    seq_len = int(response_mask.shape[-1])
    positions = torch.arange(seq_len, dtype=torch.float32, device=response_mask.device)
    configured_domains = list(dict.fromkeys(domains + sorted(set(labels))))

    for domain in configured_domains:
        indices = [idx for idx, label in enumerate(labels) if label == domain]
        if not indices:
            continue

        valid = response_mask[indices].detach().bool()
        token_count = int(valid.sum().detach().cpu().item())
        domain_scores = scores[indices]
        domain_abs_losses = abs_losses[indices]
        domain_teacher_diffs = teacher_diffs[indices]
        domain_student_diffs = student_diffs[indices]
        proxy_mass = float(domain_scores.sum().detach().cpu().item())
        teacher_diff_mass = float(domain_teacher_diffs.sum().detach().cpu().item())
        student_diff_mass = float(domain_student_diffs.sum().detach().cpu().item())
        valid_teacher_diffs = domain_teacher_diffs[valid] if token_count else None
        valid_student_diffs = domain_student_diffs[valid] if token_count else None
        valid_scores = domain_scores[valid] if token_count else None
        valid_abs_losses = domain_abs_losses[valid] if token_count else None
        summary = {
            "proxy_mass": proxy_mass,
            "proxy_mean": proxy_mass / token_count if token_count else 0.0,
            "proxy_mass_frac": proxy_mass / total_proxy if total_proxy > 0.0 else 0.0,
            "combined_diff_mass": proxy_mass,
            "combined_diff_mean": proxy_mass / token_count if token_count else 0.0,
            "combined_diff_mass_frac": proxy_mass / total_proxy if total_proxy > 0.0 else 0.0,
            "combined_diff_p95": 0.0
            if valid_scores is None
            else float(torch.quantile(valid_scores.float(), 0.95).detach().cpu().item()),
            "combined_diff_max": 0.0
            if valid_scores is None
            else float(valid_scores.max().detach().cpu().item()),
            "teacher_teacher_diff_mass": teacher_diff_mass,
            "teacher_teacher_diff_mean": teacher_diff_mass / token_count if token_count else 0.0,
            "teacher_teacher_diff_mass_frac": teacher_diff_mass / total_teacher_diff if total_teacher_diff > 0.0 else 0.0,
            "teacher_teacher_diff_p95": 0.0
            if valid_teacher_diffs is None
            else float(torch.quantile(valid_teacher_diffs.float(), 0.95).detach().cpu().item()),
            "teacher_teacher_diff_max": 0.0
            if valid_teacher_diffs is None
            else float(valid_teacher_diffs.max().detach().cpu().item()),
            "student_teacher_diff_mass": student_diff_mass,
            "student_teacher_diff_mean": student_diff_mass / token_count if token_count else 0.0,
            "student_teacher_diff_p95": 0.0
            if valid_student_diffs is None
            else float(torch.quantile(valid_student_diffs.float(), 0.95).detach().cpu().item()),
            "student_teacher_diff_max": 0.0
            if valid_student_diffs is None
            else float(valid_student_diffs.max().detach().cpu().item()),
            "teacher_disagreement_mean": 0.0
            if token_count == 0
            else float(valid_teacher_diffs.mean().detach().cpu().item()),
            "token_abs_opd_loss_mean": 0.0
            if token_count == 0
            else float(valid_abs_losses.mean().detach().cpu().item()),
            "opd_signal_abs_mean": 0.0
            if token_count == 0
            else float(valid_abs_losses.mean().detach().cpu().item()),
            "top1_token_share": 0.0,
            "top10_token_share": 0.0,
            "unique_token_count": 0.0,
        }

        if token_ids is not None and token_count > 0 and (proxy_mass > 0.0 or teacher_diff_mass > 0.0):
            domain_token_ids = token_ids[indices]
            flat_ids = domain_token_ids[valid.cpu()].long()
            flat_scores = domain_scores.detach().cpu()[valid.cpu()].float()
            flat_abs_losses = domain_abs_losses.detach().cpu()[valid.cpu()].float()
            flat_teacher_diffs = domain_teacher_diffs.detach().cpu()[valid.cpu()].float()
            flat_student_diffs = domain_student_diffs.detach().cpu()[valid.cpu()].float()
            flat_positions = positions.expand(len(indices), -1).detach().cpu()[valid.cpu()].float()
            positive_scores = (flat_scores > 0) | (flat_teacher_diffs > 0)
            flat_ids = flat_ids[positive_scores]
            flat_scores = flat_scores[positive_scores]
            flat_abs_losses = flat_abs_losses[positive_scores]
            flat_teacher_diffs = flat_teacher_diffs[positive_scores]
            flat_student_diffs = flat_student_diffs[positive_scores]
            flat_positions = flat_positions[positive_scores]
            if flat_ids.numel() > 0:
                unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
                unique_count = int(unique_ids.numel())
                score_sums = torch.zeros(unique_count, dtype=torch.float64)
                teacher_diff_sums = torch.zeros(unique_count, dtype=torch.float64)
                count_sums = torch.zeros(unique_count, dtype=torch.float64)
                loss_sums = torch.zeros(unique_count, dtype=torch.float64)
                student_diff_sums = torch.zeros(unique_count, dtype=torch.float64)
                position_sums = torch.zeros(unique_count, dtype=torch.float64)
                ones = torch.ones_like(flat_scores, dtype=torch.float64)
                score_sums.scatter_add_(0, inverse, flat_scores.double())
                teacher_diff_sums.scatter_add_(0, inverse, flat_teacher_diffs.double())
                count_sums.scatter_add_(0, inverse, ones)
                loss_sums.scatter_add_(0, inverse, flat_abs_losses.double())
                student_diff_sums.scatter_add_(0, inverse, flat_student_diffs.double())
                position_sums.scatter_add_(0, inverse, flat_positions.double())

                top_count = min(max(1, top_k), unique_count)
                top_teacher_scores, top_indices = torch.topk(teacher_diff_sums, top_count)
                top_score_values = [float(value) for value in torch.topk(score_sums, top_count).values.tolist()]
                summary["unique_token_count"] = float(unique_count)
                summary["top1_token_share"] = (
                    top_score_values[0] / proxy_mass if proxy_mass > 0.0 and top_score_values else 0.0
                )
                summary["top10_token_share"] = (
                    sum(top_score_values[:10]) / proxy_mass if proxy_mass > 0.0 else 0.0
                )
                summary["top1_teacher_diff_share"] = (
                    float(top_teacher_scores[0].item()) / teacher_diff_mass
                    if teacher_diff_mass > 0.0 and top_teacher_scores.numel() > 0
                    else 0.0
                )
                summary["top10_teacher_diff_share"] = (
                    float(top_teacher_scores[:10].sum().item()) / teacher_diff_mass
                    if teacher_diff_mass > 0.0
                    else 0.0
                )
                for rank, token_index in enumerate(top_indices.tolist(), start=1):
                    token_count_value = float(count_sums[token_index].item())
                    score_sum = float(score_sums[token_index].item())
                    teacher_diff_sum = float(teacher_diff_sums[token_index].item())
                    rows.append(
                        {
                            "domain": domain,
                            "rank": rank,
                            "token_id": int(unique_ids[token_index].item()),
                            "token_count": token_count_value,
                            "conflict_proxy_sum": score_sum,
                            "conflict_proxy_mean": score_sum / token_count_value if token_count_value else 0.0,
                            "conflict_proxy_frac": score_sum / proxy_mass if proxy_mass > 0.0 else 0.0,
                            "combined_diff_sum": score_sum,
                            "combined_diff_mean": score_sum / token_count_value if token_count_value else 0.0,
                            "combined_diff_frac": score_sum / proxy_mass if proxy_mass > 0.0 else 0.0,
                            "teacher_teacher_diff_sum": teacher_diff_sum,
                            "teacher_teacher_diff_mean": teacher_diff_sum / token_count_value
                            if token_count_value
                            else 0.0,
                            "teacher_teacher_diff_frac": teacher_diff_sum / teacher_diff_mass
                            if teacher_diff_mass > 0.0
                            else 0.0,
                            "student_teacher_diff_mean": float(student_diff_sums[token_index].item() / token_count_value)
                            if token_count_value
                            else 0.0,
                            "token_abs_opd_loss_mean": float(loss_sums[token_index].item() / token_count_value)
                            if token_count_value
                            else 0.0,
                            "teacher_disagreement_mean": float(
                                teacher_diff_sums[token_index].item() / token_count_value
                            )
                            if token_count_value
                            else 0.0,
                            "response_position_mean": float(position_sums[token_index].item() / token_count_value)
                            if token_count_value
                            else 0.0,
                        }
                    )

        summaries[domain] = summary

    return summaries, rows


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


def _equal_workload_partitions(
    indices: list[int],
    workloads: list[int],
    partition_count: int,
) -> list[list[int]]:
    """Greedily balance workload while keeping equal sample counts."""

    capacity = len(indices) // partition_count
    partitions: list[list[int]] = [[] for _ in range(partition_count)]
    partition_workloads = [0 for _ in range(partition_count)]
    ordered_indices = sorted(indices, key=lambda idx: (-workloads[idx], idx))
    for sample_idx in ordered_indices:
        candidates = [rank for rank in range(partition_count) if len(partitions[rank]) < capacity]
        target_rank = min(
            candidates,
            key=lambda rank: (partition_workloads[rank], len(partitions[rank]), rank),
        )
        partitions[target_rank].append(sample_idx)
        partition_workloads[target_rank] += workloads[sample_idx]

    for rank, partition in enumerate(partitions):
        partition.sort(key=lambda idx: (workloads[idx], idx))
        partitions[rank] = partition[::2] + partition[1::2][::-1]
    return partitions


def _ensure_meta_info(batch: Any) -> dict[str, Any]:
    meta_info = getattr(batch, "meta_info", None)
    if not isinstance(meta_info, dict):
        meta_info = {}
        setattr(batch, "meta_info", meta_info)
    return meta_info


class MOPDAuditLogger:
    """Writes per-domain audit JSONL rows and TensorBoard-compatible scalars."""

    def __init__(self, config: Any):
        self.config = config
        audit_config = _cfg_get(config, "mopd_audit", {})
        self.enabled = bool(_cfg_get(audit_config, "enabled", False))
        self.output_dir = Path(str(_cfg_get(audit_config, "output_dir", "mopd_audit")))
        self.domains = list(_cfg_get(audit_config, "domains", ["math", "code"]))
        self.prefix = str(_cfg_get(audit_config, "tensorboard_prefix", "mopd"))
        self.tensorboard_layout = str(_cfg_get(audit_config, "tensorboard_layout", "domain_category"))
        self.tensorboard_prune_mode = str(_cfg_get(audit_config, "tensorboard_prune_mode", "none")).lower()
        self.max_samples_per_domain = int(_cfg_get(audit_config, "max_samples_per_domain", 32))
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
        self.full_gradient_train_max_samples_per_domain = _optional_positive_int(
            _cfg_get(audit_config, "full_gradient_train_max_samples_per_domain", None)
        )
        self.full_gradient_micro_batch_size_per_gpu = max(
            1,
            int(_cfg_get(audit_config, "full_gradient_micro_batch_size_per_gpu", 1)),
        )
        self.full_gradient_storage_dtype = str(_cfg_get(audit_config, "full_gradient_storage_dtype", "float32"))
        self.sample_gradient_enabled = bool(_cfg_get(audit_config, "sample_gradient_enabled", False))
        self.sample_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "sample_gradient_freq_steps", 1)))
        self.sample_gradient_norm_enabled = bool(_cfg_get(audit_config, "sample_gradient_norm_enabled", True))
        self.sample_gradient_cos_enabled = bool(_cfg_get(audit_config, "sample_gradient_cos_enabled", False))
        self.sample_gradient_cos_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "sample_gradient_cos_freq_steps", 1)),
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
        self.entropy_enabled = bool(_cfg_get(audit_config, "entropy_enabled", True))
        self.entropy_freq_steps = max(1, int(_cfg_get(audit_config, "entropy_freq_steps", 1)))
        self.token_conflict_enabled = bool(_cfg_get(audit_config, "token_conflict_enabled", True))
        self.token_conflict_freq_steps = max(1, int(_cfg_get(audit_config, "token_conflict_freq_steps", 1)))
        self.token_conflict_top_k = max(1, int(_cfg_get(audit_config, "token_conflict_top_k", 50)))
        self.token_gradient_enabled = bool(_cfg_get(audit_config, "token_gradient_enabled", False))
        self.token_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "token_gradient_freq_steps", 10)))
        self.token_gradient_max_samples_per_domain = max(
            1,
            int(_cfg_get(audit_config, "token_gradient_max_samples_per_domain", 8)),
        )
        self.token_gradient_top_k_per_sample = max(
            1,
            int(_cfg_get(audit_config, "token_gradient_top_k_per_sample", 4)),
        )
        token_gradient_top_p = _cfg_get(audit_config, "token_gradient_top_p", 0.10)
        self.token_gradient_top_p = min(
            1.0,
            max(0.0, float(0.10 if token_gradient_top_p is None else token_gradient_top_p)),
        )
        self.token_gradient_min_teacher_diff = float(
            _cfg_get(audit_config, "token_gradient_min_teacher_diff", 0.0)
        )
        self.token_gradient_strict_grad_restore = bool(
            _cfg_get(audit_config, "token_gradient_strict_grad_restore", False)
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

    def should_log_entropy(self, step: int) -> bool:
        return self._freq_active(self.entropy_enabled, self.entropy_freq_steps, step)

    def should_log_token_conflict(self, step: int) -> bool:
        return self._freq_active(self.token_conflict_enabled, self.token_conflict_freq_steps, step)

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
        return self.enabled and (full_gradient_active or sample_gradient_active or self.should_compute_token_gradient(step))

    def should_compute_domain_gradient(self, step: int) -> bool:
        full_gradient_active = self.full_gradient_enabled and step % self.full_gradient_freq_steps == 0
        return self.enabled and (full_gradient_active or self.should_compute_token_gradient(step))

    def should_compute_token_gradient(self, step: int) -> bool:
        return self._freq_active(self.token_gradient_enabled, self.token_gradient_freq_steps, step)

    def balance_domain_gradient_batch(
        self,
        batch: Any,
        *,
        step: int,
        world_size: int,
    ) -> dict[str, float]:
        """Align domain counts across contiguous actor-rank dispatch chunks."""

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
        }
        meta_info[_DOMAIN_PARTITION_META_KEY] = partition_meta
        metrics = {
            "global/audit/full_gradient_domain_partition_aligned": 0.0,
            "global/audit/full_gradient_domain_partition_unsupported": 1.0,
        }
        if world_size <= 1:
            partition_meta["aligned"] = True
            partition_meta["unsupported_reason"] = ""
            metrics["global/audit/full_gradient_domain_partition_aligned"] = 1.0
            metrics["global/audit/full_gradient_domain_partition_unsupported"] = 0.0
            return metrics
        if len(self.domains) != 2 or "attention_mask" not in batch.batch:
            partition_meta["unsupported_reason"] = "requires_two_domains_and_attention_mask"
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

        lengths = attention_mask.detach().view(batch_size, -1).sum(dim=-1).to(device="cpu").long()
        workloads = [24576 * int(length) + int(length) ** 2 for length in lengths.tolist()]
        domain_partitions = {
            domain: _equal_workload_partitions(indices, workloads, world_size)
            for domain, indices in domain_indices.items()
        }
        rank_partitions = [
            [
                sample_idx
                for domain in self.domains
                for sample_idx in domain_partitions[domain][rank]
            ]
            for rank in range(world_size)
        ]
        expected_rank_size = batch_size // world_size
        if any(len(partition) != expected_rank_size for partition in rank_partitions):
            partition_meta["unsupported_reason"] = "rank_partition_size_mismatch"
            return metrics

        import torch

        global_idx = torch.tensor(
            [sample_idx for partition in rank_partitions for sample_idx in partition],
            dtype=torch.long,
        )
        batch.reorder(global_idx)
        domain_block_sample_counts = {
            domain: len(domain_partitions[domain][0]) for domain in self.domains
        }
        rank_domain_sample_counts = [
            {domain: len(domain_partitions[domain][rank]) for domain in self.domains}
            for rank in range(world_size)
        ]
        partition_meta.update(
            {
                "aligned": True,
                "unsupported_reason": "",
                "rank_sample_count": int(expected_rank_size),
                "domain_block_sample_counts": domain_block_sample_counts,
                "rank_domain_sample_counts": rank_domain_sample_counts,
            }
        )
        metrics["global/audit/full_gradient_domain_partition_aligned"] = 1.0
        metrics["global/audit/full_gradient_domain_partition_unsupported"] = 0.0
        return metrics

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
                "learning_rate": self._current_learning_rate_value(),
                "sample_gradient_enabled": self.should_compute_sample_gradient(step) and mode == "train",
                "sample_gradient_freq_steps": self.sample_gradient_freq_steps,
                "sample_gradient_norm_enabled": self.sample_gradient_norm_enabled,
                "sample_gradient_cos_enabled": self.sample_gradient_cos_enabled
                and mode == "train"
                and step % self.sample_gradient_cos_freq_steps == 0,
                "sample_gradient_cos_freq_steps": self.sample_gradient_cos_freq_steps,
                "sample_gradient_log_sample_level": self.should_log_sample_gradient_level(step),
                "sample_gradient_log_sample_level_freq_steps": self.sample_gradient_log_sample_level_freq_steps,
                "offload_domain_gradients": self.full_gradient_offload_domain_gradients,
                "token_gradient_enabled": self.should_compute_token_gradient(step) and mode == "train",
                "token_gradient_freq_steps": self.token_gradient_freq_steps,
                "token_gradient_max_samples_per_domain": self.token_gradient_max_samples_per_domain,
                "token_gradient_top_k_per_sample": self.token_gradient_top_k_per_sample,
                "token_gradient_top_p": self.token_gradient_top_p,
                "token_gradient_min_teacher_diff": self.token_gradient_min_teacher_diff,
                "token_gradient_strict_grad_restore": self.token_gradient_strict_grad_restore,
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
                token_conflict_rows,
                token_gap_rows,
                entropy_distribution_rows,
            ) = self._compute_training_rows(batch, step, lr)
        except Exception as exc:  # pragma: no cover - defensive remote logging
            self._write_jsonl("audit_errors.jsonl", [{"step": step, "stage": "training", "error": repr(exc)}])
            return {self._global_tag("audit", "error"): 1.0}

        metrics[self._global_tag("audit", "wall_time_step")] = time.perf_counter() - started_at
        self._write_jsonl("domain_step_metrics.jsonl", domain_rows)
        self._write_jsonl("loss_variance_domain_step.jsonl", variance_rows)
        self._write_jsonl("loss_variance_sample.jsonl", sample_rows)
        if token_conflict_rows:
            self._write_jsonl("token_conflict_attribution.jsonl", token_conflict_rows)
        if token_gap_rows:
            self._write_jsonl("token_gap_vectors.jsonl", token_gap_rows)
        if entropy_distribution_rows:
            self._write_jsonl("entropy_distribution_vectors.jsonl", entropy_distribution_rows)
        return metrics

    def log_validation_metrics(self, val_metrics: dict[str, Any], step: int) -> dict[str, float]:
        if not self.should_log_validation_metrics(step):
            return {}
        return _log_validation_metrics(self, val_metrics, step)

    def log_training_cost(self, metrics: dict[str, Any], step: int, n_gpus: int = 1) -> dict[str, float]:
        return _log_training_cost(self, metrics, step, n_gpus)

    def _compute_training_rows(
        self, batch: Any, step: int, lr: Any
    ) -> tuple[dict[str, float], list, list, list, list, list, list]:
        import torch

        tensor_batch = batch.batch
        non_tensor = batch.non_tensor_batch
        old_log_probs = tensor_batch["old_log_probs"].detach().float()
        response_mask = response_mask_from_batch(tensor_batch, old_log_probs)
        batch_keys = set(tensor_batch.keys())
        math_teacher_log_prob = (tensor_batch["math_teacher_log_prob"] if "math_teacher_log_prob" in batch_keys else old_log_probs).detach().float()
        base_log_prob = (tensor_batch["base_log_prob"] if "base_log_prob" in batch_keys else old_log_probs).detach().float()
        code_teacher_log_prob = (
            tensor_batch["code_teacher_log_prob"] if "code_teacher_log_prob" in batch_keys else math_teacher_log_prob
        ).detach().float()
        batch_size = int(old_log_probs.shape[0])

        labels = extract_teacher_domains(non_tensor, batch_size)
        sample_ids = extract_sample_ids(non_tensor, batch_size, step)

        teacher_log_probs = torch.zeros_like(old_log_probs)
        teacher_teacher_diff = torch.zeros_like(old_log_probs)
        student_teacher_diff = torch.zeros_like(old_log_probs)
        reverse_kl = torch.zeros_like(old_log_probs)
        has_alternate_teacher = "code_teacher_log_prob" in batch_keys
        for idx, label in enumerate(labels):
            teacher_log_prob = code_teacher_log_prob[idx] if label == "code" else math_teacher_log_prob[idx]
            if has_alternate_teacher:
                alternate_teacher_log_prob = math_teacher_log_prob[idx] if label == "code" else code_teacher_log_prob[idx]
                teacher_teacher_diff[idx] = (teacher_log_prob - alternate_teacher_log_prob).abs()
            else:
                teacher_teacher_diff[idx] = (teacher_log_prob - old_log_probs[idx]).abs()
            student_teacher_diff[idx] = (old_log_probs[idx] - teacher_log_prob).abs()
            teacher_log_probs[idx] = teacher_log_prob
            if self.lambda_vals == 1.0:
                reverse_kl[idx] = old_log_probs[idx] - teacher_log_prob
            else:
                reverse_kl[idx] = (
                    old_log_probs[idx]
                    - base_log_prob[idx]
                    - (teacher_log_prob - base_log_prob[idx]) * self.lambda_vals
                )
        combined_diff = student_teacher_diff * teacher_teacher_diff
        gap_signed = teacher_log_probs - old_log_probs
        gap_abs = gap_signed.abs()

        student_entropy = (
            tensor_batch["student_entropy"].detach().float() if "student_entropy" in batch_keys else None
        )
        teacher_entropy = None
        if "math_teacher_entropy" in batch_keys:
            math_teacher_entropy = tensor_batch["math_teacher_entropy"].detach().float()
            code_teacher_entropy = (
                tensor_batch["code_teacher_entropy"].detach().float()
                if "code_teacher_entropy" in batch_keys
                else math_teacher_entropy
            )
            teacher_entropy = torch.zeros_like(old_log_probs)
            for idx, label in enumerate(labels):
                teacher_entropy[idx] = code_teacher_entropy[idx] if label == "code" else math_teacher_entropy[idx]
        teacher_student_cross_entropy = None
        if "math_teacher_student_cross_entropy" in batch_keys:
            math_teacher_student_cross_entropy = tensor_batch["math_teacher_student_cross_entropy"].detach().float()
            code_teacher_student_cross_entropy = (
                tensor_batch["code_teacher_student_cross_entropy"].detach().float()
                if "code_teacher_student_cross_entropy" in batch_keys
                else math_teacher_student_cross_entropy
            )
            teacher_student_cross_entropy = torch.zeros_like(old_log_probs)
            for idx, label in enumerate(labels):
                teacher_student_cross_entropy[idx] = (
                    code_teacher_student_cross_entropy[idx]
                    if label == "code"
                    else math_teacher_student_cross_entropy[idx]
                )
        elif "teacher_student_cross_entropy" in batch_keys:
            teacher_student_cross_entropy = tensor_batch["teacher_student_cross_entropy"].detach().float()

        sample_token_opd_loss_mean = _mask_mean(reverse_kl, response_mask)
        sample_opd_loss = (reverse_kl * response_mask).sum(dim=-1)
        sample_loss_sq_mean = _mask_mean(reverse_kl.square(), response_mask)
        sample_loss_var = torch.clamp(sample_loss_sq_mean - sample_token_opd_loss_mean.square(), min=0.0)
        sample_loss_std = torch.sqrt(sample_loss_var)
        sample_loss_cv = sample_loss_std / (sample_token_opd_loss_mean.abs() + 1e-8)
        effective_tokens = response_mask.sum(dim=-1).detach().cpu().tolist()
        teacher_student_gap = _mask_mean(teacher_log_probs - old_log_probs, response_mask)
        teacher_logprob_mean = _mask_mean(teacher_log_probs, response_mask)
        advantages = tensor_batch["advantages"].detach().float() if "advantages" in batch_keys else -reverse_kl
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
        token_conflict_rows: list[dict[str, Any]] = []
        token_gap_rows: list[dict[str, Any]] = []
        entropy_distribution_rows: list[dict[str, Any]] = []
        token_gap_active = self.should_log_token_gap(step)
        entropy_active = self.should_log_entropy(step)
        token_conflict_active = self.should_log_token_conflict(step)
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
        token_conflict_summaries: dict[str, dict[str, float]] = {}
        if token_conflict_active:
            token_ids = _response_token_id_matrix(tensor_batch, batch_keys, response_mask)
            token_conflict_summaries, token_conflict_rows = _token_conflict_attribution(
                labels=labels,
                domains=configured_domains,
                token_ids=token_ids,
                response_mask=response_mask,
                reverse_kl=reverse_kl,
                teacher_teacher_diff=teacher_teacher_diff,
                student_teacher_diff=student_teacher_diff,
                combined_diff=combined_diff,
                top_k=self.token_conflict_top_k,
            )
            for token_row in token_conflict_rows:
                token_row["step"] = step
                token_row["learning_rate"] = learning_rate

        for domain in configured_domains:
            indices = indices_by_domain[domain]
            safe_domain = safe_name(domain)
            domain_token_count = token_count_by_domain[domain]
            domain_sample_count = sample_count_by_domain[domain]
            domain_loss_vars = [sample_loss_vars[idx] for idx in indices]
            domain_cvs = [loss_cvs[idx] for idx in indices]
            domain_gaps = [gap_means[idx] for idx in indices]
            domain_teacher_logprobs = [teacher_logprob_means[idx] for idx in indices]
            domain_advantages = [advantage_means[idx] for idx in indices]
            domain_rewards = [reward_values[idx] for idx in indices] if reward_values is not None else []
            domain_sample_ids = [sample_ids[idx] for idx in indices]
            domain_token_counts = [token_counts[idx] for idx in indices]
            domain_token_stats = (
                _masked_token_stats(reverse_kl[indices], response_mask[indices])
                if indices
                else {"mean": None, "std": None, "variance": None}
            )
            signed_gap_stats: dict[str, float | None] = {}
            abs_gap_stats: dict[str, float | None] = {}
            if token_gap_active:
                domain_gap_vector = None
                domain_gap_abs_vector = None
                if indices:
                    domain_valid_mask = response_mask[indices].detach().bool()
                    domain_gap_vector = gap_signed[indices][domain_valid_mask]
                    domain_gap_abs_vector = gap_abs[indices][domain_valid_mask]
                signed_gap_stats = _token_distribution_stats(domain_gap_vector, "gap_signed")
                abs_gap_stats = _token_distribution_stats(domain_gap_abs_vector, "gap_abs")
                if domain_gap_vector is not None and int(domain_gap_vector.numel()) > 0:
                    token_gap_rows.append(
                        {
                            "step": step,
                            "domain": domain,
                            "learning_rate": learning_rate,
                            "token_count": int(domain_gap_vector.numel()),
                            "gap_signed_vector_domain": _tensor_to_float_list(domain_gap_vector),
                            "gap_abs_vector_domain": _tensor_to_float_list(domain_gap_abs_vector),
                            "gap_vector_domain": _tensor_to_float_list(domain_gap_vector),
                        }
                    )
            entropy_metrics: dict[str, float | None] = {}
            teacher_entropy_stats: dict[str, float | None] = {}
            student_entropy_stats: dict[str, float | None] = {}
            cross_entropy_stats: dict[str, float | None] = {}
            if entropy_active:
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
                teacher_entropy_stats = _token_distribution_stats(teacher_entropy_vector, "teacher_entropy")
                student_entropy_stats = _token_distribution_stats(student_entropy_vector, "student_entropy")
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
                    entropy_row["teacher_entropy_vector_domain"] = _tensor_to_float_list(teacher_entropy_vector)
                if student_entropy_vector is not None and int(student_entropy_vector.numel()) > 0:
                    entropy_row["student_entropy_vector_domain"] = _tensor_to_float_list(student_entropy_vector)
                if cross_entropy_vector is not None and int(cross_entropy_vector.numel()) > 0:
                    entropy_row["teacher_student_cross_entropy_vector_domain"] = _tensor_to_float_list(
                        cross_entropy_vector
                    )
                if len(entropy_row) > 4:
                    entropy_distribution_rows.append(entropy_row)
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
                "token_opd_loss_mean": domain_token_stats["mean"],
                "token_opd_loss_std": domain_token_stats["std"],
                "token_opd_loss_variance": domain_token_stats["variance"],
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
                "teacher_student_gap_mean": _mean(domain_gaps),
                "teacher_confidence_mean": _mean(confidence_values),
                "calibration_error": calibration_error,
                "duplicate_rate": duplicate_rate,
            }
            row.update(signed_gap_stats)
            row.update(abs_gap_stats)
            row.update(entropy_metrics)
            row.update(teacher_entropy_stats)
            row.update(student_entropy_stats)
            row.update(cross_entropy_stats)
            row.update(token_conflict_summaries.get(domain, {}))
            domain_rows.append(row)
            variance_rows.append(
                {
                    "step": step,
                    "domain": domain,
                    "learning_rate": learning_rate,
                    "metric_scope": "domain_step",
                    "loss_name": "opd_loss_token",
                    "domain_sample_count": domain_sample_count,
                    "domain_token_count": domain_token_count,
                    "token_opd_loss_mean": row["token_opd_loss_mean"],
                    "token_opd_loss_std": row["token_opd_loss_std"],
                    "token_opd_loss_variance": row["token_opd_loss_variance"],
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
                "proxy_mass",
                "proxy_mean",
                "proxy_mass_frac",
                "teacher_disagreement_mean",
                "token_abs_opd_loss_mean",
                "opd_signal_abs_mean",
                "combined_diff_mass",
                "combined_diff_mean",
                "combined_diff_mass_frac",
                "combined_diff_p95",
                "combined_diff_max",
                "teacher_teacher_diff_mass",
                "teacher_teacher_diff_mean",
                "teacher_teacher_diff_mass_frac",
                "teacher_teacher_diff_p95",
                "teacher_teacher_diff_max",
                "student_teacher_diff_mass",
                "student_teacher_diff_mean",
                "student_teacher_diff_p95",
                "student_teacher_diff_max",
                "top1_teacher_diff_share",
                "top10_teacher_diff_share",
                "top1_token_share",
                "top10_token_share",
                "unique_token_count",
            }
            for key in domain_metric_keys:
                numeric = finite_float(row.get(key))
                if numeric is not None:
                    metrics[self._domain_tag(safe_domain, domain_metric_category(key), key)] = numeric

            if sample_level_active and indices:
                for idx in indices[: self.max_samples_per_domain]:
                    sample_rows.append(
                        {
                            "step": step,
                            "domain": domain,
                            "sample_id": sample_ids[idx],
                            "learning_rate": learning_rate,
                            "metric_scope": "sample_token",
                            "loss_name": "opd_loss_token",
                            "effective_tokens": token_counts[idx],
                            "opd_loss": opd_losses[idx],
                            "sample_token_opd_loss_mean": sample_token_opd_loss_means[idx],
                            "sample_token_opd_loss_variance": float(sample_loss_var[idx].detach().cpu().item()),
                            "training_reward": None if reward_values is None else reward_values[idx],
                            "training_correctness": None if correctness_values is None else correctness_values[idx],
                        }
                    )

        if total_tokens:
            global_token_stats = _masked_token_stats(reverse_kl, response_mask)
            global_sample_stats = _sample_value_stats(opd_losses)
            global_loss_metrics = {
                "token_opd_loss_mean": global_token_stats["mean"],
                "token_opd_loss_std": global_token_stats["std"],
                "token_opd_loss_variance": global_token_stats["variance"],
                "sample_opd_loss_mean": global_sample_stats["mean"],
                "sample_opd_loss_std": global_sample_stats["std"],
                "sample_opd_loss_variance": global_sample_stats["variance"],
            }
            for key, value in global_loss_metrics.items():
                numeric = finite_float(value)
                if numeric is not None:
                    metrics[self._global_tag("loss", key)] = numeric
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
            token_conflict_rows,
            token_gap_rows,
            entropy_distribution_rows,
        )
