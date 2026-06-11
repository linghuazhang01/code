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


def _sample_value_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "variance": _var(values),
    }


def _tensor_to_float_list(tensor: Any) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().tolist()]


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
        self.log_validation = bool(_cfg_get(audit_config, "log_validation_metrics", True))
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
        self.sample_gradient_norm_enabled = bool(_cfg_get(audit_config, "sample_gradient_norm_enabled", True))
        self.sample_gradient_cos_enabled = bool(_cfg_get(audit_config, "sample_gradient_cos_enabled", False))
        self.sample_gradient_cos_freq_steps = max(
            1,
            int(_cfg_get(audit_config, "sample_gradient_cos_freq_steps", 1)),
        )
        self.sample_gradient_log_sample_level = bool(
            _cfg_get(audit_config, "sample_gradient_log_sample_level", True)
        )
        self.full_gradient_offload_domain_gradients = bool(
            _cfg_get(audit_config, "full_gradient_offload_domain_gradients", True)
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

    def should_compute_full_gradient(self, step: int) -> bool:
        full_gradient_active = self.should_compute_domain_gradient(step)
        sample_gradient_active = self.sample_gradient_enabled and (
            self.sample_gradient_norm_enabled
            or (
                self.sample_gradient_cos_enabled
                and step % self.sample_gradient_cos_freq_steps == 0
            )
        )
        return self.enabled and (full_gradient_active or sample_gradient_active)

    def should_compute_domain_gradient(self, step: int) -> bool:
        return self.enabled and self.full_gradient_enabled and step % self.full_gradient_freq_steps == 0

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
                "sample_gradient_enabled": self.sample_gradient_enabled and mode == "train",
                "sample_gradient_norm_enabled": self.sample_gradient_norm_enabled,
                "sample_gradient_cos_enabled": self.sample_gradient_cos_enabled
                and mode == "train"
                and step % self.sample_gradient_cos_freq_steps == 0,
                "sample_gradient_cos_freq_steps": self.sample_gradient_cos_freq_steps,
                "sample_gradient_log_sample_level": self.sample_gradient_log_sample_level,
                "offload_domain_gradients": self.full_gradient_offload_domain_gradients,
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
            metrics, domain_rows, variance_rows, sample_rows = self._compute_training_rows(batch, step, lr)
        except Exception as exc:  # pragma: no cover - defensive remote logging
            self._write_jsonl("audit_errors.jsonl", [{"step": step, "stage": "training", "error": repr(exc)}])
            return {self._global_tag("audit", "error"): 1.0}

        metrics[self._global_tag("audit", "wall_time_step")] = time.perf_counter() - started_at
        self._write_jsonl("domain_step_metrics.jsonl", domain_rows)
        self._write_jsonl("loss_variance_domain_step.jsonl", variance_rows)
        self._write_jsonl("loss_variance_sample.jsonl", sample_rows)
        return metrics

    def log_validation_metrics(self, val_metrics: dict[str, Any], step: int) -> dict[str, float]:
        return _log_validation_metrics(self, val_metrics, step)

    def log_training_cost(self, metrics: dict[str, Any], step: int, n_gpus: int = 1) -> dict[str, float]:
        return _log_training_cost(self, metrics, step, n_gpus)

    def _compute_training_rows(self, batch: Any, step: int, lr: Any) -> tuple[dict[str, float], list, list, list]:
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
        reverse_kl = torch.zeros_like(old_log_probs)
        for idx, label in enumerate(labels):
            teacher_log_prob = code_teacher_log_prob[idx] if label == "code" else math_teacher_log_prob[idx]
            teacher_log_probs[idx] = teacher_log_prob
            if self.lambda_vals == 1.0:
                reverse_kl[idx] = old_log_probs[idx] - teacher_log_prob
            else:
                reverse_kl[idx] = (
                    old_log_probs[idx]
                    - base_log_prob[idx]
                    - (teacher_log_prob - base_log_prob[idx]) * self.lambda_vals
                )

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
            }
            for key in domain_metric_keys:
                numeric = finite_float(row.get(key))
                if numeric is not None:
                    metrics[self._domain_tag(safe_domain, domain_metric_category(key), key)] = numeric

            if self.log_sample_level and indices:
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

        return metrics, domain_rows, variance_rows, sample_rows
