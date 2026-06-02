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
from mopd_verl.audit_validation import (
    log_validation_anchor_batch as _log_validation_anchor_batch,
    should_update_validation_anchor as _should_update_validation_anchor,
)
from mopd_verl.tensorboard_filter import (
    filter_tensorboard_metrics as _filter_tensorboard_metrics,
    is_direct_audit_metric_key,
)
from mopd_verl.tensorboard_tags import domain_metric_category, safe_name


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


def _var(values: list[float]) -> float | None:
    return float(np.var(values)) if values else None


def _optional_positive_int(value: Any) -> int | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return max(1, int(value))


def _mask_mean(matrix: Any, mask: Any) -> Any:
    import torch

    denom = mask.sum(dim=-1).clamp(min=1)
    return (matrix * mask).sum(dim=-1) / denom


def _tensor_to_float_list(tensor: Any) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().tolist()]


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
        self.validation_anchor_enabled = bool(_cfg_get(audit_config, "validation_anchor_enabled", True))
        self.validation_anchor_refresh_steps = int(_cfg_get(audit_config, "validation_anchor_refresh_steps", 0))
        self.full_gradient_enabled = bool(_cfg_get(audit_config, "full_gradient_enabled", False))
        self.full_gradient_freq_steps = max(1, int(_cfg_get(audit_config, "full_gradient_freq_steps", 1)))
        self.full_gradient_train_max_samples_per_domain = _optional_positive_int(
            _cfg_get(audit_config, "full_gradient_train_max_samples_per_domain", None)
        )
        self.full_gradient_validation_max_samples_per_domain = _optional_positive_int(
            _cfg_get(audit_config, "full_gradient_validation_max_samples_per_domain", None)
        )
        self.full_gradient_micro_batch_size_per_gpu = max(
            1,
            int(_cfg_get(audit_config, "full_gradient_micro_batch_size_per_gpu", 1)),
        )
        self.full_gradient_storage_dtype = str(_cfg_get(audit_config, "full_gradient_storage_dtype", "float32"))
        policy_loss = _cfg_get(_cfg_get(_cfg_get(config, "actor_rollout_ref", {}), "actor", {}), "policy_loss", {})
        self.lambda_vals = float(_cfg_get(policy_loss, "lambda_vals", 1.0))
        self._last_validation_metrics: dict[str, float] = {}
        self._validation_anchor_step: int | None = None
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
        return self.enabled and self.full_gradient_enabled and step % self.full_gradient_freq_steps == 0

    def full_gradient_meta(self, mode: str, step: int) -> dict[str, Any]:
        max_samples = (
            self.full_gradient_validation_max_samples_per_domain
            if mode == "validation_anchor"
            else self.full_gradient_train_max_samples_per_domain
        )
        return {
            "mopd_full_gradient": {
                "enabled": self.should_compute_full_gradient(step),
                "mode": mode,
                "step": step,
                "domains": self.domains,
                "max_samples_per_domain": max_samples,
                "micro_batch_size_per_gpu": self.full_gradient_micro_batch_size_per_gpu,
                "storage_dtype": self.full_gradient_storage_dtype,
                "learning_rate": self._current_learning_rate_value(),
                "validation_anchor_refresh_steps": self.validation_anchor_refresh_steps,
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
        return finite_float(policy_lr) or 0.0

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

    def should_update_validation_anchor(self, step: int) -> bool:
        return _should_update_validation_anchor(self, step)

    def log_validation_anchor_batch(self, batch: Any, step: int) -> dict[str, float]:
        return _log_validation_anchor_batch(self, batch, step)

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
        ref_log_prob = (tensor_batch["ref_log_prob"] if "ref_log_prob" in batch_keys else old_log_probs).detach().float()
        base_log_prob = (tensor_batch["base_log_prob"] if "base_log_prob" in batch_keys else old_log_probs).detach().float()
        base_ref_log_prob = (
            tensor_batch["base_ref_log_prob"] if "base_ref_log_prob" in batch_keys else ref_log_prob
        ).detach().float()
        batch_size = int(old_log_probs.shape[0])

        labels = extract_teacher_domains(non_tensor, batch_size)
        sample_ids = extract_sample_ids(non_tensor, batch_size, step)

        teacher_log_probs = torch.zeros_like(old_log_probs)
        reverse_kl = torch.zeros_like(old_log_probs)
        for idx, label in enumerate(labels):
            teacher_log_prob = base_ref_log_prob[idx] if label == "code" else ref_log_prob[idx]
            teacher_log_probs[idx] = teacher_log_prob
            if self.lambda_vals == 1.0:
                reverse_kl[idx] = old_log_probs[idx] - teacher_log_prob
            else:
                reverse_kl[idx] = (
                    old_log_probs[idx]
                    - base_log_prob[idx]
                    - (teacher_log_prob - base_log_prob[idx]) * self.lambda_vals
                )

        sample_loss_mean = _mask_mean(reverse_kl, response_mask)
        sample_loss_sq_mean = _mask_mean(reverse_kl.square(), response_mask)
        sample_loss_var = torch.clamp(sample_loss_sq_mean - sample_loss_mean.square(), min=0.0)
        sample_loss_std = torch.sqrt(sample_loss_var)
        sample_loss_cv = sample_loss_std / (sample_loss_mean.abs() + 1e-8)
        effective_tokens = response_mask.sum(dim=-1).detach().cpu().tolist()
        teacher_student_gap = _mask_mean(teacher_log_probs - old_log_probs, response_mask)
        teacher_logprob_mean = _mask_mean(teacher_log_probs, response_mask)
        advantages = tensor_batch["advantages"].detach().float() if "advantages" in batch_keys else -reverse_kl
        sample_advantage_mean = _mask_mean(advantages, response_mask)

        token_scores = tensor_batch["token_level_scores"].detach().float() if "token_level_scores" in batch_keys else None
        sample_correctness = None
        if token_scores is not None:
            sample_correctness = (token_scores * response_mask).sum(dim=-1).gt(0).detach().float()

        configured_domains = list(dict.fromkeys(self.domains + sorted(set(labels))))
        total_tokens = float(response_mask.sum().item())
        total_samples = float(batch_size)
        metrics: dict[str, float] = {}
        domain_rows: list[dict[str, Any]] = []
        variance_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []

        loss_means = _tensor_to_float_list(sample_loss_mean)
        sample_loss_vars = _tensor_to_float_list(sample_loss_var)
        loss_cvs = _tensor_to_float_list(sample_loss_cv)
        token_counts = [float(x) for x in effective_tokens]
        gap_means = _tensor_to_float_list(teacher_student_gap)
        teacher_logprob_means = _tensor_to_float_list(teacher_logprob_mean)
        advantage_means = _tensor_to_float_list(sample_advantage_mean)
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
            domain_losses = [loss_means[idx] for idx in indices]
            domain_loss_vars = [sample_loss_vars[idx] for idx in indices]
            domain_cvs = [loss_cvs[idx] for idx in indices]
            domain_gaps = [gap_means[idx] for idx in indices]
            domain_teacher_logprobs = [teacher_logprob_means[idx] for idx in indices]
            domain_advantages = [advantage_means[idx] for idx in indices]
            domain_sample_ids = [sample_ids[idx] for idx in indices]

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
                "domain_sample_count": domain_sample_count,
                "domain_token_count": domain_token_count,
                "domain_token_frac": domain_token_count / total_tokens if total_tokens else 0.0,
                "opd_loss_mean": _mean(domain_losses),
                "opd_loss_variance": _var(domain_losses),
                "sample_loss_variance_mean": _mean(domain_loss_vars),
                "high_variance_sample_rate": None
                if not domain_cvs
                else float(np.mean([cv > self.high_variance_cv_threshold for cv in domain_cvs])),
                "advantage_mean": _mean(domain_advantages),
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
                    "metric_scope": "domain_step",
                    "loss_name": "opd_loss_token",
                    "domain_sample_count": domain_sample_count,
                    "domain_token_count": domain_token_count,
                    "opd_loss_mean": row["opd_loss_mean"],
                    "opd_loss_variance": row["opd_loss_variance"],
                    "sample_loss_variance_mean": row["sample_loss_variance_mean"],
                    "high_variance_sample_rate": row["high_variance_sample_rate"],
                }
            )

            domain_metric_keys = {
                "domain_sample_count",
                "domain_token_count",
                "domain_token_frac",
                "opd_loss_mean",
                "opd_loss_variance",
                "sample_loss_variance_mean",
                "high_variance_sample_rate",
                "advantage_mean",
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
                            "metric_scope": "sample_token",
                            "loss_name": "opd_loss_token",
                            "effective_tokens": token_counts[idx],
                            "sample_loss_mean": loss_means[idx],
                            "sample_loss_variance": float(sample_loss_var[idx].detach().cpu().item()),
                        }
                    )

        if total_tokens:
            mix = [row["domain_token_frac"] for row in domain_rows if row["domain_token_frac"]]
            entropy = -sum(frac * math.log(frac) for frac in mix)
            metrics[self._global_tag("data", "domain_mix_entropy")] = entropy
            metrics[self._global_tag("data", "total_tokens")] = total_tokens
            metrics[self._global_tag("data", "total_samples")] = total_samples

        return metrics, domain_rows, variance_rows, sample_rows
