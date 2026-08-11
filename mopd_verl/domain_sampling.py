"""Domain-aware weighted training sampler helpers for MOPD verl runs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from mopd_verl.domain_sampler_state import DomainSamplerStateMixin

logger = logging.getLogger(__name__)

DOMAIN_LABEL_KEYS = ("domain", "opd_teacher", "source_domain", "ability")


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def normalize_domain_sampling_weights(raw_weights: Any) -> dict[str, float]:
    if raw_weights is None:
        return {}
    if not hasattr(raw_weights, "items"):
        raise ValueError(
            "data.domain_sampling_weights must be a mapping from domain to positive weight."
        )

    weights: dict[str, float] = {}
    for domain, value in raw_weights.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(
                f"Domain sampling weight for {domain!r} must be positive and finite."
            )
        weights[str(domain)] = numeric
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {domain: weight / total for domain, weight in weights.items()}


def normalize_domain_train_files(raw_files: Any) -> dict[str, list[str]]:
    if raw_files is None:
        return {}
    if not hasattr(raw_files, "items"):
        raise ValueError(
            "data.domain_train_files must be a mapping from domain to file path list."
        )

    output: dict[str, list[str]] = {}
    for domain, value in raw_files.items():
        files = [value] if isinstance(value, str) else list(value)
        if not files or not all(isinstance(item, str) and item for item in files):
            raise ValueError(
                f"data.domain_train_files.{domain} must contain at least one file path."
            )
        output[str(domain)] = [str(item) for item in files]
    return output


def _normalize_path_key(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _domain_file_lookup(raw_files: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for domain, files in normalize_domain_train_files(raw_files).items():
        for file_path in files:
            lookup[str(file_path)] = domain
            lookup[_normalize_path_key(file_path)] = domain
    return lookup


def domain_for_data_file(config: Any, file_path: str) -> str | None:
    raw_files = _cfg_get(config, "domain_train_files", None)
    lookup = _domain_file_lookup(raw_files)
    if not lookup:
        return None
    return lookup.get(str(file_path)) or lookup.get(_normalize_path_key(file_path))


def _domain_row_sample_ids(
    domain: str, source_file: str | os.PathLike[str] | None, row_count: int
) -> list[str]:
    source_name = "unknown" if source_file is None else os.fspath(source_file)
    source_fingerprint = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
    return [
        f"{domain}:{source_fingerprint}:{row_index}" for row_index in range(row_count)
    ]


def annotate_hf_dataset_domain(
    dataframe: Any,
    domain: str,
    source_file: str | os.PathLike[str] | None = None,
) -> Any:
    row_count = len(dataframe)
    for column in ("domain", "opd_teacher", "source_domain"):
        values = [domain] * row_count
        if column in getattr(dataframe, "column_names", []):
            dataframe = dataframe.remove_columns([column])
        dataframe = dataframe.add_column(column, values)
    if "sample_id" in getattr(dataframe, "column_names", []):
        dataframe = dataframe.remove_columns(["sample_id"])
    dataframe = dataframe.add_column(
        "sample_id",
        _domain_row_sample_ids(domain, source_file, row_count),
    )
    return dataframe


def domain_label_from_row(row: Mapping[str, Any]) -> str:
    for key in DOMAIN_LABEL_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)

    extra_info = row.get("extra_info")
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            extra_info = None
    if isinstance(extra_info, Mapping):
        for key in DOMAIN_LABEL_KEYS:
            value = extra_info.get(key)
            if value is not None:
                return str(value)
    return "unknown"


def allocate_domain_batch_counts(
    batch_size: int,
    raw_weights: Any,
    domains: Sequence[str] | None = None,
    min_samples_per_domain: int = 0,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    target_weights = normalize_domain_sampling_weights(raw_weights)
    if not target_weights:
        return {}

    ordered_domains = [str(domain) for domain in (domains or target_weights.keys())]
    if not ordered_domains or len(set(ordered_domains)) != len(ordered_domains):
        raise ValueError("Configured domains must be non-empty and unique.")
    if set(ordered_domains) != set(target_weights):
        raise ValueError(
            "Configured domains must exactly match data.domain_sampling_weights."
        )
    minimum = int(min_samples_per_domain)
    if minimum < 0:
        raise ValueError("min_samples_per_domain must be non-negative.")
    required_batch_size = max(len(ordered_domains), len(ordered_domains) * minimum)
    if batch_size < required_batch_size:
        raise ValueError(
            f"Batch size {batch_size} cannot provide {minimum} samples to "
            f"each of {len(ordered_domains)} domains."
        )

    allocatable = batch_size - len(ordered_domains) * minimum
    residual_weights = {
        domain: max(batch_size * target_weights[domain] - minimum, 0.0)
        for domain in ordered_domains
    }
    residual_total = sum(residual_weights.values())
    if residual_total <= 0.0:
        residual_weights = {
            domain: target_weights[domain] for domain in ordered_domains
        }
        residual_total = 1.0
    exact_extras = {
        domain: allocatable * residual_weights[domain] / residual_total
        for domain in ordered_domains
    }
    counts = {
        domain: minimum + int(math.floor(exact_extras[domain]))
        for domain in ordered_domains
    }
    remainder = batch_size - sum(counts.values())
    ranked_domains = sorted(
        ordered_domains,
        key=lambda domain: (
            exact_extras[domain] - math.floor(exact_extras[domain]),
            target_weights[domain],
            domain,
        ),
        reverse=True,
    )
    for domain in ranked_domains[:remainder]:
        counts[domain] += 1
    return counts


class DomainBatchSampler(DomainSamplerStateMixin):
    """Yield full batches with exact domain counts derived from target weights."""

    def __init__(
        self,
        labels: Sequence[str],
        target_weights: Mapping[str, float],
        batch_size: int,
        *,
        replacement: bool = True,
        seed: int | None = None,
    ) -> None:
        self.labels = [str(label) for label in labels]
        self.target_weights = normalize_domain_sampling_weights(target_weights)
        self.batch_size = int(batch_size)
        self.replacement = bool(replacement)
        self.seed = None if seed is None else int(seed)
        self.batch_counts = allocate_domain_batch_counts(
            self.batch_size,
            self.target_weights,
            domains=list(self.target_weights.keys()),
        )
        self.allocation_version = 0
        self.batches_yielded = 0
        self.indices_by_domain: dict[str, list[int]] = {
            domain: [idx for idx, label in enumerate(self.labels) if label == domain]
            for domain in self.target_weights
        }

        missing_domains = [
            domain
            for domain, quota in self.batch_counts.items()
            if quota > 0 and not self.indices_by_domain[domain]
        ]
        if missing_domains:
            raise ValueError(
                f"No training samples found for configured domains: {missing_domains}"
            )

        if self.replacement:
            self.length = len(self.labels) // self.batch_size
        else:
            lengths = [
                len(self.indices_by_domain[domain]) // quota
                for domain, quota in self.batch_counts.items()
                if quota > 0
            ]
            self.length = min(lengths) if lengths else 0

        if self.length <= 0:
            raise ValueError("DomainBatchSampler would produce zero batches.")

        import torch

        self._generator = torch.Generator()
        self._generator.manual_seed(0 if self.seed is None else self.seed)
        labels_hash = hashlib.sha256()
        for label in self.labels:
            labels_hash.update(label.encode("utf-8"))
            labels_hash.update(b"\0")
        self._labels_fingerprint = labels_hash.hexdigest()

    def update_target_weights(
        self,
        target_weights: Mapping[str, float],
        *,
        min_samples_per_domain: int = 1,
    ) -> dict[str, float]:
        """Atomically replace the integer allocation used by future batches."""

        if not self.replacement:
            raise ValueError(
                "Runtime domain allocation updates require replacement sampling."
            )
        normalized = normalize_domain_sampling_weights(target_weights)
        if set(normalized) != set(self.target_weights):
            raise ValueError(
                "Updated domain weights must preserve the configured domain set."
            )
        counts = allocate_domain_batch_counts(
            self.batch_size,
            normalized,
            domains=list(self.target_weights),
            min_samples_per_domain=min_samples_per_domain,
        )
        weights_unchanged = all(
            math.isclose(
                normalized[domain],
                self.target_weights[domain],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for domain in normalized
        )
        if weights_unchanged and counts == self.batch_counts:
            return {domain: count / self.batch_size for domain, count in counts.items()}
        self.target_weights = normalized
        self.batch_counts = counts
        self.allocation_version += 1
        logger.info(
            "Updated MOPD domain allocation version=%d batch_counts=%s",
            self.allocation_version,
            counts,
        )
        return {domain: count / self.batch_size for domain, count in counts.items()}

    def __len__(self) -> int:
        return self.length

    def _normalize_checkpoint_weights(self, raw_weights: Any) -> dict[str, float]:
        return normalize_domain_sampling_weights(raw_weights)

    def __iter__(self) -> Iterator[list[int]]:
        import torch

        generator = self._generator

        if self.replacement:
            for _ in range(self.length):
                batch: list[int] = []
                batch_counts = dict(self.batch_counts)
                for domain, quota in batch_counts.items():
                    if quota <= 0:
                        continue
                    pool = self.indices_by_domain[domain]
                    sampled = torch.randint(
                        len(pool), (quota,), generator=generator
                    ).tolist()
                    batch.extend(pool[idx] for idx in sampled)
                self.batches_yielded += 1
                yield self._shuffle_batch(batch, generator)
            return

        domain_orders: dict[str, list[int]] = {}
        for domain, pool in self.indices_by_domain.items():
            order = torch.randperm(len(pool), generator=generator).tolist()
            domain_orders[domain] = [pool[idx] for idx in order]
        for batch_idx in range(self.length):
            batch = []
            batch_counts = dict(self.batch_counts)
            for domain, quota in batch_counts.items():
                if quota <= 0:
                    continue
                start = batch_idx * quota
                end = start + quota
                batch.extend(domain_orders[domain][start:end])
            self.batches_yielded += 1
            yield self._shuffle_batch(batch, generator)

    @staticmethod
    def _shuffle_batch(batch: list[int], generator: Any) -> list[int]:
        import torch

        order = torch.randperm(len(batch), generator=generator).tolist()
        return [batch[idx] for idx in order]


def _dataset_rows(dataset: Any) -> Iterable[Mapping[str, Any]]:
    dataframe = getattr(dataset, "dataframe", dataset)
    return (dataframe[idx] for idx in range(len(dataframe)))


def create_domain_batch_sampler(
    data_config: Any, dataset: Any, batch_size: int
) -> DomainBatchSampler | None:
    raw_files = _cfg_get(data_config, "domain_train_files", None)
    domain_files = normalize_domain_train_files(raw_files)
    if not domain_files:
        return None

    target_weights = normalize_domain_sampling_weights(
        _cfg_get(data_config, "domain_sampling_weights", None)
    )
    if not target_weights:
        target_weights = {domain: 1.0 / len(domain_files) for domain in domain_files}

    missing_weight_domains = [
        domain for domain in domain_files if domain not in target_weights
    ]
    if missing_weight_domains:
        raise ValueError(
            f"Domains missing from data.domain_sampling_weights: {missing_weight_domains}"
        )

    labels = [domain_label_from_row(row) for row in _dataset_rows(dataset)]
    replacement = bool(_cfg_get(data_config, "domain_sampling_replacement", True))
    seed = _cfg_get(data_config, "seed", None)
    sampler = DomainBatchSampler(
        labels=labels,
        target_weights={domain: target_weights[domain] for domain in domain_files},
        batch_size=batch_size,
        replacement=replacement,
        seed=None if seed is None else int(seed),
    )
    logger.info(
        "Using MOPD exact domain batch sampler with batch_counts=%s, replacement=%s",
        sampler.batch_counts,
        sampler.replacement,
    )
    return sampler
