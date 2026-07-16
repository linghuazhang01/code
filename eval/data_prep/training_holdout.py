"""Create deterministic, prompt-grouped holdouts from domain training parquet files."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainSpec:
    """Describe one domain training source and its output directory name."""

    name: str
    source_relative_path: str
    output_name: str


@dataclass(frozen=True)
class HoldoutConfig:
    """Immutable parameters controlling deterministic holdout creation."""

    eval_size: int = 1_000
    seed: int = 42
    batch_size: int = 1_024
    write_remainder: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class HoldoutResult:
    """Serializable audit result for one domain split."""

    domain: str
    source_path: str
    source_sha256: str
    source_rows: int
    unique_prompt_groups: int
    duplicate_rows: int
    requested_eval_rows: int
    eval_rows: int
    remainder_rows: int
    eval_path: str
    remainder_path: str | None
    eval_sha256: str
    remainder_sha256: str | None
    selected_group_sha256: tuple[str, ...]


DEFAULT_DOMAIN_SPECS = (
    DomainSpec("math", "DeepMath-103K/train_filtered_level6.parquet", "math"),
    DomainSpec("code", "Eurus/code_train.parquet", "code"),
    DomainSpec("if", "IF/train.parquet", "if"),
    DomainSpec("science", "Science/train.parquet", "science"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _normalize_prompt(prompt: Any) -> Any:
    if isinstance(prompt, list):
        normalized_messages: list[Any] = []
        for message in prompt:
            if isinstance(message, Mapping):
                normalized_messages.append(
                    {
                        "role": str(message.get("role", "")).strip().lower(),
                        "content": " ".join(
                            str(message.get("content", "")).split()
                        ).casefold(),
                    }
                )
            else:
                normalized_messages.append(" ".join(str(message).split()).casefold())
        return normalized_messages
    return " ".join(str(prompt).split()).casefold()


def _prompt_group_sha256(row: Mapping[str, Any]) -> str:
    """Hash the problem identity while keeping exact duplicate prompts together."""

    prompt = row.get("prompt")
    if prompt not in (None, [], ""):
        # Deliberately omit data_source: Code contains the same prompt under both
        # taco and codecontests, sometimes with different test-case bundles.
        identity = {"prompt": _normalize_prompt(prompt)}
    else:
        extra_info = row.get("extra_info")
        identity = {
            "data_source": row.get("data_source"),
            "extra_info": extra_info,
            "reward_model": row.get("reward_model"),
        }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _selection_rank(group_sha256: str, domain: str, seed: int) -> str:
    payload = f"{seed}\0{domain}\0{group_sha256}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_group_hashes(batch: pa.RecordBatch) -> list[str]:
    prompt_index = batch.schema.get_field_index("prompt")
    if prompt_index >= 0:
        return [
            _prompt_group_sha256({"prompt": prompt})
            for prompt in batch.column(prompt_index).to_pylist()
        ]
    return [_prompt_group_sha256(row) for row in batch.to_pylist()]


def _select_groups(
    source: Path,
    domain: str,
    config: HoldoutConfig,
) -> tuple[set[str], int, int]:
    parquet_file = pq.ParquetFile(source)
    group_counts: Counter[str] = Counter()
    for batch in parquet_file.iter_batches(
        batch_size=config.batch_size, columns=["prompt"]
    ):
        group_counts.update(_batch_group_hashes(batch))
    source_rows = sum(group_counts.values())
    requested_rows = min(config.eval_size, source_rows)
    ranked_groups = sorted(
        group_counts,
        key=lambda group: (_selection_rank(group, domain, config.seed), group),
    )

    selected: set[str] = set()
    selected_rows = 0
    for group in ranked_groups:
        if selected_rows >= requested_rows:
            break
        selected.add(group)
        selected_rows += group_counts[group]
    return selected, len(group_counts), source_rows


def _prepare_destination(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    if temporary.exists():
        temporary.unlink()
    return temporary


def _write_partitioned_parquet(
    source: Path,
    selected_groups: set[str],
    eval_path: Path,
    remainder_path: Path | None,
    config: HoldoutConfig,
) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(source)
    eval_temporary = _prepare_destination(eval_path, config.overwrite)
    remainder_temporary = (
        _prepare_destination(remainder_path, config.overwrite)
        if remainder_path is not None
        else None
    )
    eval_rows = 0
    remainder_rows = 0
    eval_writer: pq.ParquetWriter | None = None
    remainder_writer: pq.ParquetWriter | None = None

    try:
        eval_writer = pq.ParquetWriter(
            eval_temporary, parquet_file.schema_arrow, compression="zstd"
        )
        if remainder_temporary is not None:
            remainder_writer = pq.ParquetWriter(
                remainder_temporary,
                parquet_file.schema_arrow,
                compression="zstd",
            )
        for batch in parquet_file.iter_batches(batch_size=config.batch_size):
            group_hashes = _batch_group_hashes(batch)
            eval_indices = [
                index
                for index, group_hash in enumerate(group_hashes)
                if group_hash in selected_groups
            ]
            eval_index_set = set(eval_indices)
            remainder_indices = [
                index for index in range(len(batch)) if index not in eval_index_set
            ]
            if eval_indices:
                eval_batch = batch.take(pa.array(eval_indices, type=pa.int64()))
                eval_writer.write_batch(eval_batch)
                eval_rows += len(eval_indices)
            if remainder_writer is not None and remainder_indices:
                remainder_batch = batch.take(
                    pa.array(remainder_indices, type=pa.int64())
                )
                remainder_writer.write_batch(remainder_batch)
                remainder_rows += len(remainder_indices)
    finally:
        if eval_writer is not None:
            eval_writer.close()
        if remainder_writer is not None:
            remainder_writer.close()

    eval_temporary.replace(eval_path)
    if remainder_temporary is not None and remainder_path is not None:
        remainder_temporary.replace(remainder_path)
    if remainder_path is None:
        remainder_rows = parquet_file.metadata.num_rows - eval_rows
    return eval_rows, remainder_rows


def create_domain_holdout(
    source: str | Path,
    output_root: str | Path,
    spec: DomainSpec,
    config: HoldoutConfig,
    remainder_root: str | Path | None = None,
) -> HoldoutResult:
    """Create one deterministic eval holdout and optionally its training remainder."""

    source_path = Path(source).resolve()
    root = Path(output_root).resolve()
    resolved_remainder_root = (
        Path(remainder_root).resolve() if remainder_root is not None else root
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing {spec.name} training data: {source_path}")
    if config.eval_size <= 0 or config.batch_size <= 0:
        raise ValueError("eval_size and batch_size must be positive integers.")

    source_hash_before = _sha256_file(source_path)
    selected, unique_groups, source_rows = _select_groups(
        source_path, spec.name, config
    )
    eval_path = root / spec.output_name / "test.parquet"
    remainder_path = (
        resolved_remainder_root / spec.output_name / "train.parquet"
        if config.write_remainder
        else None
    )
    eval_rows, remainder_rows = _write_partitioned_parquet(
        source_path,
        selected,
        eval_path,
        remainder_path,
        config,
    )
    source_hash_after = _sha256_file(source_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError(f"Source file changed during split: {source_path}")
    if eval_rows + remainder_rows != source_rows:
        raise RuntimeError(f"Row-count invariant failed for {spec.name}.")

    result = HoldoutResult(
        domain=spec.name,
        source_path=str(source_path),
        source_sha256=source_hash_before,
        source_rows=source_rows,
        unique_prompt_groups=unique_groups,
        duplicate_rows=source_rows - unique_groups,
        requested_eval_rows=min(config.eval_size, source_rows),
        eval_rows=eval_rows,
        remainder_rows=remainder_rows,
        eval_path=str(eval_path),
        remainder_path=str(remainder_path) if remainder_path is not None else None,
        eval_sha256=_sha256_file(eval_path),
        remainder_sha256=_sha256_file(remainder_path)
        if remainder_path is not None
        else None,
        selected_group_sha256=tuple(sorted(selected)),
    )
    LOGGER.info(
        "Created %s holdout: %s eval / %s remainder rows",
        spec.name,
        eval_rows,
        remainder_rows,
    )
    return result


def create_all_holdouts(
    data_root: str | Path,
    output_root: str | Path,
    specs: Sequence[DomainSpec],
    config: HoldoutConfig,
    remainder_root: str | Path | None = None,
) -> list[HoldoutResult]:
    """Create requested domain holdouts and write a root audit manifest."""

    source_root = Path(data_root).resolve()
    target_root = Path(output_root).resolve()
    resolved_remainder_root = (
        Path(remainder_root).resolve() if remainder_root is not None else target_root
    )
    manifest_path = target_root / "manifest.json"
    if manifest_path.exists() and not config.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {manifest_path}")
    for spec in specs:
        source_path = source_root / spec.source_relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing {spec.name} training data: {source_path}")
        destinations = [target_root / spec.output_name / "test.parquet"]
        if config.write_remainder:
            destinations.append(
                resolved_remainder_root / spec.output_name / "train.parquet"
            )
        for destination in destinations:
            if destination.exists() and not config.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {destination}"
                )
    results = [
        create_domain_holdout(
            source_root / spec.source_relative_path,
            target_root,
            spec,
            config,
            remainder_root=resolved_remainder_root,
        )
        for spec in specs
    ]
    payload = {
        "format_version": 1,
        "selection": "sha256-ranked whitespace-normalized casefolded prompt groups",
        "config": asdict(config),
        "domains": [asdict(result) for result in results],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
