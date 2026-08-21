"""Download pinned WildSci data and prepare an OPD-compatible science parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import string
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

DATASET_ID = "JustinTX/WildSci"
DATASET_REVISION = "8890d6697bb48a5deebd7469ccdbcf2559499342"
SOURCE_ARTIFACT_REVISION = "f0117a6e4d5f4a196ab9ed933afd4c8d71970532"
SOURCE_FILENAME = "default/train/0000.parquet"
SOURCE_LOCAL_FILENAME = "wildsci-source.parquet"
SOURCE_SHA256 = "f296bb4cde088c933df26547beb7d86bf2db46c58fad2de489237bb6a8e8b660"
SOURCE_SIZE_BYTES = 73_980_916
UPSTREAM_JSONL_SHA256 = (
    "de09e66d9c74399e22b5d942be55c7b1d3c778d3a0b7878da345bc42978d75b5"
)
SOURCE_LICENSE = "CC BY 4.0"
ALIGNED_VOTING_TYPES = frozenset({"all_aligned", "majority_aligned"})
EXCLUDED_DISCIPLINES = frozenset({"Social Sciences"})
OPTION_LETTERS = tuple(string.ascii_uppercase[:10])
DATA_SOURCE = "wildsci_science"
PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. Think through the problem "
    "carefully and select exactly one option. The last line of your response "
    "must be in the format 'Answer: \\boxed{X}', where X is one of A through J."
)


@dataclass(frozen=True)
class PreparationStats:
    raw_rows: int
    selected_rows: int
    exact_duplicates_removed: int
    excluded_discipline_counts: dict[str, int]
    voting_type_counts: dict[str, int]
    discipline_counts: dict[str, int]
    answer_counts: dict[str, int]


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url() -> str:
    """Return the immutable URL for the pinned official Parquet artifact."""

    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{SOURCE_ARTIFACT_REVISION}/{SOURCE_FILENAME}"
    )


def download_source(output_path: Path, *, force: bool = False) -> str:
    """Download the pinned source atomically and verify its published digest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        if output_path.stat().st_size != SOURCE_SIZE_BYTES:
            raise ValueError(
                f"Existing source size mismatch: {output_path} has "
                f"{output_path.stat().st_size} bytes"
            )
        digest = sha256_path(output_path)
        if digest != SOURCE_SHA256:
            raise ValueError(
                f"Existing source checksum mismatch: {output_path} has {digest}"
            )
        LOGGER.info("Using verified cached source: %s", output_path)
        return digest

    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()
    request = urllib.request.Request(
        source_url(),
        headers={"User-Agent": "OPD-WildSci-preparation/1.0"},
    )
    LOGGER.info("Downloading %s", source_url())
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial_path.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if partial_path.stat().st_size != SOURCE_SIZE_BYTES:
            raise ValueError(
                f"Downloaded source size mismatch: expected {SOURCE_SIZE_BYTES}, "
                f"got {partial_path.stat().st_size}"
            )
        digest = sha256_path(partial_path)
        if digest != SOURCE_SHA256:
            raise ValueError(
                f"Downloaded source checksum mismatch: expected {SOURCE_SHA256}, "
                f"got {digest}"
            )
        os.replace(partial_path, output_path)
        return digest
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def iter_source_parquet(path: Path) -> Iterable[dict[str, Any]]:
    """Yield records from the pinned official Parquet without loading it all."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the source parquet") from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=1_024):
        yield from batch.to_pylist()


def _ordered_options(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    raw_options = record.get("options")
    if not isinstance(raw_options, Mapping):
        raise ValueError("options must be an A-J mapping")
    labels = [str(label).strip().upper() for label in raw_options]
    if set(labels) != set(OPTION_LETTERS) or len(labels) != len(OPTION_LETTERS):
        raise ValueError(f"expected exactly A-J options, got {labels}")
    normalized = {str(label).strip().upper(): value for label, value in raw_options.items()}
    choices = [str(normalized[label]).strip() for label in OPTION_LETTERS]
    if any(not choice for choice in choices):
        raise ValueError("option text must be non-empty")
    return list(OPTION_LETTERS), choices


def _normalize_votes(record: Mapping[str, Any]) -> list[str | None]:
    raw_votes = record.get("voting_answers")
    if not isinstance(raw_votes, Sequence) or isinstance(raw_votes, (str, bytes)):
        raise ValueError("voting_answers must be a sequence")
    votes = [None if vote is None else str(vote).strip().upper() for vote in raw_votes]
    if len(votes) != 8:
        raise ValueError(f"expected 8 voting answers, got {len(votes)}")
    return votes


def _validate_alignment(voting_type: str, answer: str, votes: Sequence[str | None]) -> int:
    correct_votes = sum(vote == answer for vote in votes)
    if voting_type == "all_aligned" and correct_votes != len(votes):
        raise ValueError(
            f"all_aligned row has only {correct_votes}/{len(votes)} matching votes"
        )
    if voting_type == "majority_aligned" and correct_votes <= len(votes) // 2:
        raise ValueError(
            f"majority_aligned row has only {correct_votes}/{len(votes)} matching votes"
        )
    return correct_votes


def _build_prompt(question: str, labels: Sequence[str], choices: Sequence[str]) -> str:
    option_block = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
    )
    return f"{PROMPT_TEMPLATE}\n\n{question}\n\n{option_block}"


def _dedup_key(question: str, choices: Sequence[str]) -> str:
    normalized = {
        "question": " ".join(question.casefold().split()),
        "choices": [" ".join(choice.casefold().split()) for choice in choices],
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_row(record: Mapping[str, Any], source_index: int) -> tuple[dict[str, Any], str]:
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError("question must be non-empty")
    voting_type = str(record.get("voting_type") or "").strip()
    answer = str(record.get("answer") or "").strip().upper()
    labels, choices = _ordered_options(record)
    if answer not in labels:
        raise ValueError(f"answer {answer!r} is not a valid option")
    votes = _normalize_votes(record)
    correct_votes = _validate_alignment(voting_type, answer, votes)
    content_hash = _dedup_key(question, choices)
    paper_id = str(record.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("paper_id must be non-empty")

    metadata = {
        "rm_type": "gpqa",
        "opd_teacher": "science",
        "domain": "science",
        "source_domain": "science",
        "split": "train",
        "sample_id": f"wildsci:{paper_id}:{content_hash[:16]}",
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_row_index": source_index,
        "source_record_hash": content_hash,
        "paper_id": paper_id,
        "discipline": str(record.get("discipline") or "").strip(),
        "nc_domain": str(record.get("nc_domain") or "").strip(),
        "nc_subdomain": str(record.get("nc_subdomain") or "").strip(),
        "voting_type": voting_type,
        "voting_answers": votes,
        "correct_vote_count": correct_votes,
        "choices": choices,
        "valid_letters": labels,
        "correct_letter": answer,
        "license": SOURCE_LICENSE,
    }
    row = {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": _build_prompt(question, labels, choices)}],
        "ability": "science",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": metadata,
    }
    return row, content_hash


def prepare_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], PreparationStats]:
    """Select aligned records, validate them, and convert them to verl rows."""

    rows: list[dict[str, Any]] = []
    raw_rows = 0
    duplicates = 0
    seen: dict[str, str] = {}
    voting_types: Counter[str] = Counter()
    disciplines: Counter[str] = Counter()
    excluded_disciplines: Counter[str] = Counter()
    answers: Counter[str] = Counter()

    for source_index, record in enumerate(records):
        raw_rows += 1
        voting_type = str(record.get("voting_type") or "").strip()
        if voting_type not in ALIGNED_VOTING_TYPES:
            continue
        discipline = str(record.get("discipline") or "").strip()
        if discipline in EXCLUDED_DISCIPLINES:
            excluded_disciplines[discipline] += 1
            continue
        try:
            row, content_hash = _prepare_row(record, source_index)
        except ValueError as exc:
            raise ValueError(f"Invalid aligned WildSci row {source_index}: {exc}") from exc
        answer = str(row["reward_model"]["ground_truth"])
        previous_answer = seen.get(content_hash)
        if previous_answer is not None:
            if previous_answer != answer:
                raise ValueError(
                    f"Conflicting answers for duplicate content at row {source_index}"
                )
            duplicates += 1
            continue
        seen[content_hash] = answer
        rows.append(row)
        metadata = row["extra_info"]
        voting_types[str(metadata["voting_type"])] += 1
        disciplines[str(metadata["discipline"])] += 1
        answers[answer] += 1

    stats = PreparationStats(
        raw_rows=raw_rows,
        selected_rows=len(rows),
        exact_duplicates_removed=duplicates,
        excluded_discipline_counts=dict(sorted(excluded_disciplines.items())),
        voting_type_counts=dict(sorted(voting_types.items())),
        discipline_counts=dict(sorted(disciplines.items())),
        answer_counts=dict(sorted(answers.items())),
    )
    return rows, stats


def write_parquet(rows: Sequence[Mapping[str, Any]], output_path: Path) -> str:
    """Write prepared rows atomically and return the output digest."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required to write parquet") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".part")
    frame = pd.DataFrame(rows)
    frame.to_parquet(partial_path, index=False)
    os.replace(partial_path, output_path)
    return sha256_path(output_path)


def write_manifest(
    manifest_path: Path,
    *,
    source_path: Path,
    source_digest: str,
    prepared_path: Path,
    prepared_digest: str,
    stats: PreparationStats,
) -> None:
    """Write a portable preparation manifest."""

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "artifact_revision": SOURCE_ARTIFACT_REVISION,
            "artifact_filename": SOURCE_FILENAME,
            "local_filename": source_path.name,
            "sha256": source_digest,
            "expected_sha256": SOURCE_SHA256,
            "expected_size_bytes": SOURCE_SIZE_BYTES,
            "license": SOURCE_LICENSE,
            "url": source_url(),
            "upstream_jsonl_sha256": UPSTREAM_JSONL_SHA256,
        },
        "prepared": {
            "filename": prepared_path.name,
            "sha256": prepared_digest,
            "schema": ["data_source", "prompt", "ability", "reward_model", "extra_info"],
            "data_source": DATA_SOURCE,
            "selection": sorted(ALIGNED_VOTING_TYPES),
            "excluded_disciplines": sorted(EXCLUDED_DISCIPLINES),
            "rationale_included": False,
        },
        "stats": asdict(stats),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = manifest_path.with_suffix(manifest_path.suffix + ".part")
    partial_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial_path, manifest_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_target = root / "data" / "G-OPD-Training-Data" / "WildSci"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, default=default_target)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    target_dir = args.target_dir.resolve()
    source_path = target_dir / "raw" / SOURCE_LOCAL_FILENAME
    prepared_path = target_dir / "train.parquet"
    manifest_path = target_dir / "manifest.json"

    source_digest = download_source(source_path, force=args.force_download)
    rows, stats = prepare_records(iter_source_parquet(source_path))
    prepared_digest = write_parquet(rows, prepared_path)
    write_manifest(
        manifest_path,
        source_path=source_path,
        source_digest=source_digest,
        prepared_path=prepared_path,
        prepared_digest=prepared_digest,
        stats=stats,
    )
    LOGGER.info(
        "WildSci ready: raw=%d selected=%d duplicates_removed=%d output=%s",
        stats.raw_rows,
        stats.selected_rows,
        stats.exact_duplicates_removed,
        prepared_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
