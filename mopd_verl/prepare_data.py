"""Prepare verl parquet files with domain-specific MOPD teacher routing."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

VALID_TEACHERS = {"math", "code"}
PAPER_MATH_EVAL_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
PAPER_MATH_EVAL_SPECS = {
    "aime24": ("AIME2024", "data/aime24/test.jsonl", "PaperEval/AIME24/test.parquet"),
    "aime25": ("AIME2025", "data/aime25/test.jsonl", "PaperEval/AIME25/test.parquet"),
    "hmmt25_feb": ("HMMT25Feb", "data/hmmt25_feb/test.jsonl", "PaperEval/HMMT25Feb/test.parquet"),
    "hmmt25_nov": ("HMMT25Nov", "data/hmmt25_nov/test.jsonl", "PaperEval/HMMT25Nov/test.parquet"),
}


@dataclass(frozen=True)
class TeacherValidation:
    counts: dict[str, int]
    invalid_rows: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_rows


@dataclass(frozen=True)
class SampleIdValidation:
    duplicate_count: int
    invalid_rows: list[dict[str, Any]]

    @property
    def is_valid(self) -> bool:
        return not self.invalid_rows and self.duplicate_count == 0


def _normalize_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"Unsupported extra_info value: {value!r}")


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def _stable_sample_id(row: Mapping[str, Any], teacher: str, row_position: int, extra_info: Mapping[str, Any]) -> str:
    data_source = str(row.get("data_source", "unknown")).replace("/", "_")
    index = extra_info.get("index", row_position)
    return f"{teacher}:{data_source}:{index}"


def add_teacher_column(frame: pd.DataFrame, teacher: str) -> pd.DataFrame:
    if teacher not in VALID_TEACHERS:
        raise ValueError(f"teacher must be one of {sorted(VALID_TEACHERS)}, got {teacher!r}")

    result = frame.copy(deep=True)
    if "extra_info" not in result.columns:
        result["extra_info"] = [{} for _ in range(len(result))]

    extra_info = []
    for row_position, (_, row) in enumerate(result.iterrows()):
        value = row.get("extra_info")
        normalized = _normalize_extra_info(value)
        normalized["opd_teacher"] = teacher
        normalized["domain"] = teacher
        normalized["source_domain"] = teacher
        normalized.setdefault("sample_id", _stable_sample_id(row, teacher, row_position, normalized))
        extra_info.append(normalized)

    result["extra_info"] = extra_info
    return result


def merge_teacher_data(math_path: str | Path, code_path: str | Path, output_path: str | Path) -> None:
    math_frame = add_teacher_column(read_parquet(math_path), "math")
    code_frame = add_teacher_column(read_parquet(code_path), "code")
    merged = pd.concat([math_frame, code_frame], ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            records.append(record)
    return records


def math_eval_jsonl_to_verl_parquet(input_path: str | Path, output_path: str | Path, data_source: str) -> int:
    """Convert paper math eval JSONL files into verl validation parquet format."""

    source = Path(input_path)
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(_load_jsonl(source)):
        problem = str(record.get("problem", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not problem or not answer:
            raise ValueError(f"{source}:{row_position + 1} must contain non-empty problem and answer fields.")
        raw_id = record.get("id", row_position)
        rows.append(
            {
                "id": f"{data_source}:{raw_id}",
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{problem}\n{PAPER_MATH_EVAL_PROMPT}",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": row_position,
                    "split": "test",
                    "sample_id": f"validation:{data_source}:{raw_id}",
                    "domain": "math",
                    "source_domain": "math",
                    "validation_dataset": data_source,
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def prepare_paper_eval_data(gopd_dir: str | Path, output_root: str | Path | None = None) -> dict[str, int]:
    """Prepare AIME/HMMT paper-eval validation parquets under a G-OPD checkout."""

    root = Path(gopd_dir)
    target_root = Path(output_root) if output_root is not None else root / "G-OPD-Training-Data"
    counts: dict[str, int] = {}
    for dataset_name, (data_source, jsonl_relative, parquet_relative) in PAPER_MATH_EVAL_SPECS.items():
        counts[dataset_name] = math_eval_jsonl_to_verl_parquet(
            input_path=root / jsonl_relative,
            output_path=target_root / parquet_relative,
            data_source=data_source,
        )
    return counts


def teacher_counts(path: str | Path) -> dict[str, int]:
    return validate_teacher_labels(path).counts


def _iter_extra_info(frame: pd.DataFrame):
    if "extra_info" not in frame.columns:
        for index in frame.index:
            yield int(index), {}
        return
    for index, value in frame["extra_info"].items():
        try:
            yield int(index), _normalize_extra_info(value)
        except ValueError:
            yield int(index), {}


def validate_teacher_labels(path: str | Path) -> TeacherValidation:
    frame = read_parquet(path)
    counts = {teacher: 0 for teacher in sorted(VALID_TEACHERS)}
    invalid_rows: list[dict[str, Any]] = []

    if "extra_info" not in frame.columns:
        return TeacherValidation(
            counts=counts,
            invalid_rows=[{"index": int(index), "teacher": None, "reason": "missing extra_info"} for index in frame.index],
        )

    for index, value in frame["extra_info"].items():
        try:
            teacher = _normalize_extra_info(value).get("opd_teacher")
        except ValueError as exc:
            invalid_rows.append({"index": int(index), "teacher": None, "reason": str(exc)})
            continue
        if teacher in counts:
            counts[teacher] += 1
        else:
            invalid_rows.append({"index": int(index), "teacher": teacher, "reason": "invalid or missing opd_teacher"})

    return TeacherValidation(counts=counts, invalid_rows=invalid_rows)


def validate_sample_ids(path: str | Path) -> SampleIdValidation:
    frame = read_parquet(path)
    invalid_rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    duplicate_count = 0

    for index, extra_info in _iter_extra_info(frame):
        sample_id = extra_info.get("sample_id")
        teacher = extra_info.get("opd_teacher")
        domain = extra_info.get("domain")
        if not sample_id:
            invalid_rows.append({"index": index, "sample_id": sample_id, "reason": "missing sample_id"})
            continue
        sample_id = str(sample_id)
        if sample_id in seen:
            duplicate_count += 1
            invalid_rows.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason": f"duplicate sample_id first seen at row {seen[sample_id]}",
                }
            )
        else:
            seen[sample_id] = index
        if teacher in VALID_TEACHERS and domain != teacher:
            invalid_rows.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason": f"domain {domain!r} does not match opd_teacher {teacher!r}",
                }
            )

    return SampleIdValidation(duplicate_count=duplicate_count, invalid_rows=invalid_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge", help="Merge math/code parquet files and add opd_teacher.")
    merge_parser.add_argument("--math-train", required=True, help="Math-domain training parquet.")
    merge_parser.add_argument("--code-train", required=True, help="Code-domain training parquet.")
    merge_parser.add_argument("--output", required=True, help="Output merged parquet.")

    inspect_parser = subparsers.add_parser("inspect", help="Count opd_teacher labels in a parquet file.")
    inspect_parser.add_argument("path", help="Parquet file to inspect.")

    paper_eval_parser = subparsers.add_parser(
        "prepare-paper-eval",
        help="Convert paper math eval JSONL files into verl validation parquet files.",
    )
    paper_eval_parser.add_argument("--gopd-dir", required=True, help="Path to the G-OPD checkout root.")
    paper_eval_parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for generated parquets. Defaults to <gopd-dir>/G-OPD-Training-Data.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-paper-eval":
        counts = prepare_paper_eval_data(args.gopd_dir, args.output_root)
        sys.stdout.write(json.dumps({"counts": counts}, sort_keys=True) + "\n")
        return 0

    if args.command == "merge":
        merge_teacher_data(args.math_train, args.code_train, args.output)
        validation = validate_teacher_labels(args.output)
    else:
        validation = validate_teacher_labels(args.path)

    sample_validation = validate_sample_ids(args.output if args.command == "merge" else args.path)
    payload = {
        "counts": validation.counts,
        "invalid_rows": validation.invalid_rows[:20],
        "sample_id_duplicate_count": sample_validation.duplicate_count,
        "sample_id_invalid_rows": sample_validation.invalid_rows[:20],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0 if validation.is_valid and sample_validation.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
