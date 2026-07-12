"""Prepare generic ToolRL-style tool-use eval JSONL into verl-compatible parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("data/eval_data/toolrl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            records.append(record)
    return records


def _first_string(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def toolrl_jsonl_to_verl_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    split: str = "test",
) -> int:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(_read_jsonl(Path(input_path))):
        prompt = _first_string(record, ("question", "instruction", "prompt", "query", "task"))
        if not prompt:
            raise ValueError(f"{input_path}:{index + 1} is missing question/instruction/prompt text.")
        raw_id = record.get("id", record.get("uid", record.get("sample_id", index)))
        ground_truth = record.get("answer", record.get("ground_truth", record.get("expected", "")))
        row: dict[str, Any] = {
            "id": f"{dataset}:{raw_id}",
            "data_source": dataset,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "tool",
            "reward_model": {"style": "external", "ground_truth": ground_truth},
            "extra_info": {
                "index": index,
                "split": split,
                "sample_id": f"toolrl:{dataset}:{raw_id}",
                "domain": "tool",
                "source_domain": "tool",
                "validation_dataset": dataset,
                "requires_external_tool_eval": True,
            },
        }
        if record.get("metadata") is not None:
            row["metadata"] = json.dumps(record["metadata"], ensure_ascii=False)
        rows.append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("BFCL", "API-Bank", "Bamboogle"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUT_DIR / args.dataset / "test.parquet"
    count = toolrl_jsonl_to_verl_parquet(args.input, output, dataset=args.dataset, split=args.split)
    print(json.dumps({"count": count, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
