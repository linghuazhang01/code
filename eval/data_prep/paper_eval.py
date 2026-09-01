"""Convert G-OPD paper-eval JSONL files into local verl parquet files."""

from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from eval.domains.code.prompting import (
    build_evalplus_prompt,
    build_lcb_qwen3_non_thinking_prompt,
)

PAPER_MATH_EVAL_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
PAPER_MATH_EVAL_SPECS = {
    "aime24": ("AIME2024", "data/aime24/test.jsonl", "math/AIME24/test.parquet"),
    "aime25": ("AIME2025", "data/aime25/test.jsonl", "math/AIME25/test.parquet"),
    "hmmt25_feb": (
        "HMMT25Feb",
        "data/hmmt25_feb/test.jsonl",
        "math/HMMT25Feb/test.parquet",
    ),
    "hmmt25_nov": (
        "HMMT25Nov",
        "data/hmmt25_nov/test.jsonl",
        "math/HMMT25Nov/test.parquet",
    ),
}
PAPER_CODE_EVAL_SPECS = {
    "humaneval_plus": (
        "HumanEvalPlus",
        "code_eval/data/HumanEvalPlus.jsonl",
        "code/HumanEvalPlus/test.parquet",
    ),
    "mbpp_plus": (
        "MBPPPlus",
        "code_eval/data/MbppPlus.jsonl",
        "code/MBPPPlus/test.parquet",
    ),
}
LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}
for _lcb_idx in range(1, 7):
    LCB_RELEASE_FILES[f"v{_lcb_idx}"] = [
        "test.jsonl" if _lcb_idx == 1 else f"test{_lcb_idx}.jsonl"
    ]
LCB_RELEASE_FILES["v5"] = [
    "v5/test-00000-of-00002.parquet",
    "v5/test-00001-of-00002.parquet",
]


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


def math_eval_jsonl_to_verl_parquet(
    input_path: str | Path, output_path: str | Path, data_source: str
) -> int:
    """Convert paper math eval JSONL files into verl validation parquet format."""

    source = Path(input_path)
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(_load_jsonl(source)):
        problem = str(record.get("problem", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not problem or not answer:
            raise ValueError(
                f"{source}:{row_position + 1} must contain non-empty problem and answer fields."
            )
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


def _evalplus_ground_truth(record: Mapping[str, Any], data_source: str) -> str:
    entry_point = str(record["entry_point"])
    if data_source == "HumanEvalPlus":
        assert_case = str(record.get("test", "")).strip()
        if "check(" in assert_case:
            assert_case = f"{assert_case}\ncheck({entry_point})"
    else:
        assert_case = str(record.get("assertion", "")).strip()
    return json.dumps(
        {
            "prompt": str(record.get("prompt", "")),
            "entry_point": entry_point,
            "assert_case": assert_case,
            "dataset": data_source,
        },
        ensure_ascii=False,
    )


def evalplus_jsonl_to_verl_parquet(
    input_path: str | Path, output_path: str | Path, data_source: str
) -> int:
    """Convert HumanEval+/MBPP+ JSONL files into verl validation parquet format."""

    source = Path(input_path)
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(_load_jsonl(source)):
        task_id = str(record.get("task_id", row_position))
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(
                f"{source}:{row_position + 1} must contain a non-empty prompt."
            )
        rows.append(
            {
                "id": f"{data_source}:{task_id}",
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": build_evalplus_prompt(prompt),
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": _evalplus_ground_truth(record, data_source),
                },
                "extra_info": {
                    "index": row_position,
                    "split": "test",
                    "sample_id": f"validation:{data_source}:{task_id}",
                    "opd_teacher": "code",
                    "domain": "code",
                    "source_domain": "code",
                    "validation_dataset": data_source,
                    "prompt_template": "paper_evalplus_qwen_chat",
                    "entry_point": record.get("entry_point"),
                    "task_id": task_id,
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def _json_loads_if_needed(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


class _RestrictedUnpickler(pickle.Unpickler):
    """Decode LCB's pickled JSON string without permitting global imports."""

    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def _decode_lcb_private_tests(value: Any) -> list[dict[str, Any]]:
    try:
        decoded = _json_loads_if_needed(value) or []
    except (json.JSONDecodeError, TypeError):
        compressed = base64.b64decode(str(value).encode("utf-8"))
        payload = zlib.decompress(compressed)
        serialized_json = _RestrictedUnpickler(io.BytesIO(payload)).load()
        decoded = json.loads(serialized_json)
    if not isinstance(decoded, list):
        raise ValueError("LiveCodeBench private_test_cases must decode to a list.")
    return decoded


def _lcb_ground_truth(record: Mapping[str, Any]) -> str:
    metadata = _json_loads_if_needed(record.get("metadata", "{}")) or {}
    public_tests = _json_loads_if_needed(record.get("public_test_cases", "[]")) or []
    private_tests = _decode_lcb_private_tests(record.get("private_test_cases", "[]"))
    all_tests = [*public_tests, *private_tests]
    return json.dumps(
        {
            "inputs": [str(test.get("input", "")) for test in all_tests],
            "outputs": [str(test.get("output", "")) for test in all_tests],
            "fn_name": metadata.get("func_name"),
        },
        ensure_ascii=False,
    )


def _lcb_records_to_verl_parquet(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    data_source: str,
) -> int:
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(records):
        question_id = str(record.get("question_id", row_position))
        question = str(record.get("question_content", ""))
        if not question.strip():
            raise ValueError(
                f"LiveCodeBench row {row_position + 1} must contain question_content."
            )
        prompt_content = build_lcb_qwen3_non_thinking_prompt(question)
        rows.append(
            {
                "id": f"{data_source}:{question_id}",
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": _lcb_ground_truth(record),
                },
                "extra_info": {
                    "index": row_position,
                    "split": "test",
                    "sample_id": f"validation:{data_source}:{question_id}",
                    "opd_teacher": "code",
                    "domain": "code",
                    "source_domain": "code",
                    "validation_dataset": data_source,
                    "prompt_template": "paper_lcb_qwen3_non_thinking",
                    "question_id": question_id,
                    "platform": record.get("platform"),
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def lcb_jsonl_to_verl_parquet(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    data_source: str = "LiveCodeBench-v6",
) -> int:
    """Convert LiveCodeBench JSONL shards into verl validation parquet format."""

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing LiveCodeBench source: {source}")
        records.extend(_load_jsonl(source))
    return _lcb_records_to_verl_parquet(records, output_path, data_source)


def lcb_source_parquet_to_verl_parquet(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    data_source: str,
) -> int:
    """Convert official LiveCodeBench source parquet shards into verl format."""

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing LiveCodeBench source: {source}")
        records.extend(pd.read_parquet(source).to_dict(orient="records"))
    return _lcb_records_to_verl_parquet(records, output_path, data_source)


def lcb_source_parquet_to_jsonl(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
) -> int:
    """Materialize the JSONL compatibility file expected by the G-OPD LCB fork."""

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing LiveCodeBench source: {source}")
        records.extend(pd.read_parquet(source).to_dict(orient="records"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for record in records:
            encoded = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            handle.write(f"{encoded}\n")
    temporary_output.replace(output)
    return len(records)


def prepare_paper_eval_data(
    gopd_dir: str | Path, output_root: str | Path | None = None
) -> dict[str, int]:
    """Prepare paper-eval validation parquets under data/eval_data."""

    root = Path(gopd_dir)
    target_root = (
        Path(output_root) if output_root is not None else root / "data/eval_data"
    )
    counts: dict[str, int] = {}
    for dataset_name, (
        data_source,
        jsonl_relative,
        parquet_relative,
    ) in PAPER_MATH_EVAL_SPECS.items():
        counts[dataset_name] = math_eval_jsonl_to_verl_parquet(
            input_path=root / jsonl_relative,
            output_path=target_root / parquet_relative,
            data_source=data_source,
        )
    for dataset_name, (
        data_source,
        jsonl_relative,
        parquet_relative,
    ) in PAPER_CODE_EVAL_SPECS.items():
        counts[dataset_name] = evalplus_jsonl_to_verl_parquet(
            input_path=root / jsonl_relative,
            output_path=target_root / parquet_relative,
            data_source=data_source,
        )
    lcb_root = root / "code_eval/coding/LiveCodeBench/code_generation_lite"
    counts["lcb_v5"] = lcb_source_parquet_to_verl_parquet(
        input_paths=[lcb_root / name for name in LCB_RELEASE_FILES["v5"]],
        output_path=target_root / "code/LiveCodeBench-v5/test.parquet",
        data_source="LiveCodeBench-v5",
    )
    counts["lcb_v6"] = lcb_jsonl_to_verl_parquet(
        input_paths=[lcb_root / name for name in LCB_RELEASE_FILES["v6"]],
        output_path=target_root / "code/LiveCodeBench/test.parquet",
        data_source="LiveCodeBench-v6",
    )
    return counts
