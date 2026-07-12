"""Convert G-OPD paper-eval JSONL files into local verl parquet files."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from eval.domains.code.prompting import build_evalplus_prompt, build_lcb_qwen3_non_thinking_prompt

PAPER_MATH_EVAL_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
PAPER_MATH_EVAL_SPECS = {
    "aime24": ("AIME2024", "data/aime24/test.jsonl", "math/AIME24/test.parquet"),
    "aime25": ("AIME2025", "data/aime25/test.jsonl", "math/AIME25/test.parquet"),
    "hmmt25_feb": ("HMMT25Feb", "data/hmmt25_feb/test.jsonl", "math/HMMT25Feb/test.parquet"),
    "hmmt25_nov": ("HMMT25Nov", "data/hmmt25_nov/test.jsonl", "math/HMMT25Nov/test.parquet"),
}
PAPER_CODE_EVAL_SPECS = {
    "humaneval_plus": ("HumanEvalPlus", "code_eval/data/HumanEvalPlus.jsonl", "code/HumanEvalPlus/test.parquet"),
    "mbpp_plus": ("MBPPPlus", "code_eval/data/MbppPlus.jsonl", "code/MBPPPlus/test.parquet"),
}
LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}
for _lcb_idx in range(1, 7):
    LCB_RELEASE_FILES[f"v{_lcb_idx}"] = ["test.jsonl" if _lcb_idx == 1 else f"test{_lcb_idx}.jsonl"]


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


def evalplus_jsonl_to_verl_parquet(input_path: str | Path, output_path: str | Path, data_source: str) -> int:
    """Convert HumanEval+/MBPP+ JSONL files into verl validation parquet format."""

    source = Path(input_path)
    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for row_position, record in enumerate(_load_jsonl(source)):
        task_id = str(record.get("task_id", row_position))
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{source}:{row_position + 1} must contain a non-empty prompt.")
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
                "reward_model": {"style": "rule", "ground_truth": _evalplus_ground_truth(record, data_source)},
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


def _lcb_ground_truth(record: Mapping[str, Any]) -> str:
    metadata = _json_loads_if_needed(record.get("metadata", "{}")) or {}
    public_tests = _json_loads_if_needed(record.get("public_test_cases", "[]")) or []
    return json.dumps(
        {
            "inputs": [str(test.get("input", "")) for test in public_tests],
            "outputs": [str(test.get("output", "")) for test in public_tests],
            "fn_name": metadata.get("func_name"),
        },
        ensure_ascii=False,
    )


def lcb_jsonl_to_verl_parquet(input_paths: Sequence[str | Path], output_path: str | Path) -> int:
    """Convert LiveCodeBench code-generation-lite JSONL shards into verl validation parquet format."""

    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        source = Path(input_path)
        if not source.exists():
            continue
        for record in _load_jsonl(source):
            row_position = len(rows)
            question_id = str(record.get("question_id", row_position))
            title = str(record.get("question_title", "")).strip()
            question = str(record.get("question_content", "")).strip()
            starter_code = str(record.get("starter_code", "") or "").strip()
            if not question and not title and not starter_code:
                raise ValueError(f"{source}:{row_position + 1} must contain question text or starter code.")
            prompt_content = build_lcb_qwen3_non_thinking_prompt(question or title or starter_code)
            rows.append(
                {
                    "id": f"LiveCodeBench:{question_id}",
                    "data_source": "LiveCodeBench",
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt_content,
                        }
                    ],
                    "ability": "code",
                    "reward_model": {"style": "rule", "ground_truth": _lcb_ground_truth(record)},
                    "extra_info": {
                        "index": row_position,
                        "split": "test",
                        "sample_id": f"validation:LiveCodeBench:{question_id}",
                        "opd_teacher": "code",
                        "domain": "code",
                        "source_domain": "code",
                        "validation_dataset": "LiveCodeBench",
                        "prompt_template": "paper_lcb_qwen3_non_thinking",
                        "question_id": question_id,
                        "platform": record.get("platform"),
                    },
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    return len(rows)


def prepare_paper_eval_data(gopd_dir: str | Path, output_root: str | Path | None = None) -> dict[str, int]:
    """Prepare paper-eval validation parquets under data/eval_data."""

    root = Path(gopd_dir)
    target_root = Path(output_root) if output_root is not None else root / "data/eval_data"
    counts: dict[str, int] = {}
    for dataset_name, (data_source, jsonl_relative, parquet_relative) in PAPER_MATH_EVAL_SPECS.items():
        counts[dataset_name] = math_eval_jsonl_to_verl_parquet(
            input_path=root / jsonl_relative,
            output_path=target_root / parquet_relative,
            data_source=data_source,
        )
    for dataset_name, (data_source, jsonl_relative, parquet_relative) in PAPER_CODE_EVAL_SPECS.items():
        counts[dataset_name] = evalplus_jsonl_to_verl_parquet(
            input_path=root / jsonl_relative,
            output_path=target_root / parquet_relative,
            data_source=data_source,
        )
    lcb_root = root / "code_eval/coding/LiveCodeBench/code_generation_lite"
    lcb_files = [lcb_root / name for name in LCB_RELEASE_FILES["release_v6"]]
    counts["lcb"] = lcb_jsonl_to_verl_parquet(
        input_paths=lcb_files,
        output_path=target_root / "code/LiveCodeBench/test.parquet",
    )
    return counts
