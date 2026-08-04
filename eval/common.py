"""Shared utilities for Qwen thinking-mode validation experiments."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eval.domains.code import is_code_dataset
from eval.domains.science import is_science_dataset
from eval.domains.search import is_search_dataset
from eval.domains.toolrl import is_toolrl_dataset

DEFAULT_PAPER_EVAL_DATA_FILES = (
    "data/eval_data/math/AIME24/test.parquet",
    "data/eval_data/math/AIME25/test.parquet",
    "data/eval_data/math/HMMT25Feb/test.parquet",
    "data/eval_data/math/HMMT25Nov/test.parquet",
    "data/eval_data/code/HumanEvalPlus/test.parquet",
    "data/eval_data/code/MBPPPlus/test.parquet",
)
DEFAULT_TOOLRL_DATA_FILES = (
    "data/eval_data/toolrl/BFCL/test.parquet",
    "data/eval_data/toolrl/API-Bank/test.parquet",
    "data/eval_data/toolrl/Bamboogle/test.parquet",
)
DEFAULT_SEARCH_DATA_FILES = ("data/SearchQA/test.parquet",)
DEFAULT_IF_DATA_FILES = (
    "data/eval_data/if/IFEval/test.parquet",
    "data/eval_data/if/IFBench/test.parquet",
)
DEFAULT_SCIENCE_DATA_FILES = ("data/eval_data/science/GPQA/test.parquet",)
DEFAULT_DATA_FILES = (
    DEFAULT_PAPER_EVAL_DATA_FILES
    + DEFAULT_IF_DATA_FILES
    + DEFAULT_SCIENCE_DATA_FILES
    + DEFAULT_TOOLRL_DATA_FILES
    + DEFAULT_SEARCH_DATA_FILES
)
THINKING_MODES = ("thinking", "non_thinking")


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dataset: str
    ability: str
    messages: list[dict[str, str]]
    ground_truth: Any
    extra_info: dict[str, Any] = field(default_factory=dict)
    sample_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    mode: str
    enable_thinking: bool
    sample_id: str
    dataset: str
    ability: str
    ground_truth: Any
    prediction: str
    score: float | None
    correct: bool | None
    prompt_tokens: int
    generated_tokens: int
    thinking_tokens: int
    answer_tokens: int
    total_tokens: int
    latency_seconds: float
    generated_tokens_per_second: float
    completion_preview: str
    rollout_index: int = 0
    generation_seed: int | None = None
    max_new_tokens: int | None = None
    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    completion: str | None = None
    reward_metadata: list[dict[str, Any]] | None = None
    sample_metadata: dict[str, Any] | None = None


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _json_safe(value: Any) -> Any:
    """Convert parquet-backed metadata into values accepted by ``json.dumps``."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = _safe_json_loads(value)
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise ValueError(f"Unsupported prompt value: {value!r}")

    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"Prompt item is not a mapping: {item!r}")
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        messages.append({"role": role, "content": content})
    return messages


def _iter_parquet_rows(
    path: Path,
    *,
    start: int,
    limit: int | None,
    batch_size: int = 1_024,
) -> Iterable[tuple[int, Mapping[str, Any]]]:
    """Yield a stable row slice without loading the full parquet into memory."""

    import pyarrow.parquet as pq

    if start < 0:
        raise ValueError("Parquet row start must be non-negative.")
    if limit is not None and limit < 1:
        raise ValueError("Parquet row limit must be positive when provided.")

    emitted = 0
    row_offset = 0
    for record_batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        rows = record_batch.to_pylist()
        batch_end = row_offset + len(rows)
        if batch_end <= start:
            row_offset = batch_end
            continue

        local_start = max(start - row_offset, 0)
        for local_index in range(local_start, len(rows)):
            if limit is not None and emitted >= limit:
                return
            yield row_offset + local_index, rows[local_index]
            emitted += 1
        row_offset = batch_end


def normalize_ability(raw_ability: str, dataset: str) -> str:
    ability = raw_ability.strip().lower()
    if ability in {"math", "code", "search", "reasoning", "tool", "if", "science"}:
        return ability
    if ability in {"searchqa", "search_qa", "qa"} or is_search_dataset(dataset):
        return "search"
    if ability in {"toolrl", "tool_call", "tool-use", "tool_use"} or is_toolrl_dataset(dataset):
        return "tool"
    if is_science_dataset(dataset):
        return "science"
    if is_code_dataset(dataset):
        return "code"
    return ability or "unknown"


def load_eval_samples(
    data_files: Sequence[str | Path],
    max_samples_per_dataset: int | None = None,
    *,
    sample_offset_per_dataset: int = 0,
    skip_missing: bool = False,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for data_file in data_files:
        path = Path(data_file)
        if not path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(path)
        for row_index, row in _iter_parquet_rows(
            path,
            start=sample_offset_per_dataset,
            limit=max_samples_per_dataset,
        ):
            reward_model = normalize_mapping(row.get("reward_model"))
            extra_info = normalize_mapping(row.get("extra_info"))
            dataset = str(extra_info.get("validation_dataset") or row.get("data_source") or path.parent.name)
            sample_id = str(extra_info.get("sample_id") or row.get("id") or f"{dataset}:{row_index}")
            raw_ability = str(extra_info.get("domain") or row.get("ability") or extra_info.get("source_domain") or "unknown")
            ability = normalize_ability(raw_ability, dataset)
            raw_ground_truth = reward_model.get("ground_truth", "")
            sample_metadata = {
                "source_file": str(path),
                "source_row_index": _json_safe(row_index),
                "source_row_position": row_index,
                "source_id": _json_safe(row.get("id")),
                "data_source": _json_safe(row.get("data_source")),
                "source_ability": _json_safe(row.get("ability")),
                "reward_style": _json_safe(reward_model.get("style")),
                "reward_model": _json_safe(reward_model),
                "extra_info": _json_safe(extra_info),
                "additional_fields": _json_safe(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "id",
                            "data_source",
                            "prompt",
                            "ability",
                            "reward_model",
                            "extra_info",
                        }
                    }
                ),
            }
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    dataset=dataset,
                    ability=ability,
                    messages=normalize_messages(row["prompt"]),
                    ground_truth=raw_ground_truth if ability in {"search", "tool"} else str(raw_ground_truth),
                    extra_info=extra_info,
                    sample_metadata=sample_metadata,
                )
            )
    return samples


def remove_think_block(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned


def count_thinking_tokens(raw_completion: str, tokenizer: Any) -> int:
    match = re.search(r"<think>(.*?)</think>", raw_completion, flags=re.DOTALL)
    if not match:
        return 0
    return len(tokenizer.encode(match.group(1), add_special_tokens=False))


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_results(results: Iterable[EvalResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[EvalResult]] = defaultdict(list)
    aggregate_groups: dict[tuple[str, str, str], list[EvalResult]] = defaultdict(list)
    for result in results:
        groups[(result.mode, result.dataset, result.ability)].append(result)
        aggregate_groups[(result.mode, "ALL", result.ability)].append(result)
        aggregate_groups[(result.mode, "ALL", "all")].append(result)

    summaries: list[dict[str, Any]] = []
    for key, items in {**groups, **aggregate_groups}.items():
        mode, dataset, ability = key
        scored = [item for item in items if item.score is not None]
        by_sample: dict[str, list[EvalResult]] = defaultdict(list)
        for item in scored:
            by_sample[item.sample_id].append(item)
        sample_sizes = [len(sample_items) for sample_items in by_sample.values()]
        summaries.append(
            {
                "mode": mode,
                "dataset": dataset,
                "ability": ability,
                "sample_count": len(items),
                "scored_count": len(scored),
                "accuracy": _mean([float(item.correct) for item in scored if item.correct is not None]),
                "avg_score": _mean([float(item.score) for item in scored if item.score is not None]),
                "unique_sample_count": len(by_sample),
                "min_samples_per_prompt": min(sample_sizes, default=0),
                "max_samples_per_prompt": max(sample_sizes, default=0),
                "avg_at_k": _mean(
                    [
                        float(sum(float(item.score) for item in sample_items) / len(sample_items))
                        for sample_items in by_sample.values()
                    ]
                ),
                "observed_pass_at_k": _mean(
                    [float(any(item.correct is True for item in sample_items)) for sample_items in by_sample.values()]
                ),
                "avg_prompt_tokens": _mean([item.prompt_tokens for item in items]),
                "avg_generated_tokens": _mean([item.generated_tokens for item in items]),
                "avg_thinking_tokens": _mean([item.thinking_tokens for item in items]),
                "avg_answer_tokens": _mean([item.answer_tokens for item in items]),
                "avg_total_tokens": _mean([item.total_tokens for item in items]),
                "avg_latency_seconds": _mean([item.latency_seconds for item in items]),
                "avg_generated_tokens_per_second": _mean([item.generated_tokens_per_second for item in items]),
            }
        )
    return sorted(summaries, key=lambda item: (item["mode"], item["dataset"], item["ability"]))


def write_outputs(results: list[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "thinking_eval_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    summaries = summarize_results(results)
    summary_path = output_dir / "thinking_eval_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output_dir / "thinking_eval_summary.csv", index=False)


def append_sample_outputs(results: Sequence[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "thinking_eval_samples.jsonl"
    with samples_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
