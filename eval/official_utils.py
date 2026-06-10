"""Shared helpers for standalone official benchmark wrappers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class OfficialEvalResult:
    dataset: str
    output_dir: Path
    summary: dict[str, Any]


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def limited(items: list[Any], max_samples: int | None) -> list[Any]:
    if max_samples is None or max_samples < 0:
        return items
    return items[:max_samples]


def load_vllm(
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("Official eval wrappers require vllm for local model generation.") from exc

    kwargs: dict[str, Any] = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def sampling_params(max_tokens: int, temperature: float, top_p: float) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("Official eval wrappers require vllm for local model generation.") from exc
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)


def retry_call(fn: Any, *, max_attempts: int = 3, sleep_seconds: float = 20.0) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable retry state")
