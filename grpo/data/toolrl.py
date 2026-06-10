"""Normalize ToolRL/RLLA parquet data for the shared GRPO launcher."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TOOLRL_TEACHER = "tool"
DEFAULT_TOOLRL_DATA_SOURCE = "toolrl_rlla"


def _normalize_extra_info(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"Unsupported extra_info value: {value!r}")


def _sample_id(row: Mapping[str, Any], row_position: int, teacher: str) -> str:
    data_source = str(row.get("data_source", DEFAULT_TOOLRL_DATA_SOURCE)).replace("/", "_")
    row_id = row.get("id", row_position)
    return f"{teacher}:{data_source}:{row_id}"


def _normalize_prompt(value: Any) -> list[dict[str, str]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if content is None:
                continue
            role = str(item.get("role", "user"))
            messages.append({"role": role, "content": str(content)})
        if messages:
            return messages
    if isinstance(value, str) and value.strip():
        return [{"role": "user", "content": value.strip()}]
    raise ValueError("ToolRL row is missing a valid prompt.")


def toolrl_frame_to_verl(
    frame: pd.DataFrame,
    *,
    split: str,
    teacher: str = DEFAULT_TOOLRL_TEACHER,
    data_source: str = DEFAULT_TOOLRL_DATA_SOURCE,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Add shared teacher-routing metadata to a ToolRL dataframe."""

    if max_samples is not None and max_samples >= 0:
        frame = frame.head(max_samples)

    result = frame.copy(deep=True)
    result["prompt"] = [_normalize_prompt(value) for value in result["prompt"]]
    if "data_source" not in result.columns:
        result["data_source"] = data_source
    else:
        result["data_source"] = result["data_source"].fillna(data_source)

    extra_infos: list[dict[str, Any]] = []
    for row_position, (_, row) in enumerate(result.iterrows()):
        extra_info = _normalize_extra_info(row.get("extra_info"))
        extra_info.update(
            {
                "split": split,
                "opd_teacher": teacher,
                "domain": teacher,
                "source_domain": teacher,
                "validation_dataset": data_source,
            }
        )
        extra_info.setdefault("sample_id", _sample_id(row, row_position, teacher))
        extra_infos.append(extra_info)

    result["extra_info"] = extra_infos
    return result


def toolrl_to_verl_parquet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    teacher: str = DEFAULT_TOOLRL_TEACHER,
    data_source: str = DEFAULT_TOOLRL_DATA_SOURCE,
    max_samples: int | None = None,
) -> int:
    """Write ToolRL data with shared metadata to a verl-compatible parquet."""

    frame = pd.read_parquet(Path(input_path))
    output_frame = toolrl_frame_to_verl(
        frame,
        split=split,
        teacher=teacher,
        data_source=data_source,
        max_samples=max_samples,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_parquet(output, index=False)
    return len(output_frame)
