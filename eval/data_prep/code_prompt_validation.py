"""Validate that prepared Code eval parquet files use the canonical G-OPD prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from eval.domains.code.prompting import (
    EVALPLUS_CODE_INSTRUCTION,
    EVALPLUS_PROMPT_TEMPLATE,
    LCB_QWEN3_PREAMBLE,
    LCB_QWEN3_PROMPT_TEMPLATE,
)

CODE_DATASET_NAMES = frozenset(
    {"HumanEvalPlus", "MBPPPlus", "LiveCodeBench-v5", "LiveCodeBench"}
)
EVALPLUS_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    EVALPLUS_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()
LCB_PROMPT_TEMPLATE_SHA256 = hashlib.sha256(
    LCB_QWEN3_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, Sequence) or isinstance(decoded, str):
            raise ValueError("Prompt JSON must decode to a message sequence.")
        return decoded
    if isinstance(value, Sequence):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, Sequence):
            return converted
    raise ValueError(f"Unsupported prompt container: {type(value).__name__}")


def _user_content(prompt: Any) -> str:
    messages = _as_sequence(prompt)
    if len(messages) != 1 or not isinstance(messages[0], Mapping):
        raise ValueError("Code eval prompt must contain exactly one user message.")
    message = messages[0]
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("Code eval prompt must contain one string-valued user message.")
    return str(message["content"])


def user_content_sha256(contents: Sequence[str]) -> str:
    """Hash ordered user contents with an unambiguous JSON encoding."""

    payload = json.dumps(
        list(contents),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_column_sha256(prompts: Sequence[Any]) -> str:
    """Hash the ordered user-content column of a prepared parquet artifact."""

    return user_content_sha256([_user_content(prompt) for prompt in prompts])


def validate_evalplus_user_content(content: str) -> None:
    """Reject EvalPlus prompts that do not preserve G-OPD's three-newline join."""

    separator = "\n\n\n" + EVALPLUS_CODE_INSTRUCTION
    if not content.endswith(separator):
        raise ValueError("EvalPlus prompt does not match the exact G-OPD user-content suffix.")
    task_prompt = content[: -len(separator)]
    if not task_prompt or task_prompt != task_prompt.strip():
        raise ValueError("EvalPlus prompt does not match the exact G-OPD user-content suffix.")


def validate_lcb_user_content(content: str) -> None:
    """Reject LCB prompts that do not match the G-OPD Qwen3NonThinking template."""

    prefix = f"{LCB_QWEN3_PREAMBLE}Question:\n"
    suffix = "\n\n\n\n" + EVALPLUS_CODE_INSTRUCTION
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise ValueError("LiveCodeBench prompt does not match the G-OPD template.")
    question = content[len(prefix) : -len(suffix)]
    if not question.strip():
        raise ValueError("LiveCodeBench question content is empty.")


def _validate_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Code prompt manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Code prompt manifest mismatch: {mismatches}")


def validate_code_prompt_artifact(data_file: str | Path) -> None:
    """Validate a canonical Code eval parquet; ignore non-Code artifacts."""

    path = Path(data_file)
    dataset_name = path.parent.name
    if dataset_name not in CODE_DATASET_NAMES:
        return
    frame = pd.read_parquet(path, columns=["data_source", "prompt"])
    contents = [_user_content(prompt) for prompt in frame["prompt"]]
    content_sha256 = user_content_sha256(contents)
    if dataset_name.startswith("LiveCodeBench"):
        release_version = "v5" if dataset_name.endswith("-v5") else "v6"
        expected_data_source = f"LiveCodeBench-{release_version}"
        if set(frame["data_source"]) != {expected_data_source}:
            raise ValueError(
                f"LiveCodeBench data_source mismatch: expected {expected_data_source!r}."
            )
        _validate_manifest(
            path,
            {
                "data_source": expected_data_source,
                "chat_template_enable_thinking": False,
                "chat_template_tokenizer": "Qwen/Qwen3-4B",
                "enable_thinking": False,
                "prompt_template": "gopd_qwen3_non_thinking",
                "prompt_template_sha256": LCB_PROMPT_TEMPLATE_SHA256,
                "release_version": release_version,
                "rows": len(contents),
                "user_content_sha256": content_sha256,
            },
        )
        for content in contents:
            validate_lcb_user_content(content)
        return
    _validate_manifest(
        path,
        {
            "data_source": dataset_name,
            "dataset": dataset_name,
            "enable_thinking": False,
            "prompt_template": "gopd_evalplus_qwen_chat",
            "prompt_template_sha256": EVALPLUS_PROMPT_TEMPLATE_SHA256,
            "rows": len(contents),
            "user_content_sha256": content_sha256,
        },
    )
    if set(frame["data_source"]) != {dataset_name}:
        raise ValueError(f"EvalPlus data_source mismatch: expected {dataset_name!r}.")
    for content in contents:
        validate_evalplus_user_content(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_files", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for data_file in args.data_files:
        validate_code_prompt_artifact(data_file)


if __name__ == "__main__":
    main()
