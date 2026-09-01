#!/usr/bin/env python3
"""Rescore archived code rollouts with the pinned G-OPD EvalPlus evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASET_CONFIG = {
    "humaneval": {
        "record_name": "HumanEvalPlus",
        "task_prefix": "HumanEval/",
        "source_env": "HUMANEVAL_OVERRIDE_PATH",
    },
    "mbpp": {
        "record_name": "MBPPPlus",
        "task_prefix": "Mbpp/",
        "source_env": "MBPP_OVERRIDE_PATH",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _source_task_id(record: Mapping[str, Any], record_name: str) -> str:
    metadata = record.get("sample_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Record is missing sample_metadata")
    source_id = metadata.get("source_id")
    prefix = f"{record_name}:"
    if not isinstance(source_id, str) or not source_id.startswith(prefix):
        raise ValueError(f"Invalid source_id for {record_name}: {source_id!r}")
    return source_id[len(prefix) :]


def _apply_sanitizer(
    payload: tuple[Callable[[str, str], str], str, str],
) -> str:
    sanitizer, response, entry_point = payload
    return sanitizer(response, entry_point)


def _official_sanitize(response: str, entry_point: str) -> str:
    from evalplus.sanitize import sanitize

    return sanitize(response, entrypoint=entry_point)


def prepare_samples(
    records_path: Path,
    source_path: Path,
    dataset: str,
    output_path: Path,
    sanitizer: Callable[[str, str], str],
    expected_rollouts: int = 8,
    parallel: int = 1,
) -> dict[str, Any]:
    """Convert archived raw responses to the exact EvalPlus solution format."""
    config = DATASET_CONFIG[dataset]
    source = {row["task_id"]: row for row in _read_jsonl(source_path)}
    selected: dict[str, dict[int, str]] = defaultdict(dict)

    for record in _read_jsonl(records_path):
        if record.get("dataset") != config["record_name"]:
            continue
        task_id = _source_task_id(record, config["record_name"])
        if task_id not in source:
            raise ValueError(f"Task {task_id!r} is absent from {source_path}")
        rollout_index = record.get("rollout_index")
        response = record.get("response")
        if not isinstance(rollout_index, int) or not isinstance(response, str):
            raise ValueError(f"Invalid rollout payload for {task_id}")
        if rollout_index in selected[task_id]:
            raise ValueError(f"Duplicate rollout {rollout_index} for {task_id}")
        selected[task_id][rollout_index] = response

    if set(selected) != set(source):
        missing = sorted(set(source) - set(selected))
        extra = sorted(set(selected) - set(source))
        raise ValueError(f"Task coverage mismatch: missing={missing}, extra={extra}")

    expected_indices = set(range(expected_rollouts))
    if parallel < 1:
        raise ValueError(f"parallel must be positive, got {parallel}")

    ordered_samples: list[tuple[str, str, str]] = []
    for task_id in source:
        rollout_map = selected[task_id]
        if set(rollout_map) != expected_indices:
            raise ValueError(
                f"Rollout mismatch for {task_id}: "
                f"expected={sorted(expected_indices)}, actual={sorted(rollout_map)}"
            )
        entry_point = source[task_id]["entry_point"]
        ordered_samples.extend(
            (task_id, rollout_map[rollout_index], entry_point)
            for rollout_index in range(expected_rollouts)
        )

    sanitizer_inputs = (
        (sanitizer, response, entry_point)
        for _, response, entry_point in ordered_samples
    )
    if parallel == 1:
        sanitized = list(map(_apply_sanitizer, sanitizer_inputs))
    else:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            sanitized = list(
                executor.map(_apply_sanitizer, sanitizer_inputs, chunksize=1)
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for (task_id, _, _), solution in zip(ordered_samples, sanitized, strict=True):
            handle.write(
                json.dumps(
                    {"task_id": task_id, "solution": solution},
                    ensure_ascii=False,
                )
                + "\n"
            )
            sample_count += 1

    return {
        "dataset": dataset,
        "record_name": config["record_name"],
        "records_path": str(records_path.resolve()),
        "records_sha256": _sha256(records_path),
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "task_count": len(source),
        "rollouts_per_task": expected_rollouts,
        "sanitize_processes": parallel,
        "sample_count": sample_count,
        "samples_path": str(output_path.resolve()),
        "samples_sha256": _sha256(output_path),
    }


def summarize_results(
    results_path: Path,
    expected_rollouts: int = 8,
) -> dict[str, Any]:
    """Compute completion accuracy and observed Pass@K for base/plus tests."""
    with results_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    evaluations = payload.get("eval")
    if not isinstance(evaluations, Mapping) or not evaluations:
        raise ValueError(f"No EvalPlus evaluations found in {results_path}")

    per_task_counts = {task_id: len(rows) for task_id, rows in evaluations.items()}
    rollout_counts = set(per_task_counts.values())
    if len(rollout_counts) != 1:
        raise ValueError(f"Inconsistent rollout counts: {per_task_counts}")
    rollouts_per_task = rollout_counts.pop()
    if rollouts_per_task != expected_rollouts:
        raise ValueError(
            f"Expected K={expected_rollouts}, found K={rollouts_per_task}"
        )
    total = sum(per_task_counts.values())
    summary: dict[str, Any] = {
        "results_path": str(results_path.resolve()),
        "results_sha256": _sha256(results_path),
        "task_count": len(evaluations),
        "rollouts_per_task": rollouts_per_task,
        "sample_count": total,
        "dataset_hash_md5": payload.get("hash"),
        "metrics": {},
    }

    for suite in ("base", "plus"):
        correct = 0
        passed_tasks = 0
        for rows in evaluations.values():
            if suite == "base":
                flags = [row.get("base_status") == "pass" for row in rows]
            else:
                flags = [
                    row.get("base_status") == row.get("plus_status") == "pass"
                    for row in rows
                ]
            correct += sum(flags)
            passed_tasks += int(any(flags))
        summary["metrics"][suite] = {
            "correct_samples": correct,
            "accuracy": correct / total,
            "passed_tasks": passed_tasks,
            "observed_pass_at_k": passed_tasks / len(evaluations),
            "k": rollouts_per_task,
        }
    return summary


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_command(args: argparse.Namespace) -> None:
    config = DATASET_CONFIG[args.dataset]
    os.environ[config["source_env"]] = str(args.source.resolve())
    payload = prepare_samples(
        records_path=args.records,
        source_path=args.source,
        dataset=args.dataset,
        output_path=args.output,
        sanitizer=_official_sanitize,
        expected_rollouts=args.expected_rollouts,
        parallel=args.parallel,
    )
    _write_json(args.manifest, payload)


def _summarize_command(args: argparse.Namespace) -> None:
    payload = summarize_results(args.results, expected_rollouts=args.expected_rollouts)
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--records", type=Path, required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--expected-rollouts", type=int, default=8)
    prepare.add_argument("--parallel", type=int, default=1)
    prepare.set_defaults(func=_prepare_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--results", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--expected-rollouts", type=int, default=8)
    summarize.set_defaults(func=_summarize_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
