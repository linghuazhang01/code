"""G-OPD-aligned LiveCodeBench generation and official scoring helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

LCB_RELEASE_COUNTS = {"v5": 167, "v6": 175}
LCB_SOURCE_SHA256 = {
    "v5": "34dc80fac0fb8c3919835079dafa7831fc10056705d9b0d242003ad3ad1e0f5c",
    "v6": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5",
}
LCB_MODEL_STYLE = "Qwen3-4B-NonThinking"
LCB_FORMATTER_TOKENIZER = "Qwen/Qwen3-4B"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one source artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def lcb_root(gopd_dir: Path) -> Path:
    """Return the vendored LiveCodeBench root in a G-OPD checkout."""

    return gopd_dir / "code_eval/coding/LiveCodeBench"


def lcb_source_path(gopd_dir: Path, release: str) -> Path:
    """Return the exact incremental source used by the G-OPD loader."""

    if release not in LCB_RELEASE_COUNTS:
        raise ValueError(f"Unsupported LiveCodeBench release: {release}")
    return lcb_root(gopd_dir) / "code_generation_lite" / f"test{release[1:]}.jsonl"


def validate_lcb_source(gopd_dir: Path, release: str) -> dict[str, Any]:
    """Fail closed unless an incremental LCB source matches the pinned artifact."""

    source = lcb_source_path(gopd_dir, release)
    if not source.is_file():
        raise FileNotFoundError(f"Missing LiveCodeBench {release} source: {source}")
    with source.open(encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    expected_rows = LCB_RELEASE_COUNTS[release]
    if row_count != expected_rows:
        raise ValueError(
            f"LiveCodeBench {release} has {row_count} rows; expected {expected_rows}."
        )
    source_hash = file_sha256(source)
    expected_hash = LCB_SOURCE_SHA256[release]
    if source_hash != expected_hash:
        raise ValueError(
            f"LiveCodeBench {release} SHA-256 differs: {source_hash} != {expected_hash}"
        )
    return {
        "source_file": str(source),
        "source_rows": row_count,
        "source_sha256": source_hash,
    }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_gopd_runtime(gopd_dir: Path) -> SimpleNamespace:
    """Import the unmodified G-OPD formatter, benchmark, extractor, and scorer."""

    root = lcb_root(gopd_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing G-OPD LiveCodeBench checkout: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    with _working_directory(root):
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.lm_styles import LanguageModelStore
        from lcb_runner.runner.scenario_router import (
            build_prompt_benchmark,
            combine_results,
            get_metrics,
        )
        from lcb_runner.utils.scenarios import Scenario

    return SimpleNamespace(
        extract_instance_results=extract_instance_results,
        model_store=LanguageModelStore,
        build_prompt_benchmark=build_prompt_benchmark,
        combine_results=combine_results,
        get_metrics=get_metrics,
        scenario=Scenario,
    )


def _gopd_args(release: str, *, num_process_evaluate: int = 64, timeout: int = 6) -> Any:
    return SimpleNamespace(
        scenario=None,
        release_version=release,
        not_fast=False,
        start_date=None,
        end_date=None,
        cot_code_execution=False,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )


@lru_cache(maxsize=8)
def load_lcb_benchmark(gopd_dir: Path, release: str) -> tuple[Any, list[Any], Any]:
    """Load the sorted official benchmark and exact Qwen3NonThinking formatter."""

    validate_lcb_source(gopd_dir, release)
    runtime = _load_gopd_runtime(gopd_dir)
    args = _gopd_args(release)
    args.scenario = runtime.scenario.codegeneration
    benchmark, format_prompt = runtime.build_prompt_benchmark(args)
    expected = LCB_RELEASE_COUNTS[release]
    if len(benchmark) != expected:
        raise ValueError(
            f"G-OPD loaded {len(benchmark)} LiveCodeBench {release} rows; expected {expected}."
        )
    return runtime, benchmark, format_prompt


@lru_cache(maxsize=8)
def load_lcb_prompts(gopd_dir: Path, release: str) -> tuple[Any, list[Any], list[str]]:
    """Render and cache the exact prompt strings once per worker and release."""

    runtime, benchmark, format_prompt = load_lcb_benchmark(gopd_dir, release)
    model_style = runtime.model_store[LCB_MODEL_STYLE].model_style
    prompts = [format_prompt(problem, model_style) for problem in benchmark]
    return runtime, benchmark, prompts


def lcb_prompt_sha256(gopd_dir: Path, release: str) -> str:
    """Hash the exact rendered prompts consumed by vLLM for one release."""

    _, benchmark, prompts = load_lcb_prompts(gopd_dir, release)
    digest = hashlib.sha256()
    for problem, prompt in zip(benchmark, prompts, strict=True):
        payload = {
            "question_id": str(problem.question_id),
            "prompt": prompt,
        }
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def generate_lcb_task(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    llm: Any,
) -> None:
    """Generate one LCB micro-shard with the persistent one-GPU vLLM engine."""

    from vllm import SamplingParams

    release = str(task["dataset"]).removeprefix("lcb_")
    gopd_dir = Path(str(manifest["lcb"]["gopd_dir"]))
    runtime, benchmark, prompts = load_lcb_prompts(gopd_dir, release)
    start = int(task["source_start"])
    end = int(task["source_end_exclusive"])
    shard = benchmark[start:end]
    shard_prompts = prompts[start:end]
    generation = manifest["generation"]
    outputs = llm.generate(
        shard_prompts,
        SamplingParams(
            n=int(task["num_samples"]),
            max_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            seed=int(task["generation_seed"]),
        ),
    )
    raw_outputs: list[list[str]] = []
    for request_output in outputs:
        candidates = [candidate.text for candidate in request_output.outputs]
        if len(candidates) != int(task["num_samples"]):
            raise RuntimeError(
                f"Task {task['task_id']} expected {task['num_samples']} outputs, "
                f"got {len(candidates)}."
            )
        raw_outputs.append(candidates)
    combined = runtime.combine_results(
        runtime.scenario.codegeneration,
        raw_outputs,
        runtime.model_store[LCB_MODEL_STYLE],
        False,
    )
    records = [
        problem.insert_output(output_list, code_list)
        for problem, (output_list, code_list) in zip(shard, combined, strict=True)
    ]
    output_dir = Path(str(task["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def merge_and_score_lcb_release(
    *,
    manifest: Mapping[str, Any],
    release: str,
    tasks: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge disjoint LCB shards and run the original G-OPD official scorer."""

    gopd_dir = Path(str(manifest["lcb"]["gopd_dir"]))
    runtime, benchmark, _ = load_lcb_benchmark(gopd_dir, release)
    records: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: int(item["sequence"])):
        shard_path = Path(str(task["output_dir"])) / "records.json"
        shard_records = json.loads(shard_path.read_text(encoding="utf-8"))
        expected_rows = int(task["source_end_exclusive"]) - int(task["source_start"])
        if len(shard_records) != expected_rows:
            raise ValueError(
                f"LCB task {task['task_id']} produced {len(shard_records)} rows; "
                f"expected {expected_rows}."
            )
        records.extend(shard_records)
    records.sort(key=lambda item: str(item["question_id"]))
    expected_ids = [str(problem.question_id) for problem in benchmark]
    actual_ids = [str(record["question_id"]) for record in records]
    if actual_ids != expected_ids:
        raise ValueError(f"LiveCodeBench {release} merged question IDs differ from G-OPD.")
    k = int(manifest["generation"]["code_samples"])
    if any(len(record.get("output_list", [])) != k for record in records):
        raise ValueError(f"LiveCodeBench {release} merged output is not K={k}.")
    raw_outputs = [record["output_list"] for record in records]
    recomputed = runtime.combine_results(
        runtime.scenario.codegeneration,
        raw_outputs,
        runtime.model_store[LCB_MODEL_STYLE],
        False,
    )
    for record, (output_list, code_list) in zip(records, recomputed, strict=True):
        if not all(isinstance(value, str) for value in output_list + code_list):
            raise ValueError(f"LiveCodeBench {release} contains a non-string candidate.")
        if len(code_list) != k or record.get("code_list") != code_list:
            raise ValueError(f"LiveCodeBench {release} extracted code differs from G-OPD.")

    output_dir.mkdir(parents=True, exist_ok=False)
    generation_path = output_dir / f"codegeneration_{k}_{manifest['generation']['temperature']}.json"
    generation_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    combined = recomputed
    args = _gopd_args(release)
    args.scenario = runtime.scenario.codegeneration
    metrics = runtime.get_metrics(args.scenario, args, benchmark, combined)
    graded = runtime.extract_instance_results(metrics[1])
    eval_records = [
        problem.insert_output_evaluation(
            output_list,
            code_list,
            graded_list,
            metadata=metadata,
        )
        for problem, (output_list, code_list), graded_list, metadata in zip(
            benchmark,
            combined,
            graded,
            metrics[2],
            strict=True,
        )
    ]
    generation_path.with_name(f"{generation_path.stem}_eval.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    eval_all_path = generation_path.with_name(f"{generation_path.stem}_eval_all.json")
    eval_all_path.write_text(
        json.dumps(eval_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir.joinpath("SUCCESS").touch()
    return {
        "release": release,
        "prompt_count": len(records),
        "rollouts_per_prompt": k,
        "generation_file": str(generation_path),
        "evaluation_file": str(eval_all_path),
        "source": validate_lcb_source(gopd_dir, release),
    }
