"""Atomic task implementations used by persistent data-parallel workers."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from eval.common import (
    EvalResult,
    append_sample_outputs,
    load_eval_samples,
    write_outputs,
)
from eval.domains.scoring import score_completion
from eval.domains.science.official_eval import run_dataset as run_science_dataset
from eval.lcb_official import generate_lcb_task
from eval.runner import generate_vllm_batch

LOGGER = logging.getLogger(__name__)


def _task_config(
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    eval_model_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task_id": task["task_id"],
        "model_path": manifest["model"]["path"],
        "eval_model_path": eval_model_path,
        "dataset": task["dataset"],
        "domain": task["domain"],
        "source_start": task["source_start"],
        "source_end_exclusive": task["source_end_exclusive"],
        "num_samples": task["num_samples"],
        "generation_seed": task["generation_seed"],
        "execution": manifest["execution"],
        "generation": manifest["generation"],
    }


def _temporary_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Evaluation shard output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.worker-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale shard staging directory exists: {temporary}")
    temporary.mkdir()
    return temporary


def run_standard_task(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    eval_model_path: str,
    llm: Any,
    tokenizer: Any,
    source_file: Path,
    resume: bool,
    scoring_executor: ThreadPoolExecutor | None = None,
) -> Callable[[], None] | None:
    """Evaluate one parquet micro-shard with batched K-way sampling."""

    output_dir = Path(str(task["output_dir"]))
    start = int(task["source_start"])
    count = int(task["source_end_exclusive"]) - start
    samples = load_eval_samples(
        [source_file],
        max_samples_per_dataset=count,
        sample_offset_per_dataset=start,
    )
    if len(samples) != count:
        raise ValueError(
            f"Task {task['task_id']} loaded {len(samples)} prompts; expected {count}"
        )
    if resume and output_dir.exists():
        raise ValueError(f"Incomplete schema-v2 shard is not resumable: {output_dir}")

    temporary = _temporary_output_dir(output_dir)
    config = _task_config(task, manifest, eval_model_path=eval_model_path)
    temporary.joinpath("eval_run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results: list[EvalResult] = []
    num_samples = int(task["num_samples"])
    execution = manifest["execution"]
    generation = manifest["generation"]
    batch_size = int(execution["batch_size"])
    deferred = scoring_executor is not None and bool(execution["score_code"]) and task["domain"] == "code"
    generation_start = time.perf_counter()
    try:
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start : batch_start + batch_size]
            LOGGER.info(
                "task=%s dataset=%s batch_rollouts=%d prompts=%d-%d/%d",
                task["task_id"],
                task["dataset"],
                num_samples,
                batch_start + 1,
                batch_start + len(batch),
                len(samples),
            )
            results.extend(
                generate_vllm_batch(
                    llm,
                    tokenizer,
                    batch,
                    mode="non_thinking",
                    max_new_tokens=int(generation["max_new_tokens"]),
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    score_code=bool(execution["score_code"] and task["domain"] == "code" and not deferred),
                    save_completion=True,
                    generation_seed=int(task["generation_seed"]) + batch_start,
                    num_return_sequences=num_samples,
                )
            )
        if len(results) != int(task["expected_records"]):
            raise RuntimeError(
                f"Task {task['task_id']} produced {len(results)} records; "
                f"expected {task['expected_records']}"
            )
        generation_seconds = time.perf_counter() - generation_start
        if deferred:
            # Raw generations survive scoring errors; never publish them as scored output.
            append_sample_outputs(results, temporary / "raw")

            def finalize() -> None:
                scoring_start = time.perf_counter()

                def score_one(index: int) -> EvalResult:
                    result = results[index]
                    if result.completion is None:
                        raise ValueError("Deferred scoring requires a saved completion")
                    score, prediction, metadata = score_completion(
                        samples[index // num_samples], result.completion, score_code=True,
                    )
                    return replace(result, score=score, prediction=prediction,
                                   correct=None if score is None else score == 1.0,
                                   reward_metadata=metadata)

                assert scoring_executor is not None
                scored = list(scoring_executor.map(score_one, range(len(results))))
                scoring_seconds = time.perf_counter() - scoring_start
                write_outputs(scored, temporary)
                temporary.joinpath("stage_timings.json").write_text(
                    json.dumps({"generation_seconds": generation_seconds,
                                "scoring_seconds": scoring_seconds,
                                "records": len(scored)}, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.joinpath("SUCCESS").touch()
                temporary.rename(output_dir)

            return finalize
        write_outputs(results, temporary)
        temporary.joinpath("SUCCESS").touch()
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_mmlupro_task(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    eval_model_path: str,
    llm: Any,
    tokenizer: Any,
) -> None:
    """Evaluate one pinned MMLU-Pro micro-shard with the shared engine."""

    execution = manifest["execution"]
    generation = manifest["generation"]
    start = int(task["source_start"])
    count = int(task["source_end_exclusive"]) - start
    output_dir = Path(str(task["output_dir"]))
    temporary = _temporary_output_dir(output_dir)
    try:
        result = run_science_dataset(
            dataset_key="mmlupro_500_seed42",
            model_path=eval_model_path,
            output_dir=temporary,
            max_samples=count,
            sample_offset=start,
            tensor_parallel_size=1,
            gpu_memory_utilization=float(execution["gpu_memory"]),
            max_model_len=int(execution["max_model_len"]),
            max_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            enable_thinking=False,
            num_samples=int(task["num_samples"]),
            seed=int(task["generation_seed"]),
            llm=llm,
            tokenizer=tokenizer,
        )
        if int(result.summary["sample_count"]) != int(task["expected_records"]):
            raise ValueError(
                f"Task {task['task_id']} produced {result.summary['sample_count']} records; "
                f"expected {task['expected_records']}"
            )
        temporary.joinpath("SUCCESS").touch()
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_lcb_task_atomic(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    llm: Any,
) -> None:
    """Generate one LCB micro-shard and publish it atomically."""

    output_dir = Path(str(task["output_dir"]))
    temporary = _temporary_output_dir(output_dir)
    try:
        generate_lcb_task(
            task={**task, "output_dir": str(temporary)},
            manifest=manifest,
            llm=llm,
        )
        temporary.joinpath("SUCCESS").touch()
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
