"""Persistent one-GPU worker for the data-parallel Slurm evaluation pool."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.common import append_sample_outputs, load_eval_samples, write_outputs
from eval.domains.science.official_eval import run_dataset as run_science_dataset
from eval.runner import (
    completed_samples_for_rollout,
    generate_vllm_batch,
    load_incremental_results,
    load_vllm_model,
    validate_resume_prefix,
)

LOGGER = logging.getLogger(__name__)


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("suite") != "parallel_slurm_eval":
        raise ValueError(f"Unsupported parallel evaluation manifest: {path}")
    return value


def _source_by_dataset(manifest: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(source["dataset"]): Path(str(source["source_file"]))
        for source in manifest["sources"]
    }


def _task_config(
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    eval_model_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
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


def _load_or_create_standard_output(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    resume: bool,
) -> list[Any]:
    config_path = output_dir / "eval_run_config.json"
    samples_path = output_dir / "thinking_eval_samples.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"Evaluation shard output already exists: {output_dir}")
        if not config_path.is_file() or not samples_path.is_file():
            raise FileNotFoundError(f"Shard is not safely resumable: {output_dir}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != dict(config):
            raise ValueError(f"Shard resume configuration differs: {output_dir}")
        return load_incremental_results(samples_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(dict(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    samples_path.write_text("", encoding="utf-8")
    return []


def run_standard_task(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    eval_model_path: str,
    llm: Any,
    tokenizer: Any,
    source_file: Path,
    resume: bool,
) -> None:
    """Evaluate one standard parquet slice while reusing the worker's vLLM engine."""
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

    config = _task_config(task, manifest, eval_model_path=eval_model_path)
    results = _load_or_create_standard_output(
        output_dir=output_dir,
        config=config,
        resume=resume,
    )
    num_samples = int(task["num_samples"])
    validate_resume_prefix(results, samples, ["non_thinking"], num_samples)
    execution = manifest["execution"]
    generation = manifest["generation"]
    batch_size = int(execution["batch_size"])
    max_new_tokens = int(generation["max_new_tokens"])
    score_code = bool(execution["score_code"] and task["domain"] == "code")
    task_seed = int(task["generation_seed"])

    for rollout_index in range(num_samples):
        resume_start = completed_samples_for_rollout(
            len(results),
            mode_index=0,
            rollout_index=rollout_index,
            sample_count=len(samples),
            num_samples=num_samples,
        )
        if resume_start == len(samples):
            continue
        if resume_start % batch_size != 0:
            raise ValueError(
                f"Task {task['task_id']} resume point {resume_start} is not aligned "
                f"to batch_size={batch_size}"
            )
        for batch_start in range(resume_start, len(samples), batch_size):
            batch = samples[batch_start : batch_start + batch_size]
            generation_seed = task_seed + rollout_index * len(samples) + batch_start
            LOGGER.info(
                "task=%s dataset=%s rollout=%d prompts=%d-%d/%d",
                task["task_id"],
                task["dataset"],
                rollout_index,
                batch_start + 1,
                batch_start + len(batch),
                len(samples),
            )
            batch_results = generate_vllm_batch(
                llm,
                tokenizer,
                batch,
                mode="non_thinking",
                max_new_tokens=max_new_tokens,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                score_code=score_code,
                save_completion=True,
                rollout_index=rollout_index,
                generation_seed=generation_seed,
            )
            results.extend(batch_results)
            append_sample_outputs(batch_results, output_dir)
    write_outputs(results, output_dir)


def run_mmlupro_task(
    *,
    task: Mapping[str, Any],
    manifest: Mapping[str, Any],
    eval_model_path: str,
    llm: Any,
    tokenizer: Any,
) -> None:
    """Evaluate one pinned MMLU-Pro slice using the already-loaded engine."""
    execution = manifest["execution"]
    generation = manifest["generation"]
    start = int(task["source_start"])
    count = int(task["source_end_exclusive"]) - start
    result = run_science_dataset(
        dataset_key="mmlupro_500_seed42",
        model_path=eval_model_path,
        output_dir=Path(str(task["output_dir"])),
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


def claim_next_task(queue_root: Path, worker_id: int) -> Path | None:
    """Atomically move the earliest pending task into this worker's running set."""
    for pending_path in sorted((queue_root / "pending").glob("*.task")):
        claimed = queue_root / "running" / f"{pending_path.stem}.gpu{worker_id}.task"
        try:
            os.replace(pending_path, claimed)
        except FileNotFoundError:
            continue
        return claimed
    return None


def _task_id_from_file(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").rstrip("\n").split("\t")
    if len(fields) != 11:
        raise ValueError(f"Invalid task file: {path}")
    return fields[1]


def run_worker(
    *,
    manifest_path: Path,
    eval_model_path: str,
    worker_id: int,
    resume: bool,
) -> int:
    """Load one vLLM engine and process queue tasks until no pending work remains."""
    manifest = _load_manifest(manifest_path)
    queue_root = manifest_path.parent / "queue"
    if not any((queue_root / "pending").glob("*.task")):
        LOGGER.info("worker=%d no pending tasks", worker_id)
        return 0

    execution = manifest["execution"]
    LOGGER.info("worker=%d loading model=%s", worker_id, eval_model_path)
    llm = load_vllm_model(
        eval_model_path,
        "auto",
        1,
        float(execution["gpu_memory"]),
        max_model_len=int(execution["max_model_len"]),
        max_num_batched_tokens=int(execution["max_num_batched_tokens"]),
        max_num_seqs=int(execution["max_num_seqs"]),
        enforce_eager=bool(execution["enforce_eager"]),
        enable_chunked_prefill=bool(execution["enable_chunked_prefill"]),
    )
    tokenizer = llm.get_tokenizer()
    task_by_id = {str(task["task_id"]): task for task in manifest["tasks"]}
    source_by_dataset = _source_by_dataset(manifest)
    had_task_failure = False

    while (claimed := claim_next_task(queue_root, worker_id)) is not None:
        try:
            task_id = _task_id_from_file(claimed)
            task = task_by_id[task_id]
            LOGGER.info(
                "worker=%d task=%s dataset=%s rows=%s:%s",
                worker_id,
                task_id,
                task["dataset"],
                task["source_start"],
                task["source_end_exclusive"],
            )
            if task["task_type"] == "standard":
                run_standard_task(
                    task=task,
                    manifest=manifest,
                    eval_model_path=eval_model_path,
                    llm=llm,
                    tokenizer=tokenizer,
                    source_file=source_by_dataset[str(task["dataset"])],
                    resume=resume,
                )
            elif task["task_type"] == "official_mmlupro":
                run_mmlupro_task(
                    task=task,
                    manifest=manifest,
                    eval_model_path=eval_model_path,
                    llm=llm,
                    tokenizer=tokenizer,
                )
            else:
                raise ValueError(f"Unsupported task type: {task['task_type']}")
            success_marker = Path(str(task["success_marker"]))
            success_marker.parent.mkdir(parents=True, exist_ok=True)
            success_marker.touch()
            os.replace(claimed, queue_root / "done" / claimed.name)
        except Exception:
            failed_path = queue_root / "failed" / claimed.name
            if claimed.exists():
                os.replace(claimed, failed_path)
            LOGGER.exception("worker=%d failed task file=%s", worker_id, claimed)
            had_task_failure = True
    return int(had_task_failure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-model-path", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    return run_worker(
        manifest_path=args.manifest,
        eval_model_path=args.eval_model_path,
        worker_id=args.worker_id,
        resume=args.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
