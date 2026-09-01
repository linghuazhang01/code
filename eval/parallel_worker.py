"""Persistent one-GPU worker for the data-parallel Slurm evaluation pool."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.parallel_tasks import run_lcb_task_atomic, run_mmlupro_task, run_standard_task
from eval.runner import load_vllm_model

LOGGER = logging.getLogger(__name__)


def _validate_single_visible_gpu() -> None:
    """Fail closed unless this replica owns exactly one CUDA device."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Each data-parallel worker requires exactly one visible GPU; "
            f"found {torch.cuda.device_count()}"
        )


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


def _wave_queue_root(queue_root: Path, wave: Mapping[str, Any]) -> Path:
    return queue_root / "waves" / f"{int(wave['wave_index']):04d}_{wave['dataset']}"


def _execute_claimed_task(
    *,
    claimed: Path,
    queue_root: Path,
    worker_id: int,
    task_by_id: Mapping[str, Mapping[str, Any]],
    source_by_dataset: Mapping[str, Path],
    manifest: Mapping[str, Any],
    eval_model_path: str,
    llm: Any,
    tokenizer: Any,
    resume: bool,
) -> bool:
    """Execute one claimed task and move its queue record to a terminal state."""

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
        elif task["task_type"] == "official_lcb":
            run_lcb_task_atomic(task=task, manifest=manifest, llm=llm)
        else:
            raise ValueError(f"Unsupported task type: {task['task_type']}")
        success_marker = Path(str(task["success_marker"]))
        if not success_marker.is_file():
            raise FileNotFoundError(
                f"Task output was published without its atomic SUCCESS marker: {success_marker}"
            )
        os.replace(claimed, queue_root / "done" / claimed.name)
        return True
    except Exception:
        failed_path = queue_root / "failed" / claimed.name
        if claimed.exists():
            os.replace(claimed, failed_path)
        LOGGER.exception("worker=%d failed task file=%s", worker_id, claimed)
        return False


def run_worker(
    *,
    manifest_path: Path,
    eval_model_path: str,
    worker_id: int,
    resume: bool,
) -> int:
    """Load one TP=1 engine and process strict dataset waves without reloading."""
    manifest = _load_manifest(manifest_path)
    queue_root = manifest_path.parent / "queue"
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError("Dataset-wave workers require a schema-v2 manifest")
    expected_eval_path = str(manifest["model"]["eval_path"])
    if eval_model_path != expected_eval_path:
        raise ValueError(
            f"Worker eval model differs from manifest: {eval_model_path} != {expected_eval_path}"
        )
    if not any(queue_root.glob("waves/*/pending/*.task")):
        LOGGER.info("worker=%d no pending tasks", worker_id)
        return 0

    _validate_single_visible_gpu()

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
    for wave in manifest["waves"]:
        wave_root = _wave_queue_root(queue_root, wave)
        expected_tasks = int(wave["expected_tasks"])
        while True:
            if any((wave_root / "failed").glob("*.task")):
                LOGGER.error("worker=%d wave=%s has a failed task", worker_id, wave["dataset"])
                return 1
            claimed = claim_next_task(wave_root, worker_id)
            if claimed is not None:
                if not _execute_claimed_task(
                    claimed=claimed,
                    queue_root=wave_root,
                    worker_id=worker_id,
                    task_by_id=task_by_id,
                    source_by_dataset=source_by_dataset,
                    manifest=manifest,
                    eval_model_path=eval_model_path,
                    llm=llm,
                    tokenizer=tokenizer,
                    resume=resume,
                ):
                    return 1
                continue
            pending = any((wave_root / "pending").glob("*.task"))
            running = any((wave_root / "running").glob("*.task"))
            if pending or running:
                time.sleep(0.2)
                continue
            done_count = len(list((wave_root / "done").glob("*.task")))
            if done_count != expected_tasks:
                LOGGER.error(
                    "worker=%d wave=%s terminal count=%d expected=%d",
                    worker_id,
                    wave["dataset"],
                    done_count,
                    expected_tasks,
                )
                return 1
            missing_markers = [
                task_id
                for task_id in wave["task_ids"]
                if not Path(str(task_by_id[str(task_id)]["success_marker"])).is_file()
            ]
            if missing_markers:
                LOGGER.error(
                    "worker=%d wave=%s missing success markers=%s",
                    worker_id,
                    wave["dataset"],
                    missing_markers[:4],
                )
                return 1
            wave_root.joinpath("SUCCESS").touch()
            LOGGER.info("worker=%d completed wave=%s", worker_id, wave["dataset"])
            break
    return 0


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
