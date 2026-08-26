"""Plan and merge data-parallel evaluation shards for a Slurm GPU pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from eval.domains.science.pinned_mmlupro import validate_mmlupro_500_artifact
from eval.report import _compact_record, _detail_record, summarize_records, write_report

SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class DatasetSpec:
    """Describe one dataset that can be split across independent GPU workers."""

    key: str
    domain: str
    relative_path: str
    task_type: str = "standard"


DATASET_SPECS = (
    DatasetSpec("aime24", "math", "data/eval_data/math/AIME24/test.parquet"),
    DatasetSpec("aime25", "math", "data/eval_data/math/AIME25/test.parquet"),
    DatasetSpec("hmmt25feb", "math", "data/eval_data/math/HMMT25Feb/test.parquet"),
    DatasetSpec("hmmt25nov", "math", "data/eval_data/math/HMMT25Nov/test.parquet"),
    DatasetSpec("humaneval_plus", "code", "data/eval_data/code/HumanEvalPlus/test.parquet"),
    DatasetSpec("mbpp_plus", "code", "data/eval_data/code/MBPPPlus/test.parquet"),
    DatasetSpec("gpqa_diamond", "science", "data/eval_data/science/GPQA/test.parquet"),
    DatasetSpec(
        "mmlupro_500_seed42",
        "science",
        "data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/test.parquet",
        task_type="official_mmlupro",
    ),
)
DATASET_SPEC_BY_KEY = {spec.key: spec for spec in DATASET_SPECS}


def model_label(model_path: str) -> str:
    """Return the output label used by the existing Slurm evaluator."""
    path = Path(model_path.rstrip("/"))
    label = f"{path.parent.name}__{path.name}" if path.name.startswith("global_step_") else path.name
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in label)
    if not safe:
        raise ValueError(f"Could not derive a model label from {model_path}")
    return safe


def file_sha256(path: Path) -> str:
    """Hash a source artifact without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_ranges(total_rows: int, shard_count: int) -> list[tuple[int, int]]:
    """Split rows into non-empty, contiguous ranges with size difference at most one."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    actual_shards = min(total_rows, shard_count)
    return [
        (total_rows * index // actual_shards, total_rows * (index + 1) // actual_shards)
        for index in range(actual_shards)
    ] if actual_shards else []


def _selected_specs(dataset_keys: Sequence[str], include_mmlupro_500: bool) -> list[DatasetSpec]:
    requested = list(dataset_keys)
    if include_mmlupro_500 and "mmlupro_500_seed42" not in requested:
        requested.append("mmlupro_500_seed42")
    unknown = sorted(set(requested) - set(DATASET_SPEC_BY_KEY))
    if unknown:
        raise ValueError(f"Unsupported parallel evaluation datasets: {unknown}")
    return [DATASET_SPEC_BY_KEY[key] for key in requested]


def build_manifest(
    *,
    code_dir: Path,
    suite_root: Path,
    run_tag: str,
    model_path: str,
    dataset_keys: Sequence[str],
    shards_per_dataset: int,
    min_rows_per_shard: int,
    math_samples: int,
    code_samples: int,
    science_samples: int,
    base_seed: int,
    max_samples_per_dataset: int | None,
    include_mmlupro_500: bool,
    max_new_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 24,
    gpu_memory: float = 0.85,
    max_model_len: int = 18432,
    max_num_batched_tokens: int = 32768,
    max_num_seqs: int = 24,
    score_code: bool = True,
    code_sandbox_image: str = "verlai/verl:vllm023.dev1",
    code_sandbox_image_id: str = "unresolved",
) -> dict[str, Any]:
    """Create a deterministic shard manifest for one model and evaluation suite."""
    if shards_per_dataset < 1:
        raise ValueError("shards_per_dataset must be positive")
    if min_rows_per_shard < 1:
        raise ValueError("min_rows_per_shard must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if max_samples_per_dataset is not None and max_samples_per_dataset < 1:
        raise ValueError("max_samples_per_dataset must be positive when provided")
    for name, value in (
        ("max_new_tokens", max_new_tokens),
        ("batch_size", batch_size),
        ("max_model_len", max_model_len),
        ("max_num_batched_tokens", max_num_batched_tokens),
        ("max_num_seqs", max_num_seqs),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive for sampled multi-rollout evaluation")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not math.isfinite(gpu_memory) or not 0 < gpu_memory <= 1:
        raise ValueError("gpu_memory must be in (0, 1]")
    samples_by_domain = {
        "math": math_samples,
        "code": code_samples,
        "science": science_samples,
    }
    if any(value < 1 for value in samples_by_domain.values()):
        raise ValueError("samples per domain must be positive")

    scorer_sources = (
        "eval/domains/scoring.py",
        "mopd_verl/code_reward.py",
    )
    scorer_source_sha256 = {
        relative_path: (
            file_sha256(code_dir / relative_path)
            if (code_dir / relative_path).is_file()
            else None
        )
        for relative_path in scorer_sources
    }

    label = model_label(model_path)
    sources: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    sequence = 0
    for spec in _selected_specs(dataset_keys, include_mmlupro_500):
        source_file = code_dir / spec.relative_path
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing evaluation parquet: {source_file}")
        pinned_validation = None
        if spec.task_type == "official_mmlupro":
            pinned_validation = validate_mmlupro_500_artifact(
                source_file,
                source_file.with_name("manifest.json"),
            )
        source_rows = pq.ParquetFile(source_file).metadata.num_rows
        selected_rows = (
            min(source_rows, max_samples_per_dataset)
            if max_samples_per_dataset is not None
            else source_rows
        )
        source_metadata: dict[str, Any] = {
            "dataset": spec.key,
            "domain": spec.domain,
            "task_type": spec.task_type,
            "source_file": str(source_file),
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "source_sha256": file_sha256(source_file),
        }
        if pinned_validation is not None:
            source_metadata["pinned_validation"] = pinned_validation.as_dict()
        sources.append(source_metadata)
        num_samples = 4 if spec.task_type == "official_mmlupro" else samples_by_domain[spec.domain]
        effective_shards = min(
            shards_per_dataset,
            max(1, selected_rows // min_rows_per_shard),
        ) if selected_rows else 0
        for shard_index, (start, end) in enumerate(balanced_ranges(selected_rows, effective_shards)):
            task_id = (
                f"{sequence:04d}_{spec.domain}_{spec.key}_"
                f"{shard_index:02d}_{start:06d}_{end:06d}"
            )
            task_root = suite_root / "shards" / task_id
            if spec.task_type == "official_mmlupro":
                output_dir = task_root / label / "mmlupro_500_seed42"
                success_marker = output_dir / "SUCCESS"
            else:
                output_dir = task_root / label / spec.domain
                success_marker = task_root / "SUCCESS"
            tasks.append(
                {
                    "sequence": sequence,
                    "task_id": task_id,
                    "task_type": spec.task_type,
                    "dataset": spec.key,
                    "domain": spec.domain,
                    "source_start": start,
                    "source_end_exclusive": end,
                    "num_samples": num_samples,
                    "generation_seed": (
                        base_seed
                        if spec.task_type == "official_mmlupro"
                        else base_seed + sequence * SEED_STRIDE
                    ),
                    "expected_records": (end - start) * num_samples,
                    "task_root": str(task_root),
                    "output_dir": str(output_dir),
                    "success_marker": str(success_marker),
                }
            )
            sequence += 1

    return {
        "schema_version": 1,
        "suite": "parallel_slurm_eval",
        "status": "running",
        "run_tag": run_tag,
        "suite_root": str(suite_root),
        "model": {"label": label, "path": model_path},
        "execution": {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "parallelism": "data_parallel_gpu_worker_pool",
            "shards_per_dataset": shards_per_dataset,
            "min_rows_per_shard": min_rows_per_shard,
            "batch_size": batch_size,
            "gpu_memory": gpu_memory,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "score_code": score_code,
            "code_scorer": {
                "backend": "docker" if score_code else "disabled",
                "image": code_sandbox_image if score_code else None,
                "image_id": code_sandbox_image_id if score_code else None,
                "source_sha256": scorer_source_sha256 if score_code else None,
            },
        },
        "generation": {
            "math_samples": math_samples,
            "code_samples": code_samples,
            "science_samples": science_samples,
            "mmlupro_samples": 4,
            "base_seed": base_seed,
            "standard_shard_seed_rule": f"base_seed + task_sequence * {SEED_STRIDE}",
            "mmlupro_seed_rule": "fixed base_seed for every prompt shard",
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "max_samples_per_dataset": max_samples_per_dataset,
        "sources": sources,
        "tasks": tasks,
        "total_shards": len(tasks),
        "expected_records_total": sum(task["expected_records"] for task in tasks),
        "completed_shards": 0,
    }


def resume_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable manifest fields used to validate a resumed run."""
    ignored = {"status", "completed_shards"}
    return {key: value for key, value in manifest.items() if key not in ignored}


def _validate_tsv_field(value: object) -> str:
    rendered = str(value)
    if "\t" in rendered or "\n" in rendered:
        raise ValueError(f"Task field contains a tab or newline: {rendered!r}")
    return rendered


def task_tsv(task: Mapping[str, Any]) -> str:
    """Serialize one controlled task for the Bash worker without shell evaluation."""
    fields = (
        task["sequence"],
        task["task_id"],
        task["task_type"],
        task["dataset"],
        task["domain"],
        task["source_start"],
        int(task["source_end_exclusive"]) - int(task["source_start"]),
        task["num_samples"],
        task["generation_seed"],
        task["output_dir"],
        task["success_marker"],
    )
    return "\t".join(_validate_tsv_field(field) for field in fields) + "\n"


def write_plan(manifest: dict[str, Any], *, resume: bool) -> Path:
    """Validate or create a suite and materialize its atomic task queue."""
    suite_root = Path(manifest["suite_root"])
    manifest_path = suite_root / "suite_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Evaluation suite already exists: {suite_root}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume_signature(existing) != resume_signature(manifest):
            raise ValueError(f"Parallel eval resume configuration differs: {manifest_path}")
        manifest = existing
        manifest["status"] = "running"
    elif resume:
        raise FileNotFoundError(f"RESUME requires the original suite manifest: {manifest_path}")
    elif suite_root.exists() and any(suite_root.iterdir()):
        raise FileExistsError(f"Evaluation suite directory is not empty: {suite_root}")

    suite_root.mkdir(parents=True, exist_ok=True)
    queue_root = suite_root / "queue"
    if queue_root.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked task queue: {queue_root}")
    if queue_root.exists():
        shutil.rmtree(queue_root)
    for queue_name in ("pending", "running", "done", "failed"):
        (queue_root / queue_name).mkdir(parents=True)

    completed = 0
    for task in manifest["tasks"]:
        file_name = f"{int(task['sequence']):04d}_{task['task_id']}.task"
        payload = task_tsv(task)
        if Path(task["success_marker"]).is_file():
            (queue_root / "done" / file_name).write_text(payload, encoding="utf-8")
            completed += 1
        else:
            if task["task_type"] == "official_mmlupro" and Path(task["output_dir"]).exists():
                if any(Path(task["output_dir"]).iterdir()):
                    raise ValueError(
                        "Incomplete MMLU-Pro shards are not resumable; use a new run tag: "
                        f"{task['output_dir']}"
                    )
            (queue_root / "pending" / file_name).write_text(payload, encoding="utf-8")
    manifest["completed_shards"] = completed
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _prepare_atomic_output(output_dir: Path) -> Path | None:
    marker = output_dir / "SUCCESS"
    if marker.is_file():
        return None
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace incomplete merged output: {output_dir}")
    temp_dir = output_dir.with_name(f".{output_dir.name}.merge-{os.getpid()}")
    if temp_dir.exists():
        raise FileExistsError(f"Stale merge directory exists: {temp_dir}")
    temp_dir.mkdir(parents=True)
    return temp_dir


def _publish_atomic_output(temp_dir: Path, output_dir: Path) -> None:
    (temp_dir / "SUCCESS").touch()
    temp_dir.rename(output_dir)


def _validate_task_records(task: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    expected = int(task["expected_records"])
    if len(records) != expected:
        raise ValueError(
            f"Shard {task['task_id']} produced {len(records)} records; expected {expected}"
        )


def _merge_standard_domain(
    *,
    manifest: Mapping[str, Any],
    domain: str,
    tasks: Sequence[Mapping[str, Any]],
) -> None:
    suite_root = Path(str(manifest["suite_root"]))
    label = str(manifest["model"]["label"])
    output_dir = suite_root / label / domain
    temp_dir = _prepare_atomic_output(output_dir)
    if temp_dir is None:
        return
    raw_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    try:
        for task in sorted(tasks, key=lambda item: int(item["sequence"])):
            if not Path(str(task["success_marker"])).is_file():
                raise FileNotFoundError(f"Shard is incomplete: {task['task_id']}")
            shard_records = _read_jsonl(Path(str(task["output_dir"])) / "thinking_eval_samples.jsonl")
            _validate_task_records(task, shard_records)
            for record in shard_records:
                key = (
                    str(record.get("dataset")),
                    str(record.get("sample_id")),
                    str(record.get("mode")),
                    int(record.get("rollout_index", 0)),
                )
                if key in seen:
                    raise ValueError(f"Duplicate merged evaluation record: {key}")
                seen.add(key)
                raw_records.append(record)

        raw_path = temp_dir / "thinking_eval_samples.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for record in raw_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        run_config = {
            "parallel_manifest": str(suite_root / "suite_manifest.json"),
            "domain": domain,
            "task_ids": [task["task_id"] for task in tasks],
            "generation": manifest["generation"],
            "execution": manifest["execution"],
        }
        (temp_dir / "eval_run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        compact_records = [_compact_record(record) for record in raw_records]
        detail_records = [
            {"run_id": f"{label}_{domain}_{manifest['run_tag']}", "model_path": manifest["model"]["path"], **detail}
            for detail in (_detail_record(record) for record in raw_records)
            if detail is not None
        ]
        payload = {
            "run_id": f"{label}_{domain}_{manifest['run_tag']}",
            "status": "final",
            "model_path": manifest["model"]["path"],
            "scoring_backend": "verl.utils.reward_score.default_compute_score",
            "record_source": "parallel_shards",
            "expected_total": sum(int(task["expected_records"]) for task in tasks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "notes": "Merged from disjoint data-parallel GPU shards.",
            "summary": summarize_records(compact_records),
            "records": compact_records,
            "run_config": run_config,
        }
        write_report(payload, temp_dir, detail_records)
        (temp_dir / "parallel_merge_manifest.json").write_text(
            json.dumps({"task_ids": [task["task_id"] for task in tasks]}, indent=2) + "\n",
            encoding="utf-8",
        )
        _publish_atomic_output(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def summarize_mmlupro_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model_path: str,
) -> dict[str, Any]:
    """Recompute the official MMLU-Pro summary after merging prompt shards."""
    correct = sum(int(record.get("correct") is True) for record in records)
    prompts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    categories: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        question_id = str(record.get("question_id", record.get("index")))
        prompts[question_id].append(record)
        categories[str(record.get("category", "unknown"))].append(record)

    def category_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        category_prompts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            category_prompts[str(item.get("question_id", item.get("index")))].append(item)
        category_correct = sum(int(item.get("correct") is True) for item in items)
        passed = sum(any(item.get("correct") is True for item in values) for values in category_prompts.values())
        return {
            "correct": category_correct,
            "total": len(items),
            "passed_prompts": passed,
            "prompt_count": len(category_prompts),
            "accuracy": category_correct / len(items) if items else None,
            "observed_pass_at_k": passed / len(category_prompts) if category_prompts else None,
        }

    passed_prompts = sum(any(item.get("correct") is True for item in values) for values in prompts.values())
    rollout_counts = {len(values) for values in prompts.values()}
    return {
        "dataset": "mmlupro_500_seed42",
        "model_path": model_path,
        "prompt_count": len(prompts),
        "rollouts_per_prompt": rollout_counts.pop() if len(rollout_counts) == 1 else None,
        "sample_count": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "passed_prompts": passed_prompts,
        "observed_pass_at_k": passed_prompts / len(prompts) if prompts else None,
        "per_category": {
            category: category_summary(items)
            for category, items in sorted(categories.items())
        },
    }


def _merge_mmlupro(
    *,
    manifest: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> None:
    if not tasks:
        return
    suite_root = Path(str(manifest["suite_root"]))
    label = str(manifest["model"]["label"])
    output_dir = suite_root / label / "mmlupro_500_seed42"
    temp_dir = _prepare_atomic_output(output_dir)
    if temp_dir is None:
        return
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    try:
        for task in sorted(tasks, key=lambda item: int(item["sequence"])):
            if not Path(str(task["success_marker"])).is_file():
                raise FileNotFoundError(f"Shard is incomplete: {task['task_id']}")
            shard_records = _read_jsonl(
                Path(str(task["output_dir"])) / "mmlupro_500_seed42" / "records.jsonl"
            )
            _validate_task_records(task, shard_records)
            for record in shard_records:
                key = (str(record.get("question_id")), int(record.get("rollout_index", 0)))
                if key in seen:
                    raise ValueError(f"Duplicate MMLU-Pro record: {key}")
                seen.add(key)
                merged_record = dict(record)
                merged_record["index"] = int(task["source_start"]) + int(record.get("index", 0))
                records.append(merged_record)

        dataset_dir = temp_dir / "mmlupro_500_seed42"
        dataset_dir.mkdir()
        with (dataset_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = summarize_mmlupro_records(records, model_path=str(manifest["model"]["path"]))
        (dataset_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        run_config = {
            "dataset": "mmlupro_500_seed42",
            "model_path": manifest["model"]["path"],
            "prompt_count": summary["prompt_count"],
            "num_samples": summary["rollouts_per_prompt"],
            "parallel_manifest": str(suite_root / "suite_manifest.json"),
            "task_ids": [task["task_id"] for task in tasks],
        }
        (dataset_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        official_index = {
            "output_dir": str(output_dir),
            "results": [
                {
                    "dataset": "mmlupro_500_seed42",
                    "output_dir": str(output_dir / "mmlupro_500_seed42"),
                    "summary": summary,
                }
            ],
        }
        (temp_dir / "official_eval_summary.json").write_text(
            json.dumps(official_index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temp_dir / "standard_ood_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark": "MMLU-Pro-500",
                    "model_path": manifest["model"]["path"],
                    "parallel_manifest": str(suite_root / "suite_manifest.json"),
                    "task_ids": [task["task_id"] for task in tasks],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish_atomic_output(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def merge_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate all shards, merge domain outputs, and mark the suite complete."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = list(manifest["tasks"])
    missing = [task["task_id"] for task in tasks if not Path(task["success_marker"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot merge incomplete shards: {missing[:8]}")
    standard_by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    mmlupro_tasks: list[Mapping[str, Any]] = []
    for task in tasks:
        if task["task_type"] == "official_mmlupro":
            mmlupro_tasks.append(task)
        else:
            standard_by_domain[str(task["domain"])].append(task)
    for domain, domain_tasks in standard_by_domain.items():
        _merge_standard_domain(manifest=manifest, domain=domain, tasks=domain_tasks)
    _merge_mmlupro(manifest=manifest, tasks=mmlupro_tasks)
    manifest["status"] = "complete"
    manifest["completed_shards"] = len(tasks)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(manifest["suite_root"]).joinpath("SUCCESS").touch()
    return manifest


def _parse_dataset_csv(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("datasets cannot be empty")
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Create or validate the shard queue.")
    plan_parser.add_argument("--code-dir", type=Path, required=True)
    plan_parser.add_argument("--suite-root", type=Path, required=True)
    plan_parser.add_argument("--run-tag", required=True)
    plan_parser.add_argument("--model-path", required=True)
    plan_parser.add_argument("--datasets", type=_parse_dataset_csv, required=True)
    plan_parser.add_argument("--shards-per-dataset", type=int, required=True)
    plan_parser.add_argument("--min-rows-per-shard", type=int, default=24)
    plan_parser.add_argument("--math-samples", type=int, required=True)
    plan_parser.add_argument("--code-samples", type=int, required=True)
    plan_parser.add_argument("--science-samples", type=int, required=True)
    plan_parser.add_argument("--base-seed", type=int, default=42)
    plan_parser.add_argument("--max-new-tokens", type=int, default=16384)
    plan_parser.add_argument("--temperature", type=float, default=1.0)
    plan_parser.add_argument("--top-p", type=float, default=1.0)
    plan_parser.add_argument("--batch-size", type=int, default=24)
    plan_parser.add_argument("--gpu-memory", type=float, default=0.85)
    plan_parser.add_argument("--max-model-len", type=int, default=18432)
    plan_parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    plan_parser.add_argument("--max-num-seqs", type=int, default=24)
    plan_parser.add_argument(
        "--code-sandbox-image",
        default="verlai/verl:vllm023.dev1",
    )
    plan_parser.add_argument("--code-sandbox-image-id", default="unresolved")
    plan_parser.add_argument("--no-score-code", action="store_true")
    plan_parser.add_argument("--max-samples-per-dataset", type=int)
    plan_parser.add_argument("--include-mmlupro-500", action="store_true")
    plan_parser.add_argument("--resume", action="store_true")

    merge_parser = subparsers.add_parser("merge", help="Merge all successful shards.")
    merge_parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        manifest = build_manifest(
            code_dir=args.code_dir,
            suite_root=args.suite_root,
            run_tag=args.run_tag,
            model_path=args.model_path,
            dataset_keys=args.datasets,
            shards_per_dataset=args.shards_per_dataset,
            min_rows_per_shard=args.min_rows_per_shard,
            math_samples=args.math_samples,
            code_samples=args.code_samples,
            science_samples=args.science_samples,
            base_seed=args.base_seed,
            max_samples_per_dataset=args.max_samples_per_dataset,
            include_mmlupro_500=args.include_mmlupro_500,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            gpu_memory=args.gpu_memory,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            score_code=not args.no_score_code,
            code_sandbox_image=args.code_sandbox_image,
            code_sandbox_image_id=args.code_sandbox_image_id,
        )
        manifest_path = write_plan(manifest, resume=args.resume)
        print(f"[parallel-eval] planned shards={manifest['total_shards']} manifest={manifest_path}")
        return 0
    manifest = merge_manifest(args.manifest)
    print(
        f"[parallel-eval] merged shards={manifest['completed_shards']} "
        f"output={manifest['suite_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
