"""Validate and record the canonical ten-dataset MOPD evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.lcb_official import LCB_RELEASE_COUNTS, LCB_SOURCE_SHA256

EXPECTED_ROLLOUTS = 8
PARALLEL_DATASET_KEYS = (
    "aime24",
    "aime25",
    "hmmt25feb",
    "hmmt25nov",
    "humaneval_plus",
    "mbpp_plus",
    "lcb_v5",
    "lcb_v6",
    "gpqa_diamond",
    "mmlupro_500_seed42",
)
CANONICAL_DATASETS = (
    "AIME2024",
    "AIME2025",
    "HMMT25Feb",
    "HMMT25Nov",
    "HumanEvalPlus",
    "MBPPPlus",
    "LiveCodeBench-v5",
    "LiveCodeBench-v6",
    "GPQA-Diamond",
    "MMLU-Pro-500-seed42",
)
@dataclass(frozen=True)
class StandardSuiteConfig:
    """Immutable provenance for one standard evaluation run."""

    suite_root: Path
    model_path: str
    eval_model_path: str
    run_tag: str
    slurm_job_id: str
    remote_host: str
    local_archive: str
    gopd_dir: Path
    gpu_count: int = 4
    reference_anchor: bool = False


def file_sha256(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _state_payload(config: StandardSuiteConfig, status: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "run_tag": config.run_tag,
        "model_path": config.model_path,
        "eval_model_path": config.eval_model_path,
        "checkpoint_step": None if config.reference_anchor else 60,
        "evaluation_role": "reference_anchor" if config.reference_anchor else "step60_candidate",
        "slurm_job_id": config.slurm_job_id,
        "remote_host": config.remote_host,
        "suite_root": str(config.suite_root),
        "local_archive": config.local_archive,
        "gpu_count": config.gpu_count,
        "canonical_datasets": list(CANONICAL_DATASETS),
        "rollouts_per_dataset": EXPECTED_ROLLOUTS,
    }


def _write_run_manifest(
    config: StandardSuiteConfig,
    *,
    status: str,
    completed_at: str | None = None,
) -> None:
    timestamp = completed_at or datetime.now(timezone.utc).isoformat()
    dataset_lines = "\n".join(f"  - {name}" for name in CANONICAL_DATASETS)
    config.suite_root.joinpath("RUN_MANIFEST.md").write_text(
        "\n".join(
            (
                "# MOPD Standard Evaluation Run",
                "",
                f"- Status: `{status}`",
                f"- Run tag: `{config.run_tag}`",
                f"- Remote host/cluster: `{config.remote_host}` / Slurm `compute`",
                f"- Remote checkpoint: `{config.model_path}`",
                f"- Prepared eval model: `{config.eval_model_path}`",
                (
                    "- Checkpoint/global step: `reference anchor (not applicable)`"
                    if config.reference_anchor
                    else "- Checkpoint/global step: `60`"
                ),
                (
                    f"- Model/training identifier: "
                    f"`{Path(config.model_path).name if config.reference_anchor else Path(config.model_path).parent.name}`"
                ),
                f"- Slurm job ID: `{config.slurm_job_id}`",
                f"- Recorded at: `{timestamp}`",
                f"- Remote output: `{config.suite_root}`",
                f"- Intended local archive: `{config.local_archive}`",
                "- Protocol: `10 datasets × K=8`, temperature `1.0`, top-p `1.0`, seed `42`",
                (
                    f"- GPU execution: `DP={config.gpu_count}`, "
                    f"{config.gpu_count} persistent `TP=1` vLLM replicas, "
                    "strict dataset waves, dynamic micro-shards"
                ),
                "- Rollout batching: each prompt request samples `n=8` candidates in one vLLM batch",
                "- LiveCodeBench formatter: `Qwen3-4B-NonThinking`, tokenizer `Qwen/Qwen3-4B`, `enable_thinking=false`",
                "- Canonical datasets:",
                dataset_lines,
                "",
            )
        ),
        encoding="utf-8",
    )


def initialize_suite(config: StandardSuiteConfig, *, resume: bool) -> None:
    """Create a fail-closed outer run directory before GPU evaluation."""

    state_path = config.suite_root / "standard_suite_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Standard evaluation suite exists: {config.suite_root}")
        existing = _load_json(state_path)
        _require(existing.get("model_path") == config.model_path, "Resume model path differs.")
        _require(existing.get("run_tag") == config.run_tag, "Resume run tag differs.")
        _require(
            int(existing.get("gpu_count", 4)) == config.gpu_count,
            "Resume GPU count differs.",
        )
        _require(
            bool(existing.get("evaluation_role") == "reference_anchor")
            == config.reference_anchor,
            "Resume evaluation role differs.",
        )
    elif config.suite_root.exists() and any(config.suite_root.iterdir()):
        raise FileExistsError(f"Standard suite directory is not empty: {config.suite_root}")
    config.suite_root.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(_state_payload(config, "running"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_run_manifest(config, status="running")


def _validate_parallel_phase(config: StandardSuiteConfig) -> dict[str, Any]:
    phase_root = config.suite_root / "parallel"
    manifest_path = phase_root / "suite_manifest.json"
    manifest = _load_json(manifest_path)
    _require(manifest.get("status") == "complete", "Parallel phase is not complete.")
    _require(manifest.get("model", {}).get("path") == config.model_path, "Model path mismatch.")
    _require(
        manifest.get("model", {}).get("eval_path") == config.eval_model_path,
        "Prepared eval model path mismatch.",
    )
    source_keys = {str(source["dataset"]) for source in manifest["sources"]}
    _require(source_keys == set(PARALLEL_DATASET_KEYS), "Parallel dataset set is not canonical.")
    execution = manifest["execution"]
    _require(
        int(execution["worker_count"]) == config.gpu_count,
        "Standard evaluation worker count differs from the recorded GPU count.",
    )
    _require(
        execution["scheduling"] == "strict_dataset_wave_dynamic_microshards",
        "Standard evaluation does not use strict dataset waves.",
    )
    waves = manifest["waves"]
    _require(
        [str(wave["dataset"]) for wave in waves] == list(PARALLEL_DATASET_KEYS),
        "Dataset wave order is not canonical.",
    )
    _require(
        all(int(wave["expected_tasks"]) >= config.gpu_count for wave in waves),
        "Every standard dataset wave must expose at least one micro-shard per GPU.",
    )
    for task in manifest["tasks"]:
        _require(int(task["num_samples"]) == EXPECTED_ROLLOUTS, "Parallel task is not K=8.")
        row_count = int(task["source_end_exclusive"]) - int(task["source_start"])
        _require(int(task["expected_records"]) == row_count * EXPECTED_ROLLOUTS, "Bad task count.")
    generation = manifest["generation"]
    _require(int(generation["base_seed"]) == 42, "Standard evaluation seed is not 42.")
    for key in ("math_samples", "code_samples", "science_samples", "mmlupro_samples"):
        _require(int(generation[key]) == EXPECTED_ROLLOUTS, f"{key} is not K=8.")
    _require((phase_root / "SUCCESS").is_file(), "Parallel SUCCESS marker is missing.")
    label = str(manifest["model"]["label"])
    for relative in ("math", "code", "science", "mmlupro_500_seed42", "lcb"):
        _require((phase_root / label / relative / "SUCCESS").is_file(), f"Missing {relative} SUCCESS.")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "sources": manifest["sources"],
        "expected_records": manifest["expected_records_total"],
    }


def _validate_lcb_release(config: StandardSuiteConfig, release: str) -> dict[str, Any]:
    expected_count = LCB_RELEASE_COUNTS[release]
    manifest = _load_json(config.suite_root / "parallel/suite_manifest.json")
    runtime_root = (
        config.suite_root
        / "parallel"
        / str(manifest["model"]["label"])
        / "lcb"
        / release
    )
    generation_files = list(runtime_root.glob("codegeneration_8_1.0.json"))
    _require(len(generation_files) == 1, f"Expected one {release} generation file.")
    generation_file = generation_files[0]
    eval_all_file = generation_file.with_name("codegeneration_8_1.0_eval_all.json")
    generations = _load_json(generation_file)
    evaluations = _load_json(eval_all_file)
    for label, records in (("generation", generations), ("evaluation", evaluations)):
        _require(isinstance(records, list), f"LCB {release} {label} payload is not a list.")
        _require(len(records) == expected_count, f"LCB {release} {label} row count differs.")
        ids = [str(record["question_id"]) for record in records]
        _require(len(ids) == len(set(ids)), f"LCB {release} has duplicate question IDs.")
        _require(
            all(len(record.get("output_list", [])) == EXPECTED_ROLLOUTS for record in records),
            f"LCB {release} {label} is not K=8.",
        )
    _require(
        all(len(record.get("graded_list", [])) == EXPECTED_ROLLOUTS for record in evaluations),
        f"LCB {release} evaluation does not contain eight grades per prompt.",
    )
    source_file = (
        config.gopd_dir
        / "code_eval/coding/LiveCodeBench/code_generation_lite"
        / f"test{release[1:]}.jsonl"
    )
    source_hash = file_sha256(source_file)
    _require(source_hash == LCB_SOURCE_SHA256[release], f"LCB {release} source hash differs.")
    return {
        "release": release,
        "prompt_count": expected_count,
        "rollouts_per_prompt": EXPECTED_ROLLOUTS,
        "generation_file": str(generation_file),
        "generation_sha256": file_sha256(generation_file),
        "evaluation_file": str(eval_all_file),
        "evaluation_sha256": file_sha256(eval_all_file),
        "source_file": str(source_file),
        "source_sha256": source_hash,
    }


def finalize_suite(config: StandardSuiteConfig) -> dict[str, Any]:
    """Validate all ten datasets and publish the outer success marker."""

    parallel = _validate_parallel_phase(config)
    lcb = [_validate_lcb_release(config, release) for release in LCB_RELEASE_COUNTS]
    completed_at = datetime.now(timezone.utc).isoformat()
    summary = {
        **_state_payload(config, "complete"),
        "completed_at": completed_at,
        "parallel_phase": parallel,
        "livecodebench": lcb,
    }
    config.suite_root.joinpath("standard_suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config.suite_root.joinpath("standard_suite_state.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_run_manifest(config, status="complete", completed_at=completed_at)
    config.suite_root.joinpath("STANDARD_SUCCESS").touch()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "finalize"))
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-model-path", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--local-archive", required=True)
    parser.add_argument("--gopd-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--reference-anchor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StandardSuiteConfig(
        suite_root=args.suite_root,
        model_path=args.model_path,
        eval_model_path=args.eval_model_path,
        run_tag=args.run_tag,
        slurm_job_id=args.slurm_job_id,
        remote_host=args.remote_host,
        local_archive=args.local_archive,
        gopd_dir=args.gopd_dir,
        gpu_count=args.gpu_count,
        reference_anchor=args.reference_anchor,
    )
    if args.command == "initialize":
        initialize_suite(config, resume=args.resume)
        return 0
    print(json.dumps(finalize_suite(config), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
