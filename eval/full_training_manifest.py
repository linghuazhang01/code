"""Build and validate the manifest for the sharded full-training eval suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

SourceMap = Mapping[str, tuple[str, Path]]


@dataclass(frozen=True)
class ManifestConfig:
    code_dir: Path
    output_root: Path
    status: str
    run_tag: str
    student_model_path: str
    teacher_model_path: str
    gpu_id: str
    shard_size: int
    minimum_free_gib: int
    max_samples_per_domain: int | None
    max_new_tokens: int
    num_samples: int
    temperature: float
    top_p: float
    seed: int
    batch_size: int
    gpu_memory: float
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    resume: bool
    manifest_initialized: bool


def default_sources(code_dir: Path) -> dict[str, tuple[str, Path]]:
    return {
        "math": (
            "training_full_math",
            code_dir / "data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
        ),
        "code": (
            "training_full_code",
            code_dir / "data/G-OPD-Training-Data/Eurus/code_train.parquet",
        ),
        "if": (
            "training_full_if",
            code_dir / "data/G-OPD-Training-Data/IF/train.parquet",
        ),
        "science": (
            "training_full_science",
            code_dir / "data/G-OPD-Training-Data/Science/train.parquet",
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    cfg: ManifestConfig,
    *,
    sources: SourceMap | None = None,
) -> dict[str, Any]:
    models = {
        "qwen3_1p7b": cfg.student_model_path,
        "goosereason_4b": cfg.teacher_model_path,
    }
    domains: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for domain, (dataset, source_file) in (sources or default_sources(cfg.code_dir)).items():
        source_rows = pq.ParquetFile(source_file).metadata.num_rows
        selected_rows = (
            min(source_rows, cfg.max_samples_per_domain)
            if cfg.max_samples_per_domain is not None
            else source_rows
        )
        domains.append(
            {
                "domain": domain,
                "dataset": dataset,
                "source_file": str(source_file),
                "source_rows": source_rows,
                "source_sha256": file_sha256(source_file),
                "selected_rows_per_model": selected_rows,
            }
        )
        for model_label in models:
            for shard_number, start in enumerate(range(0, selected_rows, cfg.shard_size)):
                end = min(start + cfg.shard_size, selected_rows)
                shard_index = f"{shard_number:05d}_{start:09d}_{end:09d}"
                shard_dir = cfg.output_root / model_label / domain / f"shard_{shard_index}"
                shards.append(
                    {
                        "model": model_label,
                        "domain": domain,
                        "dataset": dataset,
                        "source_start": start,
                        "source_end_exclusive": end,
                        "generation_seed": cfg.seed,
                        "expected_records": (end - start) * cfg.num_samples,
                        "output_dir": str(shard_dir),
                        "success": (shard_dir / "SUCCESS").is_file(),
                    }
                )

    return {
        "schema_version": 1,
        "suite": "full_training",
        "status": cfg.status,
        "run_tag": cfg.run_tag,
        "models": models,
        "execution": {
            "gpu_id": cfg.gpu_id,
            "tensor_parallel_size": 1,
            "backend": "vllm",
            "mode": "non_thinking",
            "batch_size": cfg.batch_size,
            "gpu_memory": cfg.gpu_memory,
            "max_model_len": cfg.max_model_len,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "max_num_seqs": cfg.max_num_seqs,
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "shard_size": cfg.shard_size,
            "minimum_free_gib_guard": cfg.minimum_free_gib,
            "max_samples_per_domain": cfg.max_samples_per_domain,
        },
        "generation": {
            "max_new_tokens": cfg.max_new_tokens,
            "num_samples": cfg.num_samples,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "base_seed": cfg.seed,
            "shard_seed_rule": "fixed base_seed for every shard",
        },
        "domains": domains,
        "expected_records_total": sum(item["expected_records"] for item in shards),
        "completed_shards": sum(1 for item in shards if item["success"]),
        "total_shards": len(shards),
        "shards": shards,
    }


def resume_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(value.get("execution", {}))
    execution.pop("gpu_id", None)
    execution.pop("minimum_free_gib_guard", None)
    return {
        "schema_version": value.get("schema_version"),
        "suite": value.get("suite"),
        "run_tag": value.get("run_tag"),
        "models": value.get("models"),
        "execution": execution,
        "generation": value.get("generation"),
        "domains": value.get("domains"),
        "shards": [
            {key: item for key, item in shard.items() if key != "success"}
            for shard in value.get("shards", [])
        ],
    }


def validate_manifest_state(cfg: ManifestConfig, manifest: Mapping[str, Any]) -> None:
    manifest_path = cfg.output_root / "suite_manifest.json"
    if cfg.manifest_initialized:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Suite manifest disappeared during the run: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume_signature(existing) != resume_signature(manifest):
            raise ValueError(
                f"Full-training configuration or source identity changed during the run: {manifest_path}"
            )
        return

    if cfg.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"RESUME=1 requires the original suite manifest: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume_signature(existing) != resume_signature(manifest):
            raise ValueError(
                "Full-training resume configuration or source identity differs "
                f"from the original suite: {manifest_path}"
            )
    elif cfg.output_root.exists() and any(cfg.output_root.iterdir()):
        raise FileExistsError(
            f"Output suite directory is not empty: {cfg.output_root}. "
            "Use a new RUN_TAG, or set RESUME=1 with identical settings."
        )


def process_manifest(cfg: ManifestConfig) -> dict[str, Any]:
    manifest = build_manifest(cfg)
    validate_manifest_state(cfg, manifest)
    if cfg.status != "dry_run":
        cfg.output_root.mkdir(parents=True, exist_ok=True)
        (cfg.output_root / "suite_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status", choices=("running", "complete", "dry_run"), required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--shard-size", type=int, required=True)
    parser.add_argument("--minimum-free-gib", type=int, required=True)
    parser.add_argument("--max-samples-per-domain", type=int)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu-memory", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-initialized", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ManifestConfig(**vars(args))
    try:
        manifest = process_manifest(cfg)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Validated full-training {cfg.status} plan: "
        f"shards={manifest['total_shards']} records={manifest['expected_records_total']}"
    )


if __name__ == "__main__":
    main()
