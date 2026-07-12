"""Download General-Reasoner paper benchmark datasets with `datasets`."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("data/eval_data/greasoner/official")
GPQA_D_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    source: str
    split: str
    loader_name: str
    data_files: dict[str, str] | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mmlupro": DatasetSpec(
        key="mmlupro",
        display_name="MMLU-Pro",
        source="TIGER-Lab/MMLU-Pro",
        split="test",
        loader_name="TIGER-Lab/MMLU-Pro",
    ),
    "gpqa_d": DatasetSpec(
        key="gpqa_d",
        display_name="GPQA-D",
        source=GPQA_D_CSV_URL,
        split="test",
        loader_name="csv",
        data_files={"test": GPQA_D_CSV_URL},
    ),
    "supergpqa": DatasetSpec(
        key="supergpqa",
        display_name="SuperGPQA",
        source="m-a-p/SuperGPQA",
        split="train",
        loader_name="m-a-p/SuperGPQA",
    ),
    "theoremqa": DatasetSpec(
        key="theoremqa",
        display_name="TheoremQA",
        source="TIGER-Lab/TheoremQA",
        split="test",
        loader_name="TIGER-Lab/TheoremQA",
    ),
    "bbeh": DatasetSpec(
        key="bbeh",
        display_name="BBEH",
        source="MrLight/bbeh-eval",
        split="train",
        loader_name="MrLight/bbeh-eval",
    ),
}


def resolve_dataset_keys(keys: list[str]) -> list[str]:
    if "all" in keys:
        return list(DATASET_SPECS)
    unknown = sorted(set(keys) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unknown General-Reasoner official datasets: {unknown}")
    return keys


def load_official_dataset(spec: DatasetSpec, cache_dir: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package before downloading official eval data.") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if spec.data_files:
        kwargs["data_files"] = spec.data_files
    return load_dataset(spec.loader_name, split=spec.split, **kwargs)


def materialize_remote_csv(spec: DatasetSpec, output_dir: Path) -> DatasetSpec:
    if spec.loader_name != "csv" or spec.data_files is None:
        return spec

    raw_dir = output_dir / "_raw" / spec.display_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    materialized_files: dict[str, str] = {}
    for split, source in spec.data_files.items():
        source_path = Path(source)
        if source.startswith(("http://", "https://")):
            local_file = raw_dir / source_path.name
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError("Remote CSV materialization requires `requests`.") from exc
            response = requests.get(source, timeout=120)
            response.raise_for_status()
            local_file.write_bytes(response.content)
            materialized_files[split] = str(local_file)
        else:
            materialized_files[split] = source
    return DatasetSpec(
        key=spec.key,
        display_name=spec.display_name,
        source=spec.source,
        split=spec.split,
        loader_name=spec.loader_name,
        data_files=materialized_files,
    )


def write_dataset(spec: DatasetSpec, dataset: Any, output_dir: Path, force: bool) -> dict[str, Any]:
    dataset_dir = output_dir / spec.display_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file = dataset_dir / "test.parquet"
    metadata_file = dataset_dir / "metadata.json"
    if output_file.exists() and not force:
        return {
            "dataset": spec.display_name,
            "status": "skipped",
            "reason": "output exists; pass --force to overwrite",
            "output_file": str(output_file),
        }

    dataset.to_parquet(str(output_file))
    metadata = {
        "dataset": spec.display_name,
        "key": spec.key,
        "source": spec.source,
        "loader_name": spec.loader_name,
        "split": spec.split,
        "rows": len(dataset),
        "columns": list(dataset.column_names),
        "output_file": str(output_file),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": "downloaded", **metadata}


def download_datasets(
    *,
    dataset_keys: list[str],
    output_dir: Path,
    cache_dir: str | None,
    force: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key in resolve_dataset_keys(dataset_keys):
        spec = materialize_remote_csv(DATASET_SPECS[key], output_dir)
        dataset = load_official_dataset(spec, cache_dir)
        results.append(write_dataset(spec, dataset, output_dir, force))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "download_summary.json"
    payload = {
        "output_dir": str(output_dir),
        "datasets": [asdict(DATASET_SPECS[key]) for key in resolve_dataset_keys(dataset_keys)],
        "results": results,
    }
    summary_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=("all", *DATASET_SPECS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = download_datasets(
        dataset_keys=args.datasets,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
