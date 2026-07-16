"""Create reproducible paper-scale subsets of large reasoning benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("data/eval_data/greasoner/official")


@dataclass(frozen=True)
class SubsetSpec:
    key: str
    dataset: str
    subset_name: str
    sample_size: int
    seed: int
    id_column: str
    group_columns: tuple[str, ...]
    paper: str
    artifact_url: str
    reproduction: str
    note: str
    expected_source_sha256: str | None = None
    expected_selected_ids_sha256: str | None = None


SUBSET_SPECS: tuple[SubsetSpec, ...] = (
    SubsetSpec(
        key="mmlupro_openprm_style_500",
        dataset="MMLU-Pro",
        subset_name="openprm_style_500_seed42",
        sample_size=500,
        seed=42,
        id_column="question_id",
        group_columns=("category",),
        paper="OpenPRM (ICLR 2025)",
        artifact_url="https://openreview.net/forum?id=fGIqGfmgkW",
        reproduction="protocol_only",
        note=(
            "OpenPRM reports a random 500-example MMLU-Pro test subset but does not "
            "publish its selected IDs or random seed. This is a reproducible "
            "OpenPRM-style sample, not the paper's exact sample."
        ),
    ),
    SubsetSpec(
        key="supergpqa_rsa_1000",
        dataset="SuperGPQA",
        subset_name="rsa_1000_seed42",
        sample_size=1000,
        seed=42,
        id_column="uuid",
        group_columns=("discipline", "difficulty"),
        paper="RSA",
        artifact_url="https://github.com/HyperPotatoNeo/RSA/tree/main/data/supergpqa",
        reproduction="exact",
        note=(
            "The public RSA 1,000-example artifact is reproduced by taking the first "
            "1,000 positions of NumPy RandomState(42).permutation(26529)."
        ),
        expected_source_sha256=(
            "b8541e06e61116ed11253776451b20da809ff489a7ca4af6388b7d846066c5c2"
        ),
        expected_selected_ids_sha256=(
            "00330f9465a13ab14871ebdd8684dd3c18d6e87ab27ea350ca61d6129f8a3b00"
        ),
    ),
)


def select_source_indices(total_rows: int, sample_size: int, seed: int) -> list[int]:
    """Return a deterministic sample without replacement in sampled order."""
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if sample_size <= 0 or sample_size > total_rows:
        raise ValueError("sample_size must be in [1, total_rows]")
    permutation = np.random.RandomState(seed).permutation(total_rows)
    return [int(index) for index in permutation[:sample_size]]


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_ids_sha256(selected_ids: list[str | int | float | bool | None]) -> str:
    """Hash an ordered ID list using a stable compact JSON representation."""
    payload = json.dumps(
        selected_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value_counts(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"Missing grouping column: {column}")
        series = frame[column].fillna("__null__").astype(str).value_counts().sort_index()
        counts[column] = {str(key): int(value) for key, value in series.items()}
    return counts


def create_subset(spec: SubsetSpec, data_root: Path, force: bool) -> dict[str, Any]:
    """Materialize one subset and its complete provenance manifest."""
    source_file = data_root / spec.dataset / "test.parquet"
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing full dataset: {source_file}")

    subset_dir = data_root / spec.dataset / "subsets" / spec.subset_name
    subset_file = subset_dir / "test.parquet"
    manifest_file = subset_dir / "manifest.json"
    source_sha256 = sha256_file(source_file)
    if (
        spec.expected_source_sha256 is not None
        and source_sha256 != spec.expected_source_sha256
    ):
        raise RuntimeError(
            f"Pinned source hash mismatch for exact subset {spec.dataset}: {source_sha256}"
        )

    if subset_file.exists() != manifest_file.exists() and not force:
        raise RuntimeError(
            f"Incomplete subset artifact under {subset_dir}; rerun with --force to regenerate"
        )
    if subset_file.is_file() and manifest_file.is_file() and not force:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("full_dataset", {}).get("sha256") != source_sha256:
            raise RuntimeError(
                f"Full dataset changed for {spec.dataset}; rerun with --force to regenerate"
            )
        if manifest.get("subset", {}).get("sha256") != sha256_file(subset_file):
            raise RuntimeError(f"Subset hash mismatch: {subset_file}")
        expected_selection = {
            "seed": spec.seed,
            "sample_size": spec.sample_size,
            "id_column": spec.id_column,
        }
        actual_selection = manifest.get("selection", {})
        if any(actual_selection.get(key) != value for key, value in expected_selection.items()):
            raise RuntimeError(
                f"Subset specification changed for {subset_dir}; rerun with --force"
            )
        if spec.expected_selected_ids_sha256 is not None:
            selected_ids = actual_selection.get("selected_ids")
            if not isinstance(selected_ids, list):
                raise RuntimeError(f"Missing selected IDs in {manifest_file}")
            if selected_ids_sha256(selected_ids) != spec.expected_selected_ids_sha256:
                raise RuntimeError(f"Pinned selected-ID hash mismatch for {subset_dir}")
        return manifest

    frame = pd.read_parquet(source_file)
    if spec.id_column not in frame.columns:
        raise ValueError(f"Missing ID column in {source_file}: {spec.id_column}")
    if frame[spec.id_column].duplicated().any():
        raise ValueError(f"ID column is not unique in {source_file}: {spec.id_column}")

    source_indices = select_source_indices(len(frame), spec.sample_size, spec.seed)
    subset = frame.iloc[source_indices].reset_index(drop=True)
    selected_ids = [_json_scalar(value) for value in subset[spec.id_column].tolist()]
    ids_sha256 = selected_ids_sha256(selected_ids)
    if (
        spec.expected_selected_ids_sha256 is not None
        and ids_sha256 != spec.expected_selected_ids_sha256
    ):
        raise RuntimeError(f"Pinned selected-ID hash mismatch for {spec.dataset}: {ids_sha256}")

    subset_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(subset_file, index=False)
    manifest = {
        "dataset": spec.dataset,
        "paper_protocol": {
            "paper": spec.paper,
            "artifact_url": spec.artifact_url,
            "reproduction": spec.reproduction,
            "note": spec.note,
        },
        "full_dataset": {
            "file": str(source_file.relative_to(data_root)),
            "rows": len(frame),
            "sha256": source_sha256,
        },
        "selection": {
            "method": "numpy.random.RandomState(seed).permutation",
            "seed": spec.seed,
            "sample_size": spec.sample_size,
            "id_column": spec.id_column,
            "source_indices": source_indices,
            "selected_ids": selected_ids,
            "selected_ids_sha256": ids_sha256,
        },
        "subset": {
            "name": spec.subset_name,
            "file": str(subset_file.relative_to(data_root)),
            "rows": len(subset),
            "sha256": sha256_file(subset_file),
            "columns": list(subset.columns),
            "distributions": _value_counts(subset, spec.group_columns),
        },
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_subsets(
    *,
    keys: list[str],
    data_root: Path,
    force: bool,
) -> dict[str, Any]:
    """Create requested subsets and a compact root-level summary."""
    known_keys = {spec.key for spec in SUBSET_SPECS}
    unknown_keys = sorted(set(keys) - known_keys)
    if unknown_keys:
        raise ValueError(f"Unknown subset specifications: {unknown_keys}")
    manifests: list[dict[str, Any]] = []
    for spec in SUBSET_SPECS:
        subset_dir = data_root / spec.dataset / "subsets" / spec.subset_name
        artifact_exists = (subset_dir / "test.parquet").exists() or (
            subset_dir / "manifest.json"
        ).exists()
        if spec.key in keys or artifact_exists:
            manifests.append(
                create_subset(
                    spec,
                    data_root,
                    force=force if spec.key in keys else False,
                )
            )
    summary = {
        "data_root": str(data_root),
        "subsets": [
            {
                "dataset": manifest["dataset"],
                "paper": manifest["paper_protocol"]["paper"],
                "reproduction": manifest["paper_protocol"]["reproduction"],
                "full_rows": manifest["full_dataset"]["rows"],
                "subset_rows": manifest["subset"]["rows"],
                "subset_file": manifest["subset"]["file"],
                "manifest_file": str(
                    Path(manifest["subset"]["file"]).parent / "manifest.json"
                ),
                "subset_sha256": manifest["subset"]["sha256"],
            }
            for manifest in manifests
        ],
    }
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "subset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=("all", *(spec.key for spec in SUBSET_SPECS)),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys = (
        [spec.key for spec in SUBSET_SPECS]
        if "all" in args.datasets
        else list(dict.fromkeys(args.datasets))
    )
    summary = prepare_subsets(keys=keys, data_root=args.data_root, force=args.force)
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
