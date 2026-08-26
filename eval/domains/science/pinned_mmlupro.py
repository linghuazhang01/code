"""Validate the pinned MMLU-Pro-500 evaluation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_DATA_SHA256 = (
    "9db4fb82f4fc59ab4514b2f3a2fe54928b3fc9d11a483bf678958261b8f6a4a6"
)
EXPECTED_SELECTED_IDS_SHA256 = (
    "ea1c19950afe4ac82a3b32c8afb39b50fa032a64e096fed61364e1d0c1c81760"
)
EXPECTED_DATASET = "MMLU-Pro"
EXPECTED_SAMPLE_SIZE = 500
EXPECTED_SEED = 42


@dataclass(frozen=True)
class PinnedMMLUValidation:
    """Hashes and protocol fields proven by pinned-artifact validation."""

    data_sha256: str
    selected_ids_sha256: str
    dataset: str = EXPECTED_DATASET
    sample_size: int = EXPECTED_SAMPLE_SIZE
    seed: int = EXPECTED_SEED

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable validation record."""

        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_ids_sha256(selected_ids: list[Any]) -> str:
    payload = json.dumps(
        selected_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"MMLU-Pro-500 manifest is missing {label}")
    return value


def validate_mmlupro_500_artifact(
    data_file: Path,
    manifest_file: Path,
    *,
    expected_data_sha256: str = EXPECTED_DATA_SHA256,
    expected_selected_ids_sha256: str = EXPECTED_SELECTED_IDS_SHA256,
) -> PinnedMMLUValidation:
    """Fail unless data and provenance match the pinned MMLU-Pro-500 contract."""

    if not data_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(
            f"Missing pinned MMLU-Pro-500 artifact under {data_file.parent}"
        )

    actual_data_sha256 = _sha256_file(data_file)
    if actual_data_sha256 != expected_data_sha256:
        raise RuntimeError(
            f"MMLU-Pro-500 data SHA-256 mismatch: {actual_data_sha256}"
        )

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid MMLU-Pro-500 manifest JSON: {manifest_file}"
        ) from exc
    manifest = _mapping(manifest, "root object")
    selection = _mapping(manifest.get("selection"), "selection")
    subset = _mapping(manifest.get("subset"), "subset")
    selected_ids = selection.get("selected_ids")
    if not isinstance(selected_ids, list):
        raise TypeError("MMLU-Pro-500 manifest is missing selected_ids")
    actual_selected_ids_sha256 = _selected_ids_sha256(selected_ids)

    expected_fields = {
        "dataset": manifest.get("dataset") == EXPECTED_DATASET,
        "seed": selection.get("seed") == EXPECTED_SEED,
        "sample_size": selection.get("sample_size") == EXPECTED_SAMPLE_SIZE,
        "subset_rows": subset.get("rows") == EXPECTED_SAMPLE_SIZE,
        "subset_sha256": subset.get("sha256") == expected_data_sha256,
        "selected_ids_sha256": (
            actual_selected_ids_sha256 == expected_selected_ids_sha256
            and selection.get("selected_ids_sha256")
            == expected_selected_ids_sha256
        ),
    }
    invalid = [name for name, valid in expected_fields.items() if not valid]
    if invalid:
        raise RuntimeError(
            "Invalid MMLU-Pro-500 manifest fields: " + ", ".join(invalid)
        )

    return PinnedMMLUValidation(
        data_sha256=actual_data_sha256,
        selected_ids_sha256=actual_selected_ids_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = validate_mmlupro_500_artifact(
            args.data_file,
            args.manifest_file,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(validation.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
