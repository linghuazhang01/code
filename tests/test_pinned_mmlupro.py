from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.domains.science.pinned_mmlupro import (
    validate_mmlupro_500_artifact,
)


def _selected_ids_sha256(selected_ids: list[int]) -> str:
    payload = json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    *,
    data_sha256: str,
    selected_ids: list[int],
    seed: int = 42,
) -> str:
    selected_ids_sha256 = _selected_ids_sha256(selected_ids)
    payload = {
        "dataset": "MMLU-Pro",
        "selection": {
            "seed": seed,
            "sample_size": 500,
            "selected_ids": selected_ids,
            "selected_ids_sha256": selected_ids_sha256,
        },
        "subset": {
            "rows": 500,
            "sha256": data_sha256,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return selected_ids_sha256


class PinnedMMLUValidationTests(unittest.TestCase):
    def test_accepts_matching_data_and_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_file = root / "test.parquet"
            manifest_file = root / "manifest.json"
            data_file.write_bytes(b"pinned-data")
            data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
            selected_ids_sha256 = _write_manifest(
                manifest_file,
                data_sha256=data_sha256,
                selected_ids=[7, 11],
            )

            validation = validate_mmlupro_500_artifact(
                data_file,
                manifest_file,
                expected_data_sha256=data_sha256,
                expected_selected_ids_sha256=selected_ids_sha256,
            )

        self.assertEqual(validation.data_sha256, data_sha256)
        self.assertEqual(validation.selected_ids_sha256, selected_ids_sha256)
        self.assertEqual(validation.sample_size, 500)
        self.assertEqual(validation.seed, 42)

    def test_rejects_data_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_file = root / "test.parquet"
            manifest_file = root / "manifest.json"
            data_file.write_bytes(b"changed-data")
            selected_ids_sha256 = _write_manifest(
                manifest_file,
                data_sha256="expected-data-sha",
                selected_ids=[7, 11],
            )

            with self.assertRaisesRegex(RuntimeError, "data SHA-256 mismatch"):
                validate_mmlupro_500_artifact(
                    data_file,
                    manifest_file,
                    expected_data_sha256="expected-data-sha",
                    expected_selected_ids_sha256=selected_ids_sha256,
                )

    def test_rejects_manifest_protocol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_file = root / "test.parquet"
            manifest_file = root / "manifest.json"
            data_file.write_bytes(b"pinned-data")
            data_sha256 = hashlib.sha256(data_file.read_bytes()).hexdigest()
            selected_ids_sha256 = _write_manifest(
                manifest_file,
                data_sha256=data_sha256,
                selected_ids=[7, 11],
                seed=7,
            )

            with self.assertRaisesRegex(RuntimeError, "seed"):
                validate_mmlupro_500_artifact(
                    data_file,
                    manifest_file,
                    expected_data_sha256=data_sha256,
                    expected_selected_ids_sha256=selected_ids_sha256,
                )


if __name__ == "__main__":
    unittest.main()
