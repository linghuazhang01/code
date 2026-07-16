from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval.domains.greasoner.prepare_subsets import (
    SubsetSpec,
    create_subset,
    prepare_subsets,
    select_source_indices,
)


class GeneralReasonerSubsetTests(unittest.TestCase):
    def test_select_source_indices_matches_randomstate_permutation(self) -> None:
        expected = np.random.RandomState(42).permutation(20)[:5].tolist()

        actual = select_source_indices(total_rows=20, sample_size=5, seed=42)

        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(set(actual)))

    def test_create_subset_records_reproducible_provenance(self) -> None:
        spec = SubsetSpec(
            key="fixture_5",
            dataset="Fixture",
            subset_name="fixture_5_seed42",
            sample_size=5,
            seed=42,
            id_column="uuid",
            group_columns=("discipline",),
            paper="Fixture paper",
            artifact_url="https://example.com/artifact",
            reproduction="exact",
            note="Test fixture.",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            dataset_dir = data_root / spec.dataset
            dataset_dir.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "uuid": [f"id-{index}" for index in range(12)],
                    "question": [f"question-{index}" for index in range(12)],
                    "discipline": ["science", "engineering"] * 6,
                }
            )
            frame.to_parquet(dataset_dir / "test.parquet", index=False)

            manifest = create_subset(spec, data_root, force=False)

            subset_file = data_root / manifest["subset"]["file"]
            subset = pd.read_parquet(subset_file)
            expected_indices = select_source_indices(12, 5, 42)
            self.assertEqual(manifest["selection"]["source_indices"], expected_indices)
            self.assertEqual(
                manifest["selection"]["selected_ids"],
                [f"id-{index}" for index in expected_indices],
            )
            self.assertEqual(subset.columns.tolist(), frame.columns.tolist())
            self.assertEqual(len(subset), 5)
            self.assertEqual(sum(manifest["subset"]["distributions"]["discipline"].values()), 5)

            manifest_file = subset_file.parent / "manifest.json"
            stored_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(stored_manifest, manifest)
            self.assertEqual(create_subset(spec, data_root, force=False), manifest)

    def test_invalid_sample_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_source_indices(total_rows=3, sample_size=4, seed=42)

    def test_selective_run_preserves_existing_subsets_in_summary(self) -> None:
        specs = tuple(
            SubsetSpec(
                key=f"fixture_{name}",
                dataset=name,
                subset_name="sample_2_seed42",
                sample_size=2,
                seed=42,
                id_column="uuid",
                group_columns=("discipline",),
                paper=f"{name} paper",
                artifact_url="https://example.com/artifact",
                reproduction="protocol_only",
                note="Test fixture.",
            )
            for name in ("FixtureA", "FixtureB")
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            for spec in specs:
                dataset_dir = data_root / spec.dataset
                dataset_dir.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "uuid": [f"{spec.dataset}-{index}" for index in range(4)],
                        "discipline": ["science", "engineering"] * 2,
                    }
                ).to_parquet(dataset_dir / "test.parquet", index=False)

            with patch(
                "eval.domains.greasoner.prepare_subsets.SUBSET_SPECS",
                specs,
            ):
                prepare_subsets(
                    keys=[spec.key for spec in specs],
                    data_root=data_root,
                    force=False,
                )
                summary = prepare_subsets(
                    keys=[specs[1].key],
                    data_root=data_root,
                    force=False,
                )

            self.assertEqual(
                [item["dataset"] for item in summary["subsets"]],
                ["FixtureA", "FixtureB"],
            )


if __name__ == "__main__":
    unittest.main()
