import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval.data_prep.training_holdout import (
    DomainSpec,
    HoldoutConfig,
    create_all_holdouts,
)


class TrainingHoldoutTests(unittest.TestCase):
    def _write_source(self, path: Path) -> None:
        rows = [
            {
                "row_id": 0,
                "data_source": "taco",
                "prompt": [{"role": "user", "content": "duplicate   problem\n"}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": "tests-a"},
                "extra_info": {"index": 0, "split": "train"},
            },
            {
                "row_id": 1,
                "data_source": "codecontests",
                "prompt": [{"role": "user", "content": "duplicate problem"}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": "tests-b"},
                "extra_info": {"index": 0, "split": "train"},
            },
            {
                "row_id": 2,
                "data_source": "taco",
                "prompt": [{"role": "user", "content": "problem two"}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": "tests-c"},
                "extra_info": {"index": 0, "split": "train"},
            },
            {
                "row_id": 3,
                "data_source": "taco",
                "prompt": [{"role": "user", "content": "problem three"}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": "tests-d"},
                "extra_info": {"index": 0, "split": "train"},
            },
        ]
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_split_is_deterministic_grouped_and_schema_preserving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "sources"
            source = source_root / "Eurus/code_train.parquet"
            self._write_source(source)
            source_hash = source.read_bytes()
            spec = DomainSpec("code", "Eurus/code_train.parquet", "code")
            config = HoldoutConfig(
                eval_size=2,
                seed=42,
                batch_size=2,
                write_remainder=True,
            )

            first_results = create_all_holdouts(
                source_root,
                root / "eval-one",
                [spec],
                config,
                remainder_root=root / "train-one",
            )
            second_results = create_all_holdouts(
                source_root,
                root / "eval-two",
                [spec],
                config,
                remainder_root=root / "train-two",
            )

            first = first_results[0]
            second = second_results[0]
            eval_table = pq.read_table(first.eval_path)
            train_table = pq.read_table(first.remainder_path)
            eval_prompts = {
                json.dumps(value, sort_keys=True)
                for value in eval_table["prompt"].to_pylist()
            }
            train_prompts = {
                json.dumps(value, sort_keys=True)
                for value in train_table["prompt"].to_pylist()
            }

            self.assertFalse(eval_prompts & train_prompts)
            self.assertEqual(first.eval_rows + first.remainder_rows, 4)
            self.assertEqual(first.unique_prompt_groups, 3)
            self.assertEqual(first.duplicate_rows, 1)
            self.assertEqual(first.selected_group_sha256, second.selected_group_sha256)
            self.assertEqual(
                eval_table.to_pylist(), pq.read_table(second.eval_path).to_pylist()
            )
            self.assertEqual(
                pq.ParquetFile(source).schema_arrow,
                pq.ParquetFile(first.eval_path).schema_arrow,
            )
            self.assertEqual(source.read_bytes(), source_hash)

    def test_existing_manifest_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "sources"
            self._write_source(source_root / "Eurus/code_train.parquet")
            output_root = root / "eval"
            output_root.mkdir()
            (output_root / "manifest.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_all_holdouts(
                    source_root,
                    output_root,
                    [DomainSpec("code", "Eurus/code_train.parquet", "code")],
                    HoldoutConfig(eval_size=1),
                )

    def test_all_destinations_are_checked_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_root = root / "sources"
            self._write_source(source_root / "first/train.parquet")
            self._write_source(source_root / "second/train.parquet")
            output_root = root / "eval"
            conflicting_output = output_root / "second/test.parquet"
            conflicting_output.parent.mkdir(parents=True)
            conflicting_output.write_bytes(b"existing")
            specs = [
                DomainSpec("first", "first/train.parquet", "first"),
                DomainSpec("second", "second/train.parquet", "second"),
            ]

            with self.assertRaises(FileExistsError):
                create_all_holdouts(
                    source_root,
                    output_root,
                    specs,
                    HoldoutConfig(eval_size=1),
                )

            self.assertFalse((output_root / "first/test.parquet").exists())
            self.assertEqual(conflicting_output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
