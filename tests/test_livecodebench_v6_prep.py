import base64
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path

import pandas as pd

from eval.data_prep.paper_eval import (
    LCB_RELEASE_FILES,
    lcb_jsonl_to_verl_parquet,
    lcb_source_parquet_to_verl_parquet,
    lcb_source_parquet_to_jsonl,
)


class LiveCodeBenchPrepTests(unittest.TestCase):
    def test_v5_and_v6_use_independent_incremental_sources(self) -> None:
        self.assertEqual(
            LCB_RELEASE_FILES["v5"],
            [
                "v5/test-00000-of-00002.parquet",
                "v5/test-00001-of-00002.parquet",
            ],
        )
        self.assertEqual(LCB_RELEASE_FILES["v6"], ["test6.jsonl"])
        self.assertEqual(len(LCB_RELEASE_FILES["release_v6"]), 6)

    def test_converter_includes_public_and_private_tests(self) -> None:
        record = {
            "question_id": "v6-example",
            "question_title": "Example",
            "question_content": "Read one integer and print it.",
            "starter_code": "",
            "platform": "codeforces",
            "metadata": json.dumps({"func_name": None}),
            "public_test_cases": json.dumps(
                [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
            ),
            "private_test_cases": base64.b64encode(
                zlib.compress(
                    pickle.dumps(
                        json.dumps(
                            [{"input": "2\n", "output": "2\n", "testtype": "stdin"}]
                        )
                    )
                )
            ).decode("utf-8"),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "test6.jsonl"
            output = root / "test.parquet"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")

            count = lcb_jsonl_to_verl_parquet([source], output)
            row = pd.read_parquet(output).iloc[0]
            ground_truth = json.loads(row["reward_model"]["ground_truth"])

            self.assertEqual(count, 1)
            self.assertEqual(row["data_source"], "LiveCodeBench-v6")
            self.assertEqual(ground_truth["inputs"], ["1\n", "2\n"])
            self.assertEqual(ground_truth["outputs"], ["1\n", "2\n"])

    def test_v5_source_parquet_uses_distinct_data_source(self) -> None:
        record = {
            "question_id": "v5-example",
            "question_title": "Example",
            "question_content": "Print one integer.",
            "starter_code": "",
            "platform": "codeforces",
            "metadata": json.dumps({"func_name": None}),
            "public_test_cases": json.dumps(
                [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
            ),
            "private_test_cases": json.dumps([]),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.parquet"
            output = root / "test.parquet"
            pd.DataFrame([record]).to_parquet(source, index=False)

            count = lcb_source_parquet_to_verl_parquet(
                [source],
                output,
                data_source="LiveCodeBench-v5",
            )
            row = pd.read_parquet(output).iloc[0]

        self.assertEqual(count, 1)
        self.assertEqual(row["data_source"], "LiveCodeBench-v5")
        self.assertEqual(row["extra_info"]["validation_dataset"], "LiveCodeBench-v5")

    def test_missing_incremental_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            with self.assertRaisesRegex(FileNotFoundError, "Missing LiveCodeBench source"):
                lcb_source_parquet_to_verl_parquet(
                    [root / "missing.parquet"],
                    root / "output.parquet",
                    data_source="LiveCodeBench-v5",
                )

    def test_v5_runner_compatibility_jsonl_preserves_rows(self) -> None:
        source_records = [
            {"question_id": "one", "question_content": "First"},
            {"question_id": "two", "question_content": "Second"},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source.parquet"
            output = root / "test5.jsonl"
            pd.DataFrame(source_records).to_parquet(source, index=False)

            count = lcb_source_parquet_to_jsonl([source], output)
            decoded = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(count, 2)
        self.assertEqual(decoded, source_records)


if __name__ == "__main__":
    unittest.main()
