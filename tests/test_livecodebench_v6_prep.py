import base64
import json
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path

import pandas as pd

from eval.data_prep.paper_eval import LCB_RELEASE_FILES, lcb_jsonl_to_verl_parquet


class LiveCodeBenchV6PrepTests(unittest.TestCase):
    def test_v6_is_incremental_test6_only(self) -> None:
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
            self.assertEqual(ground_truth["inputs"], ["1\n", "2\n"])
            self.assertEqual(ground_truth["outputs"], ["1\n", "2\n"])


if __name__ == "__main__":
    unittest.main()
