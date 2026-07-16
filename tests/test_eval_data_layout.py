from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EvalDataLayoutTests(unittest.TestCase):
    def test_science_and_if_use_dataset_directories(self) -> None:
        expected_files = (
            "data/eval_data/science/GPQA/test.parquet",
            "data/eval_data/science/HLE/test.parquet",
            "data/eval_data/science/MMLU-Pro/test.parquet",
            "data/eval_data/science/SuperGPQA/test.parquet",
            "data/eval_data/if/IFBench/test.parquet",
            "data/eval_data/if/IFEval/test.parquet",
        )

        for relative_path in expected_files:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_legacy_eval_directories_are_removed(self) -> None:
        self.assertFalse((ROOT / "data/eval_data/ifbench").exists())
        self.assertFalse((ROOT / "data/eval_data/greasoner").exists())
        self.assertFalse((ROOT / "eval/domains/greasoner").exists())


if __name__ == "__main__":
    unittest.main()
