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

    def test_training_ceiling_is_exposed_by_public_launcher(self) -> None:
        launcher = (ROOT / "scripts/run_local_eval.sh").read_text(encoding="utf-8")
        readme = (ROOT / "eval/README.md").read_text(encoding="utf-8")
        readme_zh = (ROOT / "eval/README.zh.md").read_text(encoding="utf-8")

        expected_routes = {
            "training_math": "data/eval_training_data/math/test.parquet",
            "training_code": "data/eval_training_data/code/test.parquet",
            "training_if": "data/eval_training_data/if/test.parquet",
            "training_science": "data/eval_training_data/science/test.parquet",
        }
        self.assertIn("training_ceiling)", launcher)
        for dataset_key, relative_path in expected_routes.items():
            with self.subTest(dataset=dataset_key):
                self.assertIn(
                    f'{dataset_key}) relative_paths=("{relative_path}") ;;',
                    launcher,
                )

        self.assertIn("--eval-size 10000 --seed 42 --overwrite", launcher)
        self.assertIn("NEEDS_TRAINING_CODE_SCORER", launcher)
        self.assertIn("simple Math fallback is disabled", launcher)
        self.assertIn("NEEDS_IF_SCORER", launcher)
        self.assertIn("scripts/prepare_ifbench_runtime.sh", launcher)

        for document in (readme, readme_zh):
            self.assertIn("--datasets training_ceiling", document)
            self.assertIn("10,000", document)
            self.assertIn("training-data", document)
            self.assertIn("performance", document)


if __name__ == "__main__":
    unittest.main()
