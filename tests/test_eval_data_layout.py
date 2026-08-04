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

    def test_full_training_routes_and_two_model_launchers_are_exposed(self) -> None:
        launcher = (ROOT / "scripts/run_local_eval.sh").read_text(encoding="utf-8")
        expected_routes = {
            "training_full_math": "data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
            "training_full_code": "data/G-OPD-Training-Data/Eurus/code_train.parquet",
            "training_full_if": "data/G-OPD-Training-Data/IF/train.parquet",
            "training_full_science": "data/G-OPD-Training-Data/Science/train.parquet",
        }
        self.assertIn("training_full)", launcher)
        self.assertIn("--sample-offset", launcher)
        self.assertIn("--resume", launcher)
        for dataset_key, relative_path in expected_routes.items():
            with self.subTest(dataset=dataset_key):
                self.assertIn(
                    f'{dataset_key}) relative_paths=("{relative_path}") ;;',
                    launcher,
                )

        scripts = (
            "run_two_model_ood_eval.sh",
            "run_two_model_training_ceiling_eval.sh",
            "run_two_model_full_training_eval.sh",
        )
        for script_name in scripts:
            with self.subTest(script=script_name):
                script_path = ROOT / "scripts" / script_name
                self.assertTrue(script_path.is_file())
                script = script_path.read_text(encoding="utf-8")
                self.assertIn("--tensor-parallel-size 1", script)
                self.assertIn("--save-completions", script)
                self.assertIn("Qwen3-1.7B", script)
                self.assertIn("Nemotron-Research-GooseReason-4B-Instruct", script)
                self.assertIn('MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"', script)
                self.assertIn('BATCH_SIZE="${BATCH_SIZE:-24}"', script)
                self.assertIn('GPU_MEMORY="${GPU_MEMORY:-0.6}"', script)
                self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"', script)
                self.assertIn(
                    'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"',
                    script,
                )
                self.assertIn('MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"', script)
                self.assertIn("--enforce-eager", script)
                self.assertIn("--disable-chunked-prefill", script)

        full_training = (ROOT / "scripts/run_two_model_full_training_eval.sh").read_text(
            encoding="utf-8"
        )
        full_manifest = (ROOT / "eval/full_training_manifest.py").read_text(
            encoding="utf-8"
        )
        ood = (ROOT / "scripts/run_two_model_ood_eval.sh").read_text(encoding="utf-8")
        ceiling = (
            ROOT / "scripts/run_two_model_training_ceiling_eval.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("SHARD_SIZE", full_training)
        self.assertIn("--sample-offset", full_training)
        self.assertIn("CONFIRM_FULL_TRAINING", full_training)
        self.assertIn("MIN_FREE_GB", full_training)
        self.assertIn("suite_manifest.json", full_manifest)
        self.assertIn("10000", full_training)
        self.assertIn("resume_signature", full_manifest)
        self.assertIn("source_sha256", full_manifest)
        self.assertIn('cfg.status != "dry_run"', full_manifest)
        self.assertIn('--seed "${SEED}"', full_training)
        self.assertNotIn('SEED + start', full_training)
        self.assertNotIn(",livecodebench,", ood)
        self.assertIn('config.get("eval_size") != 10_000', ceiling)
        self.assertIn('config.get("seed") != 42', ceiling)
        self.assertIn("eval_sha256", ceiling)


if __name__ == "__main__":
    unittest.main()
