import unittest
from pathlib import Path


class DownloadMopdDataTests(unittest.TestCase):
    def test_training_data_defaults_to_versionable_public_hub_dataset(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn(
            'DATASET_ID="${DATASET_ID:-icemoon28/MOPD-Training-Data}"',
            source,
        )
        self.assertIn('DATASET_REVISION="${DATASET_REVISION:-main}"', source)
        self.assertIn("revision=dataset_revision", source)

    def test_eval_data_comes_from_pinned_official_gopd_sources(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn(
            'GOPD_REPO_URL="${GOPD_REPO_URL:-https://github.com/RUCBM/G-OPD.git}"',
            source,
        )
        self.assertIn(
            "37371a4c31ad7947746200d234161769191f4748",
            source,
        )
        self.assertIn("math_eval_jsonl_to_verl_parquet", source)
        self.assertIn("evalplus_jsonl_to_verl_parquet", source)
        self.assertNotIn("${DATA_DIR}/PaperEval/", source)

    def test_livecodebench_is_optional_for_the_h200_profile(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"', source)
        self.assertIn(
            'LCB_REVISION="${LCB_REVISION:-48d36ed304dca42cf8ab20e941262ccd096518a3}"',
            source,
        )
        self.assertIn("2cafe2a842652f6aca997755a8150c348fbe25c040c0fb2ac7e63e400e10e5cb", source)
        self.assertIn("3558c5766089965eda005c39647ccf0b42be2bffe35665fecfaaa90d355b5d59", source)
        self.assertIn("bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5", source)
        self.assertIn('allow_patterns=["v5/*.parquet", "test6.jsonl"]', source)
        self.assertNotIn('source_root.glob("test*.jsonl")', source)
        self.assertIn(
            '"code/LiveCodeBench-v5/test.parquet"',
            source,
        )
        self.assertIn('"code/LiveCodeBench/test.parquet"', source)

    def test_livecodebench_manifest_uses_portable_source_name(self) -> None:
        root = Path(__file__).resolve().parents[1]
        download_source = (root / "scripts" / "download_mopd_data.sh").read_text(
            encoding="utf-8"
        )
        prepare_source = (
            root / "eval" / "scripts" / "prepare_paper_eval_data.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('"name": str(source.relative_to(source_root))', download_source)
        self.assertIn('"name": str(source.relative_to(lcb_root))', prepare_source)
        self.assertNotIn('"name": str(source)', download_source)
        self.assertNotIn('"name": str(source)', prepare_source)

    def test_parquet_dependencies_use_the_download_python_environment(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("ensure_parquet_support", source)
        self.assertIn(
            '"${PYTHON_BIN}" -m pip install "pandas>=2.0" "pyarrow>=19.0.0"', source
        )
        self.assertIn("import pyarrow", source)

    def test_paper_suite_runs_both_lcb_releases_with_k8(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "eval"
            / "scripts"
            / "run_paper_eval_suite.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('MATH_N="${MATH_N:-8}"', source)
        self.assertIn('EVALPLUS_N="${EVALPLUS_N:-8}"', source)
        self.assertIn('LCB_N="${LCB_N:-8}"', source)
        self.assertIn('LCB_RELEASE_VERSIONS="${LCB_RELEASE_VERSIONS:-v5,v6}"', source)
        self.assertIn(
            'MODEL_NAME="${MODEL_NAME:-${MODEL_PARENT_NAME}__${MODEL_BASENAME}}"',
            source,
        )
        self.assertIn(
            'paper_suite/${SAFE_MODEL_NAME}',
            source,
        )
        self.assertIn(
            'LCB_MODEL_STYLE_NAME="${LCB_MODEL_STYLE_NAME:-Qwen3-4B-NonThinking}"',
            source,
        )
        self.assertIn('run_lcb_data_parallel', source)
        self.assertIn('run_lcb_data_parallel.sh', source)
        self.assertIn('--model_path "${MODEL_RUNTIME_PATH}"', source)
        self.assertIn('--checkpoint_path "${MODEL_PATH}"', source)
        self.assertIn('scripts/prepare_eval_model.sh', source)
        self.assertIn('"${MODEL_RUNTIME_PATH}"', source)
        self.assertIn('eval_model_aliases', source)
        self.assertIn('--releases "${LCB_RELEASE_VERSIONS}"', source)
        self.assertIn('--gpus 4', source)
        self.assertIn('--shards_per_dataset 16', source)
        self.assertNotIn('--tensor_parallel_size', source)
        self.assertNotIn('--use_cache', source)
        self.assertIn('LCB_CHAT_TEMPLATE_TOKENIZER="Qwen/Qwen3-4B"', source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn(
            '"livecodebench_chat_template_enable_thinking": False',
            source,
        )

    def test_lcb_manifests_record_gopd_formatter_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "data/eval_data/code/LiveCodeBench-v5/manifest.json",
            "data/eval_data/code/LiveCodeBench/manifest.json",
        ):
            source = (root / relative_path).read_text(encoding="utf-8")
            self.assertIn('"chat_template_tokenizer": "Qwen/Qwen3-4B"', source)
            self.assertIn('"chat_template_enable_thinking": false', source)


if __name__ == "__main__":
    unittest.main()
