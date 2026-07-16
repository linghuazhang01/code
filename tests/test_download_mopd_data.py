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
            'GOPD_REPO_URL="${GOPD_REPO_URL:-http://github.com/RUCBM/G-OPD.git}"',
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
        self.assertIn(
            'LCB_SHA256="${LCB_SHA256:-bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5}"',
            source,
        )
        self.assertIn('allow_patterns=["test6.jsonl"]', source)
        self.assertNotIn('source_root.glob("test*.jsonl")', source)
        self.assertIn(
            'eval_required_files+=("code/LiveCodeBench/test.parquet")',
            source,
        )

    def test_livecodebench_manifest_uses_portable_source_name(self) -> None:
        root = Path(__file__).resolve().parents[1]
        download_source = (root / "scripts" / "download_mopd_data.sh").read_text(
            encoding="utf-8"
        )
        prepare_source = (
            root / "eval" / "scripts" / "prepare_paper_eval_data.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('"source_file": source_paths[0].name', download_source)
        self.assertIn('"source_file": lcb_source.name', prepare_source)
        self.assertNotIn('"source_file": str(source_paths[0])', download_source)
        self.assertNotIn('"source_file": str(lcb_source)', prepare_source)

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


if __name__ == "__main__":
    unittest.main()
