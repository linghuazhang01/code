import unittest
from pathlib import Path


class DownloadMopdDataTests(unittest.TestCase):
    def test_eval_data_comes_from_pinned_official_gopd_sources(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn(
            "GOPD_REPO_URL="
            '"${GOPD_REPO_URL:-https://github.com/RUCBM/G-OPD.git}"',
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
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('DOWNLOAD_LCB="${DOWNLOAD_LCB:-0}"', source)
        self.assertIn(
            'eval_required_files+=("code/data/LiveCodeBench/test.parquet")',
            source,
        )

    def test_parquet_dependencies_use_the_download_python_environment(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn("ensure_parquet_support", source)
        self.assertIn('python -m pip install "pandas>=2.0" "pyarrow>=19.0.0"', source)
        self.assertIn("import pyarrow", source)


if __name__ == "__main__":
    unittest.main()
