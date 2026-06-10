import unittest
from pathlib import Path


class NotebookEnvironmentSetupTests(unittest.TestCase):
    def test_script_avoids_anaconda_tos_and_registers_kernel(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_notebook_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"', source)
        self.assertIn("--override-channels", source)
        self.assertIn('--channel "${CONDA_CHANNEL}"', source)
        self.assertIn("setup_remote_training_env.sh", source)
        self.assertIn("python -m ipykernel install", source)

    def test_vendored_installer_prefers_packaged_flash_attention_wheel(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "verl"
            / "scripts"
            / "install_vllm_sglang_mcore.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('if [ ! -f "${FLASH_ATTN_WHEEL}" ]', source)
        self.assertIn('pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"', source)


if __name__ == "__main__":
    unittest.main()
