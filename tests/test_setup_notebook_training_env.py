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

    def test_remote_setup_installs_m2rl_if_reward_dependencies(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_remote_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")
        requirement_source = (
            Path(__file__).resolve().parents[1] / "requirement.txt"
        ).read_text(encoding="utf-8")

        self.assertIn('INSTALL_M2RL_IF_DEPS="${INSTALL_M2RL_IF_DEPS:-1}"', source)
        self.assertIn('python -m pip install --upgrade -r "${REQUIREMENT_FILE}"', source)
        self.assertIn(
            "git+https://github.com/abukharin-nv/verifiable-instructions.git",
            requirement_source,
        )
        self.assertIn("length_constraints:nth_paragraph_first_word", source)
        self.assertIn("last_word:last_word_answer", source)
        self.assertIn("emoji", requirement_source)
        self.assertIn("syllapy", requirement_source)
        self.assertIn("tensorboard==2.20.0", requirement_source)
        self.assertIn("protobuf<5.0,>=3.20.3", requirement_source)
        self.assertIn("opentelemetry-exporter-prometheus==0.47b0", requirement_source)

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
