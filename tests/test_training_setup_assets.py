import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TrainingSetupAssetScriptTests(unittest.TestCase):
    def test_setup_training_env_syncs_the_single_environment_file(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('ENV_NAME="${ENV_NAME:-mopd-verl}"', source)
        self.assertIn('ENV_FILE="${ENV_FILE:-${CODE_DIR}/environment.yml}"', source)
        self.assertIn('UPDATE_ENV="${UPDATE_ENV:-1}"', source)
        self.assertIn('conda env create --name "${ENV_NAME}" --file "${ENV_FILE}"', source)
        self.assertIn('conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune', source)
        self.assertIn('conda run --no-capture-output -n "${ENV_NAME}"', source)
        self.assertIn('if [[ -n "${CONDA_ROOT:-}" ]]', source)
        self.assertIn("ensure_git_lfs", source)
        self.assertIn('INSTALL_GIT_LFS="${INSTALL_GIT_LFS:-1}"', source)
        self.assertIn("verify_environment", source)
        self.assertNotIn("install_training_deps.sh", source)
        self.assertNotIn("pip install", source)
        self.assertIn("download_training_assets.sh", source)

    def test_environment_yaml_is_the_only_dependency_definition(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment_source = (root / "environment.yml").read_text(encoding="utf-8")
        self.assertFalse((root / "requirement.txt").exists())
        self.assertFalse((root / "scripts" / "install_training_deps.sh").exists())
        for expected in (
            "python=3.10",
            "torch==2.6.0",
            "vllm==0.8.5.post1",
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE",
            "transformers[hf_xet]==4.51.3",
            "tokenizers>=0.21.1,<0.22",
            "huggingface_hub>=0.30.0,<1.0",
            "tensorboard==2.20.0",
            "protobuf>=3.20.3,<5.0",
            "click",
            "modelscope",
            "langdetect",
            "nltk",
            "git+https://github.com/abukharin-nv/verifiable-instructions.git@f46a5ac87b1400a4f8973039844b6be9b56e3faf",
        ):
            self.assertIn(expected, environment_source)

    def test_blackwell_environment_pins_sm120_flash_attention_wheel(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment_data = yaml.safe_load(
            (root / "environment.blackwell.yml").read_text(encoding="utf-8")
        )
        dependencies = environment_data["dependencies"]
        pip_dependencies = next(
            dependency["pip"]
            for dependency in dependencies
            if isinstance(dependency, dict) and "pip" in dependency
        )
        flash_attention_dependencies = [
            dependency
            for dependency in pip_dependencies
            if "flash_attn" in dependency or "flash-attn" in dependency
        ]

        self.assertIn("python=3.10", dependencies)
        self.assertIn("torch==2.8.0+cu128", pip_dependencies)
        self.assertEqual(
            flash_attention_dependencies,
            [
                "https://github.com/Dao-AILab/flash-attention/releases/download/"
                "v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8"
                "cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
            ],
        )

    def test_asset_script_targets_qwen30b_instruct_2507_four_domain_training(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_training_assets.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            "DeepMath-103K/train_filtered_level6.parquet",
            "Eurus/code_train.parquet",
            "IF/train.parquet",
            "Science/train.parquet",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            'REQUIRE_MATH_CODE_TRAIN_DATA="${REQUIRE_MATH_CODE_TRAIN_DATA:-1}"',
            'REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"',
            "PYTHON_BIN",
            "download_mopd_data.sh",
            "download_mopd_models.sh",
            "download_qwen30b_teacher.sh",
            "prepare_m2rl_eval_data.sh",
        ):
            self.assertIn(expected, source)

    def test_download_mopd_data_validates_four_domain_files_by_default(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('REQUIRE_4DOMAIN_TRAIN_DATA="${REQUIRE_4DOMAIN_TRAIN_DATA:-1}"', source)
        self.assertIn('"IF/train.parquet"', source)
        self.assertIn('"Science/train.parquet"', source)

    def test_download_data_has_lfs_pointer_fallback_with_timeout(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_mopd_data.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('PULL_REPO_LFS_FALLBACK="${PULL_REPO_LFS_FALLBACK:-1}"', source)
        self.assertIn('GIT_LFS_TIMEOUT_SECONDS="${GIT_LFS_TIMEOUT_SECONDS:-300}"', source)
        self.assertIn("is_lfs_pointer", source)
        self.assertIn('git -C "${CODE_DIR}" lfs pull', source)

    def test_local_training_script_launches_profile_with_local_env(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_local_mopd_training.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            "scripts/run_local_mopd_training.sh",
            'CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"',
            'ENV_NAME="${ENV_NAME:-mopd-verl}"',
            "MOPD_LOCAL_CONDA_ENV",
            "Local training launch",
            "scripts/run_mopd.sh",
            "CUDA_VISIBLE_DEVICES",
            "GPU_IDLE_MEMORY_LIMIT_MB",
            "REQUIRED_GPUS",
            "SCREEN_SESSION_NAME_MAX_LENGTH=80",
            "config_checksum=",
            "config_hash",
        ):
            self.assertIn(expected, source)

    def test_local_training_default_run_id_fits_screen_limit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script_path = root / "scripts" / "run_local_mopd_training.sh"
        source_config_path = (
            root
            / "test_grad_configs"
            / (
                "mopd_grad_reliability_qwen0p6b_0p6b_aw2_fsdpsize2_"
                "audit_freq2_b16_4step_smoke.yaml"
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_dir = temp_path / "model"
            model_dir.mkdir()
            config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
            config["model"]["student_path"] = str(model_dir)
            config["model"]["math_teacher_path"] = str(model_dir)
            config["model"]["code_teacher_path"] = str(model_dir)
            config["model"]["domain_teacher_paths"] = {
                domain: str(model_dir)
                for domain in config["model"]["domain_teacher_paths"]
            }
            config_path = temp_path / source_config_path.name
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )

            bin_dir = temp_path / "bin"
            bin_dir.mkdir()
            fake_screen = bin_dir / "screen"
            fake_screen.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-ls\" ]]; then exit 0; fi\n"
                "if [[ \"${1:-}\" == \"-dmS\" ]]; then\n"
                "  [[ \"${#2}\" -le 80 ]] || exit 42\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_screen.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "GPU_IDS": "0,1,2",
                    "LOG_DIR": str(temp_path / "logs"),
                    "MOPD_LOCAL_CONDA_ENV": str(temp_path / "missing-env"),
                    "MOPD_LOCAL_CONDA_ROOT": str(temp_path / "missing-conda"),
                    "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{env['PATH']}",
                    "STOP_STALE_RAY": "0",
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(script_path),
                    str(config_path),
                    "--dry-run",
                    "--",
                    "trainer.save_freq=-1",
                ],
                cwd=root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            run_id = (temp_path / "logs" / "opd_target_run_id").read_text(
                encoding="utf-8",
            ).strip()

        self.assertLessEqual(len(run_id), 80)
        self.assertRegex(run_id, r"_[0-9a-f]{8}_[0-9]{8}_[0-9]{6}$")

    def test_grad_config_start_script_is_a_single_config_command(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script_path = root / "test_grad_configs" / "start.sh"
        source = script_path.read_text(encoding="utf-8")

        self.assertEqual(len(source.splitlines()), 1)
        self.assertIn('GPU_IDS="${GPU_IDS:-0,1,2}"', source)
        self.assertIn("scripts/run_local_mopd_training.sh", source)
        self.assertIn('"$1"', source)

    def test_qwen30b_teacher_checks_disk_before_requiring_python(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "download_qwen30b_teacher.sh"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bin_dir = temp_path / "bin"
            bin_dir.mkdir()
            for command_name in ("awk", "df", "dirname", "mkdir"):
                command_path = shutil.which(command_name)
                self.assertIsNotNone(command_path, command_name)
                (bin_dir / command_name).symlink_to(command_path)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(bin_dir),
                    "MODEL_ROOT": str(temp_path / "models"),
                    "MIN_FREE_GB": "999999",
                    "DOWNLOAD_QWEN30B": "0",
                    "REQUIRE_QWEN30B": "0",
                }
            )

            result = subprocess.run(
                ["/bin/bash", str(script_path)],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Insufficient free space", result.stderr)
        self.assertNotIn("python or python3 is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
