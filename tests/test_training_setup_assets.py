import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TrainingSetupAssetScriptTests(unittest.TestCase):
    def test_setup_training_env_creates_conda_env_and_runs_remote_setup(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('ENV_NAME="${ENV_NAME:-mopd-verl}"', source)
        self.assertIn('PYTHON_VERSION="${PYTHON_VERSION:-3.10}"', source)
        self.assertIn("--override-channels", source)
        self.assertIn('--channel "${CONDA_CHANNEL}"', source)
        self.assertIn('conda run --no-capture-output -n "${ENV_NAME}"', source)
        self.assertIn('if [[ -n "${CONDA_ROOT:-}" ]]', source)
        self.assertIn("ensure_git_lfs", source)
        self.assertIn('INSTALL_GIT_LFS="${INSTALL_GIT_LFS:-1}"', source)
        self.assertIn('REQUIREMENT_FILE="${REQUIREMENT_FILE:-${CODE_DIR}/requirement.txt}"', source)
        self.assertIn("setup_remote_training_env.sh", source)
        self.assertIn("download_training_assets.sh", source)

    def test_remote_setup_installs_dependencies_from_requirements_files(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_remote_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('REQUIREMENT_FILE="${REQUIREMENT_FILE:-${CODE_DIR}/requirement.txt}"', source)
        self.assertIn('python -m pip install --upgrade -r "${REQUIREMENT_FILE}"', source)
        self.assertIn("IF/science dependencies are missing after installing", source)
        self.assertIn('if os.environ["INSTALL_VERL_DEPS"] == "1"', source)

        requirement_source = (
            Path(__file__).resolve().parents[1] / "requirement.txt"
        ).read_text(encoding="utf-8")
        for expected in (
            "transformers[hf_xet]==4.51.3",
            "tokenizers>=0.21.1,<0.22",
            "huggingface_hub>=0.30.0,<1.0",
            "tensorboard==2.20.0",
            "protobuf<5.0,>=3.20.3",
            "click",
            "modelscope",
            "langdetect",
            "nltk",
            "git+https://github.com/abukharin-nv/verifiable-instructions.git",
        ):
            self.assertIn(expected, requirement_source)

    def test_asset_script_targets_qwen30b_four_domain_training(self) -> None:
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
            "Qwen/Qwen3-30B-A3B",
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

    def test_notebook_setup_runs_remote_setup_inside_target_env(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "setup_notebook_training_env.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        self.assertIn('conda run --no-capture-output -n "${ENV_NAME}"', source)
        self.assertIn('bash "${SCRIPT_DIR}/setup_remote_training_env.sh"', source)
        self.assertIn('REQUIREMENT_FILE="${REQUIREMENT_FILE:-${CODE_DIR}/requirement.txt}"', source)

    def test_bootstrap_script_clones_assets_and_launches_profile_overrides(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "bootstrap_qwen30b_mopd_training.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            'REPO_URL="${REPO_URL:-http://github.com/linghuazhang01/code.git}"',
            'REPO_REF="${REPO_REF:-bowen}"',
            'BUNDLE_ZIP="${BUNDLE_ZIP:-}"',
            "unpack_bundle",
            "git clone",
            "STEP 1/4 git clone/update",
            "STEP 1/4 code/data bundle unpack",
            "STEP 2/4 environment install",
            "STEP 3/4 asset preparation",
            "STEP 4/4 launch training",
            'GIT_TIMEOUT_SECONDS="${GIT_TIMEOUT_SECONDS:-300}"',
            'GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"',
            'GIT_CLONE_DEPTH="${GIT_CLONE_DEPTH:-1}"',
            "clone_args=(clone)",
            'run_git "${clone_args[@]}"',
            "--single-branch",
            "scripts/setup_training_env.sh",
            "scripts/download_training_assets.sh",
            "configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_fsdp.yaml",
            'GPU_PROFILE="4gpu"',
            'PROFILE_ACTOR_GPUS=3',
            'PROFILE_REF_GPUS=1',
            'PROFILE_TRAIN_BATCH_SIZE=384',
            'GPU_PROFILE="8gpu"',
            'PROFILE_ACTOR_GPUS=6',
            'PROFILE_REF_GPUS=2',
            'PROFILE_TRAIN_BATCH_SIZE=768',
            'PROFILE_ROLLOUT_TP=1',
            'CONDA_ROOT="${CONDA_ROOT:-/root/autodl-tmp/opd_mopd/miniconda3}"',
            'MODEL_BACKEND="${MODEL_BACKEND:-modelscope}"',
            'MIN_FREE_GB="${MIN_FREE_GB:-100}"',
            'DOWNLOAD_DATA=0',
            "actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node",
            "actor_rollout_ref.worker_placement.ref_policy.n_gpus_per_node",
            "scripts/start_remote_mopd_training.sh",
        ):
            self.assertIn(expected, source)

    def test_package_script_bundles_code_and_required_data_without_models(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "package_qwen30b_mopd_bundle.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            "BUNDLE_ROOT_NAME",
            "DeepMath-103K/train_filtered_level6.parquet",
            "Eurus/code_train.parquet",
            "IF/train.parquet",
            "Science/train.parquet",
            "AIME24/test.parquet",
            "HumanEvalPlus/test.parquet",
            "is_lfs_pointer",
            "--exclude '.git/'",
            "--exclude 'models/'",
            "--exclude 'checkpoints/'",
            "zip -qr",
        ):
            self.assertIn(expected, source)

    def test_zip_deploy_script_uploads_bundle_and_runs_remote_bootstrap(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "deploy_qwen30b_mopd_zip_remote.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            "package_qwen30b_mopd_bundle.sh",
            "scp",
            "BUNDLE_ZIP",
            "REMOTE_WORKDIR",
            "MODEL_BACKEND=modelscope",
            "DOWNLOAD_DATA=0",
            "DOWNLOAD_MODELS=1",
            "INSTALL_VERL_DEPS",
            "BUNDLE_REPLACE_EXISTING",
            "bootstrap_qwen30b_mopd_training.sh",
        ):
            self.assertIn(expected, source)

    def test_full_zip_dryrun_script_resets_remote_then_runs_complete_flow(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_qwen30b_mopd_zip_full_dryrun_remote.sh"
        )
        source = script_path.read_text(encoding="utf-8")

        for expected in (
            "REMOTE_WORKDIR=/root/autodl-tmp/opd_mopd_full_dryrun",
            "RESET_REMOTE=1",
            "rm -rf",
            "/root/autodl-tmp/opd_*",
            "deploy_qwen30b_mopd_zip_remote.sh",
            "DRY_RUN=1",
            "INSTALL_ENV=1",
            "DOWNLOAD_DATA=0",
            "DOWNLOAD_MODELS=1",
            "REQUIRE_MODELS=1",
            "MODEL_BACKEND=modelscope",
        ):
            self.assertIn(expected, source)

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
