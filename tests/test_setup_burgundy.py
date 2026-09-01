from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class SetupBurgundyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.script_path = cls.repo_root / "scripts" / "setup_burgundy.sh"

    def run_script(
        self,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(self.script_path), *arguments],
            cwd=self.repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_documents_complete_bootstrap_flow(self) -> None:
        result = self.run_script("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("repository -> Conda environment", result.stdout)
        self.assertIn("training data -> models -> verification", result.stdout)
        self.assertIn("--gres=gpu:a100:1", result.stdout)

    def test_dry_run_has_no_filesystem_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            remote_root = home / "scratch" / "opd"
            home.mkdir()

            result = self.run_script(
                "--dry-run",
                extra_env={
                    "HOME": str(home),
                    "REMOTE_ROOT": str(remote_root),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("phases=all", result.stdout)
            self.assertIn("dry_run=1", result.stdout)
            self.assertFalse(remote_root.exists())

    def test_unknown_phase_fails_closed(self) -> None:
        result = self.run_script(
            "--dry-run",
            extra_env={"PHASES": "repo,unknown"},
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported phase: unknown", result.stderr)

    def test_source_never_reads_repo_local_ssh_credentials(self) -> None:
        source = self.script_path.read_text(encoding="utf-8")

        self.assertNotIn("ssh2.sh", source)
        self.assertNotIn("sshpass", source)
        self.assertNotIn("lzhan37", source)
        self.assertIn("Do not place credentials in this script", source)

    def test_existing_asset_scripts_are_reused(self) -> None:
        source = self.script_path.read_text(encoding="utf-8")

        self.assertIn("scripts/setup_training_env.sh", source)
        self.assertIn("scripts/download_training_assets.sh", source)
        self.assertIn("DOWNLOAD_MODELS=1", source)
        self.assertIn("REQUIRE_4DOMAIN_TRAIN_DATA=1", source)
        self.assertIn('PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REMOTE_ROOT}/pip_cache}"', source)

    def test_miniforge_avoids_anaconda_defaults_tos(self) -> None:
        source = self.script_path.read_text(encoding="utf-8")

        self.assertIn("Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh", source)
        self.assertIn("14db468222ad564658656f769506056209b6dc375f5e7dfd31eb5ebbf08fa529", source)
        self.assertIn('CONDA_ROOT="${CONDA_ROOT:-${REMOTE_ROOT}/miniforge3}"', source)
        self.assertNotIn("conda tos accept", source)

    def test_explicit_conda_root_precedes_path_discovery(self) -> None:
        setup_source = (
            self.repo_root / "scripts" / "setup_training_env.sh"
        ).read_text(encoding="utf-8")
        function_source = setup_source.split("find_conda_root() {", maxsplit=1)[1]
        function_source = function_source.split("install_miniconda() {", maxsplit=1)[0]

        explicit_root = function_source.index('if [[ -n "${CONDA_ROOT:-}" ]]')
        path_discovery = function_source.index("if command -v conda")
        self.assertLess(explicit_root, path_discovery)

    def test_private_repo_url_is_not_written_to_dry_run_output(self) -> None:
        secret = "do-not-log-this-token"
        result = self.run_script(
            "--dry-run",
            extra_env={"REPO_URL": f"https://{secret}@example.invalid/repo.git"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)

    def test_write_paths_cannot_escape_scratch_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            remote_root = home / "scratch" / "opd"
            home.mkdir()
            result = self.run_script(
                extra_env={
                    "HOME": str(home),
                    "REMOTE_ROOT": str(remote_root),
                    "CODE_DIR": str(remote_root / "mopd_code"),
                    "MODEL_ROOT": str(home / "models"),
                    "PHASES": "repo",
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("MODEL_ROOT must resolve under REMOTE_ROOT", result.stderr)

    def test_non_a100_partition_fails_before_repository_access(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir) / "home"
            remote_root = home / "scratch" / "opd"
            home.mkdir()
            result = self.run_script(
                extra_env={
                    "HOME": str(home),
                    "REMOTE_ROOT": str(remote_root),
                    "CODE_DIR": str(remote_root / "mopd_code"),
                    "PHASES": "env",
                    "SLURM_JOB_ID": "1",
                    "SLURM_JOB_PARTITION": "batch",
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected gpu_a100 partition", result.stderr)

    def test_initial_clone_uses_temporary_directory(self) -> None:
        source = self.script_path.read_text(encoding="utf-8")

        self.assertIn('mktemp -d "$(dirname "${CODE_DIR}")/.mopd-clone.', source)
        self.assertIn('mv "${clone_dir}" "${CODE_DIR}"', source)


if __name__ == "__main__":
    unittest.main()
