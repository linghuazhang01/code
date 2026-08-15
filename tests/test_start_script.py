"""Behavioral tests for the root local/Slurm training entrypoint."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = ROOT / "start.sh"
TEST_CONFIG = ROOT / "configs" / "mopd_formal_audit_all_2gpu.yaml"


class StartScriptTests(unittest.TestCase):
    """Verify launcher routing without starting Python or submitting jobs."""

    def _run_with_fake_commands(
        self,
        args: Iterable[str],
        *,
        include_sbatch: bool,
        isolated_path: bool = False,
        launch_mode: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            fake_commands = ["bash"]
            if include_sbatch:
                fake_commands.append("sbatch")
            for command_name in fake_commands:
                fake_command = bin_dir / command_name
                fake_command.write_text(
                    "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
                    encoding="utf-8",
                )
                fake_command.chmod(0o755)

            if isolated_path:
                for command_name in ("basename", "dirname"):
                    command_path = shutil.which(command_name)
                    self.assertIsNotNone(command_path, command_name)
                    (bin_dir / command_name).symlink_to(command_path)

            env = os.environ.copy()
            env.pop("MOPD_CONFIG", None)
            env.pop("MOPD_LAUNCH_MODE", None)
            env.pop("SLURM_JOB_ID", None)
            if launch_mode is not None:
                env["MOPD_LAUNCH_MODE"] = launch_mode
            if extra_env is not None:
                env.update(extra_env)
            env["PATH"] = (
                str(bin_dir)
                if isolated_path
                else f"{bin_dir}:{env['PATH']}"
            )
            return subprocess.run(
                ["/bin/bash", str(START_SCRIPT), *args],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_auto_mode_uses_slurm_when_sbatch_is_available(self) -> None:
        result = self._run_with_fake_commands(
            ["--config", str(TEST_CONFIG)],
            include_sbatch=True,
            launch_mode="auto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=slurm", forwarded)
        self.assertIn(str(ROOT / "scripts" / "run_mopd.sh"), forwarded)
        self.assertIn(str(TEST_CONFIG), forwarded)
        self.assertIn("--slurm", forwarded)
        self.assertIn("++data.seed=42", forwarded)
        self.assertIn("++actor_rollout_ref.rollout.seed=42", forwarded)
        self.assertIn("++trainer.seed=42", forwarded)

    def test_default_mode_stays_local_when_sbatch_is_available(self) -> None:
        result = self._run_with_fake_commands(
            ["--config", str(TEST_CONFIG)],
            include_sbatch=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=local", forwarded)
        self.assertIn(
            str(ROOT / "scripts" / "run_local_mopd_training.sh"),
            forwarded,
        )
        self.assertNotIn("--slurm", forwarded)

    def test_auto_mode_does_not_resubmit_inside_slurm_allocation(self) -> None:
        result = self._run_with_fake_commands(
            ["--config", str(TEST_CONFIG)],
            include_sbatch=True,
            launch_mode="auto",
            extra_env={"SLURM_JOB_ID": "12345"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=local", forwarded)
        self.assertNotIn("--slurm", forwarded)

    def test_custom_seed_is_forwarded_in_local_and_slurm_modes(self) -> None:
        cases = (
            (["--local"], False),
            (["--slurm"], True),
        )
        for mode_args, include_sbatch in cases:
            with self.subTest(mode_args=mode_args):
                result = self._run_with_fake_commands(
                    ["--config", str(TEST_CONFIG), *mode_args],
                    include_sbatch=include_sbatch,
                    extra_env={"MOPD_SEED": "123"},
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                forwarded = result.stdout.splitlines()
                self.assertIn("++data.seed=123", forwarded)
                self.assertIn(
                    "++actor_rollout_ref.rollout.seed=123",
                    forwarded,
                )
                self.assertIn("++trainer.seed=123", forwarded)

    def test_auto_mode_uses_local_launcher_without_sbatch(self) -> None:
        result = self._run_with_fake_commands(
            ["--config", str(TEST_CONFIG)],
            include_sbatch=False,
            isolated_path=True,
            launch_mode="auto",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=local", forwarded)
        self.assertIn(
            str(ROOT / "scripts" / "run_local_mopd_training.sh"),
            forwarded,
        )
        self.assertNotIn("--slurm", forwarded)

    def test_explicit_local_mode_overrides_sbatch_and_forwards_hydra(self) -> None:
        result = self._run_with_fake_commands(
            [
                "--config",
                str(TEST_CONFIG),
                "--local",
                "--",
                "trainer.total_epochs=1",
            ],
            include_sbatch=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=local", forwarded)
        self.assertNotIn("--slurm", forwarded)
        separator_index = forwarded.index("--")
        self.assertEqual(
            forwarded[separator_index + 1],
            "trainer.total_epochs=1",
        )

    def test_slurm_dry_run_renders_on_login_node_without_submitting(self) -> None:
        result = self._run_with_fake_commands(
            [
                "--config",
                str(TEST_CONFIG),
                "--slurm",
                "--dry-run",
                "--",
                "trainer.total_epochs=1",
            ],
            include_sbatch=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        forwarded = result.stdout.splitlines()
        self.assertIn("LAUNCH_MODE=slurm", forwarded)
        self.assertIn("--dry-run", forwarded)
        self.assertIn("--slurm", forwarded)
        separator_index = forwarded.index("--")
        self.assertEqual(
            forwarded[separator_index + 1],
            "trainer.total_epochs=1",
        )

    def test_run_mopd_slurm_dry_run_generates_script_without_sbatch(self) -> None:
        run_mopd_script = ROOT / "scripts" / "run_mopd.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_dir = temp_path / "slurm"
            fake_python = temp_path / "fake-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"${1:-}\" = \"-\" ]; then\n"
                "  printf 'dry_run\\t2\\t8\\t0,1\\t0\\t/opt/mopd-env/bin\\n'\n"
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            sbatch_marker = temp_path / "sbatch-called"
            fake_sbatch = temp_path / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/sh\n"
                f"touch {shlex.quote(str(sbatch_marker))}\n"
                "echo 'Submitted batch job 12345'\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "MOPD_LAUNCH_PYTHON": str(fake_python),
                    "PATH": f"{temp_path}:{env['PATH']}",
                    "SLURM_LOG_DIR": str(log_dir),
                }
            )

            result = subprocess.run(
                [
                    "/bin/bash",
                    str(run_mopd_script),
                    str(TEST_CONFIG),
                    "--slurm",
                    "--dry-run",
                    "--slurm-args",
                    "--partition=debug",
                    "--",
                    "trainer.total_epochs=1",
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(sbatch_marker.exists())
            generated_scripts = list(log_dir.glob("mopd_dry_run_*.sbatch"))
            self.assertEqual(len(generated_scripts), 1)
            source = generated_scripts[0].read_text(encoding="utf-8")

        self.assertIn("#SBATCH --partition=debug", source)
        self.assertIn("--dry-run", source)
        self.assertIn("trainer.total_epochs=1", source)
        self.assertIn(
            "export PATH=/opt/mopd-env/bin:${PATH:-}",
            source,
        )
        self.assertIn('${CUDA_VISIBLE_DEVICES:-}', source)
        self.assertIn("unset ROCR_VISIBLE_DEVICES", source)

    def test_run_mopd_rejects_multiline_slurm_directive(self) -> None:
        result = subprocess.run(
            [
                "/bin/bash",
                str(ROOT / "scripts" / "run_mopd.sh"),
                str(TEST_CONFIG),
                "--slurm",
                "--dry-run",
                "--slurm-args",
                "--partition=debug\nmalicious-command",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid Slurm directive", result.stderr)

    def test_run_mopd_slurm_derives_path_from_runtime_python(self) -> None:
        run_mopd_script = ROOT / "scripts" / "run_mopd.sh"
        cases = (
            (
                "/opt/mopd-env/bin/python",
                "export PATH=/opt/mopd-env/bin:${PATH:-}",
            ),
            (
                ".venv/bin/python",
                "export PATH=.venv/bin:${PATH:-}",
            ),
            ("python", None),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for index, (python_bin, expected_export) in enumerate(cases):
                with self.subTest(python_bin=python_bin):
                    config_path = temp_path / f"runtime-path-{index}.yaml"
                    config_path.write_text(
                        "runtime:\n"
                        f"  python_bin: {python_bin}\n"
                        "trainer:\n"
                        f"  experiment_name: runtime-path-{index}\n"
                        "  n_gpus_per_node: 1\n"
                        "  nnodes: 1\n"
                        "worker_placement:\n"
                        "  separate_ref_policy: false\n"
                        "  actor_rollout:\n"
                        "    n_gpus_per_node: 1\n"
                        "    nnodes: 1\n"
                        "ray_kwargs:\n"
                        "  ray_init:\n"
                        "    num_cpus: 8\n",
                        encoding="utf-8",
                    )
                    log_dir = temp_path / f"slurm-{index}"
                    env = os.environ.copy()
                    env.update(
                        {
                            "MOPD_LAUNCH_PYTHON": sys.executable,
                            "SLURM_LOG_DIR": str(log_dir),
                        }
                    )
                    result = subprocess.run(
                        [
                            "/bin/bash",
                            str(run_mopd_script),
                            str(config_path),
                            "--slurm",
                            "--dry-run",
                            "--",
                            "trainer.total_epochs=1",
                        ],
                        cwd=ROOT,
                        env=env,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    generated_scripts = list(log_dir.glob("*.sbatch"))
                    self.assertEqual(len(generated_scripts), 1)
                    source = generated_scripts[0].read_text(encoding="utf-8")

                    if expected_export is None:
                        self.assertNotIn("\nexport PATH=", source)
                    else:
                        self.assertIn(expected_export, source)

    def test_run_mopd_rejects_unsafe_slurm_directive_contracts(self) -> None:
        cases = (
            (
                ["--slurm-args", "--partition=debug"],
                "requires --slurm",
            ),
            (
                ["--slurm", "--dry-run", "--slurm-args", "--nodes=2"],
                "cannot be overridden",
            ),
            (
                ["--slurm", "--dry-run", "--slurm-args", "--export=NONE"],
                "cannot be overridden",
            ),
        )
        for args, expected_error in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    [
                        "/bin/bash",
                        str(ROOT / "scripts" / "run_mopd.sh"),
                        str(TEST_CONFIG),
                        *args,
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)

    def test_root_launcher_rejects_ambiguous_arguments(self) -> None:
        cases = (
            (
                ["--config", str(TEST_CONFIG), "--local", "--slurm"],
                "cannot be used together",
            ),
            (
                ["--config", str(TEST_CONFIG), str(TEST_CONFIG)],
                "Only one config path is allowed",
            ),
            (
                ["--config", f"{TEST_CONFIG}::"],
                "Config profile cannot be empty",
            ),
        )
        for args, expected_error in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    ["/bin/bash", str(START_SCRIPT), *args],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)

    def test_slurm_mode_rejects_local_only_arguments(self) -> None:
        result = subprocess.run(
            [
                "/bin/bash",
                str(START_SCRIPT),
                "--config",
                str(TEST_CONFIG),
                "--slurm",
                "--foreground",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("is local-only", result.stderr)


if __name__ == "__main__":
    unittest.main()
