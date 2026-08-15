from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from mopd_verl.code_reward import _input_output_score, _run_docker_assert_case


class DockerCodeSandboxTest(unittest.TestCase):
    @patch("mopd_verl.code_reward.subprocess.run")
    def test_pass_uses_restricted_ephemeral_container(self, run_mock: object) -> None:
        run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0)  # type: ignore[attr-defined]

        passed, metadata = _run_docker_assert_case("assert 1 + 1 == 2")

        self.assertTrue(passed)
        self.assertEqual(metadata["sandbox"], "docker")
        command = run_mock.call_args.args[0]  # type: ignore[attr-defined]
        self.assertIn("--interactive", command)
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--memory=512m", command)
        self.assertIn("--cpus=1", command)
        self.assertFalse(any("/home/" in argument for argument in command))

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_failure_is_reported_without_host_fallback(self, run_mock: object) -> None:
        run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=1)  # type: ignore[attr-defined]

        passed, metadata = _run_docker_assert_case("raise RuntimeError")

        self.assertFalse(passed)
        self.assertEqual(metadata["error"], "assertion_failed")
        self.assertEqual(metadata["returncode"], 1)

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_docker_infrastructure_error_aborts_scoring(self, run_mock: object) -> None:
        run_mock.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
            args=[],
            returncode=125,
            stdout="",
            stderr="daemon unavailable",
        )

        with self.assertRaisesRegex(RuntimeError, "returncode=125"):
            _run_docker_assert_case("assert True")

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_resource_limit_exit_is_scored_as_failure(self, run_mock: object) -> None:
        for returncode in (137, 152):
            with self.subTest(returncode=returncode):
                run_mock.return_value = subprocess.CompletedProcess(  # type: ignore[attr-defined]
                    args=[],
                    returncode=returncode,
                    stdout="",
                    stderr="",
                )
                passed, metadata = _run_docker_assert_case("while True: pass")
                self.assertFalse(passed)
                self.assertEqual(metadata["error"], "sandboxed_program_failure")

    @patch.dict("os.environ", {"MOPD_CODE_SANDBOX": "docker"})
    def test_livecodebench_is_rejected_in_docker_mode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "LiveCodeBench scoring is disabled"):
            _input_output_score("pass", {})

    @patch("mopd_verl.code_reward._remove_docker_container")
    @patch("mopd_verl.code_reward.subprocess.run")
    def test_timeout_force_removes_named_container(
        self,
        run_mock: object,
        remove_mock: object,
    ) -> None:
        run_mock.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=15)  # type: ignore[attr-defined]

        passed, metadata = _run_docker_assert_case("while True: pass")

        self.assertFalse(passed)
        self.assertEqual(metadata["error"], "timeout")
        remove_mock.assert_called_once()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
