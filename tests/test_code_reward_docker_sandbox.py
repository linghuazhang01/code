from __future__ import annotations

import subprocess
import unittest
from unittest.mock import Mock, patch

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

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_cleanup_short_timeout_retries_only_the_scoring_container(self, run_mock: Mock) -> None:
        run_mock.side_effect = [
            subprocess.TimeoutExpired(cmd="docker run", timeout=15),
            subprocess.TimeoutExpired(cmd="docker rm", timeout=5),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="container-id\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="removed\n", stderr=""),
        ]

        passed, metadata = _run_docker_assert_case("while True: pass")

        self.assertFalse(passed)
        self.assertEqual(metadata, {"error": "timeout", "sandbox": "docker"})
        run_call, first_rm, lookup, second_rm = run_mock.call_args_list
        command = run_call.args[0]
        name = command[command.index("--name") + 1]
        self.assertEqual(run_call.kwargs["timeout"], 15)
        self.assertEqual(first_rm.args[0], ["docker", "rm", "-f", name])
        self.assertEqual(first_rm.kwargs["timeout"], 5)
        self.assertEqual(second_rm.args[0], first_rm.args[0])
        self.assertEqual(second_rm.kwargs["timeout"], 15)
        self.assertEqual(
            lookup.args[0],
            ["docker", "ps", "--all", "--quiet", "--filter", f"name=^/{name}$"],
        )

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_cleanup_race_requires_successful_absence_confirmation(self, run_mock: Mock) -> None:
        for removal in (
            subprocess.TimeoutExpired(cmd="docker rm", timeout=5),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="No such container"),
        ):
            with self.subTest(removal=removal):
                run_mock.reset_mock()
                run_mock.side_effect = [
                    subprocess.TimeoutExpired(cmd="docker run", timeout=15),
                    removal,
                    subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
                ]

                passed, metadata = _run_docker_assert_case("while True: pass")

                self.assertFalse(passed)
                self.assertEqual(metadata["error"], "timeout")
                self.assertEqual(run_mock.call_count, 3)

    @patch("mopd_verl.code_reward.subprocess.run")
    def test_unconfirmed_cleanup_aborts_instead_of_scoring_timeout(self, run_mock: Mock) -> None:
        cases = (
            (
                subprocess.TimeoutExpired(cmd="docker rm", timeout=5),
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="daemon unavailable"),
            ),
            (
                subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="removal failed"),
                subprocess.CompletedProcess(args=[], returncode=0, stdout="container-id\n", stderr=""),
            ),
            (
                subprocess.TimeoutExpired(cmd="docker rm", timeout=5),
                subprocess.TimeoutExpired(cmd="docker ps", timeout=5),
            ),
            (FileNotFoundError("docker"), FileNotFoundError("docker")),
        )
        for removal, lookup in cases:
            with self.subTest(removal=removal, lookup=lookup):
                run_mock.reset_mock()
                run_mock.side_effect = [
                    subprocess.TimeoutExpired(cmd="docker run", timeout=15),
                    *[result for _ in range(3) for result in (removal, lookup)],
                ]

                with self.assertRaisesRegex(RuntimeError, "cleanup could not be confirmed.*container='mopd-code-"):
                    _run_docker_assert_case("while True: pass")

                self.assertEqual(run_mock.call_count, 7)
                self.assertEqual(
                    [call.kwargs["timeout"] for call in run_mock.call_args_list[1::2]],
                    [5, 15, 30],
                )


if __name__ == "__main__":
    unittest.main()
