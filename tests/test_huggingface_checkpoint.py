from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from mopd_verl.huggingface_checkpoint import (
    HuggingFaceCheckpointConfig,
    checkpoint_save_required,
    checkpoint_upload_completed,
    clear_checkpoint_upload_receipt,
    require_huggingface_token,
    upload_checkpoint_to_huggingface,
)
from mopd_verl.launch import build_overrides, run_command
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "mopd_formal_audit_off_2gpu.yaml"


class _FakeHfApi:
    def __init__(self, token: str) -> None:
        self.token = token
        self.create_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.create_calls.append(kwargs)

    def upload_folder(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        return type(
            "UploadResult",
            (),
            {"commit_url": "https://huggingface.co/org/run/commit/test"},
        )()


class _FailingHfApi(_FakeHfApi):
    def upload_folder(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        raise RuntimeError("network failure")


class HuggingFaceCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HuggingFaceCheckpointConfig.from_mapping(
            {
                "enabled": True,
                "steps": [2, 7],
                "repo_id": "org/run",
                "private": True,
                "path_prefix": "checkpoints",
                "token_env_var": "HF_TOKEN",
            }
        )

    def test_selected_step_forces_save_when_periodic_saving_is_disabled(self) -> None:
        self.assertTrue(
            checkpoint_save_required(
                step=7,
                save_freq=-1,
                is_last_step=False,
                esi_close_to_expiration=False,
                upload_config=self.config,
            )
        )

    def test_periodic_save_behavior_is_preserved(self) -> None:
        disabled = HuggingFaceCheckpointConfig()
        self.assertTrue(
            checkpoint_save_required(
                step=10,
                save_freq=5,
                is_last_step=False,
                esi_close_to_expiration=False,
                upload_config=disabled,
            )
        )
        self.assertFalse(
            checkpoint_save_required(
                step=9,
                save_freq=5,
                is_last_step=False,
                esi_close_to_expiration=False,
                upload_config=disabled,
            )
        )
        self.assertTrue(
            checkpoint_save_required(
                step=9,
                save_freq=5,
                is_last_step=False,
                esi_close_to_expiration=True,
                upload_config=disabled,
            )
        )
        self.assertFalse(
            checkpoint_save_required(
                step=9,
                save_freq=-1,
                is_last_step=True,
                esi_close_to_expiration=True,
                upload_config=disabled,
            )
        )
        self.assertTrue(
            checkpoint_save_required(
                step=9,
                save_freq=5,
                is_last_step=True,
                esi_close_to_expiration=False,
                upload_config=disabled,
            )
        )

    def test_runtime_omegaconf_step_array_is_supported(self) -> None:
        runtime_config = OmegaConf.create(
            {
                "enabled": True,
                "steps": [2, 7],
                "repo_id": "org/run",
            }
        )

        parsed = HuggingFaceCheckpointConfig.from_mapping(runtime_config)

        self.assertEqual(parsed.steps, (2, 7))
        self.assertFalse(
            checkpoint_save_required(
                step=6,
                save_freq=-1,
                is_last_step=False,
                esi_close_to_expiration=False,
                upload_config=self.config,
            )
        )

    def test_upload_uses_token_without_placing_it_in_upload_metadata(self) -> None:
        fake_api = _FakeHfApi("unused")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "data.pt").touch()

            commit_url = upload_checkpoint_to_huggingface(
                self.config,
                checkpoint_dir,
                7,
                environ={"HF_TOKEN": "secret-token"},
                api_factory=lambda token: self._capture_api(
                    fake_api,
                    token,
                ),
            )
            self.assertTrue(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

            clear_checkpoint_upload_receipt(checkpoint_dir)
            self.assertFalse(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

        self.assertEqual(fake_api.token, "secret-token")
        self.assertEqual(
            commit_url,
            "https://huggingface.co/org/run/commit/test",
        )
        self.assertEqual(fake_api.create_calls[0]["repo_id"], "org/run")
        self.assertEqual(fake_api.create_calls[0]["repo_type"], "model")
        self.assertTrue(fake_api.create_calls[0]["private"])
        self.assertTrue(fake_api.create_calls[0]["exist_ok"])
        self.assertEqual(
            fake_api.upload_calls[0]["folder_path"],
            str(checkpoint_dir),
        )
        self.assertEqual(fake_api.upload_calls[0]["repo_type"], "model")
        self.assertEqual(
            fake_api.upload_calls[0]["path_in_repo"],
            "checkpoints/global_step_7",
        )
        self.assertNotIn("secret-token", str(fake_api.upload_calls))

    @staticmethod
    def _capture_api(api: _FakeHfApi, token: str) -> _FakeHfApi:
        api.token = token
        return api

    def test_missing_token_fails_with_env_file_guidance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"\.env\.local"):
            require_huggingface_token(self.config, {})

    def test_failed_upload_does_not_write_success_receipt(self) -> None:
        fake_api = _FailingHfApi("unused")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "network failure"):
                upload_checkpoint_to_huggingface(
                    self.config,
                    checkpoint_dir,
                    7,
                    environ={"HF_TOKEN": "secret-token"},
                    api_factory=lambda token: self._capture_api(fake_api, token),
                )

            self.assertFalse(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

    def test_task_runner_reads_token_from_env_file_without_ray_secret_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env.local"
            env_file.write_text(
                "# local secrets\nexport HF_TOKEN='secret-token'\n",
                encoding="utf-8",
            )
            config = replace(self.config, env_file=str(env_file))

            token = require_huggingface_token(config, {})

        self.assertEqual(token, "secret-token")

    def test_invalid_step_arrays_are_rejected(self) -> None:
        for steps in ([0], [-1], [2, 2], ["2"], [True], 2):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                HuggingFaceCheckpointConfig.from_mapping(
                    {
                        "enabled": True,
                        "steps": steps,
                        "repo_id": "org/run",
                    }
                )

        for value in (
            {"enabled": True, "steps": [], "repo_id": "org/run"},
            {"enabled": True, "steps": [2], "repo_id": None},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HuggingFaceCheckpointConfig.from_mapping(value)

    def test_launcher_renders_hydra_upload_configuration(self) -> None:
        base = load_config(BASE_CONFIG)
        configured = replace(base, huggingface_checkpoint=self.config)

        overrides = build_overrides(configured)

        self.assertIn("trainer.huggingface_checkpoint.enabled=True", overrides)
        self.assertIn("trainer.huggingface_checkpoint.steps=[2, 7]", overrides)
        self.assertIn(
            "trainer.huggingface_checkpoint.repo_id='org/run'",
            overrides,
        )
        self.assertIn(
            "trainer.huggingface_checkpoint.env_file='.env.local'",
            overrides,
        )
        self.assertNotIn("secret-token", " ".join(overrides))

    def test_runtime_env_file_transfers_hf_token_outside_the_command(self) -> None:
        base = load_config(BASE_CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env.local"
            env_file.write_text("HF_TOKEN=secret-token\n", encoding="utf-8")
            runtime = replace(base.runtime, env_file=str(env_file))
            configured = replace(base, runtime=runtime)

            with patch("mopd_verl.launch.subprocess.call", return_value=0) as call:
                result = run_command(["python", "train.py"], configured)

        self.assertEqual(result, 0)
        command = call.call_args.args[0]
        child_env = call.call_args.kwargs["env"]
        self.assertEqual(command, ["python", "train.py"])
        self.assertEqual(child_env["HF_TOKEN"], "secret-token")
        self.assertNotIn("secret-token", command)


if __name__ == "__main__":
    unittest.main()
