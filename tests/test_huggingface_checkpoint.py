from __future__ import annotations

import json
import tempfile
import threading
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
from mopd_verl.huggingface_upload_async import AsyncHuggingFaceUploader
from mopd_verl.launch import build_overrides, run_command
from mopd_verl.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "mopd_formal_audit_off_2gpu.yaml"


def _create_huggingface_model(checkpoint_dir: Path) -> Path:
    model_dir = checkpoint_dir / "actor" / "huggingface"
    return _create_model_directory(model_dir)


def _create_model_directory(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").touch()
    return model_dir


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


class _BlockingHfApi(_FakeHfApi):
    def __init__(self, token: str) -> None:
        super().__init__(token)
        self.started = threading.Event()
        self.release = threading.Event()

    def upload_folder(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the background upload")
        return type(
            "UploadResult",
            (),
            {"commit_url": "https://huggingface.co/org/run/commit/async"},
        )()


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
            model_dir = _create_huggingface_model(checkpoint_dir)

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
            str(model_dir),
        )
        self.assertEqual(fake_api.upload_calls[0]["repo_type"], "model")
        self.assertEqual(
            fake_api.upload_calls[0]["path_in_repo"],
            "checkpoints/global_step_7",
        )
        self.assertEqual(
            fake_api.upload_calls[0]["delete_patterns"],
            ["*", "**/*"],
        )
        self.assertNotIn("secret-token", str(fake_api.upload_calls))

    def test_async_upload_does_not_wait_for_network(self) -> None:
        fake_api = _BlockingHfApi("unused")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "data.pt").touch()
            _create_huggingface_model(checkpoint_dir)
            uploader = AsyncHuggingFaceUploader(
                self.config,
                environ={"HF_TOKEN": "secret-token"},
                api_factory=lambda token: self._capture_api(fake_api, token),
            )

            self.assertTrue(uploader.schedule(checkpoint_dir, 7))
            self.assertTrue(fake_api.started.wait(timeout=1))
            staged_model_dir = (
                checkpoint_dir.parent / ".huggingface_uploads" / "global_step_7"
            )
            self.assertTrue(staged_model_dir.is_dir())
            self.assertEqual(
                fake_api.upload_calls[0]["folder_path"],
                str(staged_model_dir),
            )
            self.assertFalse(checkpoint_upload_completed(self.config, checkpoint_dir, 7))

            fake_api.release.set()
            results = uploader.wait()

            self.assertEqual(results[0].step, 7)
            self.assertFalse(staged_model_dir.exists())
            self.assertTrue(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

    def test_async_failure_retains_model_only_snapshot(self) -> None:
        fake_api = _FailingHfApi("unused")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "data.pt").touch()
            _create_huggingface_model(checkpoint_dir)
            uploader = AsyncHuggingFaceUploader(
                self.config,
                environ={"HF_TOKEN": "secret-token"},
                api_factory=lambda token: self._capture_api(fake_api, token),
            )

            self.assertTrue(uploader.schedule(checkpoint_dir, 7))
            with self.assertRaisesRegex(RuntimeError, "steps: 7"):
                uploader.wait()

            staged_model_dir = (
                checkpoint_dir.parent / ".huggingface_uploads" / "global_step_7"
            )
            self.assertTrue((staged_model_dir / "config.json").is_file())
            self.assertTrue((staged_model_dir / "model.safetensors").is_file())
            self.assertFalse((staged_model_dir / "data.pt").exists())
            self.assertFalse(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

    def test_retained_snapshot_is_retried_without_source_checkpoint(self) -> None:
        fake_api = _FakeHfApi("unused")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir)
            staged_model_dir = _create_model_directory(
                checkpoint_root / ".huggingface_uploads" / "global_step_7"
            )
            uploader = AsyncHuggingFaceUploader(
                self.config,
                environ={"HF_TOKEN": "secret-token"},
                api_factory=lambda token: self._capture_api(fake_api, token),
            )

            self.assertEqual(uploader.schedule_retained(checkpoint_root), (7,))
            results = uploader.wait()

            self.assertEqual(results[0].step, 7)
            self.assertEqual(
                fake_api.upload_calls[0]["folder_path"],
                str(staged_model_dir),
            )
            self.assertFalse(staged_model_dir.exists())

    def test_hardlink_failure_never_falls_back_to_synchronous_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            _create_huggingface_model(checkpoint_dir)
            uploader = AsyncHuggingFaceUploader(
                self.config,
                environ={"HF_TOKEN": "secret-token"},
                api_factory=_FakeHfApi,
            )

            with (
                patch(
                    "mopd_verl.huggingface_upload_async.os.link",
                    side_effect=OSError("unsupported"),
                ),
                patch(
                    "mopd_verl.huggingface_upload_async.shutil.copy2"
                ) as copy_file,
                self.assertRaisesRegex(RuntimeError, "refusing to copy"),
            ):
                uploader.schedule(checkpoint_dir, 7)

            uploader.wait()
            copy_file.assert_not_called()

    def test_legacy_whole_checkpoint_receipt_does_not_skip_model_upload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            checkpoint_dir.mkdir()
            receipt = {
                "step": 7,
                "repo_id": "org/run",
                "path_in_repo": "checkpoints/global_step_7",
                "commit_url": "https://huggingface.co/org/run/commit/legacy",
            }
            (checkpoint_dir / ".huggingface_upload_complete.json").write_text(
                json.dumps(receipt),
                encoding="utf-8",
            )

            self.assertFalse(
                checkpoint_upload_completed(self.config, checkpoint_dir, 7)
            )

    def test_upload_rejects_checkpoint_without_huggingface_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "global_step_7"
            model_dir = checkpoint_dir / "actor" / "huggingface"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "model weights"):
                upload_checkpoint_to_huggingface(
                    self.config,
                    checkpoint_dir,
                    7,
                    environ={"HF_TOKEN": "secret-token"},
                    api_factory=_FakeHfApi,
                )

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
            _create_huggingface_model(checkpoint_dir)

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

    def test_trainer_queues_upload_and_waits_only_after_training(self) -> None:
        trainer_source = (
            ROOT
            / "third_party"
            / "verl"
            / "verl"
            / "trainer"
            / "ppo"
            / "ray_trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn("AsyncHuggingFaceUploader", trainer_source)
        self.assertIn("_schedule_huggingface_model_upload", trainer_source)
        fit_start = trainer_source.index("    def fit(self) -> None:")
        fit_impl_start = trainer_source.index("    def _fit_impl(self) -> None:")
        fit_wrapper = trainer_source[fit_start:fit_impl_start]
        fit_impl = trainer_source[fit_impl_start:]
        self.assertIn("finally:", fit_wrapper)
        self.assertIn("self._wait_for_huggingface_model_uploads()", fit_wrapper)
        self.assertNotIn("add_note", fit_wrapper)
        self.assertIn("schedule_retained", fit_impl)
        self.assertNotIn("self._wait_for_huggingface_model_uploads()", fit_impl)

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
