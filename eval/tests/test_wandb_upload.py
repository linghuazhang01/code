from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from eval.wandb_upload import (
    WandbUploadConfig,
    apply_wandb_environment,
    artifact_paths,
    flatten_summary_metrics,
    stable_wandb_run_id,
    upload_eval_output,
)


class _FakeArtifact:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))


class _FakeRun:
    def __init__(self) -> None:
        self.id = "fake-run-id"
        self.url = "https://wandb.example/mopd-eval/fake-run-id"
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.artifacts: list[tuple[_FakeArtifact, list[str]]] = []
        self.finished = False

    def log(self, payload: dict[str, Any]) -> None:
        self.logged.append(payload)

    def log_artifact(self, artifact: _FakeArtifact, aliases: list[str]) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_kwargs: dict[str, Any] = {}

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name: str, **kwargs: Any) -> _FakeArtifact:
        return _FakeArtifact(name, **kwargs)

    def Table(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


class WandbUploadTests(unittest.TestCase):
    def test_env_file_maps_legacy_wandb_key_without_overriding_standard_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env.local"
            env_file.write_text("export Wandb_Key='legacy-test-key'\n", encoding="utf-8")

            legacy_environment: dict[str, str] = {}
            self.assertTrue(apply_wandb_environment(env_file, legacy_environment))

            standard_environment = {"WANDB_API_KEY": "standard-test-key"}
            self.assertTrue(apply_wandb_environment(env_file, standard_environment))

            empty_standard_environment = {"WANDB_API_KEY": ""}
            self.assertTrue(apply_wandb_environment(env_file, empty_standard_environment))

        self.assertEqual(legacy_environment["Wandb_Key"], "legacy-test-key")
        self.assertEqual(legacy_environment["WANDB_API_KEY"], "legacy-test-key")
        self.assertEqual(standard_environment["WANDB_API_KEY"], "standard-test-key")
        self.assertEqual(empty_standard_environment["WANDB_API_KEY"], "legacy-test-key")

    def test_flatten_summary_metrics_preserves_scalar_results(self) -> None:
        metrics = flatten_summary_metrics(
            [
                {
                    "mode": "non_thinking",
                    "dataset": "AIME2025",
                    "ability": "math",
                    "sample_count": 30,
                    "accuracy": 0.5,
                    "avg_score": None,
                }
            ]
        )

        self.assertEqual(metrics["eval/non_thinking/math/AIME2025/sample_count"], 30)
        self.assertEqual(metrics["eval/non_thinking/math/AIME2025/accuracy"], 0.5)
        self.assertNotIn("eval/non_thinking/math/AIME2025/avg_score", metrics)

    def test_stable_run_id_is_retry_safe_and_bounded(self) -> None:
        local_run_id = "qwen3_1p7b/full training:math/" + "x" * 100

        first = stable_wandb_run_id(local_run_id)
        second = stable_wandb_run_id(local_run_id)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 59)
        self.assertRegex(first, r"^[A-Za-z0-9_-]+$")

    def test_upload_logs_metrics_and_complete_rollout_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            payload = {
                "run_id": "qwen3_1p7b_ood_test",
                "status": "final",
                "model_path": "/models/Qwen3-1.7B",
                "record_source": "thinking_eval_samples.jsonl",
                "scoring_backend": "official",
                "expected_total": 1,
                "summary": [
                    {
                        "mode": "non_thinking",
                        "dataset": "AIME2025",
                        "ability": "math",
                        "sample_count": 1,
                        "scored_count": 1,
                        "accuracy": 1.0,
                    }
                ],
                "records": [{"sample_id": "aime25:0", "score": 1.0}],
                "run_config": {"gpu_memory_utilization": 0.9},
            }
            (output_dir / "thinking_eval_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            for filename in (
                "thinking_eval_samples.jsonl",
                "prompt_response_records.jsonl",
                "thinking_eval_summary.json",
                "thinking_eval_summary.csv",
                "eval_run_config.json",
            ):
                (output_dir / filename).write_text("{}\n", encoding="utf-8")

            fake_wandb = _FakeWandb()
            result = upload_eval_output(
                WandbUploadConfig(
                    output_dir=output_dir,
                    project="mopd-eval",
                    group="two_model_ood_test",
                    mode="offline",
                    upload_raw=True,
                ),
                wandb_module=fake_wandb,
            )

        self.assertEqual(fake_wandb.init_kwargs["project"], "mopd-eval")
        self.assertEqual(fake_wandb.init_kwargs["group"], "two_model_ood_test")
        self.assertEqual(fake_wandb.init_kwargs["resume"], "allow")
        self.assertEqual(fake_wandb.run.logged[0]["eval/record_count"], 1)
        artifact, aliases = fake_wandb.run.artifacts[0]
        uploaded_names = {name for _, name in artifact.files}
        self.assertIn("thinking_eval_samples.jsonl", uploaded_names)
        self.assertIn("prompt_response_records.jsonl", uploaded_names)
        self.assertIn("thinking_eval_results.json", uploaded_names)
        self.assertEqual(aliases, ["latest", "final"])
        self.assertTrue(fake_wandb.run.finished)
        self.assertEqual(result.run_url, fake_wandb.run.url)
        self.assertEqual(result.mode, "offline")
        self.assertTrue(result.local_run_dir.endswith("wandb"))

    def test_non_raw_artifact_excludes_record_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for filename in ("thinking_eval_summary.json", "thinking_eval_samples.jsonl"):
                (output_dir / filename).write_text("{}\n", encoding="utf-8")

            paths = artifact_paths(output_dir, upload_raw=False)

        self.assertEqual([path.name for path in paths], ["thinking_eval_summary.json"])

    def test_raw_upload_rejects_missing_prompt_response_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            payload = {
                "run_id": "missing_raw",
                "expected_total": 1,
                "summary": [],
                "records": [],
            }
            (output_dir / "thinking_eval_results.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (output_dir / "thinking_eval_samples.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "prompt_response_records.jsonl"):
                upload_eval_output(
                    WandbUploadConfig(output_dir=output_dir, upload_raw=True),
                    wandb_module=_FakeWandb(),
                )


if __name__ == "__main__":
    unittest.main()
