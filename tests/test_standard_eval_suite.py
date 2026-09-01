from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.standard_suite import (
    LCB_RELEASE_COUNTS,
    PARALLEL_DATASET_KEYS,
    StandardSuiteConfig,
    file_sha256,
    finalize_suite,
    initialize_suite,
)

CODE_DIR = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = CODE_DIR / "slurm_standard_eval.sh"
WORKER_SCRIPT = CODE_DIR / "scripts/slurm_standard_eval_worker.sh"


def _config(
    root: Path,
    *,
    gpu_count: int = 4,
    reference_anchor: bool = False,
) -> StandardSuiteConfig:
    return StandardSuiteConfig(
        suite_root=root / "standard-run",
        model_path="/checkpoints/method/global_step_60",
        eval_model_path="/checkpoints/method/global_step_60/actor/hf",
        run_tag="standard-run",
        slurm_job_id="123",
        remote_host="test-host",
        local_archive="/local/eval/standard-run",
        gopd_dir=root / "G-OPD",
        gpu_count=gpu_count,
        reference_anchor=reference_anchor,
    )


def _write_parallel_fixture(config: StandardSuiteConfig, *, rollouts: int = 8) -> None:
    phase_root = config.suite_root / "parallel"
    label = "method__global_step_60"
    tasks = []
    sources = []
    waves = []
    sequence = 0
    for wave_index, dataset in enumerate(PARALLEL_DATASET_KEYS):
        sources.append(
            {
                "dataset": dataset,
                "source_file": f"/data/{dataset}.parquet",
                "selected_rows": 4,
                "source_sha256": f"source-{wave_index}",
            }
        )
        task_ids = []
        for source_index in range(4):
            task_id = f"task-{sequence}"
            task_ids.append(task_id)
            tasks.append(
                {
                    "task_id": task_id,
                    "wave_index": wave_index,
                    "source_start": source_index,
                    "source_end_exclusive": source_index + 1,
                    "num_samples": rollouts,
                    "expected_records": rollouts,
                }
            )
            sequence += 1
        waves.append(
            {
                "wave_index": wave_index,
                "dataset": dataset,
                "task_ids": task_ids,
                "expected_tasks": 4,
            }
        )
    manifest = {
        "status": "complete",
        "model": {
            "path": config.model_path,
            "eval_path": config.eval_model_path,
            "label": label,
        },
        "execution": {
            "worker_count": config.gpu_count,
            "scheduling": "strict_dataset_wave_dynamic_microshards",
        },
        "sources": sources,
        "tasks": tasks,
        "waves": waves,
        "generation": {
            "base_seed": 42,
            "math_samples": rollouts,
            "code_samples": rollouts,
            "science_samples": rollouts,
            "mmlupro_samples": rollouts,
        },
        "expected_records_total": len(tasks) * rollouts,
    }
    phase_root.mkdir(parents=True)
    (phase_root / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (phase_root / "SUCCESS").touch()
    for relative in ("math", "code", "science", "mmlupro_500_seed42", "lcb"):
        output_dir = phase_root / label / relative
        output_dir.mkdir(parents=True)
        (output_dir / "SUCCESS").touch()


def _write_lcb_fixture(config: StandardSuiteConfig) -> dict[str, str]:
    source_root = (
        config.gopd_dir
        / "code_eval/coding/LiveCodeBench/code_generation_lite"
    )
    source_root.mkdir(parents=True)
    source_hashes = {}
    for release, count in LCB_RELEASE_COUNTS.items():
        source = source_root / f"test{release[1:]}.jsonl"
        source.write_text(f"fixture-{release}\n", encoding="utf-8")
        source_hashes[release] = file_sha256(source)
        output_root = config.suite_root / "parallel/method__global_step_60/lcb" / release
        output_root.mkdir(parents=True)
        generation = [
            {
                "question_id": f"{release}-{index}",
                "output_list": [f"answer-{rollout}" for rollout in range(8)],
            }
            for index in range(count)
        ]
        evaluations = [
            {**record, "graded_list": [True] * 8}
            for record in generation
        ]
        (output_root / "codegeneration_8_1.0.json").write_text(
            json.dumps(generation),
            encoding="utf-8",
        )
        (output_root / "codegeneration_8_1.0_eval_all.json").write_text(
            json.dumps(evaluations),
            encoding="utf-8",
        )
        (output_root / "SUCCESS").touch()
    return source_hashes


class StandardEvalSuiteTest(unittest.TestCase):
    def test_finalize_requires_all_ten_datasets_at_k8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            initialize_suite(config, resume=False)
            _write_parallel_fixture(config)
            source_hashes = _write_lcb_fixture(config)
            with patch.dict("eval.standard_suite.LCB_SOURCE_SHA256", source_hashes):
                summary = finalize_suite(config)

            success = (config.suite_root / "STANDARD_SUCCESS").is_file()
            run_manifest = (config.suite_root / "RUN_MANIFEST.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(summary["canonical_datasets"]), 10)
        self.assertEqual(summary["rollouts_per_dataset"], 8)
        self.assertTrue(success)
        self.assertIn("10 datasets × K=8", run_manifest)
        self.assertIn(config.model_path, run_manifest)

    def test_finalize_rejects_legacy_mmlupro_k4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir))
            initialize_suite(config, resume=False)
            _write_parallel_fixture(config, rollouts=4)
            source_hashes = _write_lcb_fixture(config)
            with (
                patch.dict("eval.standard_suite.LCB_SOURCE_SHA256", source_hashes),
                self.assertRaisesRegex(ValueError, "not K=8"),
            ):
                finalize_suite(config)

    def test_submit_dry_run_is_four_gpu_step60_and_400g(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "method/global_step_60"
            model.mkdir(parents=True)
            completed = subprocess.run(
                [
                    str(SUBMIT_SCRIPT),
                    "--model_path",
                    str(model),
                    "--run_tag",
                    "standard-test",
                    "--dry_run",
                ],
                cwd=CODE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gpus=4", completed.stdout)
        self.assertIn("--mem=400G", completed.stdout)
        self.assertIn("slurm_standard_eval_worker.sh", completed.stdout)
        self.assertIn("global_step_60", completed.stdout)

    def test_submit_dry_run_accepts_explicit_two_gpu_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "method/global_step_60"
            model.mkdir(parents=True)
            completed = subprocess.run(
                [
                    str(SUBMIT_SCRIPT),
                    "--model_path",
                    str(model),
                    "--run_tag",
                    "standard-test-dp2",
                    "--gpus",
                    "2",
                    "--dry_run",
                ],
                cwd=CODE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gpus=2", completed.stdout)
        self.assertIn("--gpus 2", completed.stdout)

    def test_finalize_accepts_recorded_two_gpu_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir), gpu_count=2)
            initialize_suite(config, resume=False)
            _write_parallel_fixture(config)
            source_hashes = _write_lcb_fixture(config)
            with patch.dict("eval.standard_suite.LCB_SOURCE_SHA256", source_hashes):
                summary = finalize_suite(config)

            run_manifest = (config.suite_root / "RUN_MANIFEST.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(summary["gpu_count"], 2)
        self.assertIn("`DP=2`", run_manifest)

    def test_submit_accepts_three_gpu_reference_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "Qwen3-1.7B"
            model.mkdir(parents=True)
            completed = subprocess.run(
                [
                    str(SUBMIT_SCRIPT),
                    "--model_path",
                    str(model),
                    "--run_tag",
                    "student-base-dp3",
                    "--gpus",
                    "3",
                    "--reference_anchor",
                    "--dry_run",
                ],
                cwd=CODE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gpus=3", completed.stdout)
        self.assertIn("--gpus 3", completed.stdout)
        self.assertIn("--reference_anchor", completed.stdout)

    def test_reference_anchor_manifest_has_no_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(Path(temp_dir), gpu_count=3, reference_anchor=True)
            initialize_suite(config, resume=False)
            state = json.loads(
                (config.suite_root / "standard_suite_state.json").read_text()
            )
            run_manifest = (config.suite_root / "RUN_MANIFEST.md").read_text()

        self.assertIsNone(state["checkpoint_step"])
        self.assertEqual(state["evaluation_role"], "reference_anchor")
        self.assertIn("reference anchor (not applicable)", run_manifest)

    def test_submit_rejects_memory_above_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "method/global_step_60"
            model.mkdir(parents=True)
            environment = os.environ.copy()
            environment["SLURM_STANDARD_MEMORY"] = "401G"
            completed = subprocess.run(
                [str(SUBMIT_SCRIPT), "--model_path", str(model), "--dry_run"],
                cwd=CODE_DIR,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("400G hard cap", completed.stderr)

    def test_worker_freezes_canonical_protocol_and_prepares_model(self) -> None:
        source = WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("SLURM_EVAL_MATH_SAMPLES=8", source)
        self.assertIn("SLURM_EVAL_CODE_SAMPLES=8", source)
        self.assertIn("SLURM_EVAL_SCIENCE_SAMPLES=8", source)
        self.assertIn("lcb_v5,lcb_v6", source)
        self.assertIn("--shards_per_dataset 16", source)
        self.assertIn("--min_rows_per_shard 1", source)
        self.assertIn("--standard_protocol", source)
        self.assertIn("MOPD_ALLOW_SIMPLE_SCORER_FALLBACK=0", source)
        self.assertIn("prepare_eval_model.sh", source)
        self.assertIn('--eval_model_path "${EVAL_MODEL}"', source)
        self.assertIn("HF_HUB_OFFLINE=1", source)
        self.assertIn("TRANSFORMERS_OFFLINE=1", source)
        self.assertIn("LCB_TOKENIZER_WITNESS", source)
        self.assertIn("eval.standard_suite finalize", source)


if __name__ == "__main__":
    unittest.main()
