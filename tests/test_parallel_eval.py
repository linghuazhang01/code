from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from eval.domains.science.pinned_mmlupro import PinnedMMLUValidation
from eval.parallel_eval import (
    balanced_ranges,
    build_manifest,
    merge_manifest,
    resume_signature,
    summarize_mmlupro_records,
    write_plan,
)
from eval.parallel_worker import run_worker


CODE_DIR = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = CODE_DIR / "slurm_parallel_eval.sh"
START_SCRIPT = CODE_DIR / "start.sh"


def _write_parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"row_id": list(range(rows))}), path)


def _standard_record(*, dataset: str, sample_id: str, rollout_index: int) -> dict[str, object]:
    return {
        "mode": "non_thinking",
        "enable_thinking": False,
        "sample_id": sample_id,
        "dataset": dataset,
        "ability": "math",
        "ground_truth": "1",
        "prediction": "1",
        "score": 1.0,
        "correct": True,
        "prompt_tokens": 1,
        "generated_tokens": 2,
        "thinking_tokens": 0,
        "answer_tokens": 2,
        "total_tokens": 3,
        "latency_seconds": 0.1,
        "generated_tokens_per_second": 20.0,
        "completion_preview": "1",
        "rollout_index": rollout_index,
        "generation_seed": 42,
        "max_new_tokens": 16,
    }


class ParallelEvalTest(unittest.TestCase):
    def test_balanced_ranges_cover_every_row_once(self) -> None:
        ranges = balanced_ranges(10, 4)

        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 10)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])))
        self.assertLessEqual(max(end - start for start, end in ranges), 3)
        self.assertGreaterEqual(min(end - start for start, end in ranges), 2)

    def test_manifest_builds_four_shards_per_dataset_and_atomic_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_parquet(root / "data/eval_data/math/AIME24/test.parquet", 10)
            _write_parquet(root / "data/eval_data/science/GPQA/test.parquet", 7)
            _write_parquet(
                root
                / "data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/test.parquet",
                8,
            )
            suite_root = root / "outputs/run"
            with patch(
                "eval.parallel_eval.validate_mmlupro_500_artifact",
                return_value=PinnedMMLUValidation(
                    data_sha256="fixture-data",
                    selected_ids_sha256="fixture-ids",
                ),
            ) as validate_pinned:
                manifest = build_manifest(
                    code_dir=root,
                    suite_root=suite_root,
                    run_tag="run",
                    model_path="/checkpoints/model/global_step_60",
                    dataset_keys=["aime24", "gpqa_diamond"],
                    shards_per_dataset=4,
                    min_rows_per_shard=1,
                    math_samples=16,
                    code_samples=4,
                    science_samples=4,
                    base_seed=42,
                    max_samples_per_dataset=None,
                    include_mmlupro_500=True,
                )
            manifest_path = write_plan(manifest, resume=False)

            pending = list((suite_root / "queue/pending").glob("*.task"))
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(stored["total_shards"], 12)
        self.assertEqual(len(pending), 12)
        self.assertEqual(stored["expected_records_total"], 10 * 16 + 7 * 4 + 8 * 4)
        validate_pinned.assert_called_once_with(
            root
            / "data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/test.parquet",
            root
            / "data/eval_data/science/MMLU-Pro/subsets/openprm_style_500_seed42/manifest.json",
        )
        self.assertEqual(
            stored["sources"][-1]["pinned_validation"]["data_sha256"],
            "fixture-data",
        )
        self.assertEqual(
            [task["source_end_exclusive"] - task["source_start"] for task in stored["tasks"][:4]],
            [2, 3, 2, 3],
        )

    def test_resume_signature_captures_scoring_and_generation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_parquet(root / "data/eval_data/math/AIME24/test.parquet", 4)
            common = {
                "code_dir": root,
                "suite_root": root / "outputs/run",
                "run_tag": "run",
                "model_path": "/models/model",
                "dataset_keys": ["aime24"],
                "shards_per_dataset": 2,
                "min_rows_per_shard": 1,
                "math_samples": 2,
                "code_samples": 1,
                "science_samples": 1,
                "base_seed": 42,
                "max_samples_per_dataset": None,
                "include_mmlupro_500": False,
            }
            scored = build_manifest(**common, score_code=True, temperature=1.0)
            unscored = build_manifest(**common, score_code=False, temperature=1.0)
            changed_sampling = build_manifest(**common, score_code=True, temperature=0.7)
            changed_runtime = build_manifest(
                **common,
                score_code=True,
                temperature=1.0,
                code_sandbox_image_id="sha256:changed",
            )

        self.assertNotEqual(scored["execution"], unscored["execution"])
        self.assertNotEqual(scored["generation"], changed_sampling["generation"])
        self.assertNotEqual(resume_signature(scored), resume_signature(unscored))
        self.assertNotEqual(resume_signature(scored), resume_signature(changed_sampling))
        self.assertNotEqual(resume_signature(scored), resume_signature(changed_runtime))

    def test_merge_validates_counts_and_produces_final_domain_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_parquet(root / "data/eval_data/math/AIME24/test.parquet", 4)
            suite_root = root / "outputs/run"
            manifest = build_manifest(
                code_dir=root,
                suite_root=suite_root,
                run_tag="run",
                model_path="/checkpoints/model/global_step_60",
                dataset_keys=["aime24"],
                shards_per_dataset=2,
                min_rows_per_shard=1,
                math_samples=2,
                code_samples=1,
                science_samples=1,
                base_seed=42,
                max_samples_per_dataset=None,
                include_mmlupro_500=False,
            )
            manifest_path = write_plan(manifest, resume=False)
            for task in manifest["tasks"]:
                output_dir = Path(task["output_dir"])
                output_dir.mkdir(parents=True)
                records = []
                for rollout_index in range(task["num_samples"]):
                    for source_index in range(task["source_start"], task["source_end_exclusive"]):
                        records.append(
                            _standard_record(
                                dataset="AIME24",
                                sample_id=f"aime24:{source_index}",
                                rollout_index=rollout_index,
                            )
                        )
                (output_dir / "thinking_eval_samples.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                Path(task["success_marker"]).touch()

            merged = merge_manifest(manifest_path)
            final_dir = suite_root / "model__global_step_60" / "math"
            merged_records = (final_dir / "thinking_eval_samples.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            success_exists = (final_dir / "SUCCESS").is_file()
            report_exists = (final_dir / "thinking_eval_results.json").is_file()

        self.assertEqual(merged["status"], "complete")
        self.assertEqual(len(merged_records), 8)
        self.assertTrue(success_exists)
        self.assertTrue(report_exists)

    def test_merge_rejects_duplicate_prompt_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_parquet(root / "data/eval_data/math/AIME24/test.parquet", 2)
            suite_root = root / "outputs/run"
            manifest = build_manifest(
                code_dir=root,
                suite_root=suite_root,
                run_tag="run",
                model_path="/models/model",
                dataset_keys=["aime24"],
                shards_per_dataset=2,
                min_rows_per_shard=1,
                math_samples=1,
                code_samples=1,
                science_samples=1,
                base_seed=42,
                max_samples_per_dataset=None,
                include_mmlupro_500=False,
            )
            manifest_path = write_plan(manifest, resume=False)
            duplicate = _standard_record(dataset="AIME24", sample_id="same", rollout_index=0)
            for task in manifest["tasks"]:
                output_dir = Path(task["output_dir"])
                output_dir.mkdir(parents=True)
                (output_dir / "thinking_eval_samples.jsonl").write_text(
                    json.dumps(duplicate) + "\n",
                    encoding="utf-8",
                )
                Path(task["success_marker"]).touch()

            with self.assertRaisesRegex(ValueError, "Duplicate merged evaluation record"):
                merge_manifest(manifest_path)

    def test_mmlupro_summary_recomputes_pass_at_k(self) -> None:
        records = [
            {"question_id": "q1", "category": "math", "correct": False},
            {"question_id": "q1", "category": "math", "correct": True},
            {"question_id": "q2", "category": "physics", "correct": False},
            {"question_id": "q2", "category": "physics", "correct": False},
        ]

        summary = summarize_mmlupro_records(records, model_path="model")

        self.assertEqual(summary["prompt_count"], 2)
        self.assertEqual(summary["rollouts_per_prompt"], 2)
        self.assertEqual(summary["accuracy"], 0.25)
        self.assertEqual(summary["observed_pass_at_k"], 0.5)

    def test_persistent_worker_continues_after_nonfatal_task_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            queue_root = root / "queue"
            for name in ("pending", "running", "done", "failed"):
                (queue_root / name).mkdir(parents=True)
            source_path = root / "source.parquet"
            _write_parquet(source_path, 2)
            tasks = []
            for sequence in range(2):
                task_root = root / "shards" / f"task-{sequence}"
                task = {
                    "sequence": sequence,
                    "task_id": f"task-{sequence}",
                    "task_type": "standard",
                    "dataset": "aime24",
                    "domain": "math",
                    "source_start": sequence,
                    "source_end_exclusive": sequence + 1,
                    "num_samples": 1,
                    "generation_seed": 42 + sequence,
                    "expected_records": 1,
                    "task_root": str(task_root),
                    "output_dir": str(task_root / "model" / "math"),
                    "success_marker": str(task_root / "SUCCESS"),
                }
                tasks.append(task)
                fields = [
                    sequence,
                    task["task_id"],
                    "standard",
                    "aime24",
                    "math",
                    sequence,
                    1,
                    1,
                    42 + sequence,
                    task["output_dir"],
                    task["success_marker"],
                ]
                (queue_root / "pending" / f"{sequence:04d}.task").write_text(
                    "\t".join(str(value) for value in fields) + "\n",
                    encoding="utf-8",
                )
            manifest = {
                "suite": "parallel_slurm_eval",
                "model": {"path": "/models/model", "label": "model"},
                "execution": {
                    "gpu_memory": 0.85,
                    "max_model_len": 18432,
                    "max_num_batched_tokens": 32768,
                    "max_num_seqs": 24,
                    "enforce_eager": True,
                    "enable_chunked_prefill": False,
                },
                "generation": {},
                "sources": [{"dataset": "aime24", "source_file": str(source_path)}],
                "tasks": tasks,
            }
            manifest_path = root / "suite_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fake_llm = SimpleNamespace(get_tokenizer=lambda: object())
            with (
                patch("eval.parallel_worker.load_vllm_model", return_value=fake_llm) as load_model,
                patch(
                    "eval.parallel_worker.run_standard_task",
                    side_effect=[ValueError("bad shard"), None],
                ) as run_task,
            ):
                status = run_worker(
                    manifest_path=manifest_path,
                    eval_model_path="/models/hf",
                    worker_id=0,
                    resume=False,
                )

            success = [Path(task["success_marker"]).is_file() for task in tasks]
            done_count = len(list((queue_root / "done").glob("*.task")))
            failed_count = len(list((queue_root / "failed").glob("*.task")))

        self.assertEqual(status, 1)
        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(run_task.call_count, 2)
        self.assertEqual(done_count, 1)
        self.assertEqual(failed_count, 1)
        self.assertEqual(success, [False, True])

    def test_submit_dry_run_requests_four_gpus_and_400g(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            completed = subprocess.run(
                [
                    str(SUBMIT_SCRIPT),
                    "--model_path",
                    str(model),
                    "--run_tag",
                    "parallel-test",
                    "--include_mmlupro_500",
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
        self.assertIn("--shards_per_dataset", completed.stdout)
        self.assertIn("--include_mmlupro_500", completed.stdout)

    def test_start_sh_forwards_slurm_eval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            completed = subprocess.run(
                [
                    "bash",
                    str(START_SCRIPT),
                    "--slurm",
                    "--eval",
                    "--model_path",
                    str(model),
                    "--run_tag",
                    "start-eval-test",
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


if __name__ == "__main__":
    unittest.main()
