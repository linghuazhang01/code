import json
import tempfile
import time
import unittest
from pathlib import Path

from eval.official_evalplus_rescore import prepare_samples, summarize_results


def _test_sanitizer(response: str, entry_point: str) -> str:
    return f"{entry_point}:{response}"


def _slow_sanitizer(response: str, entry_point: str) -> str:
    if "loop" in response:
        time.sleep(30)
    return f"{entry_point}:{response}"


class OfficialEvalPlusRescoreTest(unittest.TestCase):
    def test_slurm_worker_allows_pinned_source_path_overrides(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "slurm_evalplus_rescore_worker.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('HUMANEVAL_SOURCE="${HUMANEVAL_SOURCE:-', script)
        self.assertIn('MBPP_SOURCE="${MBPP_SOURCE:-', script)

    def test_preparation_preserves_k8_and_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            records = root / "records.jsonl"
            output = root / "samples.jsonl"
            source.write_text(
                json.dumps({"task_id": "HumanEval/1", "entry_point": "solve"})
                + "\n",
                encoding="utf-8",
            )
            rows = []
            for rollout_index in reversed(range(8)):
                rows.append(
                    {
                        "dataset": "HumanEvalPlus",
                        "rollout_index": rollout_index,
                        "response": f"answer-{rollout_index}",
                        "sample_metadata": {
                            "source_id": "HumanEvalPlus:HumanEval/1"
                        },
                    }
                )
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            manifest = prepare_samples(
                records,
                source,
                "humaneval",
                output,
                _test_sanitizer,
                parallel=2,
            )

            samples = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(manifest["sample_count"], 8)
            self.assertEqual(manifest["sanitize_processes"], 2)
            self.assertEqual(samples[0]["solution"], "solve:answer-0")
            self.assertEqual(samples[-1]["solution"], "solve:answer-7")
            self.assertEqual(manifest["sanitize_timeout_count"], 0)

    def test_sanitize_timeout_writes_empty_solution_and_records_it(self) -> None:
        for parallel in (1, 2):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "source.jsonl"
                records = root / "records.jsonl"
                output = root / "samples.jsonl"
                source.write_text(
                    json.dumps({"task_id": "Mbpp/2", "entry_point": "solve"}) + "\n",
                    encoding="utf-8",
                )
                rows = [
                    {
                        "dataset": "MBPPPlus",
                        "rollout_index": rollout_index,
                        "response": "loop" if rollout_index in (3, 5) else f"answer-{rollout_index}",
                        "sample_metadata": {"source_id": "MBPPPlus:Mbpp/2"},
                    }
                    for rollout_index in range(8)
                ]
                records.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

                start = time.monotonic()
                manifest = prepare_samples(
                    records,
                    source,
                    "mbpp",
                    output,
                    _slow_sanitizer,
                    parallel=parallel,
                    sanitize_timeout=0.5,
                )

                self.assertLess(time.monotonic() - start, 20)
                samples = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(samples), 8)
                self.assertEqual(samples[3]["solution"], "")
                self.assertEqual(samples[5]["solution"], "")
                self.assertEqual(samples[0]["solution"], "solve:answer-0")
                self.assertEqual(manifest["sanitize_timeout_seconds"], 0.5)
                self.assertEqual(manifest["sanitize_timeout_count"], 2)
                self.assertEqual(
                    [(t["task_id"], t["rollout_index"]) for t in manifest["sanitize_timeouts"]],
                    [("Mbpp/2", 3), ("Mbpp/2", 5)],
                )

    def test_summary_reports_base_and_plus_accuracy_and_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = Path(temp_dir) / "eval_results.json"
            results.write_text(
                json.dumps(
                    {
                        "hash": "dataset-md5",
                        "eval": {
                            "HumanEval/0": [
                                {"base_status": "pass", "plus_status": "fail"},
                                {"base_status": "fail", "plus_status": "fail"},
                            ],
                            "HumanEval/1": [
                                {"base_status": "fail", "plus_status": "pass"},
                                {"base_status": "pass", "plus_status": "pass"},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_results(results, expected_rollouts=2)

            self.assertEqual(summary["metrics"]["base"]["correct_samples"], 2)
            self.assertEqual(summary["metrics"]["base"]["observed_pass_at_k"], 1.0)
            self.assertEqual(summary["metrics"]["plus"]["correct_samples"], 1)
            self.assertEqual(summary["metrics"]["plus"]["observed_pass_at_k"], 0.5)

            with self.assertRaisesRegex(ValueError, "Expected K=8"):
                summarize_results(results)


if __name__ == "__main__":
    unittest.main()
