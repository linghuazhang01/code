from __future__ import annotations

import dataclasses
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eval.common import (
    DEFAULT_DATA_FILES,
    DEFAULT_IF_DATA_FILES,
    DEFAULT_SCIENCE_DATA_FILES,
    DEFAULT_SEARCH_DATA_FILES,
    DEFAULT_TOOLRL_DATA_FILES,
    EvalResult,
    EvalSample,
    append_sample_outputs,
    load_eval_samples,
    normalize_ability,
    remove_think_block,
    summarize_results,
)
from eval.full_training_manifest import resume_signature
from eval.domains.math import extract_boxed_answer, extract_final_answer, normalize_answer, simple_score_math_answer
from eval.domains.search import extract_search_answer, is_search_dataset
from eval.domains.toolrl.prepare_data import toolrl_jsonl_to_verl_parquet
from eval.report import _ability
from eval.report import _compact_record, _detail_record
from eval.runner import (
    _temporary_vllm_v1_chunked_prefill_setting,
    completed_samples_for_rollout,
    load_incremental_results,
    resolve_max_new_tokens,
    validate_output_directory,
    validate_resume_config,
    validate_resume_prefix,
)
from eval.domains.scoring import score_completion


class ThinkingEvalTest(unittest.TestCase):
    def test_vllm_v1_respects_explicit_disabled_chunked_prefill(self) -> None:
        class FakeEngineArgs:
            def _set_default_args(self, _usage_context: object, _model_config: object) -> None:
                self.enable_chunked_prefill = True

        engine_args = FakeEngineArgs()

        with _temporary_vllm_v1_chunked_prefill_setting(
            False,
            max_num_batched_tokens=32768,
            max_num_seqs=24,
            engine_args_cls=FakeEngineArgs,
        ):
            engine_args._set_default_args(object(), object())
            self.assertFalse(engine_args.enable_chunked_prefill)

        engine_args._set_default_args(object(), object())
        self.assertTrue(engine_args.enable_chunked_prefill)

    def test_vllm_v1_disabled_chunked_prefill_requires_explicit_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            with _temporary_vllm_v1_chunked_prefill_setting(
                False,
                max_num_batched_tokens=None,
                max_num_seqs=24,
                engine_args_cls=object,
            ):
                pass

    def test_default_data_files_include_small_code_validation(self) -> None:
        self.assertIn("data/eval_data/code/HumanEvalPlus/test.parquet", DEFAULT_DATA_FILES)
        self.assertIn("data/eval_data/code/MBPPPlus/test.parquet", DEFAULT_DATA_FILES)
        self.assertIn("data/eval_data/toolrl/BFCL/test.parquet", DEFAULT_TOOLRL_DATA_FILES)
        self.assertIn("data/SearchQA/test.parquet", DEFAULT_SEARCH_DATA_FILES)
        self.assertIn("data/SearchQA/test.parquet", DEFAULT_DATA_FILES)
        self.assertIn("data/eval_data/if/IFEval/test.parquet", DEFAULT_IF_DATA_FILES)
        self.assertIn("data/eval_data/science/GPQA/test.parquet", DEFAULT_SCIENCE_DATA_FILES)

    def test_extracts_nested_boxed_answer(self) -> None:
        self.assertEqual(extract_boxed_answer(r"Thus \boxed{\frac{1}{2}}."), r"\frac{1}{2}")

    def test_extracts_last_numeric_answer_without_box(self) -> None:
        self.assertEqual(extract_final_answer("Try 3 first.\nFinal answer: 204."), "204.")

    def test_simple_math_fallback_normalizes_commas_and_boxed_answers(self) -> None:
        score, prediction = simple_score_math_answer(r"After solving, \boxed{1,024}.", "1024")
        self.assertEqual(prediction, "1,024")
        self.assertEqual(score, 1.0)
        self.assertEqual(normalize_answer(r"\text{A}"), "a")

    def test_remove_think_block_for_scoring(self) -> None:
        self.assertEqual(remove_think_block("<think>reason</think>\nanswer"), "answer")
        self.assertEqual(remove_think_block("<think>unfinished reasoning"), "")

    def test_load_eval_samples_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.parquet"
            pd.DataFrame(
                [
                    {
                        "id": "sample-1",
                        "data_source": "AIME2025",
                        "prompt": [{"role": "user", "content": "Question?"}],
                        "ability": "math",
                        "reward_model": {"ground_truth": "42", "style": "rule"},
                        "extra_info": {"sample_id": "validation:AIME2025:0", "validation_dataset": "AIME2025"},
                    }
                ]
            ).to_parquet(path, index=False)

            samples = load_eval_samples([path])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_id, "validation:AIME2025:0")
        self.assertEqual(samples[0].messages[0]["content"], "Question?")
        self.assertEqual(samples[0].ground_truth, "42")
        self.assertEqual(samples[0].extra_info["validation_dataset"], "AIME2025")
        self.assertEqual(samples[0].sample_metadata["source_row_position"], 0)
        self.assertEqual(samples[0].sample_metadata["source_id"], "sample-1")
        self.assertEqual(samples[0].sample_metadata["reward_model"]["style"], "rule")
        self.assertEqual(
            samples[0].sample_metadata["extra_info"]["validation_dataset"],
            "AIME2025",
        )

    def test_load_eval_samples_streams_a_stable_row_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.parquet"
            pd.DataFrame(
                [
                    {
                        "id": f"sample-{index}",
                        "data_source": "AIME2025",
                        "prompt": [{"role": "user", "content": str(index)}],
                        "ability": "math",
                        "reward_model": {"ground_truth": str(index)},
                        "extra_info": {"sample_id": f"validation:AIME2025:{index}"},
                    }
                    for index in range(5)
                ]
            ).to_parquet(path, index=False)

            samples = load_eval_samples(
                [path],
                max_samples_per_dataset=2,
                sample_offset_per_dataset=2,
            )

        self.assertEqual([sample.sample_id for sample in samples], ["validation:AIME2025:2", "validation:AIME2025:3"])
        self.assertEqual([sample.sample_metadata["source_row_position"] for sample in samples], [2, 3])

    def test_gpqa_scoring_preserves_m2rl_metadata(self) -> None:
        sample = EvalSample(
            sample_id="science:gpqa:0",
            dataset="m2rl_gpqa_diamond",
            ability="science",
            messages=[{"role": "user", "content": "Question?"}],
            ground_truth="B",
            extra_info={"rm_type": "gpqa", "correct_letter": "B", "valid_letters": np.array(list("ABCD"))},
        )

        score, prediction, metadata = score_completion(
            sample,
            "<think>work</think>\nAnswer: B",
            score_code=False,
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(prediction, "Answer: B")
        self.assertEqual(metadata[0]["m2rl_gpqa"], 1.0)

    def test_humaneval_plus_executes_generated_code(self) -> None:
        samples = load_eval_samples(
            [Path("data/eval_data/code/HumanEvalPlus/test.parquet")],
            max_samples_per_dataset=1,
        )
        completion = """```python
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])
```"""

        score, _, metadata = score_completion(samples[0], completion, score_code=True)

        self.assertEqual(score, 1.0)
        self.assertTrue(metadata[0]["passed"])

    @patch("eval.domains.scoring._load_default_compute_score", return_value=None)
    def test_training_code_never_uses_simple_math_fallback(self, _mock_scorer) -> None:
        sample = EvalSample(
            sample_id="training-code-0",
            dataset="taco",
            ability="code",
            messages=[{"role": "user", "content": "Write a program."}],
            ground_truth='{"inputs": ["1\\n"], "outputs": ["1\\n"]}',
        )

        with self.assertRaisesRegex(RuntimeError, "simple Math fallback is invalid"):
            score_completion(sample, "```python\nprint(1)\n```", score_code=True)

    def test_hle_is_not_silently_scored_with_gpqa(self) -> None:
        sample = EvalSample(
            sample_id="science:hle:0",
            dataset="m2rl_hle",
            ability="science",
            messages=[{"role": "user", "content": "Question?"}],
            ground_truth="answer",
            extra_info={"rm_type": "hle"},
        )

        score, _, metadata = score_completion(sample, "answer", score_code=False)

        self.assertIsNone(score)
        self.assertEqual(metadata[0]["scorer"], "official_hle_judge_required")

    def test_fractional_score_is_not_complete_prompt_success(self) -> None:
        result = EvalResult(
            mode="non_thinking",
            enable_thinking=False,
            sample_id="if-0",
            dataset="m2rl_ifeval",
            ability="if",
            ground_truth="",
            prediction="response",
            score=0.5,
            correct=False,
            prompt_tokens=1,
            generated_tokens=2,
            thinking_tokens=0,
            answer_tokens=2,
            total_tokens=3,
            latency_seconds=0.1,
            generated_tokens_per_second=20.0,
            completion_preview="response",
        )

        summary = next(row for row in summarize_results([result]) if row["dataset"] == "m2rl_ifeval")

        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["avg_score"], 0.5)

    def test_summary_reports_avg_at_k_and_pass_at_k(self) -> None:
        base = {
            "mode": "thinking",
            "enable_thinking": True,
            "dataset": "AIME2024",
            "ability": "math",
            "ground_truth": "1",
            "prediction": "1",
            "prompt_tokens": 1,
            "generated_tokens": 2,
            "thinking_tokens": 1,
            "answer_tokens": 1,
            "total_tokens": 3,
            "latency_seconds": 0.1,
            "generated_tokens_per_second": 20.0,
            "completion_preview": "",
        }
        results = [
            EvalResult(sample_id="q1", score=1.0, correct=True, rollout_index=0, **base),
            EvalResult(sample_id="q1", score=0.0, correct=False, rollout_index=1, **base),
            EvalResult(sample_id="q2", score=0.0, correct=False, rollout_index=0, **base),
            EvalResult(sample_id="q2", score=0.0, correct=False, rollout_index=1, **base),
        ]

        summary = next(row for row in summarize_results(results) if row["dataset"] == "AIME2024")

        self.assertEqual(summary["unique_sample_count"], 2)
        self.assertEqual(summary["min_samples_per_prompt"], 2)
        self.assertEqual(summary["avg_at_k"], 0.25)
        self.assertEqual(summary["observed_pass_at_k"], 0.5)

    def test_load_eval_samples_normalizes_searchqa_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.parquet"
            pd.DataFrame(
                [
                    {
                        "id": "search-1",
                        "data_source": "searchR1_nq",
                        "prompt": [{"role": "user", "content": "Question?"}],
                        "ability": "searchqa",
                        "reward_model": {"ground_truth": {"target": ["Paris"]}, "style": "rule"},
                        "extra_info": {"sample_id": "search:searchR1_nq:1", "validation_dataset": "searchR1_nq"},
                    }
                ]
            ).to_parquet(path, index=False)

            samples = load_eval_samples([path])

        self.assertEqual(samples[0].ability, "search")
        self.assertEqual(samples[0].ground_truth, {"target": ["Paris"]})
        self.assertEqual(normalize_ability("qa", "searchR1_nq"), "search")

    def test_normalizes_science_and_toolrl_domains(self) -> None:
        self.assertEqual(normalize_ability("", "MMLU-Pro"), "science")
        self.assertEqual(normalize_ability("", "SuperGPQA"), "science")
        self.assertEqual(normalize_ability("", "BFCL"), "tool")
        self.assertEqual(_ability("MMLU-Pro"), "science")
        self.assertEqual(_ability("BFCL"), "tool")

    def test_toolrl_jsonl_to_verl_parquet_writes_external_eval_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = Path(tmp)
            input_path = temp_path / "bfcl.jsonl"
            output_path = temp_path / "test.parquet"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "bfcl0",
                        "question": "Call the weather API for Paris.",
                        "answer": {"name": "get_weather"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count = toolrl_jsonl_to_verl_parquet(input_path, output_path, dataset="BFCL")
            frame = pd.read_parquet(output_path)

        self.assertEqual(count, 1)
        self.assertEqual(frame.iloc[0]["ability"], "tool")
        self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "BFCL")
        self.assertTrue(frame.iloc[0]["extra_info"]["requires_external_tool_eval"])

    def test_summarize_results_groups_modes_and_all(self) -> None:
        results = [
            EvalResult(
                mode="thinking",
                enable_thinking=True,
                sample_id="a",
                dataset="AIME2025",
                ability="math",
                ground_truth="1",
                prediction="1",
                score=1.0,
                correct=True,
                prompt_tokens=10,
                generated_tokens=100,
                thinking_tokens=80,
                answer_tokens=20,
                total_tokens=110,
                latency_seconds=2.0,
                generated_tokens_per_second=50.0,
                completion_preview="",
            ),
            EvalResult(
                mode="thinking",
                enable_thinking=True,
                sample_id="b",
                dataset="AIME2025",
                ability="math",
                ground_truth="2",
                prediction="3",
                score=0.0,
                correct=False,
                prompt_tokens=10,
                generated_tokens=50,
                thinking_tokens=40,
                answer_tokens=10,
                total_tokens=60,
                latency_seconds=1.0,
                generated_tokens_per_second=50.0,
                completion_preview="",
            ),
        ]

        summaries = summarize_results(results)
        by_key = {(item["mode"], item["dataset"], item["ability"]): item for item in summaries}

        self.assertEqual(by_key[("thinking", "AIME2025", "math")]["accuracy"], 0.5)
        self.assertEqual(by_key[("thinking", "AIME2025", "math")]["avg_generated_tokens"], 75.0)
        self.assertEqual(by_key[("thinking", "ALL", "all")]["sample_count"], 2)

    def test_result_dataclass_is_json_serializable(self) -> None:
        result = EvalResult(
            mode="non_thinking",
            enable_thinking=False,
            sample_id="a",
            dataset="AIME2025",
            ability="math",
            ground_truth="1",
            prediction="1",
            score=1.0,
            correct=True,
            prompt_tokens=1,
            generated_tokens=2,
            thinking_tokens=0,
            answer_tokens=2,
            total_tokens=3,
            latency_seconds=0.1,
            generated_tokens_per_second=20.0,
            completion_preview="ok",
        )
        json.dumps(result.__dict__)

    def test_append_sample_outputs_writes_incremental_jsonl(self) -> None:
        result = EvalResult(
            mode="non_thinking",
            enable_thinking=False,
            sample_id="a",
            dataset="AIME2025",
            ability="math",
            ground_truth="1",
            prediction="1",
            score=1.0,
            correct=True,
            prompt_tokens=1,
            generated_tokens=2,
            thinking_tokens=0,
            answer_tokens=2,
            total_tokens=3,
            latency_seconds=0.1,
            generated_tokens_per_second=20.0,
            completion_preview="ok",
            prompt="prompt",
            completion="response",
        )
        with tempfile.TemporaryDirectory() as tmp:
            append_sample_outputs([result], Path(tmp))
            append_sample_outputs([result], Path(tmp))
            lines = (Path(tmp) / "thinking_eval_samples.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["completion"], "response")

    def test_resume_helpers_validate_strict_prefix_and_progress(self) -> None:
        samples = [
            EvalSample(
                sample_id=f"sample-{index}",
                dataset="dataset",
                ability="math",
                messages=[{"role": "user", "content": str(index)}],
                ground_truth=str(index),
            )
            for index in range(3)
        ]
        results = [
            EvalResult(
                mode="non_thinking",
                enable_thinking=False,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                ability=sample.ability,
                ground_truth=sample.ground_truth,
                prediction=sample.ground_truth,
                score=1.0,
                correct=True,
                prompt_tokens=1,
                generated_tokens=1,
                thinking_tokens=0,
                answer_tokens=1,
                total_tokens=2,
                latency_seconds=0.1,
                generated_tokens_per_second=10.0,
                completion_preview=sample.ground_truth,
            )
            for sample in samples[:2]
        ]

        validate_resume_prefix(results, samples, ["non_thinking"], 1)
        self.assertEqual(
            completed_samples_for_rollout(
                len(results),
                mode_index=0,
                rollout_index=0,
                sample_count=len(samples),
                num_samples=1,
            ),
            2,
        )

        invalid = [dataclasses.replace(results[0], sample_id="wrong")]
        with self.assertRaisesRegex(ValueError, "not a strict prefix"):
            validate_resume_prefix(invalid, samples, ["non_thinking"], 1)

    def test_load_incremental_results_round_trip(self) -> None:
        result = EvalResult(
            mode="non_thinking",
            enable_thinking=False,
            sample_id="a",
            dataset="dataset",
            ability="math",
            ground_truth="1",
            prediction="1",
            score=1.0,
            correct=True,
            prompt_tokens=1,
            generated_tokens=2,
            thinking_tokens=0,
            answer_tokens=2,
            total_tokens=3,
            latency_seconds=0.1,
            generated_tokens_per_second=20.0,
            completion_preview="ok",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thinking_eval_samples.jsonl"
            path.write_text(json.dumps(dataclasses.asdict(result)) + "\n", encoding="utf-8")
            loaded = load_incremental_results(path)

        self.assertEqual(loaded, [result])

    def test_resume_config_rejects_changed_generation_settings(self) -> None:
        existing = {
            "model_path": "model-a",
            "temperature": 0.0,
            "sample_offset_per_dataset": 0,
            "resume": False,
        }
        requested = {**existing, "resume": True}
        validate_resume_config(existing, requested)

        with self.assertRaisesRegex(ValueError, "temperature"):
            validate_resume_config(existing, {**requested, "temperature": 1.0})

    def test_non_resume_rejects_existing_evaluation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "thinking_eval_samples.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                validate_output_directory(output_dir, resume=False)
            validate_output_directory(output_dir, resume=True)

    def test_full_training_resume_signature_rejects_config_and_source_drift(self) -> None:
        manifest = {
            "schema_version": 1,
            "suite": "full_training",
            "run_tag": "run-a",
            "models": {"qwen": "/models/qwen"},
            "execution": {
                "gpu_id": "0",
                "minimum_free_gib_guard": 50,
                "batch_size": 4,
            },
            "generation": {"temperature": 0.0, "base_seed": 42},
            "domains": [{"domain": "math", "source_sha256": "abc"}],
            "shards": [{"source_start": 0, "success": False}],
        }
        dynamic_update = copy.deepcopy(manifest)
        dynamic_update["execution"]["gpu_id"] = "1"
        dynamic_update["shards"][0]["success"] = True
        self.assertEqual(
            resume_signature(manifest),
            resume_signature(dynamic_update),
        )

        config_drift = copy.deepcopy(manifest)
        config_drift["generation"]["temperature"] = 1.0
        self.assertNotEqual(
            resume_signature(manifest),
            resume_signature(config_drift),
        )

        source_drift = copy.deepcopy(manifest)
        source_drift["domains"][0]["source_sha256"] = "changed"
        self.assertNotEqual(
            resume_signature(manifest),
            resume_signature(source_drift),
        )

    def test_resolve_max_new_tokens_prefers_ability_override(self) -> None:
        sample = EvalSample(
            sample_id="code-1",
            dataset="HumanEvalPlus",
            ability="code",
            messages=[{"role": "user", "content": "Write code."}],
            ground_truth="",
        )

        self.assertEqual(
            resolve_max_new_tokens(
                sample,
                "thinking",
                {"thinking": 40960, "non_thinking": 10240},
                {
                    ("thinking", "math"): 40960,
                    ("thinking", "code"): 8192,
                    ("non_thinking", "math"): 10240,
                    ("non_thinking", "code"): 1024,
                },
            ),
            8192,
        )
        self.assertEqual(
            resolve_max_new_tokens(
                sample,
                "thinking",
                {"thinking": 65536, "non_thinking": 16384},
                {
                    ("thinking", "math"): None,
                    ("thinking", "code"): None,
                    ("non_thinking", "math"): None,
                    ("non_thinking", "code"): None,
                },
            ),
            65536,
        )

    def test_report_splits_compact_records_from_prompt_response_details(self) -> None:
        record = {
            "sample_id": "sample-1",
            "mode": "thinking",
            "dataset": "HumanEvalPlus",
            "ability": "code",
            "score": 1.0,
            "correct": True,
            "generated_tokens": 10,
            "thinking_tokens": 4,
            "completion_preview": "def f(): pass",
            "messages": [{"role": "user", "content": "Write code."}],
            "prompt": "<|im_start|>user\nWrite code.",
            "completion": "<think>short</think>\ndef f(): pass",
            "rollout_index": 2,
            "generation_seed": 44,
            "sample_metadata": {
                "source_file": "data.parquet",
                "source_row_position": 7,
                "extra_info": {"task_id": "task-7"},
            },
            "reward_metadata": [{"scorer": "code"}],
        }

        compact = _compact_record(record)
        detail = _detail_record(record)

        self.assertNotIn("prompt", compact)
        self.assertNotIn("completion", compact)
        self.assertEqual(compact["answer_tokens"], 6)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertEqual(detail["prompt"], "<|im_start|>user\nWrite code.")
        self.assertEqual(detail["response"], "<think>short</think>\ndef f(): pass")
        self.assertEqual(detail["rollout_index"], 2)
        self.assertEqual(detail["sample_metadata"]["extra_info"]["task_id"], "task-7")

    def test_search_dataset_and_answer_extraction(self) -> None:
        self.assertTrue(is_search_dataset("searchR1_nq"))
        self.assertEqual(_ability("searchR1_nq"), "search")
        self.assertEqual(extract_search_answer("<answer>first</answer><answer>second</answer>"), "second")


if __name__ == "__main__":
    unittest.main()
