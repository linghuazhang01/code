from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eval.domains.science.official_eval import extract_solution, get_prediction, run_dataset
from eval.domains.toolrl.common import extract_tool_calls, score_single_tool_call
from eval.official_runner import _resolve_datasets


class OfficialEvalHelpersTest(unittest.TestCase):
    def test_science_eval_extracts_boxed_answer(self) -> None:
        self.assertEqual(extract_solution("work\n\\boxed{A}\n"), "A")
        self.assertEqual(get_prediction("reasoning\n\\boxed{C}"), "C")

    def test_toolrl_extracts_and_scores_tool_calls(self) -> None:
        response = '<tool_call>\n{"name": "Search", "parameters": {"query": "Ada"}}\n</tool_call>'
        calls = extract_tool_calls(response)
        score = score_single_tool_call(calls, {"name": "Search", "parameters": {"query": "Ada"}})

        self.assertEqual(calls, [{"name": "Search", "parameters": {"query": "Ada"}}])
        self.assertEqual(score, 1)

    def test_resolves_dataset_selection(self) -> None:
        self.assertIn("api_bank", _resolve_datasets(["toolrl"], ["all"]))
        self.assertEqual(_resolve_datasets(["science"], ["mmlupro"]), ["mmlupro"])
        self.assertEqual(
            _resolve_datasets(["science"], ["mmlupro_500_seed42"]),
            ["mmlupro_500_seed42"],
        )
        self.assertIn("supergpqa", _resolve_datasets(["science"], ["all"]))
        with self.assertRaises(ValueError):
            _resolve_datasets(["science"], ["api_bank"])

    def test_science_eval_scores_and_saves_multiple_rollouts(self) -> None:
        entries = [
            {
                "question_id": 10,
                "question": "Question one?",
                "options": ["one", "two", "three"],
                "answer": "A",
                "category": "math",
            },
            {
                "question_id": 11,
                "question": "Question two?",
                "options": ["one", "two", "three"],
                "answer": "C",
                "category": "physics",
            },
        ]
        request_outputs = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(text="\\boxed{A}"),
                    SimpleNamespace(text="\\boxed{B}"),
                ]
            ),
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(text="\\boxed{C}"),
                    SimpleNamespace(text="\\boxed{C}"),
                ]
            ),
        ]
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **_: messages[0]["content"]
        )
        llm = SimpleNamespace(generate=lambda prompts, params: request_outputs)
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_, **__: tokenizer)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(sys.modules, {"transformers": transformers}),
                patch(
                    "eval.domains.science.official_eval._dataset_entries",
                    return_value=(entries, "category", "answer"),
                ),
                patch("eval.domains.science.official_eval.load_vllm", return_value=llm),
                patch("eval.domains.science.official_eval.sampling_params", return_value=object()),
            ):
                result = run_dataset(
                    dataset_key="mmlupro_500_seed42",
                    model_path="model",
                    output_dir=temp_dir,
                    max_samples=None,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.85,
                    max_model_len=18432,
                    max_tokens=16384,
                    temperature=1.0,
                    top_p=1.0,
                    enable_thinking=False,
                    num_samples=2,
                    seed=42,
                )

            self.assertEqual(result.summary["prompt_count"], 2)
            self.assertEqual(result.summary["sample_count"], 4)
            self.assertEqual(result.summary["correct"], 3)
            self.assertEqual(result.summary["accuracy"], 0.75)
            self.assertEqual(result.summary["observed_pass_at_k"], 1.0)
            records_path = Path(temp_dir) / "mmlupro_500_seed42" / "records.jsonl"
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
            self.assertEqual([record["rollout_index"] for record in records], [0, 1, 0, 1])
            self.assertTrue(all(record["prompt"] and record["completion"] for record in records))

    def test_science_eval_applies_offset_before_limit(self) -> None:
        entries = [
            {
                "question_id": index,
                "question": f"Question {index}?",
                "options": ["one", "two"],
                "answer": "A",
                "category": "math",
            }
            for index in range(4)
        ]
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **_: messages[0]["content"]
        )
        llm = SimpleNamespace(
            generate=lambda prompts, params: [
                SimpleNamespace(outputs=[SimpleNamespace(text="\\boxed{A}")])
                for _ in prompts
            ]
        )
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_, **__: tokenizer)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(sys.modules, {"transformers": transformers}),
                patch(
                    "eval.domains.science.official_eval._dataset_entries",
                    return_value=(entries, "category", "answer"),
                ),
                patch("eval.domains.science.official_eval.load_vllm", return_value=llm),
                patch("eval.domains.science.official_eval.sampling_params", return_value=object()),
            ):
                run_dataset(
                    dataset_key="mmlupro_500_seed42",
                    model_path="model",
                    output_dir=temp_dir,
                    max_samples=2,
                    sample_offset=1,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.85,
                    max_model_len=18432,
                    max_tokens=16384,
                    temperature=1.0,
                    top_p=1.0,
                    enable_thinking=False,
                    num_samples=1,
                    seed=42,
                )

            records_path = Path(temp_dir) / "mmlupro_500_seed42" / "records.jsonl"
            records = [json.loads(line) for line in records_path.read_text().splitlines()]

        self.assertEqual([record["question_id"] for record in records], [1, 2])

    def test_science_fallback_predictions_are_shard_invariant(self) -> None:
        entries = [
            {
                "question_id": index,
                "question": f"Question {index}?",
                "options": ["one", "two", "three"],
                "answer": "A",
                "category": "math",
            }
            for index in range(4)
        ]
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **_: messages[0]["content"]
        )

        def make_llm():
            return SimpleNamespace(
                generate=lambda prompts, params: [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(text="unparseable"),
                            SimpleNamespace(text="still unparseable"),
                        ]
                    )
                    for _ in prompts
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch(
                    "eval.domains.science.official_eval._dataset_entries",
                    return_value=(entries, "category", "answer"),
                ),
                patch("eval.domains.science.official_eval.sampling_params", return_value=object()),
            ):
                run_dataset(
                    dataset_key="mmlupro_500_seed42",
                    model_path="model",
                    output_dir=root / "full",
                    max_samples=4,
                    sample_offset=0,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.85,
                    max_model_len=18432,
                    max_tokens=16384,
                    temperature=1.0,
                    top_p=1.0,
                    enable_thinking=False,
                    num_samples=2,
                    seed=42,
                    llm=make_llm(),
                    tokenizer=tokenizer,
                )
                for offset in (0, 2):
                    run_dataset(
                        dataset_key="mmlupro_500_seed42",
                        model_path="model",
                        output_dir=root / f"shard-{offset}",
                        max_samples=2,
                        sample_offset=offset,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.85,
                        max_model_len=18432,
                        max_tokens=16384,
                        temperature=1.0,
                        top_p=1.0,
                        enable_thinking=False,
                        num_samples=2,
                        seed=42,
                        llm=make_llm(),
                        tokenizer=tokenizer,
                    )

            full_records = [
                json.loads(line)
                for line in (root / "full/mmlupro_500_seed42/records.jsonl").read_text().splitlines()
            ]
            shard_records = []
            for offset in (0, 2):
                shard_records.extend(
                    json.loads(line)
                    for line in (
                        root / f"shard-{offset}/mmlupro_500_seed42/records.jsonl"
                    ).read_text().splitlines()
                )

        full_predictions = {
            (record["question_id"], record["rollout_index"]): record["prediction"]
            for record in full_records
        }
        shard_predictions = {
            (record["question_id"], record["rollout_index"]): record["prediction"]
            for record in shard_records
        }
        self.assertEqual(full_predictions, shard_predictions)


if __name__ == "__main__":
    unittest.main()
