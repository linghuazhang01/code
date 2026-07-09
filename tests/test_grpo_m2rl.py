from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from grpo.data.m2rl import m2rl_frame_to_verl, validate_m2rl_frame
from grpo.data.nemotron import normalize_nemotron_record
from grpo.rewards.mixed import compute_score as compute_mixed_score
from grpo.rewards.m2rl import compute_gpqa_reward, compute_ifbench_reward, compute_score


class M2RLRewardTests(unittest.TestCase):
    def test_gpqa_reward_extracts_final_letter_without_think_tags(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        self.assertEqual(compute_gpqa_reward("The answer is C.", "C", metadata), 1.0)
        self.assertEqual(compute_gpqa_reward("Final answer: B", "C", metadata), 0.0)

    def test_compute_score_dispatches_science_from_data_source(self) -> None:
        score = compute_score(
            data_source="m2rl_science",
            solution_str="Option B is correct.",
            ground_truth="B",
            extra_info={"choices": ["A text", "B text"], "rm_type": "gpqa"},
        )

        self.assertEqual(score["score"], 1.0)
        self.assertEqual(score["m2rl_gpqa"], 1.0)

    def test_ifbench_reward_uses_instruction_metadata(self) -> None:
        class FakeInputExample:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        fake_eval = SimpleNamespace(
            InputExample=FakeInputExample,
            test_instruction_following_strict=lambda _example, responses: SimpleNamespace(
                follow_all_instructions=next(iter(responses.values())) == "ok"
            ),
        )
        metadata = {
            "record_id": 7,
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 2, "relation": "exactly"}],
            "prompt_text": "Write exactly two words.",
        }

        with (
            patch("grpo.rewards.m2rl._ensure_verifiable_instruction_registry", side_effect=ImportError),
            patch("grpo.rewards.m2rl._ensure_ifbench_importable", return_value=fake_eval),
        ):
            self.assertEqual(compute_ifbench_reward("ok", metadata), 1.0)
            self.assertEqual(compute_ifbench_reward("bad", metadata), 0.0)

    def test_ifbench_reward_prefers_verifiable_instructions(self) -> None:
        class FakeInstruction:
            def __init__(self, instruction_id: str) -> None:
                self.instruction_id = instruction_id

            def build_description(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def check_following(self, value: str) -> bool:
                return self.kwargs == {"num_words": 2} and value == "ok"

        fake_registry = SimpleNamespace(INSTRUCTION_DICT={"length_constraints:number_words": FakeInstruction})
        metadata = {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 2.0}],
            "prompt_text": "Write exactly two words.",
        }

        with (
            patch("grpo.rewards.m2rl._ensure_verifiable_instruction_registry", return_value=fake_registry),
            patch("grpo.rewards.m2rl._ensure_ifbench_importable") as ifbench_import,
        ):
            self.assertEqual(compute_ifbench_reward("ok", metadata), 1.0)
            self.assertEqual(compute_ifbench_reward("bad", metadata), 0.0)

        ifbench_import.assert_not_called()

    def test_ifbench_reward_accepts_parquet_array_like_metadata(self) -> None:
        captured: dict[str, object] = {}

        class ArrayLike:
            def __init__(self, values: list[object]) -> None:
                self._values = values

            def __iter__(self):
                return iter(self._values)

        class FakeInputExample:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_eval = SimpleNamespace(
            InputExample=FakeInputExample,
            test_instruction_following_strict=lambda _example, _responses: SimpleNamespace(
                follow_all_instructions=True
            ),
        )
        metadata = {
            "record_id": 7,
            "instruction_id_list": ArrayLike(["length_constraints:number_words"]),
            "kwargs": ArrayLike([{"num_words": 2.0, "relation": "exactly", "unused": None}]),
            "prompt_text": "Write exactly two words.",
        }

        with (
            patch("grpo.rewards.m2rl._ensure_verifiable_instruction_registry", side_effect=ImportError),
            patch("grpo.rewards.m2rl._ensure_ifbench_importable", return_value=fake_eval),
        ):
            self.assertEqual(compute_ifbench_reward("ok", metadata), 1.0)

        self.assertEqual(captured["instruction_id_list"], ["length_constraints:number_words"])
        self.assertEqual(captured["kwargs"], [{"num_words": 2, "relation": "exactly"}])

    def test_mixed_reward_routes_ifbench_to_m2rl(self) -> None:
        with patch("grpo.rewards.mixed.compute_m2rl_score", return_value={"score": 1.0}) as mocked:
            score = compute_mixed_score(
                data_source="m2rl_ifbench",
                solution_str="ok",
                ground_truth="",
                extra_info={"rm_type": "ifbench"},
            )

        self.assertEqual(score["score"], 1.0)
        mocked.assert_called_once()

    def test_mixed_reward_delegates_math_to_default_reward(self) -> None:
        with patch("grpo.rewards.mixed._compute_default_score", return_value=0.5) as mocked:
            score = compute_mixed_score(
                data_source="DeepMath-test",
                solution_str="x",
                ground_truth="y",
                extra_info={"domain": "math"},
            )

        self.assertEqual(score, 0.5)
        mocked.assert_called_once()


class M2RLDataTests(unittest.TestCase):
    def test_science_frame_converts_to_verl_schema(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is right?",
                    "label": "B",
                    "metadata": {"choices": ["wrong", "right"], "correct_letter": "B"},
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="gpqa")
        self.assertTrue(report.is_valid)

        output = m2rl_frame_to_verl(frame, rm_type="gpqa", split="train", domain="science")
        row = output.iloc[0]
        self.assertEqual(row["data_source"], "m2rl_gpqa")
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["reward_model"]["ground_truth"], "B")
        self.assertEqual(row["extra_info"]["rm_type"], "gpqa")
        self.assertEqual(row["extra_info"]["opd_teacher"], "science")

    def test_ifbench_validation_rejects_missing_instruction_metadata(self) -> None:
        frame = pd.DataFrame([{"prompt": "Obey two constraints.", "label": ""}])

        report = validate_m2rl_frame(frame, rm_type="ifbench")

        self.assertFalse(report.is_valid)
        self.assertIn("missing IFBench instruction_id_list metadata", report.invalid_rows[0]["reasons"])

    def test_ifbench_frame_with_metadata_converts(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Write exactly two words.",
                    "label": "",
                    "metadata": {
                        "instruction_id_list": ["length_constraints:number_words"],
                        "kwargs": [{"num_words": 2, "relation": "exactly"}],
                        "prompt_text": "Write exactly two words.",
                    },
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="ifbench")
        self.assertTrue(report.is_valid)
        normalized = m2rl_frame_to_verl(frame, rm_type="ifbench", split="train", domain="if")
        self.assertEqual(normalized.iloc[0]["extra_info"]["rm_type"], "ifbench")
        self.assertEqual(normalized.iloc[0]["extra_info"]["opd_teacher"], "if")

    def test_validate_accepts_verl_parquet_roundtrip_schema(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is correct?",
                    "label": "B",
                    "metadata": {"choices": ["wrong", "right"], "correct_letter": "B"},
                }
            ]
        )
        normalized = m2rl_frame_to_verl(frame, rm_type="gpqa", split="validation", domain="science")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "science.parquet"
            normalized.to_parquet(path, index=False)
            roundtrip = pd.read_parquet(path)

        report = validate_m2rl_frame(roundtrip, rm_type="gpqa")
        self.assertTrue(report.is_valid)

    def test_nemotron_instruction_following_row_normalizes_for_ifbench(self) -> None:
        row = {
            "id": 17616,
            "instruction_id_list": ["length_constraints:nth_paragraph_first_word"],
            "kwargs": [{"nth_paragraph": 3, "first_word": "crash"}],
            "prompt": "Write four paragraphs.",
            "responses_create_params": {"input": [{"role": "user", "content": "Write four paragraphs."}]},
        }

        domain, normalized = normalize_nemotron_record(row)

        self.assertEqual(domain, "if")
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["metadata"]["record_id"], 17616)
        self.assertEqual(normalized["metadata"]["instruction_id_list"], row["instruction_id_list"])
        self.assertEqual(normalized["metadata"]["prompt_text"], "Write four paragraphs.")


class M2RLConfigTests(unittest.TestCase):
    def test_if_smoke_config_points_reward_and_training_data(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "grpo" / "configs" / "m2rl_if_smoke.yaml"
        text = config_path.read_text(encoding="utf-8")

        self.assertIn("data/G-OPD-Training-Data/IF/train.parquet", text)
        self.assertIn("total_training_steps: 2", text)
        self.assertIn("custom_reward_function.path=grpo/rewards/m2rl.py", text)
        self.assertIn("custom_reward_function.name=compute_score", text)

    def test_science_config_points_reward_and_training_data(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "grpo" / "configs" / "m2rl_science.yaml"
        text = config_path.read_text(encoding="utf-8")

        self.assertIn("data/G-OPD-Training-Data/Science/train.parquet", text)
        self.assertIn("test_freq: -1", text)
        self.assertIn("val_before_train: false", text)
        self.assertIn("custom_reward_function.path=grpo/rewards/m2rl.py", text)
        self.assertIn("custom_reward_function.name=compute_score", text)


if __name__ == "__main__":
    unittest.main()
