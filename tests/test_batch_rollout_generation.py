from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from eval.common import EvalSample
from eval.runner import generate_vllm_batch


class _SamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _Tokenizer:
    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        return "rendered-prompt"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2]


class BatchRolloutGenerationTest(unittest.TestCase):
    def test_one_vllm_request_returns_all_eight_rollouts(self) -> None:
        captured: dict[str, object] = {}

        class FakeLlm:
            def generate(self, prompts: list[str], params: _SamplingParams) -> list[object]:
                captured["prompts"] = prompts
                captured["params"] = params.kwargs
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(text=f"answer-{index}", token_ids=[index])
                            for index in range(8)
                        ]
                    )
                ]

        sample = EvalSample(
            sample_id="sample-1",
            dataset="AIME2024",
            ability="math",
            messages=[{"role": "user", "content": "question"}],
            ground_truth="1",
        )
        fake_vllm = SimpleNamespace(SamplingParams=_SamplingParams)
        with (
            patch.dict(sys.modules, {"vllm": fake_vllm}),
            patch(
                "eval.runner.score_completion",
                return_value=(1.0, "1", []),
            ),
        ):
            results = generate_vllm_batch(
                FakeLlm(),
                _Tokenizer(),
                [sample],
                mode="non_thinking",
                max_new_tokens=16,
                temperature=1.0,
                top_p=1.0,
                score_code=False,
                save_completion=True,
                generation_seed=42,
                num_return_sequences=8,
            )

        self.assertEqual(captured["params"]["n"], 8)
        self.assertEqual(captured["params"]["seed"], 42)
        self.assertEqual(len(captured["prompts"]), 1)
        self.assertEqual(len(results), 8)
        self.assertEqual([result.rollout_index for result in results], list(range(8)))


if __name__ == "__main__":
    unittest.main()
