from __future__ import annotations

import unittest

from eval.domains.code.prompting import (
    EVALPLUS_CODE_INSTRUCTION,
    build_evalplus_prompt,
    build_lcb_qwen3_non_thinking_prompt,
)
from mopd_verl.code_reward import _assert_score, _extract_code


GOPD_TRAINING_CODE_SUFFIX = (
    "Write Python code to solve the problem. Present the code in \n"
    "```python\n"
    "Your code\n"
    "```\n"
    "at the end.\n"
    "You need to think first then write the Python code."
)


class CodeEvalAlignmentTest(unittest.TestCase):
    def test_evalplus_uses_gopd_training_response_contract(self) -> None:
        prompt = build_evalplus_prompt("Implement solve().")

        self.assertEqual(EVALPLUS_CODE_INSTRUCTION, GOPD_TRAINING_CODE_SUFFIX)
        self.assertEqual(
            prompt,
            f"Implement solve().\n\n{GOPD_TRAINING_CODE_SUFFIX}",
        )

    def test_livecodebench_allows_reasoning_before_final_code(self) -> None:
        prompt = build_lcb_qwen3_non_thinking_prompt("Read two integers.")

        self.assertNotIn("return anything except for the program", prompt)
        self.assertTrue(
            prompt.endswith(f"Read two integers.\n\n{GOPD_TRAINING_CODE_SUFFIX}")
        )

    def test_extractor_uses_last_python_block(self) -> None:
        completion = (
            "First attempt:\n"
            "```python\nprint('draft')\n```\n"
            "The draft is wrong. The corrected answer is:\n"
            "```python\nprint('final')\n```"
        )

        self.assertEqual(_extract_code(completion).strip(), "print('final')")

    def test_extractor_keeps_pure_code_completion(self) -> None:
        completion = "import sys\nprint(sys.version_info.major)"

        self.assertEqual(_extract_code(completion), completion)

    def test_evalplus_scorer_executes_last_python_block(self) -> None:
        completion = (
            "First attempt:\n"
            "```python\ndef answer():\n    return 0\n```\n"
            "After checking, the corrected answer is:\n"
            "```python\ndef answer():\n    return 1\n```"
        )

        score, _ = _assert_score(
            completion,
            {"assert_case": "assert answer() == 1"},
        )

        self.assertEqual(score, 1.0)

    def test_extractor_matches_training_for_non_python_fence(self) -> None:
        completion = "Reasoning first.\n```py\nprint('answer')\n```"

        self.assertEqual(_extract_code(completion), "Reasoning first.\n")


if __name__ == "__main__":
    unittest.main()
