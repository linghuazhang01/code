from __future__ import annotations

import unittest

from eval.domains.code.prompting import (
    EVALPLUS_CODE_INSTRUCTION,
    EVALPLUS_PROMPT_TEMPLATE,
    LCB_QWEN3_PROMPT_TEMPLATE,
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
GOPD_LCB_QWEN3_PREAMBLE = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests. You will NOT "
    "return anything except for the program."
)


class CodeEvalAlignmentTest(unittest.TestCase):
    def test_evalplus_uses_gopd_training_response_contract(self) -> None:
        prompt = build_evalplus_prompt("Implement solve().")

        self.assertEqual(EVALPLUS_CODE_INSTRUCTION, GOPD_TRAINING_CODE_SUFFIX)
        self.assertEqual(
            prompt,
            f"Implement solve().\n\n\n{GOPD_TRAINING_CODE_SUFFIX}",
        )
        self.assertEqual(
            EVALPLUS_PROMPT_TEMPLATE.format(task_prompt="Implement solve()."),
            prompt,
        )

    def test_livecodebench_uses_exact_gopd_qwen3_non_thinking_prompt(self) -> None:
        prompt = build_lcb_qwen3_non_thinking_prompt("Read two integers.")

        self.assertEqual(
            prompt,
            f"{GOPD_LCB_QWEN3_PREAMBLE}\n\n"
            f"Question:\nRead two integers.\n\n\n\n"
            f"{GOPD_TRAINING_CODE_SUFFIX}",
        )
        self.assertEqual(
            LCB_QWEN3_PROMPT_TEMPLATE.format(question_content="Read two integers."),
            prompt,
        )

    def test_livecodebench_preserves_question_whitespace_like_gopd(self) -> None:
        question_content = "\nRead two integers.\n"

        self.assertIn(
            f"Question:\n{question_content}\n\n\n\n",
            build_lcb_qwen3_non_thinking_prompt(question_content),
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
