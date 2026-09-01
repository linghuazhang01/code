"""Regression tests for canonical G-OPD Code prompt artifact validation."""

from __future__ import annotations

import unittest
from pathlib import Path

from eval.data_prep.code_prompt_validation import (
    validate_code_prompt_artifact,
    validate_evalplus_user_content,
    validate_lcb_user_content,
)
from eval.domains.code.prompting import (
    EVALPLUS_CODE_INSTRUCTION,
    build_evalplus_prompt,
    build_lcb_qwen3_non_thinking_prompt,
)


class CodePromptValidationTest(unittest.TestCase):
    def test_accepts_canonical_user_content(self) -> None:
        validate_evalplus_user_content(build_evalplus_prompt("Implement solve()."))
        validate_lcb_user_content(
            build_lcb_qwen3_non_thinking_prompt("Read two integers.")
        )

    def test_rejects_previous_evalplus_two_newline_prompt(self) -> None:
        stale_prompt = f"Implement solve().\n\n{EVALPLUS_CODE_INSTRUCTION}"

        with self.assertRaisesRegex(ValueError, "exact G-OPD"):
            validate_evalplus_user_content(stale_prompt)

    def test_rejects_evalplus_four_newline_prompt(self) -> None:
        stale_prompt = f"Implement solve().\n\n\n\n{EVALPLUS_CODE_INSTRUCTION}"

        with self.assertRaisesRegex(ValueError, "exact G-OPD"):
            validate_evalplus_user_content(stale_prompt)

    def test_evalplus_builder_matches_active_gopd_call_chain(self) -> None:
        source_prompt = "\n\ndef solve():\n    pass\n\n"
        codegen_argument = source_prompt.strip() + "\n"
        gopd_user_content = (
            codegen_argument + "\n\n" + EVALPLUS_CODE_INSTRUCTION
        )

        self.assertEqual(build_evalplus_prompt(source_prompt), gopd_user_content)

    def test_rejects_previous_lcb_preamble(self) -> None:
        stale_prompt = (
            "You will be given a question (problem specification) and will generate a "
            "correct Python program that matches the specification and passes all tests.\n\n"
            "Question:\nRead two integers.\n\n\n\n"
            f"{EVALPLUS_CODE_INSTRUCTION}"
        )

        with self.assertRaisesRegex(ValueError, "G-OPD template"):
            validate_lcb_user_content(stale_prompt)

    def test_current_code_artifacts_pass_preflight(self) -> None:
        root = Path(__file__).resolve().parents[1] / "data/eval_data/code"
        for dataset_name in (
            "HumanEvalPlus",
            "MBPPPlus",
            "LiveCodeBench-v5",
            "LiveCodeBench",
        ):
            with self.subTest(dataset=dataset_name):
                validate_code_prompt_artifact(root / dataset_name / "test.parquet")


if __name__ == "__main__":
    unittest.main()
