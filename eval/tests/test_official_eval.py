from __future__ import annotations

import unittest

from eval.domains.greasoner.official_eval import evaluate_bbeh_correctness, extract_solution, get_prediction
from eval.domains.toolrl.common import extract_tool_calls, score_single_tool_call
from eval.official_runner import _resolve_datasets


class OfficialEvalHelpersTest(unittest.TestCase):
    def test_greasoner_extracts_boxed_answer(self) -> None:
        self.assertEqual(extract_solution("work\n\\boxed{A}\n"), "A")
        self.assertEqual(get_prediction("reasoning\n\\boxed{C}"), "C")

    def test_bbeh_fuzzy_correctness(self) -> None:
        self.assertTrue(evaluate_bbeh_correctness("(a)", "a"))
        self.assertTrue(evaluate_bbeh_correctness("1.0", "1"))

    def test_toolrl_extracts_and_scores_tool_calls(self) -> None:
        response = '<tool_call>\n{"name": "Search", "parameters": {"query": "Ada"}}\n</tool_call>'
        calls = extract_tool_calls(response)
        score = score_single_tool_call(calls, {"name": "Search", "parameters": {"query": "Ada"}})

        self.assertEqual(calls, [{"name": "Search", "parameters": {"query": "Ada"}}])
        self.assertEqual(score, 1)

    def test_resolves_dataset_selection(self) -> None:
        self.assertIn("api_bank", _resolve_datasets(["toolrl"], ["all"]))
        self.assertEqual(_resolve_datasets(["greasoner"], ["mmlupro"]), ["mmlupro"])
        self.assertIn("gpqa_d", _resolve_datasets(["greasoner"], ["all"]))
        self.assertIn("theoremqa", _resolve_datasets(["greasoner"], ["all"]))
        with self.assertRaises(ValueError):
            _resolve_datasets(["greasoner"], ["api_bank"])


if __name__ == "__main__":
    unittest.main()
