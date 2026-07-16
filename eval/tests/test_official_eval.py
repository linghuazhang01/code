from __future__ import annotations

import unittest

from eval.domains.science.official_eval import extract_solution, get_prediction
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
        self.assertIn("supergpqa", _resolve_datasets(["science"], ["all"]))
        with self.assertRaises(ValueError):
            _resolve_datasets(["science"], ["api_bank"])


if __name__ == "__main__":
    unittest.main()
