import unittest

from scripts.prepare_wildsci_data import OPTION_LETTERS, prepare_records


def _record(
    *,
    question: str,
    answer: str,
    voting_type: str,
    votes: list[str | None],
) -> dict[str, object]:
    return {
        "paper_id": "paper-1",
        "discipline": "Physics",
        "nc_domain": "Physical sciences",
        "nc_subdomain": "Physics",
        "question": question,
        "options": {letter: f"option {letter}" for letter in OPTION_LETTERS},
        "answer": answer,
        "rationale": "This field must not enter the prepared row.",
        "rationale_answer": "J",
        "voting_type": voting_type,
        "voting_answers": votes,
    }


class PrepareWildSciDataTests(unittest.TestCase):
    def test_prepares_only_aligned_rows_in_verl_schema(self) -> None:
        records = [
            _record(
                question="Aligned question?",
                answer="B",
                voting_type="all_aligned",
                votes=["B"] * 8,
            ),
            _record(
                question="Majority question?",
                answer="C",
                voting_type="majority_aligned",
                votes=["C", "C", "C", "C", "C", "A", None, "K"],
            ),
            _record(
                question="Divergent question?",
                answer="D",
                voting_type="majority_divergent",
                votes=["A"] * 8,
            ),
        ]

        rows, stats = prepare_records(records)

        self.assertEqual(len(rows), 2)
        self.assertEqual(stats.raw_rows, 3)
        self.assertEqual(stats.selected_rows, 2)
        self.assertEqual(
            set(rows[0]),
            {"data_source", "prompt", "ability", "reward_model", "extra_info"},
        )
        self.assertEqual(rows[0]["reward_model"]["ground_truth"], "B")
        self.assertEqual(rows[0]["extra_info"]["valid_letters"], list(OPTION_LETTERS))
        self.assertNotIn("rationale", rows[0]["extra_info"])
        self.assertIn("Answer: \\boxed{X}", rows[0]["prompt"][0]["content"])

    def test_removes_exact_duplicates_with_the_same_answer(self) -> None:
        duplicate = _record(
            question="Duplicate question?",
            answer="A",
            voting_type="all_aligned",
            votes=["A"] * 8,
        )

        rows, stats = prepare_records([duplicate, duplicate])

        self.assertEqual(len(rows), 1)
        self.assertEqual(stats.exact_duplicates_removed, 1)

    def test_excludes_aligned_social_science_rows(self) -> None:
        record = _record(
            question="Social science question?",
            answer="A",
            voting_type="all_aligned",
            votes=["A"] * 8,
        )
        record["discipline"] = "Social Sciences"

        rows, stats = prepare_records([record])

        self.assertEqual(rows, [])
        self.assertEqual(stats.excluded_discipline_counts, {"Social Sciences": 1})

    def test_rejects_invalid_majority_alignment(self) -> None:
        record = _record(
            question="Invalid vote question?",
            answer="A",
            voting_type="majority_aligned",
            votes=["A", "A", "A", "A", "B", "B", "B", "B"],
        )

        with self.assertRaisesRegex(ValueError, "majority_aligned"):
            prepare_records([record])


if __name__ == "__main__":
    unittest.main()
