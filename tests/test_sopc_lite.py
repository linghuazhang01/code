from __future__ import annotations

import random
import unittest

from mopd_verl.sopc_lite import GapVector, compute_collision_result, compute_pair_collision, top_mass_support


class SOPCLiteTests(unittest.TestCase):
    def test_top_mass_support_uses_largest_absolute_entries(self) -> None:
        support = top_mass_support([0.0, -4.0, 1.0, 3.0], 0.5)

        self.assertEqual(support, {1})

    def test_negative_sparse_collision_is_positive_when_signs_disagree(self) -> None:
        left = GapVector(step=1, domain="math", values=(4.0, 1.0, 0.0), token_count=3)
        right = GapVector(step=1, domain="code", values=(-3.0, 1.0, 0.0), token_count=3)

        row = compute_pair_collision(
            left,
            right,
            top_mass_fraction=0.8,
            random_trials=0,
            rng=random.Random(0),
        )

        self.assertGreater(row.sparse_negative_collision, 0.0)
        self.assertLess(row.sparse_cosine or 0.0, 0.0)

    def test_vocab_vectors_are_comparable(self) -> None:
        left = GapVector(
            step=1,
            domain="math",
            values=(1.0, -2.0, 0.0),
            token_count=3,
            coordinate_space="vocab",
            source_field="gap_signed_sum_vector_vocab",
        )
        right = GapVector(
            step=1,
            domain="code",
            values=(-1.0, 2.0, 0.0),
            token_count=3,
            coordinate_space="vocab",
            source_field="gap_signed_sum_vector_vocab",
        )

        result = compute_collision_result([left, right], top_mass_fraction=1.0, random_trials=0)

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(len(result.skipped_pairs), 0)
        self.assertGreater(result.rows[0].sparse_negative_collision, 0.0)

    def test_mismatched_occurrence_vectors_are_skipped(self) -> None:
        left = GapVector(step=1, domain="math", values=(1.0, -2.0), token_count=2)
        right = GapVector(step=1, domain="code", values=(-1.0, 2.0, 0.5), token_count=3)

        result = compute_collision_result([left, right], top_mass_fraction=1.0, random_trials=0)

        self.assertEqual(len(result.rows), 0)
        self.assertEqual(result.skipped_pairs[0].reason, "vector_size_mismatch")


if __name__ == "__main__":
    unittest.main()
