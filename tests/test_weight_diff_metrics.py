from __future__ import annotations

import unittest

from mopd_verl.weight_diff_metrics import (
    pair_stats_from_sums,
    sparse_conflict_stats,
)
from mopd_verl.weight_diff_conflict import TopKSupport


class WeightDiffMetricsTests(unittest.TestCase):
    def test_negative_cosine_becomes_conflict_strength(self) -> None:
        stats = pair_stats_from_sums(
            teacher_a="math",
            teacher_b="code",
            tensor_count=1,
            parameter_count=3,
            dot=-6.0,
            norm_sq_a=9.0,
            norm_sq_b=4.0,
        )

        self.assertEqual(stats.cosine, -1.0)
        self.assertEqual(stats.conflict_strength, 1.0)
        self.assertEqual(stats.alignment_strength, 0.0)
        self.assertEqual(stats.negative_dot, 6.0)
        self.assertEqual(stats.negative_dot_per_param, 2.0)

    def test_positive_cosine_has_no_normalized_conflict(self) -> None:
        stats = pair_stats_from_sums(
            teacher_a="math",
            teacher_b="code",
            tensor_count=1,
            parameter_count=2,
            dot=3.0,
            norm_sq_a=9.0,
            norm_sq_b=1.0,
        )

        self.assertEqual(stats.cosine, 1.0)
        self.assertEqual(stats.conflict_strength, 0.0)
        self.assertEqual(stats.alignment_strength, 1.0)
        self.assertEqual(stats.negative_dot, 0.0)

    def test_sparse_conflict_uses_overlapping_top_coordinates(self) -> None:
        left = {("layer.weight", 0): 2.0, ("layer.weight", 1): -1.0}
        right = {("layer.weight", 0): -3.0, ("other.weight", 4): 5.0}

        stats = sparse_conflict_stats(left, right)

        self.assertEqual(stats.overlap_size, 1)
        self.assertEqual(stats.support_jaccard, 1 / 3)
        self.assertEqual(stats.sparse_dot, -6.0)
        self.assertEqual(stats.sparse_cosine, -1.0)
        self.assertEqual(stats.sparse_conflict_strength, 1.0)
        self.assertEqual(stats.sparse_negative_dot, 6.0)
        self.assertEqual(stats.sparse_abs_overlap_mass, 2.0)

    def test_topk_support_ignores_zero_diffs(self) -> None:
        support = TopKSupport(max_items=2)

        support.offer(("layer.weight", 0), 0.0)
        support.offer(("layer.weight", 1), -0.5)

        self.assertEqual(support.as_mapping(), {("layer.weight", 1): -0.5})


if __name__ == "__main__":
    unittest.main()
