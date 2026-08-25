from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from mopd_verl.domain_gradient.adaptive_neighborhood import (
    PerTokenAdaptiveNeighborhoodSpec,
    build_per_token_adaptive_neighborhood,
)
from mopd_verl.domain_gradient.adaptive_neighborhood_metrics import (
    adaptive_neighborhood_metric_components,
    adaptive_neighborhood_metrics,
    aggregate_adaptive_neighborhood_metrics,
)


def _spec(**overrides: object) -> PerTokenAdaptiveNeighborhoodSpec:
    values: dict[str, object] = {
        "domains": ("math",),
        "domain_token_ids": (("math", (10,)),),
        "max_distance": 1,
        "epsilon": 1e-8,
        "clip_max": 1.5,
        "threshold": 0.3,
        "min_far_tokens": 1,
        "control_weight": 4.0,
        "normalize_per_response": False,
    }
    values.update(overrides)
    return PerTokenAdaptiveNeighborhoodSpec(**values)


class PerTokenAdaptiveNeighborhoodTests(unittest.TestCase):
    def test_empty_online_cold_start_is_identity(self) -> None:
        losses = torch.tensor([[0.2, 0.4, 0.6]])
        token_ids = torch.tensor([[1, 2, 3]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(domain_token_ids=(("math", ()),)),
        )

        self.assertFalse(result.center_mask.any())
        self.assertFalse(result.selected_neighbor_mask.any())
        torch.testing.assert_close(result.multiplier, torch.ones_like(losses))

    def test_individual_neighbor_scores_use_response_local_lower_median(
        self,
    ) -> None:
        losses = torch.tensor([[0.71, 1.4, 1.4, 0.2, 0.4, 0.6, 0.8]])
        token_ids = torch.tensor([[1, 10, 2, 3, 4, 5, 6]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(),
        )

        self.assertAlmostEqual(float(result.far_baselines[0]), 0.4)
        self.assertEqual(result.far_token_counts.tolist(), [4])
        self.assertEqual(
            result.center_mask.tolist(),
            [[False, True, False, False, False, False, False]],
        )
        self.assertEqual(
            result.candidate_neighbor_mask.tolist(),
            [[True, False, True, False, False, False, False]],
        )
        self.assertTrue(
            torch.equal(
                result.selected_neighbor_mask,
                result.candidate_neighbor_mask,
            )
        )
        self.assertAlmostEqual(float(result.relative_scores[0, 0]), 0.31)
        self.assertAlmostEqual(float(result.relative_scores[0, 2]), 1.0)
        torch.testing.assert_close(
            result.raw_multiplier,
            torch.tensor([[4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0]]),
        )

    def test_overlap_uses_max_pair_score_without_bucket_averaging(self) -> None:
        losses = torch.tensor([[0.6, 0.7, 1.4, 1.4, 2.4, 0.8, 0.9, 0.4, 0.5]])
        token_ids = torch.tensor([[1, 2, 10, 3, 10, 4, 5, 6, 7]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(max_distance=2),
        )

        self.assertAlmostEqual(float(result.far_baselines[0]), 0.4)
        self.assertAlmostEqual(float(result.relative_scores[0, 3]), 1.0)
        self.assertEqual(float(result.raw_multiplier[0, 3]), 4.0)
        self.assertEqual(int(result.selected_neighbor_mask[0, 3]), 1)

    def test_negative_denominator_pass_uses_fixed_control_weight(self) -> None:
        losses = torch.tensor([[0.1, 0.2, 0.5, 0.4]])
        token_ids = torch.tensor([[1, 10, 2, 3]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(),
        )

        self.assertLess(float(result.center_denominators[0, 1]), 0.0)
        self.assertEqual(float(result.relative_scores[0, 0]), 1.5)
        self.assertEqual(float(result.raw_multiplier[0, 0]), 4.0)
        self.assertEqual(
            float(result.raw_multiplier[0, 0]),
            float(result.raw_multiplier[0, 1]),
        )

    def test_missing_far_baseline_keeps_center_only(self) -> None:
        losses = torch.tensor([[0.7, 1.4, 1.2]])
        token_ids = torch.tensor([[1, 10, 2]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(),
        )

        self.assertEqual(result.far_token_counts.tolist(), [0])
        self.assertFalse(result.selected_neighbor_mask.any())
        torch.testing.assert_close(
            result.raw_multiplier,
            torch.tensor([[1.0, 4.0, 1.0]]),
        )

    def test_threshold_is_inclusive_and_normalization_is_per_response(
        self,
    ) -> None:
        losses = torch.tensor([[0.7, 1.4, 1.4, 0.4]])
        token_ids = torch.tensor([[1, 10, 2, 3]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        initial = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(threshold=0.0),
        )
        exact_threshold = float(initial.relative_scores[0, 0])
        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(
                threshold=exact_threshold,
                normalize_per_response=True,
            ),
        )

        self.assertTrue(result.selected_neighbor_mask[0, 0])
        torch.testing.assert_close(
            result.multiplier[valid].mean(),
            torch.tensor(1.0),
            rtol=1e-6,
            atol=1e-6,
        )
        center_multiplier = result.multiplier[0, 1]
        torch.testing.assert_close(result.multiplier[0, 0], center_multiplier)
        torch.testing.assert_close(result.multiplier[0, 2], center_multiplier)

    def test_zero_threshold_does_not_select_zero_score_neighbors(self) -> None:
        losses = torch.tensor([[0.4, 1.4, 0.4, 0.4]])
        token_ids = torch.tensor([[1, 10, 2, 3]])
        valid = torch.ones_like(losses, dtype=torch.bool)

        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(threshold=0.0),
        )

        self.assertTrue(result.eligible_neighbor_mask[0, 0])
        self.assertEqual(float(result.relative_scores[0, 0]), 0.0)
        self.assertFalse(result.selected_neighbor_mask[0, 0])
        self.assertEqual(float(result.raw_multiplier[0, 0]), 1.0)

    def test_metrics_report_extra_coverage_against_fixed_d0(self) -> None:
        losses = torch.tensor([[0.71, 1.4, 1.4, 0.2, 0.4, 0.6, 0.8]])
        token_ids = torch.tensor([[1, 10, 2, 3, 4, 5, 6]])
        valid = torch.ones_like(losses, dtype=torch.bool)
        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(),
        )

        metrics = adaptive_neighborhood_metrics(result, valid, threshold=0.3)

        self.assertEqual(metrics["actor/adaptive_fixed_d0_token_count"], 1.0)
        self.assertEqual(metrics["actor/adaptive_extra_weighted_token_count"], 2.0)
        self.assertEqual(metrics["actor/adaptive_threshold_pass_token_count"], 2.0)
        self.assertEqual(
            metrics["actor/adaptive_threshold_eligible_token_count"],
            2.0,
        )
        self.assertEqual(
            metrics["actor/adaptive_threshold_pass_token_fraction"],
            1.0,
        )
        self.assertEqual(metrics["actor/adaptive_filtered_neighbor_token_count"], 0.0)
        self.assertEqual(metrics["actor/adaptive_extra_to_d0_ratio"], 2.0)
        self.assertAlmostEqual(
            metrics["actor/adaptive_total_weighted_token_fraction"],
            3.0 / 7.0,
        )

    def test_step_metrics_sum_micro_batches_and_actor_ranks_once(self) -> None:
        losses = torch.tensor([[0.71, 1.4, 1.4, 0.2, 0.4, 0.6, 0.8]])
        token_ids = torch.tensor([[1, 10, 2, 3, 4, 5, 6]])
        valid = torch.ones_like(losses, dtype=torch.bool)
        result = build_per_token_adaptive_neighborhood(
            losses,
            valid,
            token_ids,
            valid,
            ("math",),
            _spec(),
        )
        components = adaptive_neighborhood_metric_components(
            result,
            valid,
            threshold=0.3,
            labels=("math",),
            domains=("math",),
        )

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=lambda tensor: tensor.mul_(3.0),
            ) as all_reduce,
        ):
            metrics = aggregate_adaptive_neighborhood_metrics(
                (components, components),
                reduce_distributed=True,
            )

        all_reduce.assert_called_once()
        self.assertEqual(
            metrics["actor/adaptive_threshold_pass_token_count"],
            12.0,
        )
        self.assertEqual(
            metrics["actor/adaptive_threshold_eligible_token_count"],
            12.0,
        )
        self.assertEqual(
            metrics["actor/adaptive_threshold_pass_token_fraction"],
            1.0,
        )
        self.assertEqual(
            metrics[
                "actor/adaptive_domain/math/threshold_pass_token_count"
            ],
            12.0,
        )
        self.assertAlmostEqual(
            metrics[
                "actor/adaptive_domain/math/"
                "threshold_pass_valid_token_fraction"
            ],
            2.0 / 7.0,
        )


if __name__ == "__main__":
    unittest.main()
