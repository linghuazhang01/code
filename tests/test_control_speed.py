from __future__ import annotations

import unittest

import torch

from mopd_verl.domain_gradient.control_speed import (
    ControlGapObservation,
    ControlSpeedState,
    domain_control_token_weights,
    initial_control_speed_state,
    piecewise_linear_weight,
    update_control_speed_state,
)


KNOTS = (
    (-0.0025, 0.0),
    (0.0, 0.2),
    (0.005, 2.0),
    (0.010, 3.0),
    (0.015, 4.0),
)


class ControlSpeedMappingTests(unittest.TestCase):
    def test_piecewise_mapping_matches_anchors_and_clamps(self) -> None:
        self.assertEqual(piecewise_linear_weight(-1.0, KNOTS), 0.0)
        self.assertEqual(piecewise_linear_weight(-0.0025, KNOTS), 0.0)
        self.assertEqual(piecewise_linear_weight(0.0, KNOTS), 0.2)
        self.assertEqual(piecewise_linear_weight(0.005, KNOTS), 2.0)
        self.assertEqual(piecewise_linear_weight(0.010, KNOTS), 3.0)
        self.assertEqual(piecewise_linear_weight(0.015, KNOTS), 4.0)
        self.assertEqual(piecewise_linear_weight(1.0, KNOTS), 4.0)
        self.assertAlmostEqual(
            piecewise_linear_weight(0.0025, KNOTS),
            1.1,
        )

    def test_domain_weights_are_independent_and_mean_one(self) -> None:
        token_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        response_mask = torch.ones_like(token_ids)

        weights = domain_control_token_weights(
            token_ids,
            response_mask,
            ["math", "code"],
            domain_token_ids={"math": [10, 11], "code": [20]},
            domain_weights={"math": 3.0, "code": 0.0},
            normalize_per_domain=True,
        )

        self.assertTrue(torch.allclose(weights[0].mean(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(weights[1].mean(), torch.tensor(1.0)))
        self.assertGreater(float(weights[0, 0]), 1.0)
        self.assertEqual(float(weights[1, 0]), 0.0)
        self.assertGreater(float(weights[1, 1]), 1.0)


class ControlSpeedStateTests(unittest.TestCase):
    @staticmethod
    def _observation(gap: float, count: int = 256) -> ControlGapObservation:
        return ControlGapObservation(
            gap=gap,
            count=count,
            applied_normalized_weight=1.0,
        )

    def test_five_step_speed_updates_weight_with_one_step_snapshot(self) -> None:
        state = initial_control_speed_state(
            ("math", "code"),
            initial_weight=3.0,
            weight_knots=KNOTS,
        )
        applied_before_update = state.weight_map()
        for step, math_gap in enumerate((0.10, 0.09, 0.08, 0.07, 0.06, 0.05)):
            state = update_control_speed_state(
                state,
                {
                    "math": self._observation(math_gap),
                    "code": self._observation(0.10),
                },
                window_steps=5,
                ema_beta=0.0,
                update_interval_steps=2,
                minimum_occurrences=128,
                step=step,
            )

        self.assertEqual(applied_before_update, {"math": 3.0, "code": 3.0})
        self.assertAlmostEqual(state.speeds[0], 0.01)
        self.assertAlmostEqual(state.weights[0], 3.0)
        self.assertAlmostEqual(state.speeds[1], 0.0)
        self.assertAlmostEqual(state.weights[1], 0.2)
        self.assertEqual(state.reference_steps, (0, 0))
        self.assertEqual(state.last_weight_update_steps, (5, 5))

    def test_update_interval_holds_weight_between_update_steps(self) -> None:
        state = initial_control_speed_state(
            ("math",),
            initial_weight=3.0,
            weight_knots=KNOTS,
        )
        for step, gap in enumerate((0.10, 0.09, 0.08, 0.07, 0.06, 0.05)):
            state = update_control_speed_state(
                state,
                {"math": self._observation(gap)},
                window_steps=5,
                ema_beta=0.0,
                update_interval_steps=2,
                minimum_occurrences=128,
                step=step,
            )
        weight_at_step_five = state.weights[0]
        state = update_control_speed_state(
            state,
            {"math": self._observation(0.10)},
            window_steps=5,
            ema_beta=0.0,
            update_interval_steps=2,
            minimum_occurrences=128,
            step=6,
        )

        self.assertEqual(state.weights[0], weight_at_step_five)
        self.assertEqual(state.last_weight_update_steps[0], 5)

    def test_insufficient_occurrences_do_not_change_state(self) -> None:
        state = initial_control_speed_state(
            ("science",),
            initial_weight=3.0,
            weight_knots=KNOTS,
        )
        updated = update_control_speed_state(
            state,
            {"science": self._observation(0.1, count=127)},
            window_steps=5,
            ema_beta=0.8,
            update_interval_steps=2,
            minimum_occurrences=128,
            step=1,
        )

        self.assertEqual(updated.weights, (3.0,))
        self.assertEqual(updated.observation_counts, (0,))
        self.assertEqual(updated.gap_histories, ((),))

    def test_checkpoint_round_trip_preserves_controller_state(self) -> None:
        state = initial_control_speed_state(
            ("math", "code"),
            initial_weight=3.0,
            weight_knots=KNOTS,
        )
        restored = ControlSpeedState.from_mapping(state.as_dict())

        self.assertEqual(restored, state)


if __name__ == "__main__":
    unittest.main()
