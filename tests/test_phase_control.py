from __future__ import annotations

import unittest

import torch

from mopd_verl.domain_gradient.phase_control import (
    PhaseControlState,
    PhaseGapObservation,
    initial_phase_control_state,
    phase_token_weights,
    successor_span_scores,
    update_phase_control_state,
)


class PhaseControlTests(unittest.TestCase):
    def test_domain_lists_only_weight_matching_domain_tokens(self) -> None:
        weights = phase_token_weights(
            torch.tensor([[10, 20, 30], [10, 20, 30]]),
            torch.ones(2, 3),
            ["math", "code"],
            domain_token_ids={"math": [10], "code": [20]},
            control_weight=2.0,
            phase_enabled=False,
            span_enabled=False,
            phase_gates={},
            span_length=2,
            span_decay_tau=1.0,
            normalize_per_domain=False,
        )

        torch.testing.assert_close(
            weights,
            torch.tensor([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]),
        )

    def test_a1_phase_gate_changes_marker_without_weighting_span(self) -> None:
        weights = phase_token_weights(
            torch.tensor([[10, 20, 30]]),
            torch.ones(1, 3),
            ["math"],
            domain_token_ids={"math": [10]},
            control_weight=2.0,
            phase_enabled=True,
            span_enabled=False,
            phase_gates={"math": 0.75},
            span_length=2,
            span_decay_tau=2.0,
            normalize_per_domain=False,
        )

        torch.testing.assert_close(
            weights,
            torch.tensor([[1.75, 1.0, 1.0]]),
        )

    def test_phase_gate_splits_weight_between_marker_and_successors(self) -> None:
        weights = phase_token_weights(
            torch.tensor([[10, 20, 30, 40]]),
            torch.ones(1, 4),
            ["math"],
            domain_token_ids={"math": [10]},
            control_weight=2.0,
            phase_enabled=True,
            span_enabled=True,
            phase_gates={"math": 0.75},
            span_length=2,
            span_decay_tau=2.0,
            normalize_per_domain=False,
        )

        expected = torch.tensor(
            [[
                1.75,
                1.0 + 0.25 * torch.exp(torch.tensor(-0.5)).item(),
                1.0 + 0.25 * torch.exp(torch.tensor(-1.0)).item(),
                1.0,
            ]]
        )
        torch.testing.assert_close(weights, expected)

    def test_overlapping_successor_spans_take_maximum(self) -> None:
        marker_mask = torch.tensor([[True, False, True, False, False]])
        scores = successor_span_scores(
            marker_mask,
            torch.ones(1, 5),
            span_length=2,
            decay_tau=1.0,
        )

        self.assertAlmostEqual(
            scores[0, 3].item(),
            torch.exp(torch.tensor(-1.0)).item(),
        )
        self.assertAlmostEqual(
            scores[0, 4].item(),
            torch.exp(torch.tensor(-2.0)).item(),
        )

    def test_per_domain_normalization_preserves_mean_one(self) -> None:
        weights = phase_token_weights(
            torch.tensor([[10, 20, 30], [20, 30, 40]]),
            torch.ones(2, 3),
            ["math", "code"],
            domain_token_ids={"math": [10], "code": [20]},
            control_weight=2.0,
            phase_enabled=False,
            span_enabled=False,
            phase_gates={},
            span_length=2,
            span_decay_tau=1.0,
            normalize_per_domain=True,
        )

        self.assertAlmostEqual(weights[0].mean().item(), 1.0)
        self.assertAlmostEqual(weights[1].mean().item(), 1.0)

    def test_gate_follows_relative_control_and_span_gap(self) -> None:
        state = initial_phase_control_state(("math",), initial_gate=0.8)
        for step, (control_gap, span_gap) in enumerate(
            (
                (2.0, 2.0),
                (1.8, 1.95),
                (1.6, 1.90),
            ),
            start=1,
        ):
            state = update_phase_control_state(
                state,
                {
                    "math": PhaseGapObservation(
                        control_gap=control_gap,
                        span_gap=span_gap,
                        control_count=10,
                        span_count=20,
                    )
                },
                window_steps=2,
                ema_beta=0.0,
                temperature=0.1,
                step=step,
            )
        span_favored_gate = state.gates[0]
        for step, (control_gap, span_gap) in enumerate(
            (
                (1.6, 1.7),
                (1.6, 1.4),
            ),
            start=4,
        ):
            state = update_phase_control_state(
                state,
                {
                    "math": PhaseGapObservation(
                        control_gap=control_gap,
                        span_gap=span_gap,
                        control_count=10,
                        span_count=20,
                    )
                },
                window_steps=2,
                ema_beta=0.0,
                temperature=0.1,
                step=step,
            )

        self.assertLess(span_favored_gate, 0.5)
        self.assertGreater(state.gates[0], 0.5)
        self.assertEqual(
            PhaseControlState.from_mapping(state.as_dict()),
            state,
        )


if __name__ == "__main__":
    unittest.main()
