from __future__ import annotations

import unittest

from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    occurrence_weighted_optimization_speed,
    paired_selection_bonus,
)


DOMAINS = ("math", "code")
CANDIDATES = (10, 20, 30, 40)
DOMAIN_CANDIDATES = {
    "math": (10, 20),
    "code": (20, 30, 40),
}


def _statistics(
    *entries: tuple[str, int, float, int],
) -> dict[str, dict[int, tuple[float, int]]]:
    result = {domain: {} for domain in DOMAINS}
    for domain, token_id, loss_sum, count in entries:
        result[domain][token_id] = (loss_sum, count)
    return result


class OnlineControlSelectionTests(unittest.TestCase):
    def test_paired_selection_bonus_supports_both_primary_signals(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")

        mask = torch.ones(1, 3, dtype=torch.bool)
        student_entropy = torch.tensor([[0.0, 2.0, 1.0]])
        kl_loss = torch.tensor([[1.0, 2.0, 0.0]])
        teacher_entropy = torch.tensor([[0.0, 1.0, 2.0]])

        kl_bonus = paired_selection_bonus(
            selection_mode="top_kl_student_entropy",
            student_entropy=student_entropy,
            response_mask=mask,
            configured_loss=kl_loss,
        )
        confidence_bonus = paired_selection_bonus(
            selection_mode="top_teacher_confidence_student_entropy",
            student_entropy=student_entropy,
            response_mask=mask,
            teacher_entropy=teacher_entropy,
        )

        torch.testing.assert_close(
            kl_bonus,
            torch.tensor([[0.5, 3.0, 0.5]]),
        )
        torch.testing.assert_close(
            confidence_bonus,
            torch.tensor([[1.0, 2.0, 0.5]]),
        )

    def test_optimization_speed_uses_occurrence_count_weights(self) -> None:
        speed = occurrence_weighted_optimization_speed(
            ((1, 10.0, 100), (2, 0.0, 1), (3, 8.0, 10))
        )

        self.assertIsNotNone(speed)
        assert speed is not None
        self.assertAlmostEqual(speed.value, 1.197080291970803)
        self.assertEqual(speed.observed_step_count, 3)

    def _state(
        self,
        *,
        audit_interval_steps: int = 3,
        window_steps: int = 3,
        minimum_frequency: float = 20.0,
        top_k: int = 2,
        budget_mode: str = "top_k",
        top_p: float = 1.0,
        selection_mode: str = "top_loss",
        weight_mode: str = "fixed",
        strict_occurrence_gate: bool = False,
    ) -> OnlineControlSelectionState:
        return initial_online_control_selection_state(
            DOMAINS,
            CANDIDATES,
            audit_interval_steps=audit_interval_steps,
            window_steps=window_steps,
            min_mean_occurrences_per_step=minimum_frequency,
            strict_occurrence_gate=strict_occurrence_gate,
            top_k=top_k,
            budget_mode=budget_mode,
            top_p=top_p,
            selection_mode=selection_mode,
            weight_mode=weight_mode,
        )

    def test_first_full_window_audits_at_step_three(self) -> None:
        state = self._state()
        for step in (1, 2):
            outcome, state = update_online_control_selection(
                state,
                _statistics(
                    ("math", 10, 30.0, 20),
                    ("code", 20, 40.0, 20),
                ),
                step=step,
            )
            self.assertFalse(outcome.audit_triggered)
            self.assertEqual(state.active_map(), {"math": (), "code": ()})

        outcome, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 30.0, 20),
                ("code", 20, 40.0, 20),
            ),
            step=3,
        )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map(), {"math": (10,), "code": (20,)})
        self.assertEqual(state.last_audit_step, 3)

    def test_frequency_threshold_is_mean_occurrences_per_step(self) -> None:
        state = self._state(top_k=4)
        for step, qualifying_count in ((1, 19), (2, 20), (3, 21)):
            outcome, state = update_online_control_selection(
                state,
                _statistics(
                    ("math", 10, 600.0, qualifying_count),
                    ("math", 20, 5900.0, qualifying_count),
                ),
                step=step,
            )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (20, 10))

        below_state = self._state(top_k=4)
        for step, count in ((1, 20), (2, 20), (3, 19)):
            _, below_state = update_online_control_selection(
                below_state,
                _statistics(("math", 30, 5900.0, count)),
                step=step,
            )
        self.assertNotIn(30, below_state.active_map()["math"])

    def test_strict_occurrence_gate_requires_more_than_threshold_each_step(
        self,
    ) -> None:
        state = self._state(
            minimum_frequency=20.0,
            top_k=4,
            strict_occurrence_gate=True,
        )
        statistics_by_step = (
            _statistics(
                ("math", 10, 21.0, 21),
                ("math", 20, 20.0, 20),
                ("math", 30, 21.0, 21),
            ),
            _statistics(
                ("math", 10, 21.0, 21),
                ("math", 20, 50.0, 50),
            ),
            _statistics(
                ("math", 10, 21.0, 21),
                ("math", 20, 50.0, 50),
                ("math", 30, 21.0, 21),
            ),
        )
        for step, statistics in enumerate(statistics_by_step, start=1):
            outcome, state = update_online_control_selection(
                state,
                statistics,
                step=step,
            )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (10,))
        self.assertTrue(state.strict_occurrence_gate)
        self.assertEqual(
            OnlineControlSelectionState.from_mapping(state.as_dict()),
            state,
        )

    def test_grouped_selector_applies_top_k_independently_per_group(self) -> None:
        groups = {
            domain: {
                "Control": (10, 20),
                "Structure": (30, 40),
            }
            for domain in DOMAINS
        }
        state = initial_online_control_selection_state(
            DOMAINS,
            CANDIDATES,
            audit_interval_steps=1,
            window_steps=1,
            min_mean_occurrences_per_step=20.0,
            strict_occurrence_gate=True,
            top_k=2,
            candidate_token_groups=groups,
            top_k_per_group=1,
        )

        outcome, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 21.0, 21),
                ("math", 20, 10.5, 21),
                ("math", 30, 210.0, 21),
                ("math", 40, 189.0, 21),
            ),
            step=1,
        )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (10, 30))
        self.assertEqual(state.candidate_group_map(), groups)
        self.assertEqual(state.top_k_per_group, 1)
        self.assertEqual(
            OnlineControlSelectionState.from_mapping(state.as_dict()),
            state,
        )

    def test_grouped_selector_rejects_overlap_and_inconsistent_budget(self) -> None:
        common = {
            "audit_interval_steps": 1,
            "window_steps": 1,
            "min_mean_occurrences_per_step": 20.0,
            "top_k_per_group": 1,
        }
        overlap = {
            domain: {"Control": (10, 20), "Structure": (20, 30, 40)}
            for domain in DOMAINS
        }
        with self.assertRaisesRegex(ValueError, "disjoint"):
            initial_online_control_selection_state(
                DOMAINS,
                CANDIDATES,
                top_k=2,
                candidate_token_groups=overlap,
                **common,
            )

        groups = {
            domain: {"Control": (10, 20), "Structure": (30, 40)}
            for domain in DOMAINS
        }
        with self.assertRaisesRegex(ValueError, "top_k must equal"):
            initial_online_control_selection_state(
                DOMAINS,
                CANDIDATES,
                top_k=3,
                candidate_token_groups=groups,
                **common,
            )

    def test_top_p_reaches_valid_token_occurrence_fraction(self) -> None:
        state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            top_k=1,
            budget_mode="top_p",
            top_p=0.05,
        )

        outcome, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 20.0, 2),
                ("math", 20, 15.0, 3),
                ("math", 30, 10.0, 10),
            ),
            step=1,
            valid_token_counts={"math": 100, "code": 0},
        )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (10, 20))
        self.assertEqual(state.budget_mode, "top_p")
        self.assertEqual(state.top_p, 0.05)
        result = next(
            item for item in outcome.domain_results if item.domain == "math"
        )
        self.assertEqual(result.valid_token_count, 100)
        self.assertEqual(result.selected_occurrence_count, 5)
        self.assertEqual(result.selected_occurrence_fraction, 0.05)
        self.assertEqual(result.target_occurrence_count, 5)
        self.assertTrue(result.top_p_target_reached)
        self.assertEqual(result.top_p_occurrence_shortfall, 0)
        empty_result = next(
            item for item in outcome.domain_results if item.domain == "code"
        )
        self.assertEqual(empty_result.valid_token_count, 0)
        self.assertIsNone(empty_result.top_p_target_reached)
        self.assertIsNone(empty_result.top_p_occurrence_shortfall)
        self.assertEqual(
            OnlineControlSelectionState.from_mapping(state.as_dict()),
            state,
        )

    def test_top_p_uses_occurrences_even_when_scores_are_zero(self) -> None:
        all_state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            top_k=1,
            budget_mode="top_p",
            top_p=1.0,
        )
        _, all_state = update_online_control_selection(
            all_state,
            _statistics(
                ("math", 10, 0.0, 1),
                ("math", 20, 0.0, 1),
            ),
            step=1,
            valid_token_counts={"math": 2, "code": 0},
        )
        self.assertEqual(all_state.active_map()["math"], (10, 20))

        fraction_state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            top_k=4,
            budget_mode="top_p",
            top_p=0.75,
        )
        _, fraction_state = update_online_control_selection(
            fraction_state,
            _statistics(
                ("math", 10, 0.0, 1),
                ("math", 20, 0.0, 1),
            ),
            step=1,
            valid_token_counts={"math": 2, "code": 0},
        )
        self.assertEqual(fraction_state.active_map()["math"], (10, 20))

    def test_top_p_denominator_is_all_valid_tokens_not_candidate_types(self) -> None:
        candidates = tuple(range(1, 51))
        state = initial_online_control_selection_state(
            DOMAINS,
            candidates,
            audit_interval_steps=1,
            window_steps=1,
            min_mean_occurrences_per_step=1.0,
            top_k=1,
            budget_mode="top_p",
            top_p=0.05,
        )

        statistics = _statistics(
            *(("math", token_id, 0.0, 1) for token_id in candidates)
        )
        outcome, state = update_online_control_selection(
            state,
            statistics,
            step=1,
            valid_token_counts={"math": 2_000, "code": 0},
        )

        self.assertEqual(len(state.active_map()["math"]), 50)
        result = next(
            item for item in outcome.domain_results if item.domain == "math"
        )
        self.assertEqual(result.target_occurrence_count, 100)
        self.assertFalse(result.top_p_target_reached)
        self.assertEqual(result.top_p_occurrence_shortfall, 50)

    def test_grouped_top_p_targets_union_occurrence_fraction(self) -> None:
        groups = {
            domain: {
                "Control": (10, 20),
                "Structure": (30, 40),
            }
            for domain in DOMAINS
        }
        state = initial_online_control_selection_state(
            DOMAINS,
            CANDIDATES,
            audit_interval_steps=1,
            window_steps=1,
            min_mean_occurrences_per_step=1.0,
            top_k=1,
            candidate_token_groups=groups,
            budget_mode="top_p",
            top_p=0.05,
        )

        _, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 30.0, 3),
                ("math", 20, 80.0, 10),
                ("math", 30, 18.0, 2),
                ("math", 40, 70.0, 10),
            ),
            step=1,
            valid_token_counts={"math": 100, "code": 0},
        )

        self.assertEqual(state.active_map()["math"], (10, 30))

    def test_top_p_accumulates_occurrences_and_denominator_across_window(
        self,
    ) -> None:
        state = self._state(
            audit_interval_steps=2,
            window_steps=2,
            minimum_frequency=1.0,
            top_k=1,
            budget_mode="top_p",
            top_p=0.1,
        )
        _, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 100.0, 10),
                ("math", 20, 45.0, 5),
            ),
            step=1,
            valid_token_counts={"math": 100, "code": 0},
        )
        outcome, state = update_online_control_selection(
            state,
            _statistics(("math", 20, 90.0, 10)),
            step=2,
            valid_token_counts={"math": 100, "code": 0},
        )

        self.assertEqual(state.active_map()["math"], (10, 20))
        result = next(
            item for item in outcome.domain_results if item.domain == "math"
        )
        self.assertEqual(result.valid_token_count, 200)
        self.assertEqual(result.selected_occurrence_count, 25)
        self.assertEqual(result.selected_occurrence_fraction, 0.125)

    def test_top_p_requires_consistent_valid_token_denominator(self) -> None:
        state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            budget_mode="top_p",
            top_p=0.05,
        )
        with self.assertRaisesRegex(ValueError, "valid_token_counts"):
            update_online_control_selection(
                state,
                _statistics(("math", 10, 1.0, 1)),
                step=1,
            )
        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            update_online_control_selection(
                state,
                _statistics(("math", 10, 2.0, 2)),
                step=1,
                valid_token_counts={"math": 1, "code": 0},
            )

    def test_top_p_budget_validation_rejects_invalid_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "budget mode"):
            self._state(budget_mode="mass")
        for invalid_top_p in (0.0, 1.1, float("inf")):
            with self.subTest(top_p=invalid_top_p):
                with self.assertRaisesRegex(ValueError, "top_p"):
                    self._state(budget_mode="top_p", top_p=invalid_top_p)

        groups = {
            domain: {
                "Control": (10, 20),
                "Structure": (30, 40),
            }
            for domain in DOMAINS
        }
        with self.assertRaisesRegex(ValueError, "grouped Top-K"):
            initial_online_control_selection_state(
                DOMAINS,
                CANDIDATES,
                audit_interval_steps=1,
                window_steps=1,
                min_mean_occurrences_per_step=1.0,
                top_k=2,
                candidate_token_groups=groups,
                top_k_per_group=1,
                budget_mode="top_p",
                top_p=0.8,
            )

    def test_ranking_uses_occurrence_mean_absolute_loss(self) -> None:
        state = self._state(top_k=1)
        for step in (1, 2, 3):
            outcome, state = update_online_control_selection(
                state,
                _statistics(
                    ("math", 10, 40.0, 20),
                    ("math", 20, 120.0, 60),
                    ("code", 30, 60.0, 20),
                    ("code", 40, 60.0, 20),
                ),
                step=step,
            )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (10,))
        self.assertEqual(state.active_map()["code"], (30,))

    def test_paired_mode_ranks_mean_bonus_and_stores_raw_weights(self) -> None:
        state = self._state(
            window_steps=1,
            audit_interval_steps=1,
            minimum_frequency=1.0,
            top_k=2,
            selection_mode="top_kl_student_entropy",
            weight_mode="paired",
        )
        outcome, state = update_online_control_selection(
            state,
            _statistics(
                ("math", 10, 2.0, 2),
                ("math", 20, 4.0, 2),
                ("math", 30, 4.0, 4),
            ),
            step=1,
        )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (20, 10))
        self.assertEqual(
            state.active_weight_map()["math"],
            {20: 3.0, 10: 2.0},
        )
        math_result = next(
            result for result in outcome.domain_results if result.domain == "math"
        )
        self.assertEqual(
            tuple(item.mean_selection_score for item in math_result.selected_tokens),
            (2.0, 1.0),
        )
        eligible_distribution = math_result.eligible_score_distribution
        selected_distribution = math_result.selected_score_distribution
        self.assertIsNotNone(eligible_distribution)
        self.assertIsNotNone(selected_distribution)
        assert eligible_distribution is not None
        assert selected_distribution is not None
        self.assertEqual(eligible_distribution.count, 3)
        self.assertAlmostEqual(eligible_distribution.mean, 4.0 / 3.0)
        self.assertAlmostEqual(eligible_distribution.p50, 1.0)
        self.assertAlmostEqual(eligible_distribution.p90, 1.8)
        self.assertEqual(selected_distribution.count, 2)
        self.assertAlmostEqual(selected_distribution.mean, 1.5)
        self.assertAlmostEqual(selected_distribution.std, 0.5)
        self.assertAlmostEqual(selected_distribution.p10, 1.1)
        self.assertAlmostEqual(selected_distribution.p90, 1.9)
        self.assertEqual(
            OnlineControlSelectionState.from_mapping(state.as_dict()),
            state,
        )

    def test_paired_selector_can_keep_fixed_weight_mode(self) -> None:
        state = self._state(
            window_steps=1,
            audit_interval_steps=1,
            minimum_frequency=1.0,
            top_k=1,
            selection_mode="top_teacher_confidence_student_entropy",
            weight_mode="fixed",
        )
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 3.0, 1)),
            step=1,
        )

        self.assertEqual(state.active_map()["math"], (10,))
        self.assertEqual(state.active_weight_map()["math"], {})

    def test_step_gap_clears_paired_ids_and_weights(self) -> None:
        state = self._state(
            window_steps=2,
            audit_interval_steps=2,
            minimum_frequency=1.0,
            top_k=1,
            selection_mode="top_kl_student_entropy",
            weight_mode="paired",
        )
        for step in (1, 2):
            _, state = update_online_control_selection(
                state,
                _statistics(("math", 10, 3.0, 1)),
                step=step,
            )
        outcome, state = update_online_control_selection(
            state,
            _statistics(("math", 20, 2.0, 1)),
            step=4,
        )

        self.assertTrue(outcome.history_reset)
        self.assertEqual(state.active_map(), {"math": (), "code": ()})
        self.assertEqual(state.active_weight_map(), {"math": {}, "code": {}})

    def test_top_speed_ranks_occurrence_weighted_loss_slope(self) -> None:
        state = self._state(
            minimum_frequency=2.0,
            top_k=2,
            selection_mode="top_speed",
        )
        step_rows = (
            ((10, 12.0), (20, 8.0), (30, 4.0), (40, 12.0)),
            ((10, 10.0), (20, 4.0), (30, 5.0), (40, 10.0)),
            ((10, 8.0), (20, 2.0), (30, 6.0), (40, 8.0)),
        )
        for step, rows in enumerate(step_rows, start=1):
            outcome, state = update_online_control_selection(
                state,
                _statistics(
                    *(
                        ("math", token_id, mean_loss * 2, 2)
                        for token_id, mean_loss in rows
                    )
                ),
                step=step,
            )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (20, 10))
        math_result = next(
            result for result in outcome.domain_results if result.domain == "math"
        )
        self.assertEqual(
            tuple(item.token_id for item in math_result.selected_tokens),
            (20, 10),
        )
        self.assertEqual(
            tuple(item.optimization_speed for item in math_result.selected_tokens),
            (3.0, 2.0),
        )
        self.assertEqual(
            tuple(item.observed_step_count for item in math_result.selected_tokens),
            (3, 3),
        )

    def test_top_speed_requires_frequency_and_two_observed_steps(self) -> None:
        state = self._state(
            minimum_frequency=2.0,
            top_k=4,
            selection_mode="top_speed",
        )
        statistics_by_step = (
            _statistics(
                ("math", 10, 10.0, 1),
                ("math", 20, 10.0, 1),
                ("math", 30, 10.0, 1),
            ),
            _statistics(
                ("math", 10, 40.0, 4),
                ("math", 20, 40.0, 4),
                ("math", 30, 40.0, 4),
                ("math", 40, 60.0, 6),
            ),
            _statistics(
                ("math", 10, 10.0, 1),
                ("math", 20, 5.0, 1),
                ("math", 30, 5.0, 1),
            ),
        )
        for step, statistics in enumerate(statistics_by_step, start=1):
            outcome, state = update_online_control_selection(
                state,
                statistics,
                step=step,
            )

        math_result = next(
            result for result in outcome.domain_results if result.domain == "math"
        )
        self.assertEqual(math_result.eligible_token_count, 3)
        self.assertEqual(set(state.active_map()["math"]), {10, 20, 30})
        self.assertNotIn(40, state.active_map()["math"])

    def test_top_speed_orders_negative_speeds_without_filtering(self) -> None:
        state = self._state(
            window_steps=2,
            audit_interval_steps=2,
            minimum_frequency=1.0,
            top_k=2,
            selection_mode="top_speed",
        )
        for step, values in (
            (1, ((10, 2.0), (20, 2.0))),
            (2, ((10, 3.0), (20, 4.0))),
        ):
            outcome, state = update_online_control_selection(
                state,
                _statistics(
                    *(
                        ("math", token_id, mean_loss, 1)
                        for token_id, mean_loss in values
                    )
                ),
                step=step,
            )

        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (10, 20))

    def test_top_speed_step_gap_clears_history_and_active_ids(self) -> None:
        state = self._state(
            window_steps=2,
            audit_interval_steps=2,
            minimum_frequency=1.0,
            top_k=1,
            selection_mode="top_speed",
        )
        for step, loss in ((1, 4.0), (2, 2.0)):
            _, state = update_online_control_selection(
                state,
                _statistics(("math", 10, loss, 1)),
                step=step,
            )
        self.assertEqual(state.active_map()["math"], (10,))

        outcome, state = update_online_control_selection(
            state,
            _statistics(("math", 20, 1.0, 1)),
            step=4,
        )

        self.assertTrue(outcome.history_reset)
        self.assertEqual(tuple(item[0] for item in state.history), (4,))
        self.assertEqual(state.active_map(), {"math": (), "code": ()})

    def test_window_rolls_and_selection_is_unchanged_between_audits(self) -> None:
        state = self._state(audit_interval_steps=3, window_steps=2, top_k=1)
        for step in (1, 2, 3):
            _, state = update_online_control_selection(
                state,
                _statistics(("math", 10, 100.0, 20)),
                step=step,
            )
        self.assertEqual(state.active_map()["math"], (10,))

        for step in (4, 5):
            outcome, state = update_online_control_selection(
                state,
                _statistics(("math", 20, 200.0, 20)),
                step=step,
            )
            self.assertFalse(outcome.audit_triggered)
            self.assertEqual(state.active_map()["math"], (10,))

        outcome, state = update_online_control_selection(
            state,
            _statistics(("math", 20, 200.0, 20)),
            step=6,
        )
        self.assertTrue(outcome.audit_triggered)
        self.assertEqual(state.active_map()["math"], (20,))
        self.assertEqual(tuple(item[0] for item in state.history), (5, 6))

    def test_duplicate_is_idempotent_and_gap_clears_stale_active_set(self) -> None:
        state = self._state(window_steps=1, audit_interval_steps=1)
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 100.0, 20)),
            step=1,
        )
        duplicate_outcome, duplicate = update_online_control_selection(
            state,
            _statistics(("math", 20, 1000.0, 100)),
            step=1,
        )
        self.assertTrue(duplicate_outcome.duplicate_step)
        self.assertEqual(duplicate, state)

        gap_outcome, gap_state = update_online_control_selection(
            state,
            _statistics(("math", 20, 100.0, 20)),
            step=3,
        )
        self.assertTrue(gap_outcome.history_reset)
        self.assertEqual(gap_state.active_map()["code"], ())

        with self.assertRaisesRegex(ValueError, "backward"):
            update_online_control_selection(
                gap_state,
                _statistics(),
                step=2,
            )

    def test_state_round_trip_preserves_window_and_active_ids(self) -> None:
        state = self._state(
            window_steps=2,
            audit_interval_steps=2,
            selection_mode="top_speed",
        )
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 100.0, 20)),
            step=1,
        )

        restored = OnlineControlSelectionState.from_mapping(state.as_dict())

        self.assertEqual(restored, state)
        statistics = _statistics(("math", 10, 50.0, 20))
        uninterrupted_outcome, uninterrupted = update_online_control_selection(
            state,
            statistics,
            step=2,
        )
        resumed_outcome, resumed = update_online_control_selection(
            restored,
            statistics,
            step=2,
        )
        self.assertEqual(resumed_outcome, uninterrupted_outcome)
        self.assertEqual(resumed, uninterrupted)

    def test_domain_candidate_state_round_trip_and_v1_migration(self) -> None:
        state = initial_online_control_selection_state(
            DOMAINS,
            DOMAIN_CANDIDATES,
            audit_interval_steps=3,
            window_steps=3,
            min_mean_occurrences_per_step=20.0,
            top_k=2,
        )

        self.assertEqual(state.candidate_map(), DOMAIN_CANDIDATES)
        self.assertEqual(
            OnlineControlSelectionState.from_mapping(state.as_dict()),
            state,
        )

        legacy = state.as_dict()
        legacy.pop("domain_candidate_token_ids")
        legacy.pop("selection_mode")
        legacy["candidate_token_ids"] = (10, 20)
        legacy["schema_version"] = 1
        migrated = OnlineControlSelectionState.from_mapping(legacy)
        self.assertEqual(migrated.selection_mode, "top_loss")
        self.assertEqual(
            migrated.candidate_map(),
            {"math": (10, 20), "code": (10, 20)},
        )

    def test_v2_checkpoint_without_selection_mode_defaults_to_top_loss(self) -> None:
        state = initial_online_control_selection_state(
            DOMAINS,
            DOMAIN_CANDIDATES,
            audit_interval_steps=3,
            window_steps=3,
            min_mean_occurrences_per_step=20.0,
            top_k=2,
        )
        legacy = state.as_dict()
        legacy.pop("selection_mode")
        legacy["schema_version"] = 2

        migrated = OnlineControlSelectionState.from_mapping(legacy)

        self.assertEqual(migrated.selection_mode, "top_loss")
        self.assertEqual(migrated.candidate_map(), DOMAIN_CANDIDATES)

    def test_v3_fixed_checkpoint_preserves_active_ids(self) -> None:
        state = self._state(
            window_steps=1,
            audit_interval_steps=1,
            minimum_frequency=1.0,
            top_k=1,
        )
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 2.0, 1)),
            step=1,
        )
        legacy = state.as_dict()
        legacy.pop("weight_mode")
        legacy.pop("active_token_weights")
        legacy["schema_version"] = 3

        migrated = OnlineControlSelectionState.from_mapping(legacy)

        self.assertEqual(migrated.weight_mode, "fixed")
        self.assertEqual(migrated.active_map()["math"], (10,))
        self.assertEqual(migrated.active_weight_map()["math"], {})

    def test_v4_checkpoint_defaults_to_mean_occurrence_gate(self) -> None:
        state = self._state(strict_occurrence_gate=True)
        legacy = state.as_dict()
        legacy.pop("strict_occurrence_gate")
        legacy["schema_version"] = 4

        migrated = OnlineControlSelectionState.from_mapping(legacy)

        self.assertFalse(migrated.strict_occurrence_gate)

    def test_v6_checkpoint_defaults_to_top_k_budget(self) -> None:
        state = self._state(budget_mode="top_p", top_p=0.8)
        legacy = state.as_dict()
        legacy.pop("budget_mode")
        legacy.pop("top_p")
        legacy["schema_version"] = 6

        migrated = OnlineControlSelectionState.from_mapping(legacy)

        self.assertEqual(migrated.budget_mode, "top_k")
        self.assertEqual(migrated.top_p, 1.0)

    def test_v7_top_p_checkpoint_resets_history_without_denominator(self) -> None:
        state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            budget_mode="top_p",
            top_p=0.05,
        )
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 5.0, 5)),
            step=1,
            valid_token_counts={"math": 100, "code": 0},
        )
        legacy = state.as_dict()
        legacy.pop("valid_token_count_history")
        legacy["schema_version"] = 7

        migrated = OnlineControlSelectionState.from_mapping(legacy)

        self.assertEqual(migrated.history, ())
        self.assertEqual(migrated.valid_token_count_history, ())
        self.assertEqual(migrated.active_map(), {"math": (), "code": ()})

    def test_checkpoint_rejects_invalid_top_p_budget(self) -> None:
        serialized = self._state(budget_mode="top_p", top_p=0.8).as_dict()
        serialized["top_p"] = float("nan")

        with self.assertRaisesRegex(ValueError, "top_p"):
            OnlineControlSelectionState.from_mapping(serialized)

    def test_checkpoint_rejects_valid_count_below_candidate_occurrences(
        self,
    ) -> None:
        state = self._state(
            audit_interval_steps=1,
            window_steps=1,
            minimum_frequency=1.0,
            budget_mode="top_p",
            top_p=0.05,
        )
        _, state = update_online_control_selection(
            state,
            _statistics(("math", 10, 2.0, 2)),
            step=1,
            valid_token_counts={"math": 2, "code": 0},
        )
        serialized = state.as_dict()
        serialized["valid_token_count_history"] = (
            (1, (("math", 1), ("code", 0))),
        )

        with self.assertRaisesRegex(ValueError, "cannot be smaller"):
            OnlineControlSelectionState.from_mapping(serialized)

    def test_top_speed_rejects_single_step_window_and_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self._state(window_steps=1, selection_mode="top_speed")
        with self.assertRaisesRegex(ValueError, "selection mode"):
            self._state(selection_mode="fastest")

    def test_runtime_aggregates_only_whitelisted_domain_occurrences(
        self,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        from mopd_verl.domain_gradient.control_top_loss_runtime import (
            global_candidate_loss_statistics_with_valid_counts,
        )

        result = global_candidate_loss_statistics_with_valid_counts(
            (
                torch.tensor(
                    [[10, 20, 99], [20, 30, 10]],
                    dtype=torch.long,
                ),
            ),
            (torch.tensor([[-2.0, 3.0, 100.0], [4.0, -5.0, 6.0]]),),
            (torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool),),
            (("math", "code"),),
            domains=DOMAINS,
            candidate_token_ids=CANDIDATES,
        )
        statistics = result.by_domain

        self.assertEqual(statistics["math"], {10: (2.0, 1), 20: (3.0, 1)})
        self.assertEqual(statistics["code"], {10: (6.0, 1), 20: (4.0, 1)})
        self.assertEqual(result.valid_token_counts, {"math": 3, "code": 2})

    def test_runtime_accepts_top_logp_diff_as_absolute_score(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        from mopd_verl.domain_gradient.control_top_loss_runtime import (
            global_candidate_loss_statistics,
        )

        statistics = global_candidate_loss_statistics(
            (torch.tensor([[10, 20]], dtype=torch.long),),
            (torch.tensor([[-1.5, 0.5]]),),
            (torch.ones(1, 2, dtype=torch.bool),),
            (("math",),),
            domains=DOMAINS,
            candidate_token_ids=CANDIDATES,
            selection_mode="top_logp_diff",
        )

        self.assertEqual(statistics["math"], {10: (1.5, 1), 20: (0.5, 1)})

    def test_runtime_filters_candidates_by_domain_on_a_shared_union_axis(
        self,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        from mopd_verl.domain_gradient.control_top_loss_runtime import (
            global_candidate_loss_statistics,
        )

        statistics = global_candidate_loss_statistics(
            (torch.tensor([[10, 30], [10, 30]], dtype=torch.long),),
            (torch.tensor([[2.0, 90.0], [80.0, 3.0]]),),
            (torch.ones(2, 2, dtype=torch.bool),),
            (("math", "code"),),
            domains=DOMAINS,
            domain_candidate_token_ids=DOMAIN_CANDIDATES,
        )

        self.assertEqual(statistics["math"], {10: (2.0, 1)})
        self.assertEqual(statistics["code"], {30: (3.0, 1)})

    def test_runtime_aggregates_paired_selection_bonus(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        from mopd_verl.domain_gradient.control_top_loss_runtime import (
            global_candidate_loss_statistics,
        )

        statistics = global_candidate_loss_statistics(
            (torch.tensor([[10, 20, 30]], dtype=torch.long),),
            (torch.tensor([[1.0, 2.0, 0.0]]),),
            (torch.ones(1, 3, dtype=torch.bool),),
            (("math",),),
            domains=DOMAINS,
            candidate_token_ids=CANDIDATES,
            selection_mode="top_kl_student_entropy",
            student_entropy_batches=(torch.tensor([[0.0, 2.0, 1.0]]),),
        )

        self.assertEqual(
            statistics["math"],
            {10: (0.5, 1), 20: (3.0, 1), 30: (0.5, 1)},
        )
        self.assertEqual(statistics["code"], {})


if __name__ == "__main__":
    unittest.main()
