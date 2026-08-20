from __future__ import annotations

import unittest

from mopd_verl.domain_gradient.control_top_loss import (
    OnlineControlSelectionState,
    initial_online_control_selection_state,
    update_online_control_selection,
)
from mopd_verl.domain_gradient.control_selection_scoring import (
    occurrence_weighted_optimization_speed,
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
        selection_mode: str = "top_loss",
    ) -> OnlineControlSelectionState:
        return initial_online_control_selection_state(
            DOMAINS,
            CANDIDATES,
            audit_interval_steps=audit_interval_steps,
            window_steps=window_steps,
            min_mean_occurrences_per_step=minimum_frequency,
            top_k=top_k,
            selection_mode=selection_mode,
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
            global_candidate_loss_statistics,
        )

        statistics = global_candidate_loss_statistics(
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

        self.assertEqual(statistics["math"], {10: (2.0, 1), 20: (3.0, 1)})
        self.assertEqual(statistics["code"], {10: (6.0, 1), 20: (4.0, 1)})

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


if __name__ == "__main__":
    unittest.main()
