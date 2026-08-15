from __future__ import annotations

import math
import unittest
from typing import Any


class RegionDPOLossTests(unittest.TestCase):
    def _torch(self) -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")
        return torch

    def test_behavior_matched_logits_start_at_log_two(self) -> None:
        torch = self._torch()
        from mopd_verl.region_dpo_loss import region_dpo_pair_loss

        current = torch.tensor(
            [[[[ -0.2, -0.3], [-0.5, -0.6]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        reference = current.detach().clone()
        result = region_dpo_pair_loss(
            current_log_probs=current,
            reference_log_probs=reference,
            loss_mask=torch.ones_like(current),
            pair_mask=torch.ones((1, 1)),
            beta=0.1,
        )

        self.assertAlmostEqual(result.loss.item(), math.log(2.0), places=6)
        self.assertEqual(result.metrics["actor/region_dpo_pair_count"], 1.0)

    def test_preferred_log_ratio_reduces_loss_and_has_gradient(self) -> None:
        torch = self._torch()
        from mopd_verl.region_dpo_loss import region_dpo_pair_loss

        current = torch.tensor(
            [[[[0.4, 0.4], [0.0, 0.0]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        reference = torch.zeros_like(current)
        result = region_dpo_pair_loss(
            current_log_probs=current,
            reference_log_probs=reference,
            loss_mask=torch.ones_like(current),
            pair_mask=torch.ones((1, 1)),
            beta=1.0,
        )
        result.loss.backward()

        self.assertLess(result.loss.item(), math.log(2.0))
        self.assertLess(current.grad[0, 0, 0].sum().item(), 0.0)
        self.assertGreater(current.grad[0, 0, 1].sum().item(), 0.0)

    def test_mask_starts_credit_at_first_divergence(self) -> None:
        torch = self._torch()
        from mopd_verl.region_dpo_loss import region_dpo_pair_loss

        current = torch.tensor(
            [[[[9.0, 0.3], [-9.0, 0.0]]]],
            dtype=torch.float32,
        )
        reference = torch.zeros_like(current)
        loss_mask = torch.tensor(
            [[[[0.0, 1.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        result = region_dpo_pair_loss(
            current_log_probs=current,
            reference_log_probs=reference,
            loss_mask=loss_mask,
            pair_mask=torch.ones((1, 1)),
            beta=1.0,
        )

        expected = -torch.nn.functional.logsigmoid(torch.tensor(0.3))
        torch.testing.assert_close(result.loss, expected)

    def test_loss_is_normalized_per_base_rollout(self) -> None:
        torch = self._torch()
        from mopd_verl.region_dpo_loss import region_dpo_pair_loss

        current = torch.zeros((2, 2, 2, 1), requires_grad=True)
        result = region_dpo_pair_loss(
            current_log_probs=current,
            reference_log_probs=torch.zeros_like(current),
            loss_mask=torch.ones_like(current),
            pair_mask=torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            beta=1.0,
        )

        self.assertAlmostEqual(
            result.loss.item(),
            0.5 * math.log(2.0),
            places=6,
        )

    def test_actor_forward_skips_inactive_packed_slots(self) -> None:
        torch = self._torch()
        from mopd_verl.region_dpo_loss import build_region_dpo_actor_loss

        class Actor:
            def __init__(self) -> None:
                self.forward_batch_size = 0

            def _forward_micro_batch(
                self,
                inputs: dict[str, Any],
                **_kwargs: Any,
            ) -> tuple[None, Any]:
                self.forward_batch_size = int(inputs["responses"].shape[0])
                self_attention = inputs["attention_mask"]
                if not bool(self_attention.bool().any(dim=-1).all().item()):
                    raise AssertionError("inactive all-padding row reached actor")
                logits = torch.zeros_like(
                    inputs["responses"], dtype=torch.float32
                )
                logits.requires_grad_(True)
                return None, logits

        actor = Actor()
        batch_size, points, branches, tokens = 2, 2, 2, 3
        pair_mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        active_attention = torch.tensor([1, 1, 1, 1, 1])
        attention = torch.zeros(
            (batch_size, points, branches, 5), dtype=torch.long
        )
        attention[0, 0, 0] = active_attention
        attention[0, 0, 1] = active_attention
        model_inputs = {
            "responses": torch.zeros((batch_size, tokens), dtype=torch.long),
            "region_dpo_responses": torch.zeros(
                (batch_size, points, branches, tokens), dtype=torch.long
            ),
            "region_dpo_input_ids": torch.zeros(
                (batch_size, points, branches, 5), dtype=torch.long
            ),
            "region_dpo_attention_mask": attention,
            "region_dpo_position_ids": torch.zeros_like(attention),
            "region_dpo_reference_log_probs": torch.zeros(
                (batch_size, points, branches, tokens)
            ),
            "region_dpo_loss_mask": torch.ones(
                (batch_size, points, branches, tokens)
            ),
            "region_dpo_pair_mask": pair_mask,
            "region_dpo_rewards": torch.tensor(
                [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
            ),
        }
        result = build_region_dpo_actor_loss(
            actor=actor,
            model_inputs=model_inputs,
            policy_loss_config={
                "region_dpo_beta": 0.1,
                "region_dpo_loss_weight": 0.5,
            },
            temperature=1.0,
            loss_scale_factor=1.0,
        )

        self.assertEqual(actor.forward_batch_size, 2)
        self.assertAlmostEqual(
            result.loss.item(),
            0.25 * math.log(2.0),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
