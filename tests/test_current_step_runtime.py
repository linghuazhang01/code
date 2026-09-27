"""Current-step integration contracts with hand-computed gradients and DP oracles."""

from dataclasses import asdict, replace
from datetime import timedelta
from importlib import import_module
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

import test_occurrence_selection as existing
from mopd_verl.domain_gradient.occurrence import occurrence_mask, prepare_occurrence_masks


TEACHER = "mopd_verl.full_gradient.loss_support.selected_teacher_log_prob"
PREFIX = "math/current_step/"


def test_full_gradient_audit_uses_actual_masks_without_static_id_error_metrics() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit
        from mopd_verl.domain_gradient.token_logging import LocalTokenCandidate

        audit = _audit()
        batch = _batch([[0, 1, 2]])
        _prepare(audit, [batch], [_result(batch, [[9., 1., 4.]])])
        audit.training_gradient_mask = lambda value: occurrence_mask(audit, value)
        audit._active_domain_control_token_ids = lambda: {}
        audit._weight_state = SimpleNamespace(weight_map=lambda: {})
        audit._shared_token_selection = SimpleNamespace(token_ids=())
        candidates = tuple(
            LocalTokenCandidate(0, 0, index, index, loss, loss)
            for index, loss in enumerate([9., 1., 4.])
        )
        metrics = DomainGradientAudit._loss_amplification_metrics(
            audit, {"math": candidates}, [batch],
        )
        prefix = "global/token_weight/"
        assert metrics[prefix + "token_weighted_configured_loss_abs_mass"] == pytest.approx(24.)
        assert metrics[prefix + "amplified_token_occurrence_fraction"] == pytest.approx(1 / 3)
        assert metrics[prefix + "mean_token_gradient_multiplier"] == pytest.approx(1.)
        assert not any("gradient_multiplier_abs_error" in key for key in metrics)
        assert prefix + "gradient_multiplier_mean_abs_error" not in metrics
        assert prefix + "control_occurrence_count" not in metrics


def _config(**changes: object) -> object:
    defaults = dict(
        control_token_online_selection_timing="current_step",
        control_token_online_selection_unit="token_id",
        control_token_candidate_ids=(0, 1, 2),
        domain_control_token_candidate_ids=(),
        domain_control_token_candidate_groups=(),
        control_token_online_min_mean_occurrences_per_step=0,
        control_token_online_strict_occurrence_gate=True,
        control_token_online_top_p=.2,
        control_token_online_top_p_by_domain=(),
        control_token_online_selection_mode="top_loss",
        control_token_online_selection_mode_by_domain=(),
        control_token_tail_top_p=.2,
        control_token_tail_weight=0.,
    )
    defaults.update(changes)
    return replace(existing.config(), **defaults)


def _audit(**changes: object) -> SimpleNamespace:
    actor = SimpleNamespace(
        actor_module=torch.nn.Linear(1, 1),
        config={"policy_loss": asdict(existing.load_config(existing.OVERLAY).actor)},
    )
    return SimpleNamespace(config=_config(**changes), actor=actor)


def _batch(ids: list[list[int]], domains: list[str] | None = None,
           mask: list[list[int]] | None = None) -> SimpleNamespace:
    tokens = torch.tensor(ids)
    batch = SimpleNamespace(
        batch={"responses": tokens, "response_mask": (
            torch.tensor(mask).float() if mask is not None else torch.ones_like(tokens).float()
        )}, non_tensor_batch={"domain": domains or ["math"] * len(ids)},
        meta_info={"temperature": 1.},
    )
    batch.to = lambda device: batch
    return batch


def _result(batch: SimpleNamespace, values: list[list[float]]) -> SimpleNamespace:
    raw = torch.tensor(values)
    return SimpleNamespace(
        selector_token_loss=raw,
        selector_token_loss_mask=batch.batch["response_mask"].clone(),
        configured_token_loss=raw.flip(-1) * 100,
    )


def _patch_forward(**kwargs: object) -> Any:
    module = import_module("mopd_verl.full_gradient.actor_loss")
    return patch.object(module, "build_actor_micro_batch_loss", **kwargs)


def _prepare(audit: SimpleNamespace, batches: list[SimpleNamespace],
             results: list[SimpleNamespace]) -> dict[str, float]:
    with _patch_forward(side_effect=results):
        return prepare_occurrence_masks(
            audit, batches, [1 / len(batches)] * len(batches),
            on_policy=True, temperature=1.,
        )


def test_first_step_raw_type_ranking_microbatch_domain_normalization_and_padding() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit = _audit()
        first = _batch([[0, 1, 2, 99999], [0, 1, 2, 99999]],
                       ["math", "code"], [[1, 1, 1, 0], [1, 1, 1, 1]])
        second = _batch([[1, 2]])
        metrics = _prepare(audit, [first, second], [
            _result(first, [[9., 1., 4., float("nan")], [1., 9., 4., 100.]]),
            _result(second, [[1., 4.]]),
        ])
        # Math ID 0 wins; ID 1 is the tail. Code chooses the opposite IDs.
        torch.testing.assert_close(occurrence_mask(audit, first), torch.tensor([
            [12 / 5, 0., 3 / 5, 0.], [0., 8 / 3, 2 / 3, 2 / 3],
        ]))
        torch.testing.assert_close(occurrence_mask(audit, second), torch.tensor([[0., 2.]]))
        assert metrics[PREFIX + "head/selected_count"] == 1
        assert metrics[PREFIX + "tail/selected_count"] == 2
        assert metrics[PREFIX + "tail/overshoot"] == 1
        assert metrics["global/current_step/prepass_forward_count"] == 2
        assert not hasattr(audit, "_online_control_selection_state")
        assert not occurrence_mask(audit, first).requires_grad


def test_mean_one_scope_combines_same_domain_rows_within_a_microbatch() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit, batch = _audit(), _batch([[0, 2], [1, 2]])
        _prepare(audit, [batch], [_result(batch, [[9., 4.], [1., 4.]])])
        # One domain denominator is (4+1+0+1)/4, not a separate mean per row.
        torch.testing.assert_close(occurrence_mask(audit, batch), torch.tensor([
            [8 / 3, 2 / 3], [0., 2 / 3],
        ]))


def test_fixed_structure_and_selected_control_share_one_normalizer() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.current_step_selection import (
            CurrentStepSelection,
        )
        from mopd_verl.domain_gradient.current_step_weights import (
            install_current_step_masks,
        )

        audit = _audit(
            structure_token_loss_weighting_enabled=True,
            structure_token_loss_weight=4.0,
            domain_structure_token_ids=(("math", (99,)), ("code", (98,))),
        )
        batch = _batch([[0, 99, 2]])
        batch.batch["mopd_structure_position_mask"] = torch.tensor([[0, 1, 0]])
        audit._occurrence_masks = {}
        selection = CurrentStepSelection(
            head={"math": frozenset({(0,)}), "code": frozenset()},
            tail={"math": frozenset(), "code": frozenset()},
            metrics={},
        )
        metrics = install_current_step_masks(
            audit,
            [(batch, batch.batch["response_mask"].bool(), ["math"])],
            selection,
            rank=0,
        )
        # Control ID 0 and fixed Structure ID 99 both receive raw weight 4;
        # (4 + 4 + 1) / 3 = 3, so normalization happens exactly once.
        torch.testing.assert_close(
            occurrence_mask(audit, batch),
            torch.tensor([[4 / 3, 4 / 3, 1 / 3]]),
        )
        prefix = "math/current_step/structure/"
        assert metrics[prefix + "candidate_occurrence_count"] == 1
        assert metrics[prefix + "applied_occurrence_count"] == 1
        assert metrics[prefix + "position_acceptance_fraction"] == 1


@pytest.mark.parametrize("unit", ["token_id", "occurrence"])
def test_alpha_one_matches_head_only_and_zero_budget_really_disables_tail(unit: str) -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        batch = _batch([[0, 0, 1, 2]])
        result = _result(batch, [[9., 1., 2., 4.]])
        masks = []
        for fraction, alpha in ((0., 0.), (.5, 1.)):
            audit = _audit(control_token_online_selection_unit=unit,
                           control_token_tail_top_p=fraction, control_token_tail_weight=alpha)
            metrics = _prepare(audit, [batch], [result])
            if fraction == 0:
                assert metrics[PREFIX + "tail/selected_count"] == 0
            masks.append(occurrence_mask(audit, batch))
        torch.testing.assert_close(masks[0], masks[1], rtol=0., atol=0.)
        expected = [1.6, 1.6, .4, .4] if unit == "token_id" else [16 / 7, 4 / 7, 4 / 7, 4 / 7]
        torch.testing.assert_close(masks[0], torch.tensor([expected]))


def test_actual_rkl_backward_zero_tail_and_reentry_after_same_batch_rescoring() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss

        audit, batch = _audit(), _batch([[0, 1, 2]])
        logits = torch.nn.Parameter(torch.tensor([[[3., 0.], [.1, 0.], [1., 0.]]]))
        audit.actor.actor_module = torch.nn.ParameterList([logits])
        audit.actor._forward_micro_batch = lambda *args, **kwargs: (
            None, torch.zeros(1, 3), None, None, logits,
        )
        batch.non_tensor_batch["opd_teacher"] = ["math"]
        batch.batch["math_teacher_topk_ids"] = torch.arange(2).expand(1, 3, 2)
        batch.batch["math_teacher_topk_logprobs"] = torch.full((1, 3, 2), -np.log(2.))
        for round_index in range(2):
            prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
            weights = occurrence_mask(audit, batch)
            expected = torch.tensor([[2.4, 0., .6] if round_index == 0 else [.6, 2.4, 0.]])
            torch.testing.assert_close(weights, expected)
            reference = logits.detach().clone().requires_grad_()
            logp = reference.log_softmax(-1)
            rkl = (logp.exp() * (logp + np.log(2.))).sum(-1)
            (rkl * expected).mean().backward()
            result = build_actor_micro_batch_loss(
                audit.actor, batch, loss_scale_factor=1., on_policy=True,
                gradient_mask_override=weights, return_configured_token_loss=True,
            )
            # The gate preserves the reported forward loss and scales only gradients.
            torch.testing.assert_close(result.loss, rkl.detach().mean())
            result.loss.backward()
            torch.testing.assert_close(logits.grad, reference.grad, rtol=1e-5, atol=1e-7)
            zero_position = 1 if round_index == 0 else 2
            assert torch.count_nonzero(logits.grad[0, zero_position]) == 0
            assert torch.count_nonzero(logits.grad[0, 1 if round_index else 0]) > 0
            logits.grad = None
            with torch.no_grad():
                logits[0, 1, 0] = 4.


def test_all_tail_microbatch_stays_zero_with_finite_gradients() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.full_gradient.loss_support import gate_tensor_gradient

        audit, first, second = _audit(), _batch([[1, 1]]), _batch([[0, 2]])
        metrics = _prepare(audit, [first, second], [
            _result(first, [[.1, .1]]), _result(second, [[8., 2.]]),
        ])
        assert metrics[PREFIX + "zero_weight_microbatch_count"] == 1
        weights = occurrence_mask(audit, first)
        assert torch.count_nonzero(weights) == 0
        raw = torch.tensor([[1., 2.]], requires_grad=True)
        gate_tensor_gradient(raw, weights).sum().backward()
        assert torch.isfinite(raw.grad).all() and torch.count_nonzero(raw.grad) == 0
        torch.testing.assert_close(occurrence_mask(audit, second), torch.tensor([[1.6, .4]]))


@pytest.mark.parametrize("invalid", [False, True])
def test_prepass_restores_rng_buffers_modes_and_existing_gradients(invalid: bool) -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit, batch = _audit(), _batch([[0, 1, 2]])
        module = audit.actor.actor_module
        module.register_buffer("counter", torch.tensor(7.))
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 3.)
        rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        audit._occurrence_masks = {123: "stale"}
        def forward(*args: object, **kwargs: object) -> SimpleNamespace:
            assert not torch.is_grad_enabled()
            torch.rand(3), random.random(), np.random.rand(3)
            module.eval()
            module.counter.add_(1)
            for parameter in module.parameters():
                parameter.grad = None
            return _result(batch, [[float("nan") if invalid else 9., 1., 4.]])
        with _patch_forward(side_effect=forward):
            if invalid:
                with pytest.raises(ValueError, match="current-step scoring failed"):
                    prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
                assert audit._occurrence_masks == {}
            else:
                prepare_occurrence_masks(audit, [batch], [1.], on_policy=True, temperature=1.)
        assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == python_rng
        current_numpy = np.random.get_state()
        assert numpy_rng[0] == current_numpy[0] and numpy_rng[2:] == current_numpy[2:]
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        assert module.training and module.counter.item() == 7
        for parameter in module.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 3.))


def test_audit_bypasses_lagged_state_and_retires_cache_after_source_metrics() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        simple = _audit()
        simple.actor._mopd_online_control_selection_state = {"incompatible_old_state": True}
        with patch("mopd_verl.domain_gradient.audit.DomainGradientConfig.from_meta", return_value=simple.config), \
             patch("mopd_verl.domain_gradient.audit.initial_online_control_selection_state",
                   side_effect=AssertionError("current-step must not initialize lagged state")):
            audit = DomainGradientAudit(simple.actor, {})
        batch = _batch([[0, 1, 2]])
        result = _result(batch, [[9., 1., 4.]])
        with _patch_forward(return_value=result), patch.object(audit, "_run_before_training_replay", return_value={}):
            audit.run_before_training([batch], [1.], on_policy=True, temperature=1.)
        def source_metrics(*args: object, **kwargs: object) -> dict[str, float]:
            torch.testing.assert_close(args[2][0], audit.training_gradient_mask(batch))
            return {"source_checked": 1.}
        with patch("mopd_verl.domain_gradient.audit.amplified_token_source_metrics", side_effect=source_metrics):
            assert audit.observe_completed_step([batch], [result.selector_token_loss],
                                                [batch.batch["response_mask"]]) == {"source_checked": 1.}
        with pytest.raises(RuntimeError, match="current-batch"):
            audit.training_gradient_mask(batch)


@pytest.mark.parametrize("confidence_head", [True, False])
def test_teacher_confidence_selects_head_or_tail_independently(confidence_head: bool) -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit = _audit(
            control_token_online_selection_mode="top_teacher_confidence" if confidence_head else "top_loss",
            control_token_tail_selection_mode="bottom_teacher_confidence",
        )
        batch = _batch([[0, 1, 2, 99999]])
        batch.batch["ref_log_prob"] = torch.tensor([[-4., -.1, -20., -1.]])
        _prepare(audit, [batch], [_result(batch, [[9., 1., 4., 100.]])])
        expected = [2 / 3, 8 / 3, 0., 2 / 3] if confidence_head else [8 / 3, 2 / 3, 0., 2 / 3]
        torch.testing.assert_close(occurrence_mask(audit, batch), torch.tensor([expected]))


def test_teacher_confidence_uses_mean_logp_not_mean_probability_for_types() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit = _audit(control_token_online_selection_mode="top_teacher_confidence",
                       control_token_tail_selection_mode="bottom_teacher_confidence")
        batch = _batch([[0, 0, 1, 1]])
        # Mean logp favors ID1 (-3 > -5); mean probability would favor ID0.
        batch.batch["ref_log_prob"] = torch.tensor([[-.1, -9.9, -3., -3.]])
        _prepare(audit, [batch], [_result(batch, [[9., 9., 1., 1.]])])
        torch.testing.assert_close(occurrence_mask(audit, batch), torch.tensor([[0., 0., 2., 2.]]))


def test_disabled_tail_confidence_never_reads_teacher() -> None:
    with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        audit = _audit(control_token_tail_top_p=0., control_token_tail_selection_mode="bottom_teacher_confidence")
        batch = _batch([[0, 1, 2]])
        with patch(TEACHER, side_effect=AssertionError("unexpected teacher fetch")):
            _prepare(audit, [batch], [_result(batch, [[9., 1., 4.]])])


def _gloo_worker(rank: int, rendezvous: str, failure: str) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with existing.contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
            audit = _audit(control_token_online_top_p=.1, control_token_tail_top_p=.1)
            ids = [[0, 0, 0, 1, 2], [0, 1, 1, 1, 2]][rank]
            values = [[0., 0., 0., 4., .1], [10., 4., 4., 4., .1]][rank]
            batch = _batch([ids])
            if failure:
                audit.config = replace(audit.config, control_token_online_selection_mode="top_teacher_confidence")
                if not (rank == 1 and failure == "missing"):
                    batch.batch["ref_log_prob"] = torch.full((1, 5), -.5)
                    if rank == 1:
                        batch.batch["ref_log_prob"][0, 4] = float("nan")
                with pytest.raises(ValueError, match="current-step scoring failed: .*teacher"):
                    _prepare(audit, [batch], [_result(batch, [values])])
                assert audit._occurrence_masks == {}
                return
            metrics = _prepare(audit, [batch], [_result(batch, [values])])
            # Global type means: ID0=10/4, ID1=4, ID2=.1. Mean-of-rank-means
            # would incorrectly promote ID0=(0+10)/2. Whole ID1 uses four slots.
            expected = [[5 / 7, 5 / 7, 5 / 7, 20 / 7, 0.],
                        [5 / 13, 20 / 13, 20 / 13, 20 / 13, 0.]][rank]
            torch.testing.assert_close(occurrence_mask(audit, batch), torch.tensor([expected]))
            assert metrics[PREFIX + "head/selected_count"] == 4
            assert metrics[PREFIX + "head/overshoot"] == 3
            assert metrics[PREFIX + "tail/selected_count"] == 2
            assert metrics["global/current_step/gathered_candidates"] == 6
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="requires torch.distributed Gloo")
@pytest.mark.parametrize("failure", ["", "missing", "nan"], ids=["global-type-means", "missing-TC", "nonfinite-TC"])
def test_real_two_rank_global_aggregation_and_collective_teacher_errors(tmp_path: Path, failure: str) -> None:
    torch.multiprocessing.spawn(
        _gloo_worker, args=("file://" + str(tmp_path / "rendezvous"), failure), nprocs=2,
    )
