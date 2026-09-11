"""Actual-step composition, canonical taxonomy and distributed merge contracts."""

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mopd_verl.domain_gradient.frozen_taxonomy import (
    CONTROL_TOKEN_IDS,
    STRUCTURE_TOKEN_IDS,
)
from mopd_verl.domain_gradient.token_source_metrics import amplified_token_source_metrics
from mopd_verl.tensorboard_filter import filter_tensorboard_metrics


def test_frozen_taxonomy_matches_canonical_source() -> None:
    source = Path(__file__).resolve().parents[1] / (
        'analysis-output/four-baseline-global-token-taxonomy/tables/global-taxonomy.csv'
    )
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    for kind, ids, count in (
        ('Control', CONTROL_TOKEN_IDS, 175),
        ('Structure', STRUCTURE_TOKEN_IDS, 634),
    ):
        assert ids == {int(row['token_id']) for row in rows if row['token_type'] == kind}
        assert len(ids) == count
    assert not CONTROL_TOKEN_IDS & STRUCTURE_TOKEN_IDS


def test_actual_occurrences_padding_domain_weights_and_unique_ids() -> None:
    control, structure = min(CONTROL_TOKEN_IDS), min(STRUCTURE_TOKEN_IDS)
    ids = torch.tensor([[control, control, structure, 999999, structure],
                        [control, structure, 999999, control, control]])
    metrics = amplified_token_source_metrics(
        [ids], [torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])],
        [torch.tensor([[4., 4., 4., 4., 4.], [0.5, 2., 0.5, 3., 3.]])],
        [['math', 'code']], domains=['math', 'code', 'science'],
        domain_weights={'math': 2., 'code': 0.5},
    )
    prefix = 'global/token_weight/amplified_source_'
    assert metrics[prefix + 'occurrence_count'] == 5
    assert metrics[prefix + 'control_occurrence_count'] == 2
    assert metrics[prefix + 'structure_occurrence_fraction'] == pytest.approx(2 / 5)
    assert metrics[prefix + 'unique_token_count'] == 3
    assert metrics[prefix + 'control_unique_token_fraction'] == pytest.approx(1 / 3)
    assert metrics['science/token_weight/amplified_source_control_occurrence_fraction'] == 0
    assert filter_tensorboard_metrics(metrics, 'core') == metrics


def test_distributed_merge_deduplicates_ids_and_uses_ratio_of_sums() -> None:
    control, structure = min(CONTROL_TOKEN_IDS), min(STRUCTURE_TOKEN_IDS)

    def gather(output: list, local: dict) -> None:
        output[:] = [local, {'math': {control: 3, structure: 1}}]

    with patch('torch.distributed.is_initialized', return_value=True), patch(
        'torch.distributed.get_world_size', return_value=2
    ), patch('torch.distributed.all_gather_object', side_effect=gather):
        metrics = amplified_token_source_metrics(
            [torch.tensor([[control]])], [torch.ones(1, 1)],
            [torch.full((1, 1), 4.)], [['math']], domains=['math'], domain_weights={},
        )
    prefix = 'global/token_weight/amplified_source_'
    assert metrics[prefix + 'control_occurrence_fraction'] == 0.8
    assert metrics[prefix + 'unique_token_count'] == 2


def test_fixed_weight_completed_step_emits_metrics_without_online_selector() -> None:
    import test_domain_gradient_optimization_contracts as contracts

    control, structure = min(CONTROL_TOKEN_IDS), min(STRUCTURE_TOKEN_IDS)
    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.domain_gradient.audit import DomainGradientAudit

        audit = DomainGradientAudit(SimpleNamespace(), {
            'domains': ['math'], 'control_token_loss_weighting_enabled': True,
            'control_token_loss_weight': 4., 'control_token_ids': [control, structure],
        })
        batch = SimpleNamespace(
            batch={'responses': torch.tensor([[control, structure, control]]),
                   'response_mask': torch.ones(1, 3)},
            non_tensor_batch={'domain': ['math']},
        )
        metrics = audit.observe_completed_step(
            [batch], [torch.ones(1, 3)], [torch.tensor([[1., 1., 0.]])],
        )
    assert metrics['math/token_weight/amplified_source_control_occurrence_count'] == 1
    assert metrics['math/token_weight/amplified_source_structure_occurrence_fraction'] == 0.5


def test_sequence_parallel_replicas_do_not_inflate_occurrences() -> None:
    control = min(CONTROL_TOKEN_IDS)

    def gather(output: list, local: dict) -> None:
        output[:] = [local, local]

    with patch('torch.distributed.is_initialized', return_value=True), patch(
        'torch.distributed.get_world_size', return_value=2
    ), patch('torch.distributed.all_gather_object', side_effect=gather):
        metrics = amplified_token_source_metrics(
            [torch.tensor([[control, control]])], [torch.ones(1, 2)],
            [torch.full((1, 2), 4.)], [['math']], domains=['math'],
            domain_weights={}, sequence_parallel_size=2,
        )
    assert metrics['global/token_weight/amplified_source_control_occurrence_count'] == 2
    assert metrics['global/token_weight/amplified_source_unique_token_count'] == 1


def test_mask_only_respects_teacher_prefix_region_without_returning_losses() -> None:
    import test_domain_gradient_optimization_contracts as contracts

    class Batch:
        batch = {
            'response_mask': torch.ones(1, 2),
            'teacher_prefix_mask': torch.tensor([[1., 0.]]),
            'math_teacher_topk_ids': torch.zeros(1, 2, 3, dtype=torch.long),
            'math_teacher_topk_logprobs': torch.log_softmax(torch.ones(1, 2, 3), -1),
        }
        non_tensor_batch = {'opd_teacher': ['math']}
        meta_info = {'temperature': 1.}

        def to(self, device: object) -> 'Batch':
            return self

    class Actor:
        config = {
            'entropy_coeff': 0., 'kl_loss_coef': 0., 'use_kl_loss': False,
            'loss_agg_mode': 'token-mean',
            'policy_loss': {
                'distill_loss_builder': 'topk_kl',
                'distill_mode': 'topk_renormalized_reverse_kl',
                'multi_teacher_distill': True,
                'topk_distill_support_source': 'teacher',
                'topk_distill_temperature': 1.,
                'teacher_prefix_enabled': True,
                'teacher_prefix_loss_region': 'suffix_only',
            },
        }

        def _forward_micro_batch(self, inputs: dict, **kwargs: object) -> tuple:
            return None, torch.zeros(1, 2), None, None, torch.log_softmax(
                torch.tensor([[[2., 1., 0.], [0., 1., 2.]]], requires_grad=True), -1
            )

    with contracts.DomainGradientOptimizationContractTests()._stubbed_verl(torch):
        from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss

        result = build_actor_micro_batch_loss(
            Actor(), Batch(), loss_scale_factor=1., on_policy=True,
            return_token_source_mask=True,
        )
    assert result.configured_token_loss is None
    torch.testing.assert_close(result.configured_token_loss_mask, torch.tensor([[0., 1.]]))
