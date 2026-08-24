from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mopd_verl.tip_native_math import (
    full_vocab_reverse_kl_per_token,
    normalized_entropy_per_token,
    select_tip_tokens,
)


def test_full_vocab_reverse_kl_matches_direct_formula_and_gradient() -> None:
    torch.manual_seed(7)
    teacher = torch.randn(5, 11)
    student = torch.randn(5, 11, requires_grad=True)
    candidate = full_vocab_reverse_kl_per_token(
        teacher,
        student,
        chunk_size=2,
    )

    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    expected = (
        student_log_probs.exp() * (student_log_probs - teacher_log_probs)
    ).sum(dim=-1)
    torch.testing.assert_close(candidate, expected)

    candidate.sum().backward()
    candidate_gradient = student.grad.detach().clone()
    student.grad = None
    expected.sum().backward()
    torch.testing.assert_close(candidate_gradient, student.grad)


def test_normalized_entropy_matches_paper_definition() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0], [8.0, -8.0, -8.0, -8.0]])
    entropy = normalized_entropy_per_token(logits, chunk_size=1)
    assert entropy[0].item() == 1.0
    assert 0.0 <= entropy[1].item() < 0.01


def test_tip_selector_clips_batch_entropy_and_uses_batch_minmax() -> None:
    entropy = torch.tensor([[0.0, 1.0, 100.0], [2.0, 3.0, 0.0]])
    divergence = torch.tensor([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    result = select_tip_tokens(
        entropy,
        divergence,
        mask,
        retention_ratio=2.0 / 3.0,
        entropy_clip_quantile=0.8,
    )

    expected_clip = torch.quantile(entropy[mask], 0.8)
    assert result.clipped_entropy[0, 2].item() == expected_clip.item()
    assert result.score[1, 0].item() > result.score[0, 1].item()
    assert not result.selected[1, 0]
    assert result.selected[1, 1]
    assert not result.selected[1, 2]


def test_tip_selector_keeps_floor_rho_per_rollout_with_stable_ties() -> None:
    entropy = torch.zeros(2, 5)
    divergence = torch.zeros(2, 5)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    result = select_tip_tokens(
        entropy,
        divergence,
        mask,
        retention_ratio=0.5,
    )

    assert result.selected.sum(dim=-1).tolist() == [math.floor(2.5), 1]
    assert result.selected[0].tolist() == [True, True, False, False, False]
    assert result.selected[1].tolist() == [True, False, False, False, False]


def test_tip_selector_never_selects_padding() -> None:
    entropy = torch.tensor([[1.0, 2.0, 1_000.0]])
    divergence = torch.tensor([[1.0, 2.0, 1_000.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result = select_tip_tokens(entropy, divergence, mask, retention_ratio=0.5)
    assert result.selected.tolist() == [[False, True, False]]
