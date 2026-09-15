"""Numerical and routing tests for the fused teacher vocabulary pass."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mopd_verl.teacher_logprob import (
    fused_logprob_entropy_topk_from_logits,
    response_prediction_mask,
)
from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_FULL_VOCAB,
    TOPK_LOGPROB_MODE_SPARSE,
    topk_log_probs_from_logits,
)


def test_fused_statistics_match_independent_fp32_references() -> None:
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(2, 7, 31, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, 31, (2, 7), generator=generator)
    gather_ids = torch.randint(0, 31, (2, 7, 5), generator=generator)

    actual = fused_logprob_entropy_topk_from_logits(
        logits,
        labels,
        topk=6,
        gather_topk_ids=gather_ids,
        normalize_gathered=True,
        chunk_size=4,
        logprob_mode=TOPK_LOGPROB_MODE_FULL_VOCAB,
    )

    log_norm = torch.logsumexp(logits, dim=-1)
    expected_chosen = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - log_norm
    probabilities = torch.softmax(logits, dim=-1)
    expected_entropy = log_norm - torch.sum(probabilities * logits, dim=-1)
    expected_topk_ids, expected_topk_log_probs, expected_gathered = (
        topk_log_probs_from_logits(
            logits,
            topk=6,
            gather_topk_ids=gather_ids,
            normalize_gathered=True,
            chunk_size=4,
            logprob_mode=TOPK_LOGPROB_MODE_FULL_VOCAB,
        )
    )

    torch.testing.assert_close(actual[0], expected_chosen, rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected_entropy, rtol=0, atol=0)
    assert torch.equal(actual[2], expected_topk_ids)
    torch.testing.assert_close(actual[3], expected_topk_log_probs, rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected_gathered, rtol=0, atol=0)


def test_fused_sparse_gather_preserves_unnormalized_contract() -> None:
    logits = torch.tensor([[[0.0, 2.0, 1.0, -2.0]]])
    labels = torch.tensor([[2]])
    gather_ids = torch.tensor([[[1, 3]]])

    chosen, entropy, topk_ids, topk_log_probs, gathered = (
        fused_logprob_entropy_topk_from_logits(
            logits,
            labels,
            gather_topk_ids=gather_ids,
            normalize_gathered=False,
            chunk_size=1,
            logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
        )
    )

    expected_log_probs = torch.log_softmax(logits, dim=-1)
    torch.testing.assert_close(chosen, expected_log_probs[..., 2])
    torch.testing.assert_close(
        entropy,
        -(expected_log_probs.exp() * expected_log_probs).sum(dim=-1),
    )
    assert topk_ids is None and topk_log_probs is None
    torch.testing.assert_close(gathered, logits.gather(-1, gather_ids))


def test_bf16_entropy_gate_matches_legacy_away_from_threshold() -> None:
    generator = torch.Generator().manual_seed(37)
    logits = (torch.randn(48, 67, generator=generator) * 6).to(torch.bfloat16)
    labels = torch.randint(0, 67, (48,), generator=generator)
    _, fused_entropy, _, _, _ = fused_logprob_entropy_topk_from_logits(
        logits,
        labels,
        topk=4,
        chunk_size=11,
        logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
    )
    legacy_logits = logits.clone()
    legacy_probabilities = torch.softmax(legacy_logits, dim=-1)
    legacy_entropy = (
        torch.logsumexp(legacy_logits, dim=-1)
        - torch.sum(legacy_probabilities * legacy_logits, dim=-1)
    ).float()
    torch.testing.assert_close(fused_entropy.float(), legacy_entropy, rtol=0, atol=0)
    assert torch.equal(fused_entropy > 0.8, legacy_entropy > 0.8)


def test_fused_statistics_use_runtime_entropy_implementation_per_chunk() -> None:
    logits = torch.arange(45, dtype=torch.float32).reshape(5, 9) / 7
    labels = torch.arange(5) % 9
    calls = []

    def entropy_fn(chunk: torch.Tensor) -> torch.Tensor:
        calls.append(len(chunk))
        return torch.full(chunk.shape[:-1], 0.75, dtype=chunk.dtype)

    _, entropy, _, _, _ = fused_logprob_entropy_topk_from_logits(
        logits,
        labels,
        topk=2,
        chunk_size=2,
        logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
        entropy_fn=entropy_fn,
    )

    assert calls == [2, 2, 1]
    assert torch.equal(entropy, torch.full((5,), 0.75))


def test_response_prediction_mask_excludes_prompts_padding_and_last_token() -> None:
    # Two padded rows of width 7. The response prediction window is columns
    # 2..5; indices represent the positions retained by remove-padding.
    indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
    mask = response_prediction_mask(
        indices,
        sequence_length=7,
        response_length=4,
    )

    assert indices[mask].tolist() == [2, 3, 4, 5, 10, 11, 12]


@pytest.mark.parametrize(
    ("logits", "labels", "kwargs", "match"),
    [
        (
            torch.ones(3),
            torch.ones((), dtype=torch.long),
            {"topk": 1},
            "at least 2 dims",
        ),
        (
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.long),
            {"topk": 1},
            "labels must have shape",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, dtype=torch.long),
            {"topk": 1, "chunk_size": 0},
            "positive",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, dtype=torch.long),
            {},
            "topk or gather_topk_ids",
        ),
    ],
)
def test_fused_statistics_reject_invalid_requests(
    logits: torch.Tensor,
    labels: torch.Tensor,
    kwargs: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "chunk_size": 2,
        "logprob_mode": TOPK_LOGPROB_MODE_SPARSE,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        fused_logprob_entropy_topk_from_logits(logits, labels, **arguments)


def test_dp_actor_routes_joint_teacher_statistics_through_fused_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "verl"
        / "verl"
        / "workers"
        / "actor"
        / "dp_actor.py"
    ).read_text(encoding="utf-8")

    assert "fused_logprob_entropy_topk_from_logits" in source
    assert source.count("use_fused_teacher_statistics") >= 8
    assert "and not self.config.entropy_checkpointing" in source
    assert source.count("entropy_fn=self.compute_entropy_from_logits") == 2
    assert 'getattr(self, "_teacher_performance_config", {})' in source
    assert 'self.config.get("teacher_performance"' not in source
    assert source.count("teacher_performance_enabled") == 3


def test_dp_actor_fused_ref_path_returns_compact_support_ids() -> None:
    pytest.importorskip("tensordict")
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class Config(dict):
        entropy_checkpointing = False

    class StaticModel(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.logits = logits

        def forward(self, **kwargs: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(logits=self.logits.clone())

    generator = torch.Generator().manual_seed(31)
    logits = torch.randn(2, 6, 17, generator=generator)
    responses = torch.randint(0, 17, (2, 3), generator=generator)
    micro_batch = {
        "input_ids": torch.randint(0, 17, (2, 6), generator=generator),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "position_ids": torch.arange(6).repeat(2, 1),
        "responses": responses,
    }
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = StaticModel(logits)
    actor.actor_optimizer = None
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.device_name = "cpu"
    actor.param_dtype = torch.bfloat16
    actor.compute_entropy_from_logits = lambda current_logits: (
        torch.logsumexp(current_logits, dim=-1)
        - torch.sum(
            torch.softmax(current_logits, dim=-1) * current_logits,
            dim=-1,
        )
    )
    actor.config = Config(
        teacher_performance={
            "fused_statistics": True,
            "compact_topk_ids": True,
        }
    )
    actor._teacher_performance_config = {
        "enabled": True,
        "fused_statistics": True,
        "compact_topk_ids": True,
    }

    entropy, chosen, topk_ids, topk_log_probs, gathered = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        calculate_entropy=True,
        topk=4,
        topk_logprob_chunk_size=2,
        return_extra=True,
    )
    expected = fused_logprob_entropy_topk_from_logits(
        logits[:, 2:5, :],
        responses,
        topk=4,
        chunk_size=2,
        logprob_mode=TOPK_LOGPROB_MODE_SPARSE,
    )

    torch.testing.assert_close(chosen, expected[0])
    torch.testing.assert_close(entropy, expected[1])
    assert torch.equal(topk_ids.long(), expected[2])
    assert topk_ids.dtype == torch.int32
    torch.testing.assert_close(topk_log_probs, expected[3])
    assert gathered is None
