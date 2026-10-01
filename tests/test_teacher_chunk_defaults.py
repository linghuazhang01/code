"""Default teacher chunking must survive disabled batching and new layouts."""

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_teacher_performance import Batch

from mopd_verl import teacher_performance as perf
from mopd_verl.topk_distill import (
    TOPK_RENORMALIZED_REVERSE_KL,
    topk_distill_loss_matrix,
    topk_log_probs_from_logits,
)


def _policy(calls: list[dict[str, Any]]) -> SimpleNamespace:
    def forward(
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        topk_logprob_chunk_size: int | None = None,
        teacher_memory_guarded_chunk: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append({
            "ids": micro_batch["ids"].tolist(),
            "chunk": topk_logprob_chunk_size or 16,
            "temperature": temperature,
        })
        return micro_batch["ids"], micro_batch["ids"] + 1

    policy = SimpleNamespace(
        _forward_micro_batch=forward, use_fused_kernels=False,
        use_remove_padding=True, ulysses_sequence_parallel_size=1,
        actor_optimizer=None,
        actor_module=SimpleNamespace(
            config=SimpleNamespace(torch_dtype=torch.bfloat16, vocab_size=100),
        ),
    )

    def compute(data: Batch) -> tuple[torch.Tensor, ...]:
        outputs = []
        for start in range(0, len(data), data.meta_info["micro_batch_size"]):
            stop = start + data.meta_info["micro_batch_size"]
            outputs.append(policy._forward_micro_batch(
                {"ids": data.ids[start:stop],
                 "ref_attention_mask": data.batch["ref_attention_mask"][start:stop]},
                temperature=data.meta_info["temperature"],
                topk_logprob_chunk_size=data.meta_info.get("topk_logprob_chunk_size"),
                teacher_memory_guarded_chunk=data.meta_info.get("teacher_memory_guarded_chunk", False),
            ))
        return tuple(torch.cat(column) for column in zip(*outputs))

    policy.compute_log_prob = compute
    return policy


def _unexpected(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Chunk-only execution must retain the original batching and collectives")


@pytest.mark.parametrize("config,world_size,dedicated", [
    ({}, 1, False), ({}, 4, False),
    ({"enabled": False}, 4, False), ({"enabled": True}, 4, False),
    ({"enabled": False}, 4, True),
])
@pytest.mark.parametrize("available,expected_chunk", [(10**10, 1024), (0, 16)])
def test_default_chunk_changes_only_postprocessing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    config: dict[str, Any], world_size: int, dedicated: bool,
    available: int, expected_chunk: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf, "_available", lambda tuning: available)
    for name in ("_batch_wrapper", "_distributed_min", "install_moe_dispatch"):
        monkeypatch.setattr(perf, name, _unexpected)
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    original_compute = policy.compute_log_prob
    with caplog.at_level(logging.INFO, logger=perf.__name__):
        state = perf.configure_teacher_performance(
            policy, dict(config, fused_statistics=True, compact_topk_ids=True),
            world_size=world_size, teacher_model_device="gpu",
            dedicated_teacher=dedicated,
        )
        data = Batch(torch.arange(4)[:, None], [5, 21, 8, 3])
        original_meta = dict(data.meta_info)
        output = policy.compute_log_prob(data)

    assert state["enabled"] is False
    assert state["topk_logprob_chunk_enabled"] is True
    assert state["chunk"] == state["requested_chunk"] == 1024
    assert state["moe_blocks"] == 0 and not state["distributed_batch_sync"]
    assert not state["fused_statistics"] and not state["compact_topk_ids"]
    assert policy.compute_log_prob is original_compute
    assert [call["ids"] for call in calls] == [[[i]] for i in range(4)]
    assert [call["chunk"] for call in calls] == [expected_chunk] * 4
    assert [call["temperature"] for call in calls] == [0.7] * 4
    assert data.meta_info == original_meta
    assert torch.equal(output[0], data.ids) and torch.equal(output[1], data.ids + 1)
    stats = policy._forward_micro_batch._teacher_chunk_stats
    assert stats["effective_chunk"] == expected_chunk
    assert stats["configured_chunk"] == 1024 and stats["tokens"] == 3
    assert "Teacher TopK chunk selected" in caplog.text
    if not available:
        assert stats["reason"] == "memory guard fallback"
        assert "retaining original chunk16" in caplog.text


@pytest.mark.parametrize("bundle,chunk", [(False, False), (False, True), (True, False), (True, True)])
def test_independent_switches_reach_actual_forward(
    monkeypatch: pytest.MonkeyPatch, bundle: bool, chunk: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf, "_available", lambda tuning: 10**10)
    moe_calls = []
    monkeypatch.setattr(perf, "install_moe_dispatch", lambda model, mode: moe_calls.append(mode) or 48)
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    original_forward, original_compute = policy._forward_micro_batch, policy.compute_log_prob
    state = perf.configure_teacher_performance(
        policy, {"enabled": bundle, "topk_logprob_chunk_enabled": chunk,
                 "adaptive_topk_chunk_size": 256},
        world_size=1, teacher_model_device="gpu",
    )
    policy.compute_log_prob(Batch(torch.arange(3)[:, None], [5, 21, 8]))

    assert state["enabled"] is bundle and state["topk_logprob_chunk_enabled"] is chunk
    assert (policy._forward_micro_batch is not original_forward) is chunk
    assert (policy.compute_log_prob is not original_compute) is bundle
    assert [call["chunk"] for call in calls] == [1024 if chunk else 16] * (1 if bundle else 3)
    assert moe_calls == (["stable_sort"] if bundle else [])
    if bundle and not chunk:
        assert state["adaptive_topk_chunk_size"] is None and state["chunk"] == 16


@pytest.mark.parametrize("bundle", [True, False])
@pytest.mark.parametrize("chunk_enabled", [True, False])
def test_explicit_metadata_override_survives_both_switches(
    monkeypatch: pytest.MonkeyPatch, bundle: bool, chunk_enabled: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf, "_available", lambda tuning: 0)
    monkeypatch.setattr(perf, "install_moe_dispatch", lambda model, mode: 0)
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    perf.configure_teacher_performance(
        policy, {"enabled": bundle, "topk_logprob_chunk_enabled": chunk_enabled},
        world_size=1, teacher_model_device="gpu",
    )
    data = Batch(torch.arange(3)[:, None], [5, 21, 8])
    data.meta_info["topk_logprob_chunk_size"] = 128
    original_meta = dict(data.meta_info)
    policy.compute_log_prob(data)

    assert [call["chunk"] for call in calls] == [128] * 3
    assert data.meta_info == original_meta


def test_chunk_only_initialization_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    config = {"enabled": False, "topk_logprob_chunk_size": 256}
    state = perf.configure_teacher_performance(policy, config, world_size=4, teacher_model_device="gpu")
    installed = policy._forward_micro_batch
    assert perf.configure_teacher_performance(
        policy, config, world_size=4, teacher_model_device="gpu",
    ) is state
    assert policy._forward_micro_batch is installed and state["chunk"] == 256
    # A second initialization cannot claim to disable an already installed wrapper.
    assert perf.configure_teacher_performance(
        policy, {"enabled": False, "topk_logprob_chunk_enabled": False},
        world_size=4, teacher_model_device="gpu",
    ) is state


def test_successful_effective_chunk_is_visible_under_warning_root(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perf, "_available", lambda tuning: 10**10)
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    logger = logging.getLogger(perf.__name__)
    previous_level = logger.level
    try:
        with caplog.at_level(logging.INFO):
            logging.getLogger().setLevel(logging.WARNING)
            logger.setLevel(logging.NOTSET)
            perf.configure_teacher_performance(
                policy, {"enabled": False}, world_size=4, teacher_model_device="gpu",
            )
            policy.compute_log_prob(Batch(torch.arange(1)[:, None], [5]))
        assert "Teacher performance configured" in caplog.text
        assert "Teacher TopK chunk selected" in caplog.text
        assert "'effective_chunk': 1024" in caplog.text
    finally:
        logger.setLevel(previous_level)


@pytest.mark.parametrize("field,value", [
    ("actor_optimizer", object()), ("torch_dtype", torch.float32),
    ("ulysses_sequence_parallel_size", 2), ("use_remove_padding", False),
    ("use_fused_kernels", True), ("cuda_available", False),
])
def test_chunk_default_keeps_reference_execution_guards(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: field != "cuda_available")
    calls: list[dict[str, Any]] = []
    policy = _policy(calls)
    original_forward, original_compute = policy._forward_micro_batch, policy.compute_log_prob
    if field == "torch_dtype":
        policy.actor_module.config.torch_dtype = value
    elif field != "cuda_available":
        setattr(policy, field, value)
    state = perf.configure_teacher_performance(
        policy, {"enabled": False}, world_size=4, teacher_model_device="gpu",
    )
    assert not state["enabled"] and not state["topk_logprob_chunk_enabled"]
    assert state["reason"]
    assert policy._forward_micro_batch is original_forward
    assert policy.compute_log_prob is original_compute


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_chunk_16_256_1024_preserves_top32_logprobs_and_loss(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(2, 1031, 131, generator=generator).to(dtype)
    student = torch.randn(2, 1031, 131, generator=generator).to(dtype)
    gather_ids = torch.randint(0, 131, (2, 1031, 32), generator=generator)
    outputs = [topk_log_probs_from_logits(
        logits, topk=32, gather_topk_ids=gather_ids,
        normalize_gathered=True, chunk_size=chunk,
    ) for chunk in (16, 256, 1024)]
    ids, log_probs, gathered = outputs[0]
    student_log_probs = student.float().gather(-1, ids)
    expected_loss = topk_distill_loss_matrix(
        student_topk_log_probs=student_log_probs, teacher_topk_log_probs=log_probs,
        mode=TOPK_RENORMALIZED_REVERSE_KL, include_tail=False, temperature=1.0,
    )
    for current_ids, current_log_probs, current_gathered in outputs[1:]:
        assert torch.equal(current_ids, ids)
        torch.testing.assert_close(current_log_probs, log_probs, rtol=0, atol=0)
        torch.testing.assert_close(current_gathered, gathered, rtol=0, atol=0)
        current_loss = topk_distill_loss_matrix(
            student_topk_log_probs=student_log_probs, teacher_topk_log_probs=current_log_probs,
            mode=TOPK_RENORMALIZED_REVERSE_KL, include_tail=False, temperature=1.0,
        )
        torch.testing.assert_close(current_loss, expected_loss, rtol=0, atol=0)
