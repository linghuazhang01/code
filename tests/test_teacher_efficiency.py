"""Regression checks for teacher batching and supported MoE dispatch."""

import copy

import pytest
import torch

from mopd_verl import teacher_performance as perf
from mopd_verl import teacher_performance_moe as moe
from test_teacher_performance import Batch


@pytest.mark.parametrize("remote_limit", [1, 2])
def test_batch_sync_preserves_local_budget_and_remote_limit(monkeypatch, remote_limit):
    reductions = []
    calls = []
    monkeypatch.setattr(perf, "_available", lambda tuning: 8000 + 16 * 100 * 16)

    def reduce_min(value, world_size):
        reductions.append(value)
        if len(reductions) <= 3 or (len(reductions) - 3) % 2 == 0:
            return value
        return min(value, remote_limit)

    def original(data):
        calls.append(data)
        lengths = perf._lengths(data)
        assert len(data) <= remote_limit
        assert len(data) == 1 or sum(lengths) <= 10
        return data.ids, None

    monkeypatch.setattr(perf, "_distributed_min", reduce_min)
    data = Batch(torch.arange(6)[:, None], [5, 3, 30, 1, 4, 2])
    wrapped = perf._batch_wrapper(original, perf._Tuning(16, 2, 10, 0), 100, 2)
    output = wrapped(data=data)
    assert torch.equal(output[0], data.ids)
    assert output[1] is None
    assert len(reductions) == 3 + 2 * len(calls)


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_installed_transformers_moe_matches_stock(top_k, normalize):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    torch.manual_seed(23)
    config = Qwen3MoeConfig(
        hidden_size=8, intermediate_size=16, moe_intermediate_size=8,
        num_experts=4, num_experts_per_tok=top_k, norm_topk_prob=normalize,
        num_attention_heads=2, num_key_value_heads=2,
    )
    stock = Qwen3MoeSparseMoeBlock(config).eval()
    model = torch.nn.ModuleList([copy.deepcopy(stock) for _ in range(48)])
    assert moe.install_moe_dispatch(model, "stable_sort") == 48
    inputs = torch.randn(2, 7, 8)
    with torch.no_grad():
        expected = stock(inputs)
        actual = model[0](inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
