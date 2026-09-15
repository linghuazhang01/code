"""Compatibility gates for the teacher MoE dispatch optimization."""

from __future__ import annotations

import hashlib

import torch

from mopd_verl import teacher_performance_moe as moe


def _fake_qwen_model() -> torch.nn.Module:
    def forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    block_type = type(
        "Qwen3MoeSparseMoeBlock",
        (torch.nn.Module,),
        {
            "__module__": "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "forward": forward,
        },
    )
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block_type() for _ in range(48)])
    return model


def test_allowlist_contains_exact_running_and_benchmarked_versions() -> None:
    assert moe.EXPECTED_SOURCE_SHA_BY_VERSION == {
        "4.51.3": "c4ae8784c667994091405761b588d9d73f8b4290f5bd4d43dcd5ea3842dc63cb",
        "4.57.6": "21b88637319fb0e3b5bbb87d98d1f04104d5ce176b9270486d638e6114b1b29a",
    }


def test_matching_version_and_source_installs_all_blocks(monkeypatch) -> None:
    source = "def forward(self, hidden_states):\n    return hidden_states\n"
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    monkeypatch.setattr(moe, "version", lambda package: "4.51.3")
    monkeypatch.setattr(moe.inspect, "getsource", lambda function: source)
    monkeypatch.setitem(moe.EXPECTED_SOURCE_SHA_BY_VERSION, "4.51.3", source_sha)
    model = _fake_qwen_model()

    assert moe.install_moe_dispatch(model, "stable_sort") == 48
    assert all("forward" in vars(block) for block in model.blocks)


def test_source_mismatch_retains_stock_forward(monkeypatch) -> None:
    monkeypatch.setattr(moe, "version", lambda package: "4.51.3")
    monkeypatch.setattr(moe.inspect, "getsource", lambda function: "changed")
    model = _fake_qwen_model()

    assert moe.install_moe_dispatch(model, "stable_sort") == 0
    assert all("forward" not in vars(block) for block in model.blocks)
