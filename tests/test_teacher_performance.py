"""Teacher performance routing, memory fallback, and output preservation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mopd_verl import teacher_performance as perf


class Batch:
    def __init__(self, ids: torch.Tensor, lengths: list[int]) -> None:
        self.ids = ids
        self.batch = {"ref_attention_mask": torch.tensor(lengths)[:, None]}
        self.meta_info = {"micro_batch_size": 1, "use_dynamic_bsz": True, "temperature": 0.7}
        self.non_tensor_batch = {}

    def __len__(self) -> int:
        return len(self.ids)

    def select_idxs(self, indices: list[int]) -> "Batch":
        return Batch(self.ids[indices], self.batch["ref_attention_mask"][indices, 0].tolist())


def test_memory_budget_includes_actual_chunk() -> None:
    assert perf.token_capacity(100000, 100, 1024) == 0
    assert perf.token_capacity(100000, 100, 16) == 93
    assert perf.token_capacity(12 * 1024**3, 151936, 1024) < perf.token_capacity(
        12 * 1024**3, 151936, 256)


@pytest.mark.parametrize("capacity", [0, 1, 8, 20, 100])
def test_groups_cover_long_rows_once(capacity: int) -> None:
    lengths, groups, start = [5, 3, 30, 1], [], 0
    while start < len(lengths):
        group = perf.next_group(lengths, start, 2, capacity)
        groups.append(group)
        start += len(group)
    assert sum(groups, []) == list(range(len(lengths)))
    assert all(len(group) <= 2 for group in groups)
    assert all(len(group) == 1 or sum(lengths[i] for i in group) <= capacity for group in groups)


@pytest.mark.parametrize("as_keyword", [True, False])
@pytest.mark.parametrize("entropy", [True, False])
def test_batch_preserves_order_kwargs_optional_outputs_and_input(
    monkeypatch: pytest.MonkeyPatch, as_keyword: bool, entropy: bool,
) -> None:
    monkeypatch.setattr(perf, "_available", lambda tuning: 8000 + 16 * 100 * 16)
    tuning = perf._Tuning(chunk=16, max_sequences=3, max_tokens=10, margin_bytes=12 * 1024**3)
    calls = []

    def original(data: Batch, calculate_entropy: bool, topk: int, gather_topk_ids_key: str) -> tuple:
        calls.append(data)
        assert data.meta_info["micro_batch_size"] == len(data)
        assert data.meta_info["use_dynamic_bsz"] is False
        assert data.meta_info["temperature"] == 0.7
        assert topk == 32 and gather_topk_ids_key == "ids"
        return data.ids, data.ids + 1 if calculate_entropy else None, data.ids + 2

    data = Batch(torch.arange(4)[:, None], [5, 3, 30, 1])
    wrapped = perf._batch_wrapper(original, tuning, 100)
    kwargs = dict(calculate_entropy=entropy, topk=32, gather_topk_ids_key="ids")
    result = wrapped(data=data, **kwargs) if as_keyword else wrapped(data, **kwargs)
    assert [len(call) for call in calls] == [2, 1, 1]
    assert torch.equal(result[0], data.ids)
    assert torch.equal(result[2], data.ids + 2)
    assert (result[1] is None) == (not entropy)
    assert data.meta_info["micro_batch_size"] == 1 and data.meta_info["use_dynamic_bsz"] is True


@pytest.mark.parametrize("available,expected", [(10**10, 1024), (0, None)])
def test_forward_guard_falls_back_without_rejecting_single_row(
    monkeypatch: pytest.MonkeyPatch, available: int, expected: int | None,
) -> None:
    monkeypatch.setattr(perf, "_available", lambda tuning: available)

    def original(micro_batch: dict, temperature: float, topk_logprob_chunk_size: int | None = None) -> tuple:
        return temperature, topk_logprob_chunk_size

    wrapped = perf._chunk_wrapper(original, perf._Tuning(1024, 32, 57344, 12 * 1024**3), 100)
    data = {"attention_mask": torch.ones(1, 20)}
    assert wrapped(data, temperature=0.7) == (0.7, expected)
    assert wrapped(data, 0.7, 128) == (0.7, 128)


@pytest.mark.parametrize("field,value", [
    ("world_size", 2), ("teacher_model_device", "cpu"), ("use_fused_kernels", True),
    ("use_remove_padding", False), ("ulysses_sequence_parallel_size", 2),
])
def test_unsupported_layout_does_not_modify_methods(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    forward, compute = object(), object()
    policy = SimpleNamespace(
        _forward_micro_batch=forward, compute_log_prob=compute, use_fused_kernels=False,
        use_remove_padding=True, ulysses_sequence_parallel_size=1, actor_optimizer=None,
        actor_module=SimpleNamespace(config=SimpleNamespace(torch_dtype=torch.bfloat16)),
    )
    layout = dict(world_size=1, teacher_model_device="gpu")
    if field in layout:
        layout[field] = value
    else:
        setattr(policy, field, value)
    state = perf.configure_teacher_performance(policy, {"enabled": True}, **layout)
    assert not state["enabled"]
    assert policy._forward_micro_batch is forward and policy.compute_log_prob is compute


def test_disabled_does_not_inspect_or_mutate_policy() -> None:
    policy = SimpleNamespace()
    assert not perf.configure_teacher_performance(
        policy, {"enabled": False}, world_size=1, teacher_model_device="gpu")["enabled"]
    assert vars(policy) == {}


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_production_moe_preserves_routing_and_accumulation(top_k: int, normalize: bool) -> None:
    from mopd_verl.teacher_performance_moe import qwen3_moe_forward_stable_sort

    torch.manual_seed(17)
    block = torch.nn.Module()
    block.gate = torch.nn.Linear(8, 4, bias=False)
    block.experts = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(4)])
    block.num_experts, block.top_k, block.norm_topk_prob = 4, top_k, normalize
    inputs = torch.randn(2, 7, 8)
    flat = inputs.view(-1, 8)
    router = block.gate(flat)
    weights, experts = torch.topk(torch.softmax(router, dim=1, dtype=torch.float), top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    mask = torch.nn.functional.one_hot(experts, num_classes=4).permute(2, 1, 0)
    expected = torch.zeros_like(flat)
    for expert in range(4):
        rank, token = torch.where(mask[expert])
        expected.index_add_(0, token, block.experts[expert](flat[token]) * weights[token, rank, None])
    actual, actual_router = qwen3_moe_forward_stable_sort(block, inputs)
    assert torch.equal(actual, expected.view_as(inputs))
    assert torch.equal(actual_router, router)


def test_configuration_installs_once_and_passes_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    allocator_calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", allocator_calls.append)
    seen = []
    monkeypatch.setattr(perf, "install_moe_dispatch", lambda model, mode: seen.append(mode) or 48)

    def forward(micro_batch: dict, topk_logprob_chunk_size: int | None = None) -> None:
        return None

    policy = SimpleNamespace(
        _forward_micro_batch=forward, compute_log_prob=lambda data: (data,), use_fused_kernels=False,
        use_remove_padding=True, ulysses_sequence_parallel_size=1, actor_optimizer=None,
        actor_module=SimpleNamespace(config=SimpleNamespace(torch_dtype=torch.bfloat16, vocab_size=100)),
    )
    state = perf.configure_teacher_performance(policy, {"enabled": True}, world_size=1, teacher_model_device="gpu")
    installed = policy.compute_log_prob
    assert state["enabled"] and state["chunk"] == 1024 and state["moe_blocks"] == 48
    assert state["max_tokens"] == 57344 and state["max_micro_batch_size"] == 32
    assert perf.configure_teacher_performance(
        policy, {"enabled": True}, world_size=1, teacher_model_device="gpu") is state
    assert policy.compute_log_prob is installed and seen == ["stable_sort"]
    assert allocator_calls == []


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("device", ["gpu", "cuda"])
def test_allocator_is_independent_of_batching(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, device: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)
    assert perf.configure_teacher_allocator(
        {"enabled": enabled}, teacher_model_device=device, dedicated_teacher=True,
    )
    assert calls == ["expandable_segments:True"]


@pytest.mark.parametrize("config,device,dedicated,available", [
    ({"expandable_segments": False}, "gpu", True, True),
    ({}, "gpu", False, True),
    ({}, "cpu", True, True),
    ({}, "gpu", True, False),
])
def test_allocator_skips_without_changing_process_settings(
    monkeypatch: pytest.MonkeyPatch, config: dict, device: str, dedicated: bool, available: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)
    assert not perf.configure_teacher_allocator(
        config, teacher_model_device=device, dedicated_teacher=dedicated,
    )
    assert calls == []


def test_allocator_failure_is_not_reported_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unsupported(settings: str) -> None:
        raise RuntimeError("allocator unavailable")

    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", unsupported)
    with pytest.raises(RuntimeError, match="allocator unavailable"):
        perf.configure_teacher_allocator({}, teacher_model_device="gpu", dedicated_teacher=True)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("enabled", [True, False])
def test_ref_initialization_sets_allocator_before_build_on_each_rank(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, rank: int, enabled: bool,
) -> None:
    from omegaconf import OmegaConf, open_dict

    source = Path(__file__).resolve().parents[1] / "third_party/verl/verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    init_method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "init_model")
    ref_block = next(
        node for node in init_method.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "_is_ref"
    )
    events = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", lambda value: events.append(value))
    forward, compute = object(), object()
    policy = SimpleNamespace(
        _forward_micro_batch=forward, compute_log_prob=compute, use_fused_kernels=False,
        use_remove_padding=True, ulysses_sequence_parallel_size=1, actor_optimizer=None,
        actor_module=SimpleNamespace(config=SimpleNamespace(torch_dtype=torch.bfloat16)),
    )

    def build_model(**kwargs: object) -> tuple:
        assert events == ["expandable_segments:True"]
        events.append("build")
        return (object(),)

    worker = SimpleNamespace(
        _is_ref=True, _is_actor=False, _is_rollout=False, rank=rank, world_size=2,
        _build_model_optimizer=build_model,
        config=OmegaConf.create({
            "model": {"path": "student"},
            "ref": {"model": {"path": "teacher", "teacher_model_device": "gpu"},
                    "fsdp_config": {}, "teacher_performance": {"enabled": enabled}},
            "worker_placement": {"separate_ref_policy": True},
        }),
    )
    namespace = dict(
        self=worker, OmegaConf=OmegaConf, open_dict=open_dict, use_shm=False,
        override_model_config={}, use_remove_padding=True, use_fused_kernels=False,
        copy_to_local=lambda path, **kwargs: path, omega_conf_to_dataclass=lambda value: value,
        DataParallelPPOActor=lambda **kwargs: policy,
    )
    exec(compile(ast.Module(body=[ref_block], type_ignores=[]), str(source), "exec"), namespace)
    assert events == ["expandable_segments:True", "build"]
    assert policy._forward_micro_batch is forward and policy.compute_log_prob is compute
    assert f"rank={rank} world_size=2 expandable_segments_applied=True" in capsys.readouterr().out


def test_multimodal_forward_retains_original_without_memory_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(tuning: object) -> int:
        raise AssertionError("Text-only guard should not inspect multimodal input")

    monkeypatch.setattr(perf, "_available", unexpected)

    def original(micro_batch: dict, topk_logprob_chunk_size: int | None = None) -> int | None:
        return topk_logprob_chunk_size

    wrapped = perf._chunk_wrapper(original, perf._Tuning(1024, 32, 57344, 12 * 1024**3), 100)
    assert wrapped({"multi_modal_inputs": {}}) is None
