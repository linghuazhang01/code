"""Shared ref layout guards: physical GPU mapping, pool aliases and the shared cache RPC."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from mopd_verl.shared_ref_placement import (
    shared_bundle_indices,
    shared_ref_pool_spec,
    validate_shared_pool_spec,
)

ROOT = Path(__file__).resolve().parents[1]


def test_shared_mapping_follows_physical_gpu_ids_when_bundles_are_shuffled() -> None:
    assert shared_bundle_indices(["4", "2", "0", "1"], [0, 1, 2]) == [2, 3, 1]


@pytest.mark.parametrize(
    "visible,requested",
    [(["0", "1", "4", "5"], [0, 1, 2]), (["0", "0", "1", "2"], [0, 1, 2]),
     (["0,1", "2", "4", "5"], [0, 1, 2]), (["not set", "1", "2", "4"], [0, 1, 2]),
     (["0", "1", "2", "4"], [0, 0, 1])],
)
def test_invalid_live_gpu_mapping_fails(visible: list[str], requested: list[int]) -> None:
    with pytest.raises(ValueError):
        shared_bundle_indices(visible, requested)


@pytest.mark.parametrize(
    "pools,shared",
    [({"a": [4], "r": [3]}, {"r": ("r", [0, 1, 2])}),
     ({"a": [4], "r": [3]}, {"r": ("missing", [0, 1, 2])}),
     ({"a": [4], "r": [3]}, {"r": ("a", [0, 1])}),
     ({"a": [4], "r": [5]}, {"r": ("a", [0, 1, 2, 3, 4])}),
     ({"a": [2, 2], "r": [3]}, {"r": ("a", [0, 1, 2])}),
     ({"a": [4], "r": [3]}, {"r": ("a", [0, 1, 2]), "a": ("r", [0, 1, 2, 4])})],
)
def test_invalid_pool_aliases_fail(pools: dict, shared: dict) -> None:
    with pytest.raises(ValueError):
        validate_shared_pool_spec(pools, shared)


def test_shared_cache_rpc_synchronizes_then_releases_and_rejects_other_roles() -> None:
    source = ROOT / "third_party/verl/verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "release_shared_ref_cache")
    method.decorator_list = []
    events: list[str] = []
    device = SimpleNamespace(synchronize=lambda: events.append("sync"), empty_cache=lambda: events.append("empty"))
    namespace = {"OmegaConf": OmegaConf, "get_torch_device": lambda: device}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - trusted repo method
    worker = SimpleNamespace(role="ref", config=OmegaConf.create({"worker_placement": {"share_ref_policy_gpus": True}}))
    namespace["release_shared_ref_cache"](worker)
    assert events == ["sync", "empty"]
    worker.role = "actor_rollout"
    with pytest.raises(ValueError, match="shared standalone ref"):
        namespace["release_shared_ref_cache"](worker)
