"""Shared ref layout guards, inherited experiment contract, and launcher accounting."""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config
from mopd_verl.shared_ref_placement import (
    shared_bundle_indices,
    shared_ref_pool_spec,
    validate_shared_pool_spec,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
BASE = CONFIGS / "mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4gpu_colocated.yaml"
SHARED = CONFIGS / "mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4student_3teacher_shared.yaml"


def resolved_config() -> object:
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None
    ):
        return hydra.compose(config_name="ppo_trainer", overrides=build_overrides(load_config(SHARED)))


def test_shared_config_preserves_entire_v6_recipe() -> None:
    base, shared = load_config(BASE), load_config(SHARED)
    for name in ("data", "model", "actor", "rollout", "rollout_correction", "teacher_performance"):
        assert asdict(getattr(shared, name)) == asdict(getattr(base, name))
    base_audit, shared_audit = asdict(base.audit), asdict(shared.audit)
    base_audit.pop("output_dir")
    shared_audit.pop("output_dir")
    assert shared_audit == base_audit
    assert shared.runtime.wandb_run_id != base.runtime.wandb_run_id
    assert shared.runtime.cuda_visible_devices == "0,1,2,4"
    assert shared.runtime.slurm_allocation_gpus == 4
    assert shared.worker_placement.share_ref_policy_gpus
    assert shared.worker_placement.ref_policy.gpu_ids == [0, 1, 2]
    assert shared.trainer.total_training_steps == 60
    assert shared.trainer.n_gpus_per_node == 4
    assert shared.data.train_batch_size == shared.actor.ppo_mini_batch_size == 528
    assert shared.actor.ppo_micro_batch_size_per_gpu == 1
    assert shared.teacher_performance.topk_logprob_chunk_enabled
    assert shared.teacher_performance.topk_logprob_chunk_size == 1024
    assert shared.teacher_performance.memory_margin_gib == 12
    assert shared.huggingface_checkpoint.private is False
    assert shared.huggingface_checkpoint.steps == (60,)
    assert shared.trainer.default_local_dir != base.trainer.default_local_dir
    assert shared.huggingface_checkpoint.path_prefix != base.huggingface_checkpoint.path_prefix
    assert len(shared.audit.domain_control_token_candidate_ids["math"]) == 274
    assert len(shared.audit.domain_control_token_candidate_ids["code"]) == 219
    assert {"math": 0.05, "code": 0.01} == shared.audit.control_token_online_top_p_by_domain
    assert [x for x in shared.extra_overrides if "ref.fsdp_config.fsdp_size" not in x] == [
        x for x in base.extra_overrides if "ref.fsdp_config.fsdp_size" not in x
    ]


def test_resolved_hydra_layout_is_four_actor_three_ref() -> None:
    config = resolved_config()
    assert shared_ref_pool_spec(config, [4], [3], "actor", "ref") == {"ref": ("actor", [0, 1, 2])}
    assert config.actor_rollout_ref.actor.fsdp_config.fsdp_size == 1
    assert config.actor_rollout_ref.ref.fsdp_config.fsdp_size == 3
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.ref.teacher_performance.topk_logprob_chunk_size == 1024


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("actor_rollout_ref.worker_placement.separate_ref_policy", False, "separate_ref_policy"),
        ("actor_rollout_ref.worker_placement.ref_policy.gpu_ids", [0, 0, 2], "gpu_ids"),
        ("actor_rollout_ref.worker_placement.ref_policy.gpu_ids", [0, 1], "gpu_ids"),
        ("actor_rollout_ref.worker_placement.ref_policy.gpu_ids", [-1, 1, 2], "gpu_ids"),
        ("actor_rollout_ref.worker_placement.ref_policy.gpu_ids", [False, 1, 2], "gpu_ids"),
        ("actor_rollout_ref.rollout.mode", "async", "sync rollout"),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2, "TP=1"),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2, "sequence parallel"),
        ("actor_rollout_ref.ref.fsdp_config.fsdp_size", 4, "fsdp_size"),
        ("actor_rollout_ref.model.lora_rank", 8, "LoRA"),
        ("actor_rollout_ref.model.lora_adapter_path", "", "LoRA"),
        ("actor_rollout_ref.model.base_model_path", "second-model", "one teacher"),
        ("actor_rollout_ref.ref.model.base_model_path", "second-teacher", "one teacher"),
        ("trainer.device", "cpu", "CUDA"),
    ],
)
def test_unsupported_shared_layouts_fail_before_allocation(key: str, value: object, error: str) -> None:
    config = resolved_config()
    OmegaConf.update(config, key, value, force_add=True)
    with pytest.raises(ValueError, match=error):
        shared_ref_pool_spec(config, [4], [3], "actor", "ref")


def test_shared_layout_rejects_no_set_cuda_and_multiple_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    config = resolved_config()
    with pytest.raises(ValueError, match="one node"):
        shared_ref_pool_spec(config, [2, 2], [3], "actor", "ref")
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        shared_ref_pool_spec(config, [4], [3], "actor", "ref")


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


def local_gpu_count(*overrides: str, visible: str = "0,1,2,4") -> subprocess.CompletedProcess[str]:
    source = (ROOT / "scripts/run_local_mopd_training.sh").read_text()
    parser = source.split('REQUIRED_GPUS="$(', 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    return subprocess.run(
        [sys.executable, "-", str(SHARED), visible, *overrides], input=parser,
        capture_output=True, text=True, cwd=ROOT, check=False,
    )


def test_local_launcher_counts_four_shared_gpus_and_seven_dedicated_gpus() -> None:
    shared = local_gpu_count()
    assert shared.returncode == 0, shared.stderr
    assert shared.stdout.strip() == "4"
    dedicated = local_gpu_count("+actor_rollout_ref.worker_placement.share_ref_policy_gpus=false")
    assert dedicated.returncode == 0, dedicated.stderr
    assert dedicated.stdout.strip() == "7"


@pytest.mark.parametrize(
    "overrides,visible",
    [(["+actor_rollout_ref.worker_placement.ref_policy.gpu_ids=[0,0,2]"], "0,1,2,4"),
     (["+actor_rollout_ref.worker_placement.ref_policy.process_on_nodes=[1,2]"], "0,1,2,4"),
     (["+actor_rollout_ref.worker_placement.separate_ref_policy=false"], "0,1,2,4"),
     ([], "0,1,2,3"), ([], "1,2,3,4")],
)
def test_local_launcher_rejects_wrong_shared_visibility(overrides: list[str], visible: str) -> None:
    assert local_gpu_count(*overrides, visible=visible).returncode != 0


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
