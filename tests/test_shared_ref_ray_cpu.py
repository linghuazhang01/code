"""Real Ray placement and Gloo dispatch witness, without model/GPU initialization."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("tensordict")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/verl"))

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role

from mopd_verl.launch import build_overrides
from mopd_verl.settings import load_config
from mopd_verl.shared_ref_placement import (
    bind_shared_ref_pool,
    validate_shared_pool_spec,
    verify_shared_ref_workers,
)


class LayoutProbe(Worker):
    """Exercise production worker placement/dispatch using CPU collectives."""

    def __init__(self, config: object, role: str) -> None:
        super().__init__()
        # CPU Worker uses the NPU visibility keyword. Mirror Ray's actual GPU
        # assignment so the parent WorkerDict can inspect the fake GPU slot.
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        self.role = role
        self.events: list[str] = []
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

    def get_cuda_visible_devices(self) -> str:
        # CPU Worker defaults to an NPU keyword; report Ray's CUDA assignment.
        return os.environ["CUDA_VISIBLE_DEVICES"]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        import datetime

        import torch.distributed as dist

        dist.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size,
            timeout=datetime.timedelta(seconds=45),
        )
        self.events.append("init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_shared_ref_cache(self) -> None:
        assert self.role == "ref"
        self.events.append("release")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def witness(self) -> dict:
        import torch.distributed as dist

        value = torch.tensor([self.rank + 1], dtype=torch.int64)
        dist.all_reduce(value)
        return {
            "rank": self.rank, "world": dist.get_world_size(),
            "sum": value.item(), "gpu": self.get_cuda_visible_devices(),
            "port": os.environ["MASTER_PORT"], "events": list(self.events),
            "local_world": os.environ["RAY_LOCAL_WORLD_SIZE"], "pid": os.getpid(),
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def route(self, data: DataProto) -> DataProto:
        data.batch["worker_rank"] = torch.full((len(data),), self.rank, dtype=torch.int64)
        return data


def production_layout_components() -> tuple[type, object]:
    """Load actual layout code without importing optional CUDA engine dependencies."""
    path = ROOT / "third_party/verl/verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ResourcePoolManager")
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
    init_workers = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "init_workers")
    namespace = {
        "__name__": __name__, "dataclass": dataclass, "field": field, "ray": ray,
        "Role": Role, "OmegaConf": OmegaConf, "RayResourcePool": RayResourcePool,
        "RayClassWithInitArgs": RayClassWithInitArgs, "bind_shared_ref_pool": bind_shared_ref_pool,
        "validate_shared_pool_spec": validate_shared_pool_spec,
        "verify_shared_ref_workers": verify_shared_ref_workers,
        "create_colocated_worker_cls": create_colocated_worker_cls,
    }
    exec(compile(ast.Module(body=[manager, init_workers], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - trusted repo layout code
    return namespace["ResourcePoolManager"], namespace["init_workers"]


def test_real_ray_shared_pool_and_independent_gloo_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if ray.is_initialized():
        pytest.skip("This witness requires its own isolated local Ray runtime.")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,2,0,1")
    # Four logical GPU slots exercise Ray's real accelerator scheduler on CPU.
    # macOS Unix socket paths cannot accommodate pytest's long temp path.
    ray_temp = tempfile.TemporaryDirectory(prefix="opd-ray-", dir="/tmp")
    try:
        ray.init(
            num_gpus=4, num_cpus=12, include_dashboard=False, _temp_dir=ray_temp.name,
            runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([
                str(ROOT / "tests"), str(ROOT), str(ROOT / "third_party/verl")
            ])}},
        )
        manager_cls, init_workers = production_layout_components()
        manager = manager_cls(
            resource_pool_spec={"actor_pool": [4], "ref_pool": [3]},
            mapping={Role.ActorRollout: "actor_pool", Role.RefPolicy: "ref_pool"},
            shared_pool_spec={"ref_pool": ("actor_pool", [0, 1, 2])},
        )
        config_path = ROOT / "configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v6_toploss_m05_c01_fixed4_4student_3teacher_shared.yaml"
        with initialize_config_dir(
            config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None
        ):
            config = compose(config_name="ppo_trainer", overrides=build_overrides(load_config(config_path)))
        remote_probe = ray.remote(LayoutProbe)
        trainer = SimpleNamespace(
            resource_pool_manager=manager, hybrid_engine=True, use_critic=False,
            use_reference_policy=True, ref_in_actor=False, use_rm=False,
            config=config, device_name="cuda", ray_worker_group_cls=RayWorkerGroup,
            role_worker_mapping={Role.ActorRollout: remote_probe, Role.RefPolicy: remote_probe},
        )
        init_workers(trainer)
        assert manager.get_n_gpus() == 4
        actor_pool, ref_pool = manager.resource_pool_dict.values()
        assert actor_pool.pgs is ref_pool.pgs
        assert len(actor_pool.pgs) == 1
        assert actor_pool.pgs[0].bundle_count == 4
        assert actor_pool.max_colocate_count == ref_pool.max_colocate_count == 2
        assert all(bundle == {"CPU": 2, "GPU": 1} for bundle in actor_pool.pgs[0].bundle_specs)
        actor = trainer.actor_rollout_wg.witness()
        teacher = trainer.ref_policy_wg.witness()
        assert {row["gpu"] for row in actor} == {"0", "1", "2", "4"}
        assert [row["gpu"] for row in teacher] == ["0", "1", "2"]
        assert [row["rank"] for row in actor] == list(range(4))
        assert [row["rank"] for row in teacher] == list(range(3))
        assert {row["world"] for row in actor} == {4}
        assert {row["world"] for row in teacher} == {3}
        assert {row["local_world"] for row in actor} == {"4"}
        assert {row["local_world"] for row in teacher} == {"3"}
        assert {row["sum"] for row in actor} == {10}
        assert {row["sum"] for row in teacher} == {6}
        assert len({row["pid"] for row in actor + teacher}) == 7
        assert actor[0]["port"] != teacher[0]["port"]
        assert all(row["events"] == ["init", "release"] for row in teacher)
        assert all(row["events"] == ["init"] for row in actor)

        markers = torch.arange(528)
        batch = DataProto.from_dict(
            tensors={"row": markers, "mask": (markers % 2).view(-1, 1),
                     "topk": markers.view(-1, 1) + torch.arange(32).view(1, -1)},
            non_tensors={"uid": np.array([f"uid-{row}" for row in range(528)], dtype=object)},
        )
        for worker_group, local_batch in [(trainer.actor_rollout_wg, 132), (trainer.ref_policy_wg, 176)]:
            routed = worker_group.route(batch)
            torch.testing.assert_close(routed.batch["row"], markers)
            torch.testing.assert_close(routed.batch["topk"], batch.batch["topk"])
            torch.testing.assert_close(routed.batch["mask"], batch.batch["mask"])
            assert routed.non_tensor_batch["uid"].tolist() == batch.non_tensor_batch["uid"].tolist()
            assert torch.bincount(routed.batch["worker_rank"]).tolist() == [local_batch] * worker_group.world_size
    finally:
        ray.shutdown()
        ray_temp.cleanup()
