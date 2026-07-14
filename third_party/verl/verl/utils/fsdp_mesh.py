# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared FSDP device-mesh and FSDP1 reshard helpers."""

from typing import Optional, Tuple

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from verl.utils.device import get_device_name


def create_device_mesh(world_size: int, fsdp_size: int) -> DeviceMesh:
    """Create verl's logical FSDP/replica mesh."""
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        return init_device_mesh(
            device_name,
            mesh_shape=(world_size,),
            mesh_dim_names=("fsdp",),
        )
    return init_device_mesh(
        device_name,
        mesh_shape=(world_size // fsdp_size, fsdp_size),
        mesh_dim_names=("ddp", "fsdp"),
    )


def get_sharding_strategy(device_mesh: DeviceMesh) -> ShardingStrategy:
    """Map a one- or two-dimensional logical mesh to its shard strategy."""
    if device_mesh.ndim == 1:
        return ShardingStrategy.FULL_SHARD
    if device_mesh.ndim == 2:
        return ShardingStrategy.HYBRID_SHARD
    raise NotImplementedError(
        f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
    )


def resolve_fsdp1_mesh_and_strategy(
    logical_device_mesh: DeviceMesh,
    fsdp_size: int,
) -> Tuple[Optional[DeviceMesh], ShardingStrategy]:
    """Resolve the process group topology used by an FSDP1 wrapper.

    The logical mesh remains available to verl for dispatch and replica
    accounting. For ``fsdp_size == 1``, FSDP1 uses the default WORLD process
    group with ``NO_SHARD`` so that every rank owns a full parameter replica
    and FSDP all-reduces gradients across all replicas after backward.
    """
    if fsdp_size == 1:
        return None, ShardingStrategy.NO_SHARD
    return logical_device_mesh, get_sharding_strategy(logical_device_mesh)


def maybe_reshard_fsdp1_root(module: FullyShardedDataParallel) -> bool:
    """Reshard an FSDP1 root only when its effective strategy is sharded."""
    handle = getattr(module, "_handle", None)
    if handle is None or not bool(handle.uses_sharded_strategy):
        return False
    handle.reshard(True)
    return True


def validate_fsdp1_size_one_topology(
    module: FullyShardedDataParallel,
    requested_fsdp_size: int,
) -> int:
    """Fail fast unless every size-one FSDP1 wrapper is WORLD NO_SHARD."""

    if requested_fsdp_size != 1:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("fsdp_size=1 requires an initialized process group.")

    world_size = int(dist.get_world_size())
    wrappers = tuple(
        child
        for child in module.modules()
        if isinstance(child, FullyShardedDataParallel)
    )
    if not wrappers:
        raise RuntimeError("fsdp_size=1 expected at least one FSDP1 wrapper.")

    for index, wrapper in enumerate(wrappers):
        strategy = wrapper.sharding_strategy
        process_group_size = int(dist.get_world_size(wrapper.process_group))
        if strategy != ShardingStrategy.NO_SHARD or process_group_size != world_size:
            raise RuntimeError(
                "Invalid fsdp_size=1 FSDP1 topology: "
                f"wrapper={index}, effective_strategy={strategy.name}, "
                f"process_group_size={process_group_size}, world_size={world_size}. "
                "Expected NO_SHARD on the worker WORLD process group."
            )
    return len(wrappers)
