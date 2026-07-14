"""FSDP checkpoint topology metadata and state-dict selection."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, exists
from verl.utils.fsdp_utils import fsdp_version


@dataclass(frozen=True)
class FSDPConfig:
    """Topology metadata required for safe rank-local checkpoint restore."""

    FSDP_version: int
    world_size: int
    schema_version: int = 2
    effective_sharding_strategy: str | None = None
    fsdp_process_group_size: int | None = None


def effective_sharding_strategy(model: torch.nn.Module) -> str:
    if isinstance(model, FSDP):
        return model.sharding_strategy.name
    return "FSDP2" if fsdp_version(model) == 2 else "UNSHARDED"


def fsdp_process_group_size(model: torch.nn.Module, world_size: int) -> int:
    if isinstance(model, FSDP):
        return int(torch.distributed.get_world_size(model.process_group))
    return int(world_size)


def runtime_fsdp_config(model: torch.nn.Module, world_size: int) -> FSDPConfig:
    return FSDPConfig(
        FSDP_version=fsdp_version(model),
        world_size=int(world_size),
        effective_sharding_strategy=effective_sharding_strategy(model),
        fsdp_process_group_size=fsdp_process_group_size(model, world_size),
    )


def checkpoint_state_dict_settings(
    model: torch.nn.Module,
    *,
    include_model: bool,
    include_optimizer: bool,
) -> tuple[StateDictType, object | None, object | None]:
    """Select explicit CPU-offloaded state-dict settings for FSDP1."""

    if isinstance(model, FSDP) and model.sharding_strategy == ShardingStrategy.NO_SHARD:
        return (
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            if include_model
            else None,
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            if include_optimizer
            else None,
        )
    return (
        StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=is_cuda_available)
        if include_model
        else None,
        ShardedOptimStateDictConfig(offload_to_cpu=is_cuda_available)
        if include_optimizer
        else None,
    )


def validate_checkpoint_topology(
    local_path: str,
    runtime: FSDPConfig,
) -> None:
    """Reject new checkpoints created with an incompatible FSDP topology."""

    config_path = os.path.join(local_path, "fsdp_config.json")
    if not exists(config_path):
        warnings.warn(
            "Checkpoint has no fsdp_config.json; FSDP topology cannot be "
            "validated before restore.",
            stacklevel=2,
        )
        return
    config_path = copy_to_local(config_path)
    with open(config_path, encoding="utf-8") as handle:
        saved = json.load(handle)

    for key in ("FSDP_version", "world_size"):
        if key in saved and int(saved[key]) != int(getattr(runtime, key)):
            raise RuntimeError(
                "Cross-topology FSDP checkpoint restore is unsupported: "
                f"checkpoint {key}={saved[key]}, runtime {key}="
                f"{getattr(runtime, key)}. Resume with the original topology "
                "or convert the checkpoint first."
            )

    optional_fields = (
        "effective_sharding_strategy",
        "fsdp_process_group_size",
    )
    missing = [key for key in optional_fields if key not in saved]
    if missing:
        warnings.warn(
            "Legacy FSDP checkpoint metadata cannot fully validate topology; "
            f"missing fields: {', '.join(missing)}.",
            stacklevel=2,
        )
        return
    for key in optional_fields:
        if saved[key] != getattr(runtime, key):
            raise RuntimeError(
                "Cross-topology FSDP checkpoint restore is unsupported: "
                f"checkpoint {key}={saved[key]}, runtime {key}="
                f"{getattr(runtime, key)}. Resume with the original fsdp_size "
                "or convert the checkpoint first."
            )
