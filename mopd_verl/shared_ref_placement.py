"""Explicit, single-node reference workers sharing an actor placement group."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from verl.single_controller.ray import RayResourcePool, RayWorkerGroup

logger = logging.getLogger(__name__)


def shared_ref_pool_spec(
    config: DictConfig,
    actor_pool: list[int],
    ref_pool: list[int],
    actor_pool_id: str,
    ref_pool_id: str,
) -> dict[str, tuple[str, list[int]]]:
    """Validate the opt-in layout before any workers or GPU bundles are created."""
    placement = "actor_rollout_ref.worker_placement"
    if not OmegaConf.select(config, f"{placement}.share_ref_policy_gpus", default=False):
        if OmegaConf.select(config, f"{placement}.ref_policy.gpu_ids") is not None:
            raise ValueError("ref_policy.gpu_ids requires share_ref_policy_gpus=true.")
        return {}
    if not OmegaConf.select(config, f"{placement}.separate_ref_policy", default=False):
        raise ValueError("Shared ref GPUs require separate_ref_policy=true.")
    if len(actor_pool) != 1 or len(ref_pool) != 1 or ref_pool[0] > actor_pool[0]:
        raise ValueError("Shared ref GPUs require one node and ref workers <= actor workers.")
    if config.trainer.device != "cuda":
        raise ValueError("Shared ref GPUs support CUDA workers only.")
    if config.actor_rollout_ref.actor.strategy != "fsdp" or config.actor_rollout_ref.ref.strategy != "fsdp":
        raise ValueError("Shared ref GPUs support FSDP workers only.")
    rollout = config.actor_rollout_ref.rollout
    if rollout.mode != "sync" or rollout.tensor_model_parallel_size != 1:
        raise ValueError("Shared ref GPUs require sync rollout and TP=1 for sequential memory handoff.")
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("Shared ref GPUs currently require sequence parallel size 1.")
    if int(config.actor_rollout_ref.ref.fsdp_config.fsdp_size) not in {-1, ref_pool[0]}:
        raise ValueError("Shared ref fsdp_size must equal its worker count or be -1.")
    model = config.actor_rollout_ref.model
    if int(model.get("lora_rank", 0)) > 0 or model.get("lora_adapter_path") is not None:
        raise ValueError("Shared ref GPUs require a standalone teacher, without LoRA reference-in-actor.")
    if (
        model.get("base_model_path") is not None
        or OmegaConf.select(config, "actor_rollout_ref.ref.model.base_model_path") is not None
    ):
        raise ValueError("Shared ref GPUs currently support one teacher model only.")
    if OmegaConf.select(config, "actor_rollout_ref.ref.model.teacher_model_device") not in {"gpu", "cuda"}:
        raise ValueError("Shared ref GPUs require a GPU-resident teacher.")
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"):
        raise ValueError("Shared ref GPUs require Ray to set each worker's CUDA_VISIBLE_DEVICES.")
    gpu_ids = OmegaConf.select(config, f"{placement}.ref_policy.gpu_ids")
    if (
        gpu_ids is None
        or len(gpu_ids) != ref_pool[0]
        or any(type(item) is not int or item < 0 for item in gpu_ids)
        or len(set(gpu_ids)) != len(gpu_ids)
    ):
        raise ValueError("Shared ref gpu_ids must name one distinct physical GPU per ref worker.")
    return {ref_pool_id: (actor_pool_id, list(gpu_ids))}


def validate_shared_pool_spec(
    pool_spec: dict[str, list[int]],
    shared_spec: dict[str, tuple[str, list[int]]],
) -> None:
    """Reject cyclic, nested, or oversubscribed pool aliases."""
    sources: set[str] = set()
    for target, (source, gpu_ids) in shared_spec.items():
        if target not in pool_spec or source not in pool_spec or target == source:
            raise ValueError("A shared pool must reference a distinct existing source pool.")
        if source in shared_spec or source in sources:
            raise ValueError("Only one shared ref pool per independent actor pool is supported.")
        actor_counts, ref_counts = pool_spec[source], pool_spec[target]
        if (
            len(actor_counts) != 1
            or len(ref_counts) != 1
            or not 0 < ref_counts[0] <= actor_counts[0]
            or len(gpu_ids) != ref_counts[0]
            or any(type(item) is not int or item < 0 for item in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)
        ):
            raise ValueError("Shared pool GPU IDs and single-node worker counts are inconsistent.")
        sources.add(source)


def shared_bundle_indices(
    visible_devices: Sequence[str], requested_gpu_ids: Sequence[int]
) -> list[int]:
    """Map physical IDs to bundles using live actor visibility, never rank order."""
    actual_ids = [str(value).strip() for value in visible_devices]
    if any(not value.isdigit() for value in actual_ids) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"Each actor bundle must expose one distinct physical GPU; got {actual_ids}.")
    requested = [str(value) for value in requested_gpu_ids]
    if len(set(requested)) != len(requested) or any(value not in actual_ids for value in requested):
        raise ValueError(f"Requested ref GPUs {requested} are not a distinct subset of actor GPUs {actual_ids}.")
    return [actual_ids.index(value) for value in requested]


def bind_shared_ref_pool(
    target_pool: RayResourcePool,
    source_pool: RayResourcePool,
    source_workers: RayWorkerGroup,
    gpu_ids: list[int],
) -> None:
    """Reuse the source PG and select the bundles exposing the requested GPUs."""
    import ray

    pgs = source_pool.get_placement_groups()
    if len(pgs) != 1 or source_workers.world_size != source_pool.world_size:
        raise ValueError("Shared ref placement requires all actor workers on one placement group.")
    visible = ray.get([worker.get_cuda_visible_devices.remote() for worker in source_workers.workers])
    indices = shared_bundle_indices(visible, gpu_ids)
    if target_pool.world_size != len(indices):
        raise ValueError("Shared ref worker count does not match the requested GPU IDs.")
    target_pool.pgs = pgs
    target_pool.bundle_indices = indices
    logger.info("Shared ref GPUs=%s, actor GPUs=%s, source bundle indices=%s", gpu_ids, visible, indices)


def verify_shared_ref_workers(worker_group: RayWorkerGroup, gpu_ids: list[int]) -> None:
    """Fail before model initialization if Ray placed a ref on the wrong GPU."""
    import ray

    visible = ray.get([worker.get_cuda_visible_devices.remote() for worker in worker_group.workers])
    if visible != [str(value) for value in gpu_ids]:
        raise ValueError(f"Shared ref placement mismatch: expected {gpu_ids}, actual visibility {visible}.")
