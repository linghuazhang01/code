"""GPU oracle for domain-gradient closure under replicated/FULL_SHARD FSDP1.

The minimum two-GPU topology matrix is:

* ``fsdp_size=1``: synchronized ``NO_SHARD`` replication across both ranks.
* ``fsdp_size=2``: FULL_SHARD across both ranks with replica count 1.

Run with:
    PYTHONPATH=.:third_party/verl torchrun --standalone --nproc-per-node=2 \
        tests/fsdp_domain_gradient_oracle.py --fsdp-size 1
    PYTHONPATH=.:third_party/verl torchrun --standalone --nproc-per-node=2 \
        tests/fsdp_domain_gradient_oracle.py --fsdp-size 2
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep the topology oracle independent of the full Ray/vLLM runtime. Geometry
# only needs VERL's current-device helper.
verl_module = ModuleType("verl")
verl_module.__path__ = []
utils_module = ModuleType("verl.utils")
utils_module.__path__ = []
device_module = ModuleType("verl.utils.device")
device_module.get_device_id = lambda: (
    torch.device("cuda", torch.cuda.current_device())
    if DEVICE_TYPE == "cuda"
    else torch.device("cpu")
)
device_module.get_device_name = lambda: DEVICE_TYPE
sys.modules.update(
    {
        "verl": verl_module,
        "verl.utils": utils_module,
        "verl.utils.device": device_module,
    }
)

FSDP_MESH_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "verl"
    / "verl"
    / "utils"
    / "fsdp_mesh.py"
)
fsdp_mesh_spec = importlib.util.spec_from_file_location(
    "mopd_test_fsdp_mesh",
    FSDP_MESH_PATH,
)
if fsdp_mesh_spec is None or fsdp_mesh_spec.loader is None:
    raise RuntimeError(f"Cannot load FSDP mesh helpers from {FSDP_MESH_PATH}")
fsdp_mesh_module = importlib.util.module_from_spec(fsdp_mesh_spec)
fsdp_mesh_spec.loader.exec_module(fsdp_mesh_module)
create_device_mesh = fsdp_mesh_module.create_device_mesh
resolve_fsdp1_mesh_and_strategy = (
    fsdp_mesh_module.resolve_fsdp1_mesh_and_strategy
)
validate_fsdp1_size_one_topology = (
    fsdp_mesh_module.validate_fsdp1_size_one_topology
)
maybe_reshard_fsdp1_root = fsdp_mesh_module.maybe_reshard_fsdp1_root

from mopd_verl.domain_gradient.geometry import (  # noqa: E402
    domain_metrics_from_gram,
    snapshot_gradients,
    training_parity_metrics,
    vector_dot,
    vector_squared_norm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-size", type=int, default=-1)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--bf16-tolerance", type=float, default=2e-2)
    return parser.parse_args()


def create_mesh(
    world_size: int,
    fsdp_size: int,
) -> tuple[object, object | None, ShardingStrategy, int]:
    logical_mesh = create_device_mesh(world_size, fsdp_size)
    model_mesh, strategy = resolve_fsdp1_mesh_and_strategy(
        logical_mesh,
        fsdp_size,
    )
    if fsdp_size <= 0 or fsdp_size >= world_size:
        replica_count = 1
    else:
        replica_count = world_size // fsdp_size
    return logical_mesh, model_mesh, strategy, replica_count


def expected_two_gpu_topology(fsdp_size: int) -> tuple[ShardingStrategy, int]:
    if fsdp_size == 1:
        return ShardingStrategy.NO_SHARD, 2
    if fsdp_size == 2:
        return ShardingStrategy.FULL_SHARD, 1
    raise ValueError(
        "The minimum two-GPU oracle only accepts fsdp_size 1 or 2, "
        f"got {fsdp_size}."
    )


class ActorView:
    def __init__(self, model: FSDP, fsdp_size: int) -> None:
        self.actor_module = model
        self.config = {"fsdp_config": {"fsdp_size": fsdp_size}}
        self.scaler = None


def replay(
    actor: ActorView,
    micro_batches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    domain: int | None,
) -> None:
    actor.actor_module.zero_grad(set_to_none=True)
    denominator = sum(int(features.shape[0]) for features, _ in micro_batches)
    for features, labels in micro_batches:
        prediction = actor.actor_module(features).squeeze(-1)
        if domain is not None:
            mask = (labels == domain).to(dtype=prediction.dtype)
            prediction = prediction * mask + prediction.detach() * (1.0 - mask)
        (prediction.square().sum() / float(denominator)).backward()


def gather_full_flat_vector(
    vector: tuple[torch.Tensor, ...],
    strategy: ShardingStrategy,
    world_size: int,
) -> torch.Tensor:
    """Gather a tiny oracle vector into logical parameter order on every rank."""
    local = torch.cat(vector).to(device=DEVICE_TYPE, dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if strategy == ShardingStrategy.NO_SHARD:
        return gathered[0]
    return torch.cat(gathered)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if DEVICE_TYPE == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError(
            "This minimum oracle requires exactly two ranks, "
            f"got world_size={world_size}."
        )
    logical_mesh, model_mesh, strategy, replica_count = create_mesh(
        world_size,
        args.fsdp_size,
    )
    expected_strategy, expected_replica_count = expected_two_gpu_topology(
        args.fsdp_size
    )
    if strategy != expected_strategy or replica_count != expected_replica_count:
        raise AssertionError(
            "Unexpected two-GPU topology: "
            f"strategy={strategy.name}, replica_count={replica_count}; "
            f"expected strategy={expected_strategy.name}, "
            f"replica_count={expected_replica_count}."
        )

    torch.manual_seed(7)
    module = nn.Linear(4, 1, bias=False, device=device)
    initial_weight = module.weight.detach().clone()
    model = FSDP(
        module,
        device_id=device,
        device_mesh=model_mesh,
        sharding_strategy=strategy,
        sync_module_states=DEVICE_TYPE == "cuda",
    )
    validated_wrapper_count = validate_fsdp1_size_one_topology(
        model,
        args.fsdp_size,
    )
    effective_strategy = model.sharding_strategy
    fsdp_process_group_size = dist.get_world_size(model.process_group)
    actor = ActorView(model, args.fsdp_size)
    rank_features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.5, -1.0],
                [0.25, 1.0, -0.5, 0.0],
                [-1.0, 0.5, 0.25, 1.0],
                [0.75, -0.25, 1.0, 0.5],
            ],
            [
                [-0.5, 1.0, 0.0, 0.25],
                [1.25, -0.75, 0.5, -0.5],
                [0.0, 0.5, -1.0, 1.5],
                [-1.0, -0.5, 0.75, 0.25],
            ],
        ],
        device=device,
    )
    rank_labels = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1]],
        device=device,
    )
    features = rank_features[rank]
    labels = rank_labels[rank]
    micro_batches = (
        (features[:2], labels[:2]),
        (features[2:], labels[2:]),
    )

    replay(actor, micro_batches, domain=None)
    reshard_applied = maybe_reshard_fsdp1_root(model)
    total = snapshot_gradients(actor)
    compact_total = snapshot_gradients(actor, "bfloat16")
    full_total = gather_full_flat_vector(total, effective_strategy, world_size)
    replica_gradient_max_abs_diff = 0.0
    replica_parameter_max_abs_diff = 0.0
    if args.fsdp_size == 1:
        local_gradient = total[0].to(device=device, dtype=torch.float64)
        gathered_gradients = [
            torch.empty_like(local_gradient)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_gradients, local_gradient)
        replica_gradient_max_abs_diff = max(
            float((gradient - gathered_gradients[0]).abs().max().item())
            for gradient in gathered_gradients[1:]
        )
    local_analytic_gradient = (
        2.0
        * (
            (features.double() @ initial_weight.double().T)
            * features.double()
        ).sum(dim=0)
        / float(features.shape[0])
    )
    dist.all_reduce(local_analytic_gradient, op=dist.ReduceOp.SUM)
    analytic_gradient = local_analytic_gradient / float(world_size)
    gradient_coordinate_max_abs_diff = float(
        (full_total - analytic_gradient).abs().max().item()
    )
    analytic_total_sq = float(analytic_gradient.square().sum().item())
    replay(actor, ((features, labels),), domain=None)
    single_backward = snapshot_gradients(actor)
    full_single_backward = gather_full_flat_vector(
        single_backward,
        effective_strategy,
        world_size,
    )
    accumulation_max_abs_diff = float(
        (full_total - full_single_backward).abs().max().item()
    )
    domain_vectors = {}
    compact_domain_vectors = {}
    for domain in (0, 1):
        replay(actor, micro_batches, domain=domain)
        domain_vectors[domain] = snapshot_gradients(actor)
        compact_domain_vectors[domain] = snapshot_gradients(actor, "bfloat16")
    total_sq = vector_squared_norm(actor, total)
    compact_total_sq = vector_squared_norm(actor, compact_total)
    domain_sq = {
        domain: vector_squared_norm(actor, vector)
        for domain, vector in domain_vectors.items()
    }
    domain_total_dot = {
        domain: vector_dot(actor, vector, total)
        for domain, vector in domain_vectors.items()
    }
    compact_domain_sq = {
        domain: vector_squared_norm(actor, vector)
        for domain, vector in compact_domain_vectors.items()
    }
    compact_domain_total_dot = {
        domain: vector_dot(actor, vector, compact_total)
        for domain, vector in compact_domain_vectors.items()
    }
    math_code_dot = vector_dot(actor, domain_vectors[0], domain_vectors[1])
    compact_math_code_dot = vector_dot(
        actor,
        compact_domain_vectors[0],
        compact_domain_vectors[1],
    )
    metrics = domain_metrics_from_gram(
        actor,
        ("math", "code"),
        total_sq=total_sq,
        domain_sq={"math": domain_sq[0], "code": domain_sq[1]},
        domain_total_dot={
            "math": domain_total_dot[0],
            "code": domain_total_dot[1],
        },
        pair_dot={("math", "code"): math_code_dot},
        closure_threshold=args.tolerance,
    )
    compact_metrics = domain_metrics_from_gram(
        actor,
        ("math", "code"),
        total_sq=compact_total_sq,
        domain_sq={
            "math": compact_domain_sq[0],
            "code": compact_domain_sq[1],
        },
        domain_total_dot={
            "math": compact_domain_total_dot[0],
            "code": compact_domain_total_dot[1],
        },
        pair_dot={("math", "code"): compact_math_code_dot},
        closure_threshold=args.bf16_tolerance,
        all_vectors_fp32=False,
        storage_dtype="bfloat16",
    )
    replay(actor, micro_batches, domain=None)
    parity = training_parity_metrics(actor, total, args.tolerance)
    compact_parity = training_parity_metrics(
        actor,
        compact_total,
        args.bf16_tolerance,
    )
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    replay(actor, micro_batches, domain=None)
    optimizer.step()
    local_parameter = next(model.parameters()).detach().reshape(-1)
    parameter_vector = (local_parameter.to(device="cpu", copy=True),)
    full_parameter = gather_full_flat_vector(
        parameter_vector,
        effective_strategy,
        world_size,
    )
    expected_parameter = initial_weight.double().reshape(-1) - (
        learning_rate * analytic_gradient
    )
    parameter_update_max_abs_diff = float(
        (full_parameter - expected_parameter).abs().max().item()
    )
    if args.fsdp_size == 1:
        gathered_parameters = [
            torch.empty_like(local_parameter)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_parameters, local_parameter)
        replica_parameter_max_abs_diff = max(
            float((parameter - gathered_parameters[0]).abs().max().item())
            for parameter in gathered_parameters[1:]
        )
    closure_prefix = (
        "global/pre_reweight_full_grad_closure/"
        "domain_sum_vs_pre_reweight_audit_total"
    )
    parity_prefix = "global/full_grad_training_parity/audit_total_vs_training_total"
    total_norm_rel_error = abs(total_sq - analytic_total_sq) / max(
        analytic_total_sq,
        1e-30,
    )
    compact_norm_rel_error = abs(compact_total_sq - analytic_total_sq) / max(
        analytic_total_sq,
        1e-30,
    )
    if rank == 0:
        print(
            {
                "fsdp_size": args.fsdp_size,
                "requested_strategy": strategy.name,
                "effective_strategy": effective_strategy.name,
                "logical_mesh_shape": tuple(logical_mesh.shape),
                "fsdp_process_group_size": fsdp_process_group_size,
                "replica_count": replica_count,
                "validated_wrapper_count": validated_wrapper_count,
                "reshard_applied": reshard_applied,
                "replica_gradient_max_abs_diff": replica_gradient_max_abs_diff,
                "replica_parameter_max_abs_diff": replica_parameter_max_abs_diff,
                "gradient_coordinate_max_abs_diff": (
                    gradient_coordinate_max_abs_diff
                ),
                "accumulation_max_abs_diff": accumulation_max_abs_diff,
                "parameter_update_max_abs_diff": parameter_update_max_abs_diff,
                "total_norm_rel_error": total_norm_rel_error,
                "bf16_total_norm_rel_error": compact_norm_rel_error,
                "closure_rel_l2": metrics[f"{closure_prefix}/rel_l2"],
                "closure_cosine": metrics[f"{closure_prefix}/cosine"],
                "bf16_closure_rel_l2": compact_metrics[
                    f"{closure_prefix}/rel_l2"
                ],
                "bf16_closure_cosine": compact_metrics[
                    f"{closure_prefix}/cosine"
                ],
                "training_parity_rel_l2": parity[f"{parity_prefix}/rel_l2"],
                "training_parity_cosine": parity[f"{parity_prefix}/cosine"],
                "bf16_training_parity_rel_l2": compact_parity[
                    f"{parity_prefix}/rel_l2"
                ],
                "bf16_training_parity_cosine": compact_parity[
                    f"{parity_prefix}/cosine"
                ],
            }
        )
    assert metrics["global/audit/full_gradient_replica_count"] == float(
        replica_count
    )
    assert reshard_applied == (args.fsdp_size == 2)
    assert validated_wrapper_count > 0 if args.fsdp_size == 1 else validated_wrapper_count == 0
    assert effective_strategy == expected_strategy
    assert fsdp_process_group_size == world_size
    assert replica_gradient_max_abs_diff <= args.tolerance
    assert replica_parameter_max_abs_diff <= args.tolerance
    assert gradient_coordinate_max_abs_diff <= args.tolerance
    assert accumulation_max_abs_diff <= args.tolerance
    assert parameter_update_max_abs_diff <= args.tolerance
    assert total_norm_rel_error <= args.tolerance
    assert compact_norm_rel_error <= args.bf16_tolerance
    assert metrics[f"{closure_prefix}/rel_l2"] <= args.tolerance
    assert not any("domain_sum_vs_training" in key for key in metrics)
    assert compact_metrics[f"{closure_prefix}/rel_l2"] <= args.bf16_tolerance
    assert compact_metrics["global/audit/gradient_correctness_storage_fp32"] == 0.0
    assert parity[f"{parity_prefix}/rel_l2"] <= args.tolerance
    assert compact_parity[f"{parity_prefix}/rel_l2"] <= args.bf16_tolerance
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
