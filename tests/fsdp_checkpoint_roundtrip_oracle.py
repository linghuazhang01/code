"""Two-rank CUDA checkpoint roundtrip oracle for FSDP1 size 1 and 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.fsdp_checkpoint_topology import (
    runtime_fsdp_config,
    validate_checkpoint_topology,
)
from verl.utils.fsdp_mesh import (
    create_device_mesh,
    resolve_fsdp1_mesh_and_strategy,
    validate_fsdp1_size_one_topology,
)


class TinyConfig:
    """Minimum Hugging Face-like config consumed by the checkpoint manager."""

    def save_pretrained(self, path: str | os.PathLike[str]) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "config.json", "w", encoding="utf-8") as handle:
            json.dump({"architectures": ["TinyCheckpointModel"]}, handle)


class TinyCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3)
        self.config = TinyConfig()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)

    def can_generate(self) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-size", type=int, required=True, choices=(1, 2))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--reject-checkpoint", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    return parser.parse_args()


def training_step(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().item())


def assert_global_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    tolerance: float,
) -> float:
    local_diff = (actual - expected).abs().max().double()
    dist.all_reduce(local_diff, op=dist.ReduceOp.MAX)
    difference = float(local_diff.item())
    if difference > tolerance:
        raise AssertionError(
            f"Checkpoint replay max_abs={difference} exceeds {tolerance}."
        )
    return difference


def _rank_batch(
    rank: int,
    device: torch.device,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [
            [1.0 + offset, -0.5, 0.25, float(rank)],
            [-0.25, 0.75 + offset, -1.0, 0.5 + float(rank)],
        ],
        device=device,
    )
    targets = torch.tensor(
        [
            [0.5, -0.25 + offset, 0.75],
            [-0.5 + float(rank), 0.25, -0.75 - offset],
        ],
        device=device,
    )
    return inputs, targets


def _read_metadata(path: Path) -> dict[str, Any]:
    with open(path / "fsdp_config.json", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError(f"Checkpoint oracle requires world_size=2, got {world_size}.")

    logical_mesh = create_device_mesh(world_size, args.fsdp_size)
    model_mesh, strategy = resolve_fsdp1_mesh_and_strategy(
        logical_mesh,
        args.fsdp_size,
    )
    torch.manual_seed(20260714)
    model = FSDP(
        TinyCheckpointModel().to(device),
        device_id=device,
        device_mesh=model_mesh,
        sharding_strategy=strategy,
        sync_module_states=True,
    )
    validated_wrappers = validate_fsdp1_size_one_topology(
        model,
        args.fsdp_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.8,
    )
    checkpoint_config = OmegaConf.create(
        {
            "load_contents": ["model", "optimizer", "extra"],
            "save_contents": ["model", "optimizer", "extra"],
        }
    )
    manager = FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        checkpoint_config=checkpoint_config,
    )

    first_inputs, first_targets = _rank_batch(rank, device, offset=0.0)
    replay_inputs, replay_targets = _rank_batch(rank, device, offset=0.2)
    first_loss = training_step(
        model,
        optimizer,
        scheduler,
        first_inputs,
        first_targets,
    )
    manager.save_checkpoint(str(args.checkpoint_dir), global_step=1)

    expected_loss = training_step(
        model,
        optimizer,
        scheduler,
        replay_inputs,
        replay_targets,
    )
    expected_lr = float(scheduler.get_last_lr()[0])
    with torch.no_grad():
        expected_output = model(replay_inputs).detach().clone()

    manager.load_checkpoint(str(args.checkpoint_dir))
    replay_loss = training_step(
        model,
        optimizer,
        scheduler,
        replay_inputs,
        replay_targets,
    )
    replay_lr = float(scheduler.get_last_lr()[0])
    with torch.no_grad():
        replay_output = model(replay_inputs).detach().clone()

    output_max_abs = assert_global_close(
        replay_output,
        expected_output,
        args.tolerance,
    )
    loss_abs_diff = abs(replay_loss - expected_loss)
    lr_abs_diff = abs(replay_lr - expected_lr)
    if loss_abs_diff > args.tolerance or lr_abs_diff > args.tolerance:
        raise AssertionError(
            "Checkpoint replay did not restore optimizer/scheduler state: "
            f"loss_abs_diff={loss_abs_diff}, lr_abs_diff={lr_abs_diff}."
        )

    runtime = runtime_fsdp_config(model, world_size)
    rejected_cross_topology = False
    if args.reject_checkpoint is not None:
        try:
            validate_checkpoint_topology(str(args.reject_checkpoint), runtime)
        except RuntimeError:
            rejected_cross_topology = True
        if not rejected_cross_topology:
            raise AssertionError("Cross-topology checkpoint restore was not rejected.")

    dist.barrier()
    if rank == 0:
        metadata = _read_metadata(args.checkpoint_dir)
        print(
            {
                "fsdp_size": args.fsdp_size,
                "effective_strategy": model.sharding_strategy.name,
                "fsdp_process_group_size": dist.get_world_size(
                    model.process_group
                ),
                "validated_wrapper_count": validated_wrappers,
                "first_loss": first_loss,
                "replay_loss": replay_loss,
                "expected_loss": expected_loss,
                "output_max_abs": output_max_abs,
                "loss_abs_diff": loss_abs_diff,
                "lr_abs_diff": lr_abs_diff,
                "metadata": metadata,
                "rejected_cross_topology": rejected_cross_topology,
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
