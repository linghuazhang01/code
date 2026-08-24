"""Loss scaling helpers shared by sharded actor updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def global_token_mean_loss_scales(
    response_masks: Sequence[torch.Tensor],
    *,
    reduction_device: torch.device | str,
    distributed: bool = True,
) -> list[float]:
    """Scale local token means into one global valid-token mean.

    FSDP averages gradients across ranks, so each local numerator is multiplied
    by the data-parallel world size after the global denominator is reduced.
    """

    token_counts = [float(mask.detach().sum().item()) for mask in response_masks]
    total = torch.tensor(
        sum(token_counts),
        dtype=torch.float32,
        device=reduction_device,
    )
    world_size = 1
    if (
        distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    global_token_count = float(total.item())
    if global_token_count <= 0.0:
        raise ValueError("Actor update requires at least one valid response token.")
    return [
        float(world_size) * count / global_token_count
        for count in token_counts
    ]
