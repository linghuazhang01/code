"""Real two-rank CPU witness for full-vocabulary loss reduction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mopd_verl.domain_gradient.control_full_vocabulary import (
    global_full_vocabulary_loss_statistics,
)


def _rank_witness(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        ids = torch.tensor([[0, 7], [7, 9]], dtype=torch.long)
        loss = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        result = global_full_vocabulary_loss_statistics(
            (ids[rank : rank + 1],),
            (loss[rank : rank + 1],),
            (torch.ones(1, 2, dtype=torch.bool),),
            (("math",),),
            domains=("math",),
            vocab_size=16,
        )

        assert result.by_domain == {"math": {0: (1.0, 1), 7: (5.0, 2), 9: (4.0, 1)}}
        assert result.valid_token_counts == {"math": 4}
        assert result.valid_score_sums == {"math": pytest.approx(10.0)}
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo backend")
def test_two_rank_full_vocabulary_reduction(tmp_path: Path) -> None:
    mp.spawn(
        _rank_witness,
        args=(str(tmp_path / "rendezvous"),),
        nprocs=2,
        join=True,
    )
