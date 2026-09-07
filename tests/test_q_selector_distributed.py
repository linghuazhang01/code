"""Real two-rank CPU witness for global Q moment reduction."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mopd_verl.domain_gradient.control_top_loss_runtime import (
    global_candidate_loss_statistics_with_valid_counts,
)


def _rank_witness(rank: int, init_file: str, empty_rank: bool) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        ids = torch.tensor([[10, 99], [10, 20]])
        loss = torch.tensor([[1.0, 3.0], [9.0, 7.0]], dtype=torch.float64)
        entropy = torch.tensor([[8.0, 2.0], [1.0, 4.0]], dtype=torch.float64)
        valid = torch.ones_like(ids).bool()
        if empty_rank:
            valid[1] = False
        result = global_candidate_loss_statistics_with_valid_counts(
            (ids[rank : rank + 1],),
            (loss[rank : rank + 1],),
            (valid[rank : rank + 1],),
            (("math",),),
            domains=("math",),
            candidate_token_ids=(10, 20),
            selection_mode="top_q_loss_entropy",
            student_entropy_batches=(entropy[rank : rank + 1],),
        )
        a, b = loss / loss[valid].mean(), entropy / entropy[valid].mean()
        q = a + b + a * b
        assert result.valid_token_counts["math"] == int(valid.sum())
        assert result.valid_score_sums["math"] == pytest.approx(float(q[valid].sum()))
        for token in (10, 20):
            matched = valid & (ids == token)
            if not bool(matched.any()):
                assert token not in result.by_domain["math"]
                continue
            total, count = result.by_domain["math"][token]
            assert count == int(matched.sum())
            assert total == pytest.approx(float(q[matched].sum()))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo backend")
@pytest.mark.parametrize("empty_rank", [False, True])
def test_two_rank_step_global_normalization(tmp_path: Path, empty_rank: bool) -> None:
    mp.spawn(
        _rank_witness,
        args=(str(tmp_path / "rendezvous"), empty_rank),
        nprocs=2,
        join=True,
    )
