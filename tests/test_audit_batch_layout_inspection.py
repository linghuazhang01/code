from __future__ import annotations

from copy import deepcopy

import torch

from mopd_verl.verl_audit import MOPDAuditLogger


class SyntheticBatch:
    def __init__(self, labels: list[str]) -> None:
        self.batch = {
            "attention_mask": torch.arange(len(labels) * 4, dtype=torch.long).reshape(
                len(labels), 4
            ),
        }
        self.non_tensor_batch = {
            "opd_teacher": list(labels),
            "sample_id": [f"sample-{idx}" for idx in range(len(labels))],
        }
        self.meta_info: dict[str, object] = {}
        self.reorder_call_count = 0

    def reorder(self, indices: object) -> None:
        self.reorder_call_count += 1
        raise AssertionError(
            f"audit inspection must not reorder production data: {indices}"
        )


def _logger(*, micro_batch_size: int) -> MOPDAuditLogger:
    return MOPDAuditLogger(
        {
            "mopd_audit": {
                "enabled": True,
                "domains": ["math", "code"],
                "full_gradient_enabled": True,
                "full_gradient_micro_batch_size_per_gpu": micro_batch_size,
            }
        }
    )


def test_layout_inspection_preserves_aligned_production_batch_payload() -> None:
    labels = ["math", "math", "code", "code"] * 2
    batch = SyntheticBatch(labels)
    original_attention_mask = batch.batch["attention_mask"]
    original_attention_mask_value = original_attention_mask.clone()
    original_non_tensor_batch = deepcopy(batch.non_tensor_batch)

    metrics = _logger(micro_batch_size=2).inspect_domain_gradient_batch_layout(
        batch,
        step=0,
        world_size=2,
    )

    assert batch.reorder_call_count == 0
    assert batch.batch["attention_mask"] is original_attention_mask
    assert torch.equal(batch.batch["attention_mask"], original_attention_mask_value)
    assert batch.non_tensor_batch == original_non_tensor_batch
    assert metrics["global/audit/full_gradient_domain_partition_aligned"] == 1.0
    assert metrics["global/audit/full_gradient_domain_partition_batch_reordered"] == 0.0
    partition = batch.meta_info["mopd_domain_gradient_partition"]
    assert isinstance(partition, dict)
    assert partition["inspection_only"] is True
    assert partition["production_batch_reordered"] is False
    assert partition["domain_block_sample_counts"] == {"math": 2, "code": 2}
    assert partition["rank_domain_sample_counts"] == [
        {"math": 2, "code": 2},
        {"math": 2, "code": 2},
    ]


def test_legacy_balance_entrypoint_only_reports_unaligned_layout() -> None:
    labels = ["math"] * 4 + ["code"] * 4
    batch = SyntheticBatch(labels)
    original_non_tensor_batch = deepcopy(batch.non_tensor_batch)

    metrics = _logger(micro_batch_size=1).balance_domain_gradient_batch(
        batch,
        step=0,
        world_size=2,
    )

    assert batch.reorder_call_count == 0
    assert batch.non_tensor_batch == original_non_tensor_batch
    assert metrics["global/audit/full_gradient_domain_partition_aligned"] == 0.0
    partition = batch.meta_info["mopd_domain_gradient_partition"]
    assert isinstance(partition, dict)
    assert partition["unsupported_reason"] == "rank_domain_counts_not_aligned"
    assert partition["rank_domain_sample_counts"] == [
        {"math": 4, "code": 0},
        {"math": 0, "code": 4},
    ]


def test_equal_counts_without_configured_domain_blocks_are_not_marked_aligned() -> None:
    labels = ["code", "code", "math", "math"] * 2
    batch = SyntheticBatch(labels)
    original_non_tensor_batch = deepcopy(batch.non_tensor_batch)

    metrics = _logger(micro_batch_size=2).inspect_domain_gradient_batch_layout(
        batch,
        step=0,
        world_size=2,
    )

    assert batch.reorder_call_count == 0
    assert batch.non_tensor_batch == original_non_tensor_batch
    assert metrics["global/audit/full_gradient_domain_partition_aligned"] == 0.0
    partition = batch.meta_info["mopd_domain_gradient_partition"]
    assert isinstance(partition, dict)
    assert partition["unsupported_reason"] == "rank_domain_blocks_not_aligned"
    assert "domain_block_sample_counts" not in partition
