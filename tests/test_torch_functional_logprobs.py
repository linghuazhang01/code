"""Shape and memory-regression tests for VERL log-probability helpers."""

from __future__ import annotations

import pytest
import torch

from verl.utils.torch_functional import logprobs_from_logits_v2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logprobs_from_logits_v2_matches_log_softmax_for_3d_inputs(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(11)
    logits = torch.randn(2, 1300, 17, dtype=dtype)
    labels = torch.randint(0, 17, (2, 1300))

    actual = logprobs_from_logits_v2(logits, labels)
    expected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1).to(dtype),
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


def test_logprobs_from_logits_v2_preserves_empty_shape() -> None:
    logits = torch.empty(0, 4, 17, dtype=torch.bfloat16)
    labels = torch.empty(0, 4, dtype=torch.long)

    result = logprobs_from_logits_v2(logits, labels)

    assert result.shape == (0, 4)
    assert result.dtype == logits.dtype
