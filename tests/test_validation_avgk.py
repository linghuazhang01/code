from __future__ import annotations

import pytest


pytest.importorskip("tensordict")

from verl.trainer.ppo.metric_utils import process_validation_metrics


def test_validation_mean_at_four_averages_rollouts_per_prompt() -> None:
    metrics = process_validation_metrics(
        data_sources=["HMMT25Feb"] * 8,
        sample_uids=["prompt-1"] * 4 + ["prompt-2"] * 4,
        infos_dict={
            "reward": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        },
    )

    # Per-prompt means are 0.5 and 0.75; validation reports their dataset mean.
    assert metrics["HMMT25Feb"]["reward"]["mean@4"] == pytest.approx(0.625)
