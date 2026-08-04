from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mopd_verl.settings import load_config
from mopd_verl.tensorboard_filter import filter_tensorboard_metrics
from mopd_verl.verl_audit import MOPDAuditLogger


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = (
    "mopd_qwen1p7b_nonthinking_goosereason4b_instruct_6gpu_"
    "math_code_science_topk32_reweight_auditall_nosamplegrad.yaml",
    "mopd_qwen1p7b_nonthinking_goosereason4b_instruct_8gpu_"
    "math_code_science_topk32_reweight_auditall_nosamplegrad.yaml",
)


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_goosereason_profiles_run_full_grad_without_token_grad(
    config_name: str,
) -> None:
    config = load_config(ROOT / "configs" / config_name)
    gpu_count = config.worker_placement.actor_rollout.n_gpus_per_node
    expected_batch_size = 1008 if gpu_count == 6 else 1024

    assert config.data.train_batch_size == expected_batch_size
    assert config.actor.ppo_mini_batch_size == expected_batch_size
    assert expected_batch_size % (gpu_count * 4) == 0
    assert config.audit.full_gradient_enabled
    assert config.audit.full_gradient_freq_steps == 1
    assert not config.audit.token_gradient_enabled
    assert not config.audit.token_gradient_tail_enabled
    assert not config.audit.token_gradient_loss_abs_selection_enabled
    assert config.audit.token_gradient_top_k is None
    assert not config.audit.token_gradient_top_p_enabled
    assert not config.audit.token_gradient_log_tokens_jsonl_enabled
    assert not config.audit.vocab_per_occurrence_mean_vector_enabled
    assert not config.audit.logp_vocab_per_occurrence_mean_vector_enabled
    assert not config.audit.logp_abs_vocab_per_occurrence_mean_vector_enabled
    assert not config.audit.entropy_vocab_per_occurrence_mean_vector_enabled
    assert not config.audit.dynamic_domain_loss_weighting_enabled
    assert "noreweight" in config.trainer.experiment_name


def test_sample_token_loss_mean_distribution_is_logged() -> None:
    logger = MOPDAuditLogger(
        {
            "mopd_audit": {
                "domains": ["math", "code", "empty"],
                "entropy_enabled": False,
                "token_gap_enabled": False,
                "log_sample_level": True,
            }
        }
    )
    batch = SimpleNamespace(
        batch={
            "old_log_probs": torch.zeros(4, 2),
            "response_mask": torch.ones(4, 2),
            "configured_token_loss": torch.tensor(
                [
                    [1.0, 3.0],
                    [3.0, 5.0],
                    [8.0, 12.0],
                    [100.0, 100.0],
                ]
            ),
            "configured_token_loss_mask": torch.tensor(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
            ),
            "math_teacher_log_prob": torch.zeros(4, 2),
            "code_teacher_log_prob": torch.zeros(4, 2),
        },
        non_tensor_batch={
            "opd_teacher": ["math", "math", "code", "math"],
            "sample_id": [
                "math-1",
                "math-2",
                "code-1",
                "math-zero-mask",
            ],
        },
        meta_info={},
    )

    metrics, domain_rows, variance_rows = logger._compute_training_rows(
        batch,
        step=1,
        lr=1e-5,
    )[:3]
    rows = {row["domain"]: row for row in domain_rows}
    variance_by_domain = {row["domain"]: row for row in variance_rows}

    assert rows["math"]["sample_opd_loss_mean"] == pytest.approx(4.0)
    assert rows["math"]["sample_token_opd_loss_mean"] == pytest.approx(3.0)
    assert rows["math"]["sample_token_opd_loss_std"] == pytest.approx(1.0)
    assert rows["math"]["sample_token_opd_loss_variance"] == pytest.approx(
        1.0
    )
    assert rows["code"]["sample_token_opd_loss_mean"] == pytest.approx(10.0)
    assert rows["code"]["sample_token_opd_loss_std"] == pytest.approx(0.0)
    assert rows["empty"]["sample_token_opd_loss_mean"] is None
    assert rows["empty"]["sample_token_opd_loss_std"] is None
    assert rows["empty"]["sample_token_opd_loss_variance"] is None
    assert variance_by_domain["math"][
        "sample_token_opd_loss_variance"
    ] == pytest.approx(1.0)

    expected_global_mean = 16.0 / 3.0
    expected_global_variance = 104.0 / 9.0
    expected_global_std = math.sqrt(104.0) / 3.0
    expected_metrics = {
        "math/loss/sample_token_opd_loss_mean": 3.0,
        "math/loss/sample_token_opd_loss_std": 1.0,
        "math/loss/sample_token_opd_loss_variance": 1.0,
        "global/loss/sample_token_opd_loss_mean": expected_global_mean,
        "global/loss/sample_token_opd_loss_std": expected_global_std,
        "global/loss/sample_token_opd_loss_variance": expected_global_variance,
    }
    for key, expected in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected)

    filtered = filter_tensorboard_metrics(metrics, "core")
    assert set(expected_metrics).issubset(filtered)
    assert not any(
        key.startswith("empty/loss/sample_token_opd_loss_")
        for key in metrics
    )


def test_sample_token_loss_metrics_omit_all_zero_mask_batch(
    tmp_path: Path,
) -> None:
    logger = MOPDAuditLogger(
        {
            "mopd_audit": {
                "enabled": True,
                "output_dir": str(tmp_path / "audit"),
                "domains": ["math"],
                "entropy_enabled": False,
                "token_gap_enabled": False,
                "log_sample_level": True,
            }
        }
    )
    batch = SimpleNamespace(
        batch={
            "old_log_probs": torch.zeros(1, 2),
            "response_mask": torch.ones(1, 2),
            "configured_token_loss": torch.tensor([[10.0, 20.0]]),
            "configured_token_loss_mask": torch.zeros(1, 2),
            "math_teacher_log_prob": torch.zeros(1, 2),
        },
        non_tensor_batch={
            "opd_teacher": ["math"],
            "sample_id": ["math-zero-mask"],
        },
        meta_info={},
    )

    metrics, domain_rows, _, sample_rows = logger._compute_training_rows(
        batch,
        step=1,
        lr=1e-5,
    )[:4]

    for suffix in ("mean", "std", "variance"):
        key = f"sample_token_opd_loss_{suffix}"
        assert domain_rows[0][key] is None
        assert f"math/loss/{key}" not in metrics
        assert f"global/loss/{key}" not in metrics
    assert sample_rows[0]["sample_token_opd_loss_mean"] is None
    assert sample_rows[0]["sample_token_opd_loss_variance"] is None
