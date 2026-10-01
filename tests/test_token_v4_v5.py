"""Contracts for the frozen Token V4/V5 taxonomy and config matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.domain_gradient.token_taxonomy_registry import (
    TOKEN_TAXONOMY_ARTIFACT_SHA256,
    TOKEN_TAXONOMY_REVISION,
    taxonomy_counts,
    token_taxonomy,
)
from mopd_verl.launch import build_command
from mopd_verl.settings import load_config
from mopd_verl.verl_audit import MOPDAuditLogger
from scripts.finalize_token_v4_v5 import validate_membership

CONFIG_DIR = Path("configs/token_selection/math_code/taxonomy")


def _metadata(config: object) -> dict[str, object]:
    logger = MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
    return logger.full_gradient_meta("train", 1)["mopd_full_gradient"]


def test_frozen_taxonomy_counts_siblings_and_disjointness() -> None:
    assert TOKEN_TAXONOMY_REVISION == 3
    assert taxonomy_counts("token_v4") == {"math": (119, 9), "code": (153, 45)}
    assert taxonomy_counts("token_v5") == {"math": (61, 9), "code": (72, 45)}
    v4, v5 = token_taxonomy("v4"), token_taxonomy("v5")
    for domain in ("math", "code"):
        assert not set(v5[domain].control) <= set(v4[domain].control)
        assert (
            len(set(v5[domain].control) & set(v4[domain].control))
            == {"math": 57, "code": 64}[domain]
        )
        assert set(v5[domain].control_sibling_added) == set(v5[domain].control) - set(
            v4[domain].control
        )
        assert v5[domain].structure == v4[domain].structure
        assert not set(v4[domain].control) & set(v4[domain].structure)
        assert len(v5[domain].control_sibling_added) == {"math": 4, "code": 8}[domain]
        answer_ids = {
            "math": {1590, 4226, 13023, 16688, 19357, 21806, 73877},
            "code": {1590, 4226, 9217, 11822, 13023, 16688, 19357, 21806},
        }[domain]
        for taxonomy in (v4, v5):
            assert answer_ids <= set(taxonomy[domain].structure)
            assert not answer_ids & set(taxonomy[domain].control)
    assert TOKEN_TAXONOMY_ARTIFACT_SHA256 == (
        "2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8"
    )
    assert 1590 in v4["math"].structure and 16141 not in v4["math"].structure
    assert 5987 in v4["code"].structure and 5138 not in v4["code"].structure


@pytest.mark.parametrize("version", ["v4", "v5"])
@pytest.mark.parametrize("code_top_p", ["05", "02", "01"])
@pytest.mark.parametrize("gpu_count", [3, 4, 8])
@pytest.mark.parametrize("weight", [4, 8])
def test_config_matrix_is_launchable(
    version: str,
    code_top_p: str,
    gpu_count: int,
    weight: int,
) -> None:
    path = CONFIG_DIR / (
        f"mopd_math_code_next_step_token_{version}_toploss_"
        f"m05_c{code_top_p}_fixed{weight}_{gpu_count}gpu_colocated.yaml"
    )
    config = load_config(path)
    assert config.data.train_batch_size == 528
    assert config.actor.ppo_mini_batch_size == 528
    assert config.data.domain_sampling_weights == {"math": 1, "code": 1}
    assert not config.worker_placement.separate_ref_policy
    assert config.worker_placement.actor_rollout.n_gpus_per_node == gpu_count
    assert config.worker_placement.ref_policy.n_gpus_per_node is None
    assert config.trainer.n_gpus_per_node == gpu_count
    assert config.teacher_performance.enabled
    assert config.audit.control_token_online_selection_timing == "next_step"
    assert config.audit.control_token_online_top_p_by_domain == {
        "math": 0.05,
        "code": int(code_top_p) / 100,
    }
    assert config.audit.structure_token_loss_weight == float(weight)
    assert config.audit.control_token_loss_weight == float(weight)
    assert not config.huggingface_checkpoint.private
    assert 0 < config.trainer.total_training_steps <= 65
    command = build_command(config)
    assert f"actor_rollout_ref.ref.fsdp_config.fsdp_size={gpu_count}" in command
    assert "+mopd_audit.structure_token_loss_weighting_enabled=true" in command
    domain = DomainGradientConfig.from_meta(_metadata(config))
    expected = token_taxonomy(f"token_{version}")
    assert domain.effective_domain_candidate_map() == {
        name: value.control for name, value in expected.items()
    }
    assert domain.effective_domain_structure_map() == {
        name: value.structure for name, value in expected.items()
    }
    logger = MOPDAuditLogger({"mopd_audit": asdict(config.audit)})
    evaluation = DomainGradientConfig.from_meta(
        logger.full_gradient_meta("eval", 1)["mopd_full_gradient"]
    )
    assert not evaluation.control_token_online_selection_enabled
    assert not evaluation.structure_token_loss_weighting_enabled


def test_versioned_config_rejects_taxonomy_drift(tmp_path: Path) -> None:
    source = CONFIG_DIR / (
        "mopd_math_code_next_step_token_v5_toploss_"
        "m05_c01_fixed4_4gpu_colocated.yaml"
    )
    config = load_config(source)
    metadata = _metadata(config)
    metadata["token_taxonomy_artifact_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA256"):
        DomainGradientConfig.from_meta(metadata)


def test_all_public_profiles_have_unique_namespaces() -> None:
    paths = sorted(
        CONFIG_DIR.glob(
            "mopd_math_code_next_step_token_v[45]_toploss_"
            "m05_c0[125]_fixed[48]_*gpu_colocated.yaml"
        )
    )
    assert len(paths) == 36
    configs = [load_config(path) for path in paths]
    for getter in (
        lambda value: value.runtime.wandb_run_id,
        lambda value: value.audit.output_dir,
        lambda value: value.paper_eval.output_dir,
        lambda value: value.trainer.experiment_name,
        lambda value: value.trainer.default_local_dir,
        lambda value: value.huggingface_checkpoint.path_prefix,
    ):
        values = [getter(config) for config in configs]
        assert len(values) == len(set(values))
        assert all("-r3-" in value and "-next-" in value for value in values)
    for config in configs:
        gpu_count = config.trainer.n_gpus_per_node
        assert config.rollout.tensor_model_parallel_size == 1
        assert config.actor.fsdp_size == 1
        assert 528 % gpu_count == 0
        assert 264 % gpu_count == 0
        assert config.runtime.wandb_resume == "never"
        assert "trainer.resume_mode=disable" in build_command(config)


def _artifact_and_rows() -> tuple[dict, list[dict[str, str]]]:
    root = Path("mopd_verl/domain_gradient")
    artifact = json.loads((root / "token_v4_v5.json").read_text())
    with (root / "token_v4_v5_membership.csv").open(newline="") as handle:
        return artifact, list(csv.DictReader(handle))


def test_frozen_json_matches_all_csv_keep_sets_and_strict_support() -> None:
    artifact, rows = _artifact_and_rows()
    summary = validate_membership(artifact, rows)
    assert len(rows) == 666
    assert summary["math"]["v5_control"] == 61
    assert summary["code"]["structure"] == 45
    changed = deepcopy(rows)
    row = next(row for row in changed if row["status"] == "keep")
    row[f"maxocc_{row['domain']}_4b_opd"] = "20"
    row["min_maxocc"] = "20"
    with pytest.raises(ValueError, match="support >20"):
        validate_membership(artifact, changed)


def test_importer_rejects_legacy_pending_surfaces_and_membership_drift() -> None:
    artifact, rows = _artifact_and_rows()
    with pytest.raises(ValueError, match="revision 3 metadata"):
        validate_membership(
            {key: value for key, value in artifact.items() if key != "meta"}, rows
        )
    artifact["v5"]["math"]["control"][0] = 123
    with pytest.raises(ValueError, match="CSV/JSON mismatch"):
        validate_membership(artifact, rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("structure_token_loss_weight", 6.0),
        ("control_token_loss_weight", 8.0),
        ("structure_token_position_profile", "token_v4_v5_answer_format"),
        (
            "structure_token_position_profile",
            "token_v4_v5_fourbaseline_r2_answer_format",
        ),
        (
            "token_taxonomy_artifact_sha256",
            "27884fadeea94145df23ede69109c7f8181c866de2f1c9f93d7659786719d817",
        ),
        (
            "token_taxonomy_artifact_sha256",
            "c45b5f7b6bd824140922706b2cf75630dce7130d580c346c638a3358833fd97b",
        ),
    ],
)
def test_worker_rejects_old_revision_and_inconsistent_weights(
    field: str, value: object
) -> None:
    config = load_config(
        CONFIG_DIR
        / "mopd_math_code_next_step_token_v5_toploss_m05_c01_fixed4_4gpu_colocated.yaml"
    )
    metadata = _metadata(config)
    metadata[field] = value
    with pytest.raises(ValueError):
        DomainGradientConfig.from_meta(metadata)


def test_revision_one_artifacts_and_namespaces_remain_recoverable() -> None:
    history = Path("mopd_verl/domain_gradient/token_taxonomy_history/revision_1")
    artifact = json.loads((history / "token_v4_v5.json").read_text())
    assert len(artifact["v5"]["math"]["control"]) == 59
    configs = json.loads((history / "configs.json").read_text())
    assert len(configs) == 26
    assert all("-r2-" not in text for text in configs.values())


def test_revision_two_artifacts_and_namespaces_remain_recoverable() -> None:
    history = Path("mopd_verl/domain_gradient/token_taxonomy_history/revision_2")
    assert hashlib.sha256((history / "token_v4_v5.json").read_bytes()).hexdigest() == (
        "c45b5f7b6bd824140922706b2cf75630dce7130d580c346c638a3358833fd97b"
    )
    assert (
        hashlib.sha256(
            (history / "token_v4_v5_membership.csv").read_bytes()
        ).hexdigest()
        == "f8efb0b8346e9145af0adabe065205b810fa04e7d9567beaa1632d9d6562ad70"
    )
    assert (
        json.loads((history / "token_v4_v5_provenance.json").read_text())["revision"]
        == 2
    )
    configs = json.loads((history / "configs.json").read_text())
    assert len(configs) == 44
    public = [
        text for path, text in configs.items() if Path(path).name.startswith("mopd")
    ]
    assert len(public) == 36 and all("-r2-" in text for text in public)


@pytest.mark.parametrize("surface", [" Final", ".answer", "[Conclusion", "\\final"])
def test_importer_rejects_answerword_disguised_as_control(surface: str) -> None:
    artifact, rows = _artifact_and_rows()
    row = next(
        row for row in rows if row["family"] == "Control_V5" and row["status"] == "keep"
    )
    row["token"] = surface
    with pytest.raises(ValueError, match="AnswerWord"):
        validate_membership(artifact, rows)
