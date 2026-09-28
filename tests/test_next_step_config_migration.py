"""The active V4/V5 matrix keeps its contract while shifting selection one step."""

from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path

from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
FIXTURES = ROOT / "tests/fixtures/next_step_migration"
NAMESPACE_FIELDS = {
    "runtime.wandb_run_id",
    "audit.output_dir",
    "paper_eval.output_dir",
    "huggingface_checkpoint.path_prefix",
    "trainer.experiment_name",
    "trainer.default_local_dir",
}
RETIRED_EXAMPLES = {
    "mopd_math_code_current_step_toploss_m05_c01_tailhalf_4gpu.yaml",
    "mopd_math_code_current_step_toploss_m05_c01_tailzero_4gpu.yaml",
    "mopd_math_code_current_step_teacher_confidence_m05_c01_tailhalf_4gpu.yaml",
    "mopd_math_code_current_step_loss_teacher_confidence_m05_c01_tailhalf_4gpu.yaml",
}


def _normalized(config: object) -> dict:
    resolved = json.loads(json.dumps(asdict(config)))
    # This optional selector must leave every pre-existing profile unchanged.
    assert resolved["audit"].pop("versioned_cs_selection_mode_by_domain") == {}
    assert resolved["audit"].pop("code_cs_position_gate_enabled") is None
    return resolved


def _fixture(name: str) -> dict:
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt") as handle:
        return json.load(handle)


def test_migrated_public_profiles_preserve_all_non_timing_parameters() -> None:
    originals = _fixture("current-step-resolved")
    active = {path: value for path, value in originals.items()
              if Path(path).name not in RETIRED_EXAMPLES}
    assert len(active) == 37
    namespace_values = {field: set() for field in NAMESPACE_FIELDS}

    for path, before in active.items():
        legacy = ROOT / path
        canonical = legacy.with_name(legacy.name.replace("current_step", "next_step"))
        after = _normalized(load_config(canonical))
        assert after == _normalized(load_config(legacy))
        changes = {
            f"{section}.{name}"
            for section in before
            if isinstance(before[section], dict)
            for name in before[section]
            if before[section][name] != after[section][name]
        }
        assert changes == NAMESPACE_FIELDS | {"audit.control_token_online_selection_timing"}
        matched = deepcopy(after)
        for field in changes:
            section, name = field.split(".", 1)
            matched[section][name] = before[section][name]
        assert matched == before
        assert after["audit"]["control_token_online_selection_timing"] == "next_step"
        assert after["audit"]["control_token_online_selection_unit"] == "token_id"
        assert after["audit"]["control_token_online_selection_mode"] == "top_loss"
        assert after["audit"]["control_token_online_selection_mode_by_domain"] == {}
        assert after["audit"]["control_token_tail_top_p"] == 0
        assert not after["audit"]["control_token_tail_top_p_by_domain"]
        assert after["audit"]["control_token_tail_weight"] == 1
        assert after["audit"]["control_token_online_audit_interval_steps"] == 1
        assert after["audit"]["control_token_online_window_steps"] == 1
        assert after["data"]["train_batch_size"] == 528
        assert after["actor"]["ppo_mini_batch_size"] == 528
        assert after["trainer"]["total_training_steps"] == 60
        for field in NAMESPACE_FIELDS:
            section, name = field.split(".", 1)
            value = after[section][name]
            assert "-next-" in value
            assert value not in namespace_values[field]
            namespace_values[field].add(value)


def test_private_helpers_are_aliased_and_retired_examples_are_archived() -> None:
    originals = _fixture("current-step-configs")
    helpers = [ROOT / path for path in originals if Path(path).name.startswith("_")]
    assert len(helpers) == 8
    for legacy in helpers:
        canonical = legacy.with_name(legacy.name.replace("current_step", "next_step"))
        assert canonical.exists()
        assert legacy.read_text().strip().endswith(f"extends: {canonical.name}")
    for name in RETIRED_EXAMPLES:
        assert not (CONFIGS / name).exists()
        assert not (CONFIGS / name.replace("current_step", "next_step")).exists()
        assert any(Path(path).name == name for path in originals)


def test_launched_v4_profile_remains_identical_to_frozen_snapshot() -> None:
    frozen = _fixture("launch-next-step-resolved")
    path = CONFIGS / "mopd_math_code_next_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml"
    assert _normalized(load_config(path)) == frozen
