from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mopd_verl.audit_proxy import extract_teacher_domains, extract_validation_datasets
from mopd_verl.domain_sampling import (
    DomainBatchSampler,
    allocate_domain_batch_counts,
    domain_sample_weights,
    normalize_domain_sampling_weights,
)
from mopd_verl.general_reasoner_data import general_reasoner_to_verl_parquet
from mopd_verl.launch import build_command, format_command
from mopd_verl.prepare_data import (
    evalplus_jsonl_to_verl_parquet,
    lcb_jsonl_to_verl_parquet,
    math_eval_jsonl_to_verl_parquet,
    merge_teacher_data,
    prepare_paper_eval_data,
    teacher_counts,
    validate_sample_ids,
    validate_teacher_labels,
)
from mopd_verl.search_retrieval_server import RetrievalService, SearchResult
from mopd_verl.searchqa_data import searchqa_to_verl_parquet
from mopd_verl.settings import WorkerPlacementConfig, WorkerPoolPlacementConfig, load_config
from mopd_verl.smoke_data import write_smoke_data
from mopd_verl.topk_distill import (
    DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    DISTILL_LOSS_BUILDER_TOPK_KL,
    TOPK_FORWARD_KL_WITH_TAIL,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    chosen_token_forward_kl_matrix,
    chosen_token_policy_gradient_reward_matrix,
    distill_loss_builder,
    resolved_topk_distill_mode,
    selected_logits_from_hidden_states,
    teacher_prefix_masks,
    topk_distill_loss_matrix,
    topk_log_probs_from_logits,
    topk_teacher_student_cross_entropy_matrix,
    uses_topk_distill_loss,
)
from mopd_verl.teacher_prefix import build_dataset_teacher_prefix
from grpo.data.toolrl import toolrl_to_verl_parquet
from grpo.rewards.toolrl import compute_score as compute_toolrl_score
from mopd_verl.verl_audit import MOPDAuditLogger


class MOPDVerlTests(unittest.TestCase):
    def test_domain_gradient_batch_balance_aligns_two_actor_ranks(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self) -> None:
                lengths = [(idx * 7) % 16 + 1 for idx in range(64)]
                self.batch = {
                    "attention_mask": torch.stack(
                        [
                            torch.cat(
                                [
                                    torch.ones(length, dtype=torch.long),
                                    torch.zeros(16 - length, dtype=torch.long),
                                ]
                            )
                            for length in lengths
                        ]
                    )
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math"] * 32 + ["code"] * 32,
                    "sample_id": [f"sample-{idx}" for idx in range(64)],
                }

            def reorder(self, indices: object) -> None:
                index_list = indices.tolist()
                self.batch = {key: value[indices] for key, value in self.batch.items()}
                self.non_tensor_batch = {
                    key: [value[idx] for idx in index_list] for key, value in self.non_tensor_batch.items()
                }

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": True,
                    "full_gradient_micro_batch_size_per_gpu": 1,
                }
            }
        )
        batch = SyntheticBatch()

        metrics = logger.balance_domain_gradient_batch(batch, step=0, world_size=2)

        self.assertEqual(metrics["global/audit/full_gradient_domain_partition_aligned"], 1.0)
        self.assertEqual(metrics["global/audit/full_gradient_domain_partition_unsupported"], 0.0)
        self.assertTrue(batch.meta_info["mopd_domain_gradient_partition"]["aligned"])
        self.assertEqual(
            batch.meta_info["mopd_domain_gradient_partition"]["domain_block_sample_counts"],
            {"math": 16, "code": 16},
        )
        for rank in range(2):
            rank_labels = batch.non_tensor_batch["opd_teacher"][rank * 32 : (rank + 1) * 32]
            self.assertEqual(rank_labels.count("math"), 16)
            self.assertEqual(rank_labels.count("code"), 16)
        self.assertEqual(len(set(batch.non_tensor_batch["sample_id"])), 64)

        rank_workloads = []
        for rank in range(2):
            rank_mask = batch.batch["attention_mask"][rank * 32 : (rank + 1) * 32]
            rank_lengths = rank_mask.sum(dim=-1)
            rank_workloads.append(int((24576 * rank_lengths + rank_lengths.square()).sum().item()))
        self.assertLess(abs(rank_workloads[0] - rank_workloads[1]), max(rank_workloads) * 0.05)

    def test_domain_gradient_batch_balance_skips_non_divisible_domains(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {"attention_mask": torch.ones((8, 4), dtype=torch.long)}
                self.non_tensor_batch = {
                    "opd_teacher": ["math"] * 5 + ["code"] * 3,
                    "sample_id": [f"sample-{idx}" for idx in range(8)],
                }

            def reorder(self, indices: object) -> None:
                raise AssertionError(f"unsupported batch must not be reordered: {indices}")

        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": True,
                }
            }
        )

        batch = SyntheticBatch()
        metrics = logger.balance_domain_gradient_batch(batch, step=0, world_size=2)

        self.assertEqual(metrics["global/audit/full_gradient_domain_partition_aligned"], 0.0)
        self.assertEqual(metrics["global/audit/full_gradient_domain_partition_unsupported"], 1.0)
        self.assertEqual(
            batch.meta_info["mopd_domain_gradient_partition"]["unsupported_reason"],
            "domain_counts_not_divisible_by_rank_micro_batch",
        )

    def test_validation_dataset_and_teacher_domain_are_separate(self) -> None:
        non_tensor = {
            "data_source": ["AIME2024", "codeforces"],
            "ability": ["math", "code"],
        }

        self.assertEqual(extract_teacher_domains(non_tensor, 2), ["math", "code"])
        self.assertEqual(extract_validation_datasets(non_tensor, 2), ["AIME2024", "codeforces"])

    def test_teacher_domains_fall_back_to_extra_info(self) -> None:
        non_tensor = {
            "extra_info": [
                {"opd_teacher": "math"},
                {"domain": "code"},
                json.dumps({"source_domain": "math"}),
            ]
        }

        self.assertEqual(extract_teacher_domains(non_tensor, 3), ["math", "code", "math"])

    def test_full_gradient_labels_fall_back_to_extra_info(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import (
                SequentialBackwardDomainGradientTracker,
                _labels_from_inputs,
                _teacher_labels,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            if exc.name == "torch":
                self.skipTest(f"torch is not installed: {exc}")
            raise

        class SyntheticBatch:
            non_tensor_batch = {
                "extra_info": [
                    {"domain": "math"},
                    {"domain": "code"},
                ]
            }

            def __len__(self) -> int:
                return 2

        self.assertEqual(_teacher_labels(SyntheticBatch()), ["math", "code"])
        self.assertEqual(
            _labels_from_inputs({"opd_teacher": ["code"], "domain": ["math"]}, 1),
            ["math"],
        )
        self.assertEqual(
            _labels_from_inputs(
                {"extra_info": [{"source_domain": "math"}, {"ability": "code"}]},
                2,
            ),
            ["math", "code"],
        )

        tracker = object.__new__(SequentialBackwardDomainGradientTracker)
        tracker.domains = ["math", "code"]
        tracker.inject_opd_teacher_from_domain_partition = False
        tracker._domain_partition_injected_domain = 0.0
        tracker._domain_partition_injected_opd_teacher = 0.0
        first = SyntheticBatch()
        second = SyntheticBatch()
        first.non_tensor_batch = {}
        second.non_tensor_batch = {}
        tracker._inject_partition_labels(
            [first, second],
            {
                "aligned": True,
                "domain_order": ["math", "code"],
                "domain_block_sample_counts": {"math": 2, "code": 2},
            },
            batch_idx_list=[[0, 3], [1, 2]],
        )

        self.assertEqual(_teacher_labels(first), ["math", "code"])
        self.assertEqual(_teacher_labels(second), ["math", "code"])
        self.assertNotIn("opd_teacher", first.non_tensor_batch)
        self.assertNotIn("opd_teacher", second.non_tensor_batch)

    def test_sequence_masked_target_records_schedule_without_direct_recompute(self) -> None:
        try:
            import torch

            from mopd_verl.full_gradient import tracker as tracker_module
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            if exc.name == "torch":
                self.skipTest(f"torch is not installed: {exc}")
            raise

        class SyntheticMicroBatch:
            batch = {"response_mask": torch.ones((1, 3), dtype=torch.float32)}
            non_tensor_batch = {
                "opd_teacher": ["math"],
                "sample_id": ["sample-0"],
            }

            def __len__(self) -> int:
                return 1

        tracker = object.__new__(SequentialBackwardDomainGradientTracker)
        tracker.sample_norm_enabled = False
        tracker.sample_cos_enabled = False
        tracker.token_gradient_enabled = False
        tracker.sequence_masked_target_enabled = True
        tracker.domain_gradient_enabled = True
        tracker.full_gradient_direct_recompute_enabled = False
        tracker._prepared_supported = True
        tracker.domains = ["math", "code"]
        tracker.step = 1
        tracker._micro_batch_index = 0
        tracker._sample_records = []
        tracker._schedule_candidates = []
        tracker._domain_recompute_candidates = {}
        tracker._sample_candidates = {}
        tracker._token_gradient_candidates = {}
        tracker._token_gradient_selected_sample_ids = {}
        tracker._sample_counts = {}

        micro_batch = SyntheticMicroBatch()
        with patch.object(tracker_module, "_copy_data_proto_rows_to_cpu", return_value=micro_batch):
            tracker.record_pre_update_micro_batch(
                "math",
                micro_batch,
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(len(tracker._schedule_candidates), 1)
        self.assertEqual(tracker._schedule_candidates[0]["domain"], "math")
        self.assertEqual(tracker._sample_counts["math"], 1)

    def test_sequence_domain_targets_use_contribution_scale(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            if exc.name == "torch":
                self.skipTest(f"torch is not installed: {exc}")
            raise

        tracker = object.__new__(SequentialBackwardDomainGradientTracker)
        tracker.sequence_masked_target_enabled = True
        tracker.training_gradient_from_domain_sum_enabled = False
        tracker.storage_dtype = "float32"
        tracker.domains = ["math", "code"]

        target_specs = []

        def fake_recompute(self, target_spec, *, storage_dtype):
            target_specs.append(dict(target_spec))
            return {}, tuple(), 0.0

        with patch.object(
            SequentialBackwardDomainGradientTracker,
            "_recompute_masked_schedule_target",
            fake_recompute,
        ):
            tracker._recompute_sequence_domain_targets()

        domain_specs = [item for item in target_specs if item.get("type") == "domain"]
        self.assertEqual([item["domain"] for item in domain_specs], ["math", "code"])
        self.assertTrue(all(item.get("apply_token_mask_contribution_scale") for item in domain_specs))

    def test_default_command_contains_multi_teacher_setting(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        command = build_command(config)
        rendered = format_command(command)

        self.assertIn("actor_rollout_ref.model.path=../models/Qwen3-4B", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.path=../models/Qwen3-4B-Non-Thinking-RL-Math-Step500", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.teacher_model_device=cpu", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.base_model_path=../models/Qwen3-4B-Non-Thinking-RL-Code-Step300", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.lambda_vals=1.0", rendered)
        self.assertIn("+data.domain_sampling_weights={math: 0.5, code: 0.5}", rendered)
        self.assertIn("+data.domain_sampling_replacement=true", rendered)
        self.assertIn("+data.domain_train_files=", rendered)
        self.assertIn("data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet", rendered)
        self.assertIn("data/G-OPD-Training-Data/Eurus/code_train.parquet", rendered)
        self.assertNotIn("math_and_code/train.parquet", rendered)
        self.assertIn("eval/domains/math/data/HMMT25Feb/test.parquet", rendered)
        self.assertIn("eval/domains/math/data/HMMT25Nov/test.parquet", rendered)
        self.assertIn("eval/domains/code/data/HumanEvalPlus/test.parquet", rendered)
        self.assertIn("eval/domains/code/data/MBPPPlus/test.parquet", rendered)
        self.assertNotIn("eval/domains/code/data/LiveCodeBench/test.parquet", rendered)

    def test_teacher_model_device_command_is_config_controlled(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        gpu_teacher_config = replace(
            config,
            model=replace(config.model, teacher_model_device="gpu"),
        )
        rendered = format_command(build_command(gpu_teacher_config))

        self.assertIn("+actor_rollout_ref.ref.model.teacher_model_device=gpu", rendered)

    def test_worker_placement_command_is_config_controlled(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        split_worker_config = replace(
            config,
            worker_placement=WorkerPlacementConfig(
                separate_ref_policy=True,
                actor_rollout=WorkerPoolPlacementConfig(n_gpus_per_node=2, nnodes=1),
                ref_policy=WorkerPoolPlacementConfig(process_on_nodes=[2]),
            ),
        )
        rendered = format_command(build_command(split_worker_config))

        self.assertIn("actor_rollout_ref.worker_placement.separate_ref_policy=true", rendered)
        self.assertIn("actor_rollout_ref.worker_placement.actor_rollout.n_gpus_per_node=2", rendered)
        self.assertIn("actor_rollout_ref.worker_placement.actor_rollout.nnodes=1", rendered)
        self.assertIn("actor_rollout_ref.worker_placement.ref_policy.process_on_nodes=[2]", rendered)

    def test_topk_distillation_command_is_config_controlled(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        topk_config = replace(
            config,
            actor=replace(
                config.actor,
                distill_mode="topk_forward_kl_with_tail",
                topk_distill_enabled=True,
                topk_distill_kl_direction="forward",
                topk_distill_k=16,
                topk_distill_temperature=2.0,
                topk_distill_loss_weight=0.5,
            ),
        )
        rendered = format_command(build_command(topk_config))

        self.assertIn("actor_rollout_ref.actor.policy_loss.distill_mode=topk_forward_kl_with_tail", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_kl_direction=forward", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_k=16", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_temperature=2.0", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_loss_weight=0.5", rendered)

    def test_policy_gradient_distill_builder_overrides_topk_flag(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        pg_config = replace(
            config,
            actor=replace(
                config.actor,
                distill_loss_builder="policy_gradient",
                topk_distill_enabled=True,
            ),
        )
        rendered = format_command(build_command(pg_config))
        policy_loss_config = {
            "distill_loss_builder": "policy_gradient",
            "topk_distill_enabled": True,
        }

        self.assertIn("actor_rollout_ref.actor.policy_loss.distill_loss_builder=policy_gradient", rendered)
        self.assertEqual(distill_loss_builder(policy_loss_config), DISTILL_LOSS_BUILDER_POLICY_GRADIENT)
        self.assertFalse(uses_topk_distill_loss(policy_loss_config))
        self.assertEqual(
            distill_loss_builder({"topk_distill_enabled": True}),
            DISTILL_LOSS_BUILDER_TOPK_KL,
        )

    def test_duplicate_teacher_path_is_not_rendered_as_base_ref_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "same_teacher.yaml"
            config_path.write_text(
                """
data:
  train_files: ["train.parquet"]
  val_files: ["val.parquet"]
model:
  student_path: student
  student_base_path: null
  math_teacher_path: teacher
  code_teacher_path: teacher/.
""".lstrip(),
                encoding="utf-8",
            )

            config = load_config(config_path)
            rendered = format_command(build_command(config))

        self.assertIsNone(config.model.secondary_teacher_path)
        self.assertIn("+actor_rollout_ref.ref.model.path=teacher", rendered)
        self.assertNotIn("+actor_rollout_ref.ref.model.base_model_path=", rendered)

    def test_teacher_prefix_command_is_config_controlled(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        prefix_config = replace(
            config,
            actor=replace(
                config.actor,
                teacher_prefix_enabled=True,
                teacher_prefix_loss_region="prefix_and_suffix",
                teacher_prefix_forward_kl_weight=0.75,
            ),
        )
        rendered = format_command(build_command(prefix_config))

        self.assertIn("actor_rollout_ref.actor.policy_loss.teacher_prefix_enabled=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.teacher_prefix_loss_region=prefix_and_suffix", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.teacher_prefix_forward_kl_weight=0.75", rendered)

        rollout_prefix_config = replace(
            config,
            rollout=replace(
                config.rollout,
                teacher_prefix_sampling_enabled=True,
                teacher_prefix_length=128,
                teacher_prefix_dataset_key="prefix",
            ),
        )
        rendered = format_command(build_command(rollout_prefix_config))

        self.assertIn("actor_rollout_ref.rollout.teacher_prefix_sampling_enabled=true", rendered)
        self.assertIn("actor_rollout_ref.rollout.teacher_prefix_length=128", rendered)
        self.assertIn("actor_rollout_ref.rollout.teacher_prefix_dataset_key=prefix", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.teacher_prefix_enabled=true", rendered)

    def test_topk_distillation_helper_uses_teacher_topk_tail_bucket(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        teacher = torch.log(torch.tensor([[[0.50, 0.25]]], dtype=torch.float32))
        student = torch.log(torch.tensor([[[0.40, 0.10]]], dtype=torch.float32))
        loss = topk_distill_loss_matrix(
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            mode=TOPK_FORWARD_KL_WITH_TAIL,
            include_tail=True,
            temperature=1.0,
        )
        expected = (
            0.50 * torch.log(torch.tensor(0.50 / 0.40))
            + 0.25 * torch.log(torch.tensor(0.25 / 0.10))
            + 0.25 * torch.log(torch.tensor(0.25 / 0.50))
        )

        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)
        cross_entropy = topk_teacher_student_cross_entropy_matrix(
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            include_tail=True,
            temperature=1.0,
        )
        expected_cross_entropy = -(
            0.50 * torch.log(torch.tensor(0.40))
            + 0.25 * torch.log(torch.tensor(0.10))
            + 0.25 * torch.log(torch.tensor(0.50))
        )
        self.assertAlmostEqual(float(cross_entropy.item()), float(expected_cross_entropy.item()), places=6)
        self.assertEqual(
            resolved_topk_distill_mode({"topk_distill_enabled": True}),
            TOPK_RENORMALIZED_REVERSE_KL,
        )

    def test_chosen_token_policy_gradient_reward_is_teacher_minus_student(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        student = torch.tensor([[-3.0, -1.5]], dtype=torch.float32)
        teacher = torch.tensor([[-2.0, -2.5]], dtype=torch.float32)
        reward = chosen_token_policy_gradient_reward_matrix(
            student_log_probs=student,
            teacher_log_probs=teacher,
        )

        self.assertTrue(torch.equal(reward, torch.tensor([[1.0, -1.0]], dtype=torch.float32)))


    def test_topk_distillation_helper_uses_renormalized_support_kl(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        teacher = torch.log(torch.tensor([[[0.50, 0.25]]], dtype=torch.float32))
        student = torch.log(torch.tensor([[[0.40, 0.10]]], dtype=torch.float32))

        reverse_loss = topk_distill_loss_matrix(
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            mode=TOPK_RENORMALIZED_REVERSE_KL,
            include_tail=False,
            temperature=1.0,
        )
        expected_reverse = (
            0.80 * torch.log(torch.tensor(0.80 / (2.0 / 3.0)))
            + 0.20 * torch.log(torch.tensor(0.20 / (1.0 / 3.0)))
        )
        self.assertAlmostEqual(float(reverse_loss.item()), float(expected_reverse.item()), places=6)

        forward_loss = topk_distill_loss_matrix(
            student_topk_log_probs=student,
            teacher_topk_log_probs=teacher,
            mode=TOPK_RENORMALIZED_FORWARD_KL,
            include_tail=False,
            temperature=1.0,
        )
        expected_forward = (
            (2.0 / 3.0) * torch.log(torch.tensor((2.0 / 3.0) / 0.80))
            + (1.0 / 3.0) * torch.log(torch.tensor((1.0 / 3.0) / 0.20))
        )
        self.assertAlmostEqual(float(forward_loss.item()), float(expected_forward.item()), places=6)

        self.assertEqual(
            resolved_topk_distill_mode({
                "topk_distill_enabled": True,
                "topk_distill_kl_direction": "forward",
            }),
            TOPK_RENORMALIZED_FORWARD_KL,
        )

    def test_teacher_prefix_masks_split_prefix_and_student_suffix(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
        teacher_prefix_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        prefix_loss_mask, suffix_mask, active = teacher_prefix_masks(
            {"teacher_prefix_mask": teacher_prefix_mask},
            response_mask,
            {"teacher_prefix_enabled": True},
        )

        self.assertTrue(active)
        self.assertTrue(torch.equal(prefix_loss_mask, torch.zeros_like(response_mask)))
        self.assertTrue(torch.equal(suffix_mask, torch.tensor([[0.0, 0.0, 1.0, 0.0]])))

        prefix_and_suffix_mask, _, _ = teacher_prefix_masks(
            {"teacher_prefix_mask": teacher_prefix_mask},
            response_mask,
            {"teacher_prefix_enabled": True, "teacher_prefix_loss_region": "prefix_and_suffix"},
        )
        self.assertTrue(torch.equal(prefix_and_suffix_mask, teacher_prefix_mask))

        prefix_only_mask, suffix_only_mask, _ = teacher_prefix_masks(
            {"teacher_prefix_mask": teacher_prefix_mask},
            response_mask,
            {"teacher_prefix_enabled": True, "teacher_prefix_loss_region": "prefix_only"},
        )
        self.assertTrue(torch.equal(prefix_only_mask, teacher_prefix_mask))
        self.assertTrue(torch.equal(suffix_only_mask, torch.zeros_like(response_mask)))

    def test_teacher_prefix_rollin_merge_builds_masks(self) -> None:
        try:
            import torch
            from tensordict import TensorDict
            from verl import DataProto

            from mopd_verl.teacher_prefix import merge_teacher_prefix_and_student_suffix
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"teacher prefix dependencies are not installed: {exc}")

        original = DataProto(
            batch=TensorDict(
                {
                    "input_ids": torch.tensor([[0, 11, 12], [0, 0, 21]], dtype=torch.long),
                    "attention_mask": torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long),
                    "position_ids": torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.long),
                },
                batch_size=2,
            )
        )
        suffix = DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([[51, 52, 0], [61, 62, 63]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
                },
                batch_size=2,
            )
        )
        output = merge_teacher_prefix_and_student_suffix(
            original_prompts=original,
            teacher_prefix_ids=torch.tensor([[31, 32, 0], [41, 0, 0]], dtype=torch.long),
            teacher_prefix_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
            student_suffix_output=suffix,
            max_response_length=4,
            pad_token_id=0,
        )

        self.assertTrue(torch.equal(output.batch["responses"], torch.tensor([[31, 32, 51, 52], [41, 61, 62, 0]])))
        self.assertTrue(torch.equal(output.batch["teacher_prefix_mask"], torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])))
        self.assertTrue(torch.equal(output.batch["student_suffix_mask"], torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]])))

    def test_teacher_prefix_chosen_token_forward_kl_matrix(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        student = torch.tensor([[-2.0, -1.0]], dtype=torch.float32)
        teacher = torch.tensor([[-0.5, -1.5]], dtype=torch.float32)
        loss = chosen_token_forward_kl_matrix(
            student_log_probs=student,
            teacher_log_probs=teacher,
        )

        self.assertTrue(torch.equal(loss, torch.tensor([[1.5, -0.5]], dtype=torch.float32)))

    def test_dataset_teacher_prefix_uses_prefix_field_and_length(self) -> None:
        try:
            import numpy as np
            import torch
            from verl import DataProto
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"required dependency is not installed: {exc}")

        class ToyTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return [ord(char) for char in text]

        prompts = DataProto.from_dict(
            tensors={
                "input_ids": torch.ones((2, 4), dtype=torch.long),
                "attention_mask": torch.ones((2, 4), dtype=torch.long),
            },
            non_tensors={"prefix": np.array(["abcd", [7, 8, 9, 10]], dtype=object)},
        )

        prefix_ids, prefix_mask = build_dataset_teacher_prefix(
            prompts=prompts,
            tokenizer=ToyTokenizer(),
            prefix_key="prefix",
            prefix_length=3,
            pad_token_id=0,
        )

        self.assertEqual(prefix_ids.tolist(), [[97, 98, 99], [7, 8, 9]])
        self.assertEqual(prefix_mask.tolist(), [[1, 1, 1], [1, 1, 1]])

    def test_selected_logits_from_hidden_states_matches_dense_lm_head(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        torch.manual_seed(0)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        weight = torch.randn(7, 4, requires_grad=True)
        bias = torch.randn(7, requires_grad=True)
        token_ids = torch.tensor(
            [
                [[0, 2, 5], [1, 3, 6], [2, 4, 0]],
                [[6, 5, 4], [3, 2, 1], [0, 1, 2]],
            ],
            dtype=torch.long,
        )

        selected = selected_logits_from_hidden_states(
            hidden,
            vocab_weights=weight,
            token_ids=token_ids,
            bias=bias,
            temperature=2.0,
            chunk_size=2,
        )
        dense_logits = (hidden @ weight.t() + bias) / 2.0
        expected = dense_logits.gather(dim=-1, index=token_ids)
        self.assertTrue(torch.allclose(selected, expected, atol=1e-6))

        selected.square().sum().backward()
        selected_grads = (hidden.grad.clone(), weight.grad.clone(), bias.grad.clone())

        hidden_dense = hidden.detach().clone().requires_grad_(True)
        weight_dense = weight.detach().clone().requires_grad_(True)
        bias_dense = bias.detach().clone().requires_grad_(True)
        dense_selected = ((hidden_dense @ weight_dense.t() + bias_dense) / 2.0).gather(-1, token_ids)
        dense_selected.square().sum().backward()

        self.assertTrue(torch.allclose(selected_grads[0], hidden_dense.grad, atol=1e-6))
        self.assertTrue(torch.allclose(selected_grads[1], weight_dense.grad, atol=1e-6))
        self.assertTrue(torch.allclose(selected_grads[2], bias_dense.grad, atol=1e-6))

    def test_topk_log_probs_from_logits_matches_full_log_softmax(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        logits = torch.tensor(
            [
                [[1.0, -0.5, 3.0, 0.2], [0.1, 2.0, -1.0, 0.0], [4.0, 1.0, -2.0, 3.0]],
                [[-0.5, 0.3, 0.7, 1.4], [2.5, -0.2, 0.0, 1.0], [-1.0, 0.5, 2.0, -0.7]],
            ],
            dtype=torch.float32,
        )
        gather_ids = torch.tensor(
            [
                [[2, 0], [1, 3], [0, 3]],
                [[3, 2], [0, 3], [2, 1]],
            ],
            dtype=torch.long,
        )
        topk_ids, topk_log_probs, gathered_log_probs = topk_log_probs_from_logits(
            logits,
            topk=2,
            gather_topk_ids=gather_ids,
            chunk_size=2,
        )

        expected = torch.log_softmax(logits, dim=-1)
        expected_topk_log_probs, expected_topk_ids = torch.topk(expected, 2, dim=-1)
        expected_gathered_log_probs = expected.gather(dim=-1, index=gather_ids)
        self.assertTrue(torch.equal(topk_ids, expected_topk_ids))
        self.assertTrue(torch.allclose(topk_log_probs, expected_topk_log_probs, atol=1e-6))
        self.assertTrue(torch.allclose(gathered_log_probs, expected_gathered_log_probs, atol=1e-6))

        _, _, gathered_logits = topk_log_probs_from_logits(
            logits,
            gather_topk_ids=gather_ids,
            normalize_gathered=False,
            chunk_size=2,
        )
        expected_gathered_logits = logits.gather(dim=-1, index=gather_ids)
        self.assertTrue(torch.allclose(gathered_logits, expected_gathered_logits, atol=1e-6))

    def test_toolrl_command_uses_custom_reward(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "grpo" / "configs" / "toolrl.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("data/ToolRL/rlla_4k/train.parquet", rendered)
        self.assertIn("+data.domain_sampling_weights={tool: 1}", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.multi_teacher_distill=false", rendered)
        self.assertIn("actor_rollout_ref.actor.use_kl_loss=False", rendered)
        self.assertIn("custom_reward_function.path=grpo/rewards/toolrl.py", rendered)
        self.assertIn("custom_reward_function.name=compute_score", rendered)
        self.assertIn("actor_rollout_ref.rollout.n=4", rendered)

    def test_general_reasoner_command_uses_external_verifier_worker(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "grpo" / "configs" / "general_reasoner.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("data/GeneralReasoner/WebInstructVerified/train.parquet", rendered)
        self.assertIn("+data.domain_sampling_weights={reasoning: 1}", rendered)
        self.assertIn("custom_reward_function.path=grpo/rewards/general_reasoner.py", rendered)
        self.assertIn("reward_model.enable=True", rendered)
        self.assertIn("reward_model.strategy=verifier", rendered)
        self.assertIn("+reward_model.worker.path=grpo/workers/general_verifier.py", rendered)
        self.assertIn("+reward_model.worker.name=RewardModelWorker", rendered)
        self.assertIn("reward_model.model.path=TIGER-Lab/general-verifier", rendered)

    def test_dp_actor_routes_non_code_teacher_labels_through_primary_ref(self) -> None:
        source_path = Path(__file__).resolve().parents[1] / "third_party" / "verl" / "verl" / "workers" / "actor" / "dp_actor.py"
        source = source_path.read_text(encoding="utf-8")

        self.assertIn(
            'if teacher_type == "code" and "code_teacher_log_prob" in model_inputs:\n'
            '                                            teacher_log_prob = model_inputs["code_teacher_log_prob"][i]\n'
            "                                        else:\n"
            '                                            teacher_log_prob = model_inputs["math_teacher_log_prob"][i]',
            source,
        )
        self.assertIn(
            "if lambda_vals == 1.0:\n"
            "                                            reverse_kl[i] = old_log_prob[i] - teacher_log_prob\n"
            "                                        else:\n"
            "                                            reverse_kl[i] = old_log_prob[i] - model_inputs[\"base_log_prob\"][i]",
            source,
        )

    def test_audit_logs_domain_gap_and_entropy_distribution_vectors(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.tensor(
                        [[-2.0, -1.0, -3.0], [-4.0, -2.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "math_teacher_log_prob": torch.tensor(
                        [[-1.0, -2.0, -3.0], [-4.0, -1.0, -2.0]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_log_prob": torch.tensor(
                        [[-2.0, -1.0, -2.0], [-3.0, -3.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "base_log_prob": torch.tensor(
                        [[-2.0, -1.0, -3.0], [-4.0, -2.0, -1.0]],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.tensor(
                        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                        dtype=torch.float32,
                    ),
                    "responses": torch.tensor(
                        [[5, 2, 0], [5, 5, 7]],
                        dtype=torch.long,
                    ),
                    "student_entropy": torch.tensor(
                        [[0.5, 0.6, 0.0], [0.7, 0.8, 0.9]],
                        dtype=torch.float32,
                    ),
                    "math_teacher_entropy": torch.tensor(
                        [[0.2, 0.3, 0.0], [0.4, 0.5, 0.6]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_entropy": torch.tensor(
                        [[0.9, 1.0, 0.0], [1.1, 1.2, 1.3]],
                        dtype=torch.float32,
                    ),
                    "teacher_student_cross_entropy": torch.tensor(
                        [[1.0, 1.5, 0.0], [2.0, 2.5, 3.0]],
                        dtype=torch.float32,
                    ),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "code"],
                    "sample_id": ["m0", "c0"],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math", "code"],
                        "log_sample_level": False,
                        "token_gap_vocab_vector_enabled": True,
                        "token_gap_vocab_size": 8,
                        "entropy_vocab_vector_enabled": True,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            metrics = logger.log_training_step(SyntheticBatch(), step=3, lr=0.01)
            rows = [
                json.loads(line)
                for line in (Path(tmpdir) / "token_gap_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            entropy_vector_path = Path(tmpdir) / "entropy_distribution_vectors.jsonl"
            entropy_rows = [
                json.loads(line)
                for line in entropy_vector_path.read_text(encoding="utf-8").splitlines()
            ]
            vocab_rows = [
                json.loads(line)
                for line in (Path(tmpdir) / "token_gap_vocab_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            entropy_vocab_rows = [
                json.loads(line)
                for line in (Path(tmpdir) / "entropy_vocab_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertAlmostEqual(metrics["math/token_gap/gap_abs_sum"], 2.0)
        self.assertAlmostEqual(metrics["math/token_gap/gap_signed_mean"], 0.0)
        self.assertAlmostEqual(metrics["math/entropy/sum_teacher_entropy"], 0.5)
        self.assertAlmostEqual(metrics["math/entropy/sum_student_entropy"], 1.1)
        self.assertAlmostEqual(metrics["math/entropy/teacher_entropy_mean"], 0.25)
        self.assertAlmostEqual(metrics["math/entropy/student_entropy_p50"], 0.55, places=5)
        self.assertAlmostEqual(metrics["code/entropy/sum_teacher_entropy"], 3.6, places=5)
        self.assertAlmostEqual(metrics["code/entropy/teacher_entropy_p95"], 1.29, places=5)
        self.assertAlmostEqual(metrics["code/entropy/sum_teacher_student_cross_entropy"], 7.5)
        self.assertAlmostEqual(metrics["code/entropy/teacher_student_cross_entropy_mean"], 2.5)
        self.assertAlmostEqual(
            metrics["global/token_gap_vocab_cosine/math_vs_code/gap_abs_sum_cosine"],
            2**-0.5,
            places=5,
        )
        self.assertAlmostEqual(
            metrics[
                "global/entropy_vocab_cosine/math_vs_code/"
                "teacher_student_cross_entropy_sum_cosine"
            ],
            4.5 / 9.75,
            places=5,
        )
        self.assertAlmostEqual(
            metrics["global/entropy_vocab_cosine/math_vs_code/student_entropy_sum_cosine"],
            0.75 / ((0.61 * 3.06) ** 0.5),
            places=5,
        )
        self.assertNotIn("global/token_gap_vocab_cosine/math_vs_code/gap_signed_sum_cosine", metrics)
        vectors = {row["domain"]: row["gap_vector_domain"] for row in rows}
        signed_vectors = {row["domain"]: row["gap_signed_vector_domain"] for row in rows}
        abs_vectors = {row["domain"]: row["gap_abs_vector_domain"] for row in rows}
        self.assertEqual(vectors["math"], [1.0, -1.0])
        self.assertEqual(vectors["code"], [1.0, -1.0, 0.0])
        self.assertEqual(signed_vectors["math"], [1.0, -1.0])
        self.assertEqual(abs_vectors["math"], [1.0, 1.0])
        vocab_vectors = {row["domain"]: row for row in vocab_rows}
        self.assertEqual(vocab_vectors["math"]["vocab_size"], 8)
        self.assertEqual(vocab_vectors["math"]["vocab_size_source"], "config")
        self.assertEqual(vocab_vectors["math"]["nonzero_token_ids"], [2, 5])
        self.assertEqual(vocab_vectors["math"]["token_count_vector_vocab"][2], 1.0)
        self.assertEqual(vocab_vectors["math"]["token_count_vector_vocab"][5], 1.0)
        self.assertEqual(vocab_vectors["math"]["gap_signed_sum_vector_vocab"][2], -1.0)
        self.assertEqual(vocab_vectors["math"]["gap_signed_sum_vector_vocab"][5], 1.0)
        self.assertEqual(vocab_vectors["math"]["gap_abs_sum_vector_vocab"][2], 1.0)
        self.assertEqual(vocab_vectors["math"]["gap_abs_sum_vector_vocab"][5], 1.0)
        self.assertEqual(vocab_vectors["code"]["nonzero_token_ids"], [5, 7])
        self.assertEqual(vocab_vectors["code"]["token_count_vector_vocab"][5], 2.0)
        self.assertEqual(vocab_vectors["code"]["gap_signed_sum_vector_vocab"][5], 0.0)
        self.assertEqual(vocab_vectors["code"]["gap_abs_sum_vector_vocab"][5], 2.0)
        entropy_vocab_vectors = {row["domain"]: row for row in entropy_vocab_rows}
        self.assertEqual(entropy_vocab_vectors["math"]["vocab_size"], 8)
        self.assertEqual(entropy_vocab_vectors["math"]["vocab_size_source"], "config")
        self.assertEqual(entropy_vocab_vectors["math"]["nonzero_token_ids"], [2, 5])
        self.assertEqual(
            entropy_vocab_vectors["math"]["teacher_student_cross_entropy_sum_vector_vocab"][2],
            1.5,
        )
        self.assertEqual(
            entropy_vocab_vectors["math"]["teacher_student_cross_entropy_sum_vector_vocab"][5],
            1.0,
        )
        self.assertAlmostEqual(entropy_vocab_vectors["math"]["student_entropy_sum_vector_vocab"][2], 0.6)
        self.assertAlmostEqual(entropy_vocab_vectors["math"]["student_entropy_sum_vector_vocab"][5], 0.5)
        self.assertEqual(
            entropy_vocab_vectors["code"]["teacher_student_cross_entropy_sum_vector_vocab"][5],
            4.5,
        )
        self.assertEqual(
            entropy_vocab_vectors["code"]["teacher_student_cross_entropy_mean_vector_vocab"][5],
            2.25,
        )
        self.assertAlmostEqual(entropy_vocab_vectors["code"]["student_entropy_sum_vector_vocab"][5], 1.5)
        self.assertAlmostEqual(entropy_vocab_vectors["code"]["student_entropy_mean_vector_vocab"][5], 0.75)
        entropy_vectors = {row["domain"]: row for row in entropy_rows}
        self.assertEqual(
            [round(value, 1) for value in entropy_vectors["math"]["teacher_entropy_vector_domain"]],
            [0.2, 0.3],
        )
        self.assertEqual(
            [round(value, 1) for value in entropy_vectors["math"]["student_entropy_vector_domain"]],
            [0.5, 0.6],
        )
        self.assertEqual(
            [round(value, 1) for value in entropy_vectors["code"]["teacher_student_cross_entropy_vector_domain"]],
            [2.0, 2.5, 3.0],
        )

    def test_vocab_vectors_prefer_model_config_vocab_size(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class ShortTokenizer:
            vocab_size = 8

            def __len__(self) -> int:
                return 8

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.tensor([[-3.0, -4.0]], dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor([[-2.5, -4.5]], dtype=torch.float32),
                    "response_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    "responses": torch.tensor([[2, 10]], dtype=torch.long),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math"],
                    "sample_id": ["m0"],
                }

        with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as tmpdir:
            (Path(model_dir) / "config.json").write_text(
                json.dumps({"vocab_size": 12}),
                encoding="utf-8",
            )
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math"],
                        "log_sample_level": False,
                        "token_gap_vocab_vector_enabled": True,
                    },
                    "actor_rollout_ref": {
                        "model": {"path": model_dir},
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                },
                tokenizer=ShortTokenizer(),
            )
            logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            vocab_row = json.loads(
                (Path(tmpdir) / "token_gap_vocab_vectors.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(vocab_row["vocab_size"], 12)
        self.assertEqual(vocab_row["vocab_size_source"], "model_config")
        self.assertEqual(len(vocab_row["token_count_vector_vocab"]), 12)
        self.assertEqual(vocab_row["nonzero_token_ids"], [2, 10])
        self.assertEqual(vocab_row["dropped_token_count"], 0)

    def test_audit_can_disable_gap_entropy_and_token_conflict_outputs(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.tensor([[-2.0, -1.0]], dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
                    "base_log_prob": torch.tensor([[-2.0, -1.0]], dtype=torch.float32),
                    "response_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    "student_entropy": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
                    "math_teacher_entropy": torch.tensor([[0.2, 0.3]], dtype=torch.float32),
                    "teacher_student_cross_entropy": torch.tensor([[1.0, 1.5]], dtype=torch.float32),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math"],
                    "sample_id": ["m0"],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math"],
                        "log_sample_level": False,
                        "token_gap_enabled": False,
                        "entropy_enabled": False,
                        "token_conflict_enabled": False,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            metrics = logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            output_dir = Path(tmpdir)
            domain_rows = [
                json.loads(line)
                for line in (output_dir / "domain_step_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            token_gap_file_exists = (output_dir / "token_gap_vectors.jsonl").exists()
            entropy_file_exists = (output_dir / "entropy_distribution_vectors.jsonl").exists()
            entropy_vocab_file_exists = (output_dir / "entropy_vocab_vectors.jsonl").exists()
            token_conflict_file_exists = (output_dir / "token_conflict_attribution.jsonl").exists()

        self.assertNotIn("math/token_gap/gap_abs_sum", metrics)
        self.assertNotIn("math/entropy/sum_teacher_entropy", metrics)
        self.assertNotIn("math/token_conflict/proxy_mass", metrics)
        self.assertFalse(token_gap_file_exists)
        self.assertFalse(entropy_file_exists)
        self.assertFalse(entropy_vocab_file_exists)
        self.assertFalse(token_conflict_file_exists)
        self.assertNotIn("gap_abs_sum", domain_rows[0])
        self.assertNotIn("sum_teacher_entropy", domain_rows[0])
        self.assertNotIn("proxy_mass", domain_rows[0])

    def test_audit_step_frequencies_gate_distribution_outputs(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self, sample_id: str) -> None:
                self.batch = {
                    "old_log_probs": torch.tensor([[-2.0, -1.0]], dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
                    "base_log_prob": torch.tensor([[-2.0, -1.0]], dtype=torch.float32),
                    "response_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    "responses": torch.tensor([[101, 102]], dtype=torch.long),
                    "advantages": torch.tensor([[0.5, 0.25]], dtype=torch.float32),
                    "token_level_scores": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    "student_entropy": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
                    "math_teacher_entropy": torch.tensor([[0.2, 0.3]], dtype=torch.float32),
                    "teacher_student_cross_entropy": torch.tensor([[1.0, 1.5]], dtype=torch.float32),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math"],
                    "sample_id": [sample_id],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math"],
                        "log_sample_level": True,
                        "log_sample_level_freq_steps": 2,
                        "token_gap_enabled": True,
                        "token_gap_freq_steps": 2,
                        "entropy_enabled": True,
                        "entropy_freq_steps": 2,
                        "entropy_vocab_vector_enabled": True,
                        "entropy_vocab_vector_freq_steps": 2,
                        "token_conflict_enabled": True,
                        "token_conflict_freq_steps": 2,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )

            step1_metrics = logger.log_training_step(SyntheticBatch("m0"), step=1, lr=0.01)
            output_dir = Path(tmpdir)
            self.assertFalse((output_dir / "loss_variance_sample.jsonl").exists())
            self.assertFalse((output_dir / "token_gap_vectors.jsonl").exists())
            self.assertFalse((output_dir / "entropy_distribution_vectors.jsonl").exists())
            self.assertFalse((output_dir / "entropy_vocab_vectors.jsonl").exists())
            self.assertFalse((output_dir / "token_conflict_attribution.jsonl").exists())
            self.assertNotIn("math/token_gap/gap_abs_sum", step1_metrics)
            self.assertNotIn("math/entropy/sum_teacher_entropy", step1_metrics)
            self.assertNotIn("math/token_conflict/proxy_mass", step1_metrics)

            step2_metrics = logger.log_training_step(SyntheticBatch("m1"), step=2, lr=0.01)
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "loss_variance_sample.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            domain_rows = [
                json.loads(line)
                for line in (output_dir / "domain_step_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            token_gap_rows = [
                json.loads(line)
                for line in (output_dir / "token_gap_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            entropy_rows = [
                json.loads(line)
                for line in (output_dir / "entropy_distribution_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            entropy_vocab_rows = [
                json.loads(line)
                for line in (output_dir / "entropy_vocab_vectors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            token_conflict_rows = [
                json.loads(line)
                for line in (output_dir / "token_conflict_attribution.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertIn("math/token_gap/gap_abs_sum", step2_metrics)
        self.assertIn("math/entropy/sum_teacher_entropy", step2_metrics)
        self.assertIn("math/token_conflict/proxy_mass", step2_metrics)
        self.assertEqual([row["step"] for row in sample_rows], [2])
        self.assertEqual([row["step"] for row in token_gap_rows], [2])
        self.assertEqual([row["step"] for row in entropy_rows], [2])
        self.assertEqual([row["step"] for row in entropy_vocab_rows], [2])
        self.assertEqual({row["step"] for row in token_conflict_rows}, {2})
        self.assertEqual([row["step"] for row in domain_rows], [1, 2])
        self.assertNotIn("gap_abs_sum", domain_rows[0])
        self.assertIn("gap_abs_sum", domain_rows[1])
        self.assertNotIn("sum_teacher_entropy", domain_rows[0])
        self.assertIn("sum_teacher_entropy", domain_rows[1])
        self.assertNotIn("proxy_mass", domain_rows[0])
        self.assertIn("proxy_mass", domain_rows[1])

    def test_null_audit_caps_log_full_sample_and_token_details(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.full((3, 2), -2.0, dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor(
                        [[-1.0, -1.5], [-1.25, -1.75], [-1.1, -1.6]],
                        dtype=torch.float32,
                    ),
                    "base_log_prob": torch.full((3, 2), -2.0, dtype=torch.float32),
                    "response_mask": torch.ones((3, 2), dtype=torch.float32),
                    "responses": torch.tensor([[101, 102], [103, 104], [105, 106]], dtype=torch.long),
                    "advantages": torch.ones((3, 2), dtype=torch.float32),
                    "token_level_scores": torch.ones((3, 2), dtype=torch.float32),
                }
                self.non_tensor_batch = {
                    "opd_teacher": ["math", "math", "math"],
                    "sample_id": ["m0", "m1", "m2"],
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math"],
                        "max_samples_per_domain": None,
                        "token_conflict_top_k": None,
                        "token_gap_enabled": False,
                        "entropy_enabled": False,
                    },
                    "actor_rollout_ref": {
                        "actor": {"policy_loss": {"lambda_vals": 1.0}},
                    },
                }
            )
            logger.log_training_step(SyntheticBatch(), step=1, lr=0.01)
            output_dir = Path(tmpdir)
            sample_rows = [
                json.loads(line)
                for line in (output_dir / "loss_variance_sample.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            token_conflict_rows = [
                json.loads(line)
                for line in (output_dir / "token_conflict_attribution.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["sample_id"] for row in sample_rows], ["m0", "m1", "m2"])
        self.assertEqual(len(token_conflict_rows), 6)
        self.assertEqual({row["token_id"] for row in token_conflict_rows}, {101, 102, 103, 104, 105, 106})

    def test_validation_metrics_respect_step_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": tmpdir,
                        "domains": ["math"],
                        "log_validation_metrics": True,
                        "log_validation_metrics_freq_steps": 2,
                    },
                }
            )

            step1_metrics = logger.log_validation_metrics({"val/math/score": 0.1}, step=1)
            output_dir = Path(tmpdir)
            self.assertEqual(step1_metrics, {})
            self.assertFalse((output_dir / "validation_probe.jsonl").exists())

            step2_metrics = logger.log_validation_metrics({"val/math/score": 0.2}, step=2)
            step3_metrics = logger.log_validation_metrics({"val/math/score": 0.3}, step=3)
            step4_metrics = logger.log_validation_metrics({"val/math/score": 0.5}, step=4)
            rows = [
                json.loads(line)
                for line in (output_dir / "validation_probe.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(step2_metrics, {})
        self.assertEqual(step3_metrics, {})
        self.assertIn("math/validation_gain/score", step4_metrics)
        self.assertAlmostEqual(step4_metrics["math/validation_gain/score"], 0.3)
        self.assertEqual([row["step"] for row in rows], [2, 4])
        self.assertIsNone(rows[0]["gain"])
        self.assertAlmostEqual(rows[1]["gain"], 0.3)

    def test_formal_command_enables_full_parameter_gradient_audit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("+mopd_audit.full_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.log_sample_level_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.log_validation_metrics_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.full_grad_training_parity_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_train_max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.full_gradient_micro_batch_size_per_gpu=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_storage_dtype=bfloat16", rendered)
        self.assertIn("+mopd_audit.sequence_masked_target_enabled=false", rendered)
        self.assertIn("+mopd_audit.sequence_masked_target_use_as_primary=false", rendered)
        self.assertIn("+mopd_audit.sequence_replay_skip_non_target_domains=false", rendered)
        self.assertIn("+mopd_audit.sequence_masked_target_closure_rel_l2_threshold=0.02", rendered)
        self.assertIn("+mopd_audit.training_gradient_from_domain_sum_enabled=false", rendered)
        self.assertIn("+mopd_audit.sample_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.sample_gradient_norm_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.sample_gradient_backward_recompute_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_backward_sync_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_log_sample_level_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_gap_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gap_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_gap_vocab_vector_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gap_vocab_vector_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_gap_vocab_size=null", rendered)
        self.assertIn("+mopd_audit.entropy_enabled=true", rendered)
        self.assertIn("+mopd_audit.entropy_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.entropy_vocab_vector_enabled=true", rendered)
        self.assertIn("+mopd_audit.entropy_vocab_vector_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_conflict_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_conflict_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_conflict_top_k=null", rendered)
        self.assertIn("+mopd_audit.token_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_gap_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_loss_abs_selection_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_k=100", rendered)
        self.assertIn("+mopd_audit.token_gradient_top_p=0.1", rendered)
        self.assertIn("+mopd_audit.token_gradient_strict_grad_restore=false", rendered)
        self.assertIn("+mopd_audit.token_gradient_backward_recompute_enabled=true", rendered)
        self.assertIn("+mopd_audit.token_gradient_backward_sync_enabled=true", rendered)
        self.assertNotIn("selected_topk_head_train_enabled", rendered)
        self.assertNotIn("token_gradient_top_k_per_sample", rendered)
        self.assertNotIn("token_gradient_max_samples_per_domain", rendered)
        self.assertNotIn("token_gradient_min_teacher_diff", rendered)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=1", rendered)
        self.assertNotIn("sample_gradient_cos_max_samples_per_domain", rendered)
        self.assertNotIn("sample_gradient_cos_selection", rendered)
        self.assertIn("+mopd_audit.tensorboard_prune_mode=core", rendered)
        self.assertIn("+data.domain_sampling_weights={math: 0.5, code: 0.5}", rendered)
        self.assertIn("+data.domain_train_files=", rendered)
        self.assertIn("DeepMath-103K/train_filtered_level6.parquet", rendered)
        self.assertIn("Eurus/code_train.parquet", rendered)
        self.assertIn("eval/domains/math/data/AIME24/test.parquet", rendered)
        self.assertIn("eval/domains/math/data/AIME25/test.parquet", rendered)
        self.assertIn("eval/domains/math/data/HMMT25Feb/test.parquet", rendered)
        self.assertIn("eval/domains/math/data/HMMT25Nov/test.parquet", rendered)
        self.assertIn("eval/domains/code/data/MBPPPlus/test.parquet", rendered)
        self.assertNotIn("eval/domains/code/data/LiveCodeBench/test.parquet", rendered)
        self.assertIn("trainer.default_local_dir=checkpoints/formal_audit_all_2gpu", rendered)
        self.assertNotIn("/root/autodl-tmp/opd_mopd/G-OPD/G-OPD-Training-Data", rendered)
        self.assertNotIn("/root/autodl-tmp/opd_mopd/OPD-code", rendered)
        self.assertNotIn("+paper_eval.enabled=true", rendered)
        self.assertNotIn("run_paper_eval_suite.sh", rendered)

    def test_audit_all_profile_uses_replicated_two_gpu_gradient_audit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 256)
        self.assertEqual(config.data.max_response_length, 16384)
        self.assertEqual(config.actor.ppo_mini_batch_size, 256)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor.gradient_checkpointing)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.rollout.gpu_memory_utilization, 0.7)
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertTrue(config.audit.enabled)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_norm_enabled)
        self.assertTrue(config.audit.sample_gradient_cos_enabled)
        self.assertTrue(config.audit.token_gap_vocab_vector_enabled)
        self.assertTrue(config.audit.entropy_vocab_vector_enabled)
        self.assertTrue(config.audit.token_gradient_enabled)
        self.assertEqual(config.trainer.total_training_steps, 200)
        self.assertEqual(config.trainer.save_freq, 5)
        self.assertIn("trainer.n_gpus_per_node=2", rendered)
        self.assertIn("trainer.save_freq=5", rendered)
        self.assertIn("trainer.total_training_steps=200", rendered)
        self.assertIn("data.train_batch_size=256", rendered)
        self.assertIn("data.max_response_length=16384", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=256", rendered)
        self.assertIn("actor_rollout_ref.model.enable_gradient_checkpointing=True", rendered)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=2", rendered)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.7", rendered)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=1", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_audit_all_2gpu", rendered)
        self.assertIn("+mopd_audit.full_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_norm_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_enabled=true", rendered)
        self.assertIn("trainer.default_local_dir=checkpoints/formal_audit_all_2gpu", rendered)

    def test_audit_all_2gpu_overrides_enable_dynamic_actor_batching(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_audit_all_2gpu.yaml"
        config = load_config(config_path)
        extra_args = [
            "mopd_audit.token_gradient_enabled=true",
            "mopd_audit.sample_gradient_cos_enabled=true",
            "mopd_audit.token_gradient_freq_steps=1",
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
            "actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true",
            "actor_rollout_ref.actor.policy_loss.topk_distill_k=32",
            "actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket=false",
        ]
        rendered = format_command(build_command(config, extra_args=extra_args))

        self.assertTrue(config.actor.topk_distill_enabled)
        self.assertEqual(config.actor.topk_distill_k, 32)
        self.assertFalse(config.actor.use_dynamic_bsz)
        self.assertTrue(config.actor.optimizer_offload)
        self.assertEqual(config.rollout.gpu_memory_utilization, 0.7)
        self.assertIn("mopd_audit.token_gradient_enabled=true", rendered)
        self.assertIn("mopd_audit.sample_gradient_cos_enabled=true", rendered)
        self.assertIn("mopd_audit.token_gradient_freq_steps=1", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_k=32", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket=false", rendered)
        self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=False", rendered)
        self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=True", rendered)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.optimizer_offload=True", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768", rendered)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.6", rendered)

    def test_domain_sampling_weights_target_domain_mass(self) -> None:
        rows = [
            {"extra_info": {"opd_teacher": "math"}},
            {"extra_info": {"opd_teacher": "math"}},
            {"extra_info": {"opd_teacher": "math"}},
            {"extra_info": {"opd_teacher": "code"}},
        ]
        weights = domain_sample_weights(rows, {"math": 0.25, "code": 0.75})

        self.assertAlmostEqual(sum(weights[:3]), 0.25, places=6)
        self.assertAlmostEqual(weights[3], 0.75, places=6)
        self.assertAlmostEqual(sum(weights), 1.0, places=6)

    def test_domain_sampling_weights_are_normalized(self) -> None:
        self.assertEqual(normalize_domain_sampling_weights({"math": 2, "code": 1}), {"math": 2 / 3, "code": 1 / 3})

    def test_domain_batch_allocation_uses_largest_remainder(self) -> None:
        self.assertEqual(
            allocate_domain_batch_counts(1024, {"math": 0.7, "code": 0.3}),
            {"math": 717, "code": 307},
        )
        self.assertEqual(
            allocate_domain_batch_counts(10, {"a": 0.5, "b": 0.3, "c": 0.2}),
            {"a": 5, "b": 3, "c": 2},
        )

    def test_domain_batch_sampler_emits_exact_domain_counts(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        labels = ["math"] * 6 + ["code"] * 2
        sampler = DomainBatchSampler(
            labels,
            {"math": 0.5, "code": 0.5},
            batch_size=4,
            replacement=True,
            seed=123,
        )
        first_batch = next(iter(sampler))
        domains = [labels[idx] for idx in first_batch]
        self.assertEqual(domains.count("math"), 2)
        self.assertEqual(domains.count("code"), 2)

    def test_toolrl_conversion_adds_teacher_metadata_and_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "toolrl.parquet"
            target = Path(tmpdir) / "toolrl_verl.parquet"
            pd.DataFrame(
                [
                    {
                        "data_source": "rlla",
                        "prompt": [{"content": "Pick the right tool."}],
                        "ability": "tool",
                        "reward_model": {"style": "rule", "ground_truth": "<think> ok </think>"},
                        "extra_info": {"index": 7},
                    }
                ]
            ).to_parquet(source, index=False)

            count = toolrl_to_verl_parquet(source, target, split="train")
            row = pd.read_parquet(target).to_dict(orient="records")[0]

        self.assertEqual(count, 1)
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["extra_info"]["opd_teacher"], "tool")
        self.assertEqual(row["extra_info"]["domain"], "tool")
        self.assertEqual(row["extra_info"]["split"], "train")

    def test_toolrl_reward_scores_exact_tool_call(self) -> None:
        response = (
            '<think> I should call the tool. </think>\n'
            '<tool_call>\n{"name": "GetNews", "parameters": {"page": "1"}}\n</tool_call>'
        )
        reward = compute_toolrl_score(
            data_source="toolrl_rlla",
            solution_str=response,
            ground_truth=response,
        )

        self.assertEqual(reward["toolrl_format"], 1.0)
        self.assertEqual(reward["toolrl_correctness"], 3.0)
        self.assertEqual(reward["score"], 4.0)

    def test_tensorboard_core_filter_keeps_high_signal_metrics(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "tensorboard_prune_mode": "core",
                }
            }
        )
        metrics = {
            "math/loss/token_opd_loss_mean": 0.1,
            "math/loss/token_opd_loss_std": 0.2,
            "math/loss/token_opd_loss_variance": 0.04,
            "math/loss/sample_opd_loss_mean": 0.3,
            "math/loss/sample_opd_loss_std": 0.4,
            "math/loss/sample_opd_loss_variance": 0.16,
            "global/loss/token_opd_loss_mean": 0.05,
            "global/loss/sample_opd_loss_variance": 0.1,
            "math/loss/opd_loss_p95": 0.9,
            "math/loss/kl_spike_rate": 0.2,
            "global/full_grad/total_grad_norm": 2.0,
            "global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k": -0.3,
            "global/full_grad_conflict/math_vs_code/full_grad_dot_train_i_k": -10.0,
            "global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total": 0.7,
            "global/full_grad_contribution/math_to_total/signed_projection_share": 0.6,
            "global/full_grad_cost/backward_seconds": 18.0,
            "global/full_grad_cost/domain_summary_seconds": 0.5,
            "global/full_grad_cost/finish_mini_batch_seconds": 2.0,
            "global/full_grad_closure/domain_sum_vs_training/cosine": 1.0,
            "global/full_grad_closure/domain_sum_vs_training/rel_l2": 0.0,
            "global/audit/full_gradient_domain_sequential_available": 1.0,
            "global/audit/full_gradient_domain_sequential_unsupported": 0.0,
            "global/audit/full_gradient_replicated_all_reduce": 1.0,
            "global/audit/full_gradient_replica_count": 2.0,
            "global/audit/pre_update_audit_used": 1.0,
            "global/audit/full_gradient_execution_timing_pre_update": 1.0,
            "global/audit/sample_gradient_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_norm_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_cos_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_distributed_world_size": 2.0,
            "math/grad_conflict/code/grad_cosine_train_i_k": -0.4,
            "global/grad_conflict/math_vs_code/grad_cosine_train_i_k": -0.4,
            "math/grad/grad_norm": 1.0,
            "math/teacher/teacher_logprob_mean": -0.5,
            "math/teacher/teacher_student_gap_mean": 0.1,
            "math/token_gap/gap_abs_sum": 3.0,
            "math/token_gap/gap_signed_p95": 0.8,
            "global/token_gap_vocab_cosine/math_vs_code/gap_abs_sum_cosine": 0.7,
            "global/entropy_vocab_cosine/math_vs_code/teacher_student_cross_entropy_sum_cosine": 0.6,
            "global/entropy_vocab_cosine/math_vs_code/student_entropy_sum_cosine": 0.5,
            "math/entropy/sum_teacher_entropy": 12.0,
            "math/entropy/teacher_entropy_mean": 0.7,
            "math/entropy/teacher_entropy_p95": 1.1,
            "math/entropy/student_entropy_mean": 0.8,
            "math/entropy/teacher_student_cross_entropy_mean": 0.9,
            "math/advantage/positive_frac": 0.5,
            "math/length/response_mean": 1024.0,
            "math/length/response_p95": 2048.0,
            "math/length/response_clip_ratio": 0.25,
            "math/sample_grad/norm_mean": 1.2,
            "math/sample_grad/norm_p95": 2.4,
            "math/sample_grad_cos/domain_cos_mean": 0.3,
            "math/sample_grad_cos/domain_cos_negative_frac": 0.25,
            "math/sample_grad_contribution/projection_share_mean": 0.05,
            "math/sample_grad_contribution/projection_share_normalized_sum": 1.0,
            "math/sample_grad_contribution/top1_abs_share": 0.2,
            "math/sample_grad_contribution/top1_abs_share_normalized": 0.2,
            "math/sample_grad_contribution/projection_share_trusted": 1.0,
            "math/sample_grad_closure/projection_share_sum_error": 0.01,
            "math/sample_grad_closure/projection_share_normalized_sum_error": 0.0,
            "math/sample_grad_closure/vector_cosine": 1.0,
            "math/sample_grad_closure/vector_rel_l2": 0.0,
            "math/sample_grad_cost/backward_recompute_count": 4.0,
            "math/sample_grad_cost/restore_post_target_rel_l2_max": 0.0,
            "global/full_grad_training_parity/audit_total_vs_training_total/cosine": 1.0,
            "global/full_grad_training_parity/audit_total_vs_training_total/rel_l2": 0.0,
            "global/full_grad_training_parity/sequence_total_vs_training_total/cosine": 1.0,
            "global/full_grad_training_parity/sequence_total_vs_training_total/rel_l2": 0.0,
            "math/token_grad_cost/seconds_sum": 12.0,
            "math/token_grad_cost/backward_fallback_count": 4.0,
            "global/token_grad_cost/seconds": 24.0,
            "global/token_grad_cost/seconds_per_selected_token": 3.0,
            "global/token_grad_cost/backward_fallback_seconds_sum": 20.0,
            "global/token_grad_cost/global_candidate_gap_mass": 21.0,
            "global/token_grad_cost/global_candidate_loss_abs_mass": 42.0,
            "global/token_grad_cost/valid_frac": 1.0,
            "global/token_grad_cost/restore_post_target_rel_l2_max": 1e-6,
            "global/token_grad_cost/restore_original_rel_l2_max": 0.0,
            "math/token_grad_cost/restore_post_target_max_abs_max": 1e-7,
            "math/token_grad_cost/restore_original_max_abs_max": 0.0,
            "math/token_grad/global_candidate_gap_mass": 21.0,
            "math/token_grad/global_candidate_loss_abs_mass": 42.0,
            "math/token_grad/top100_gap_gap_mass_frac": 0.31,
            "math/token_grad/top100_gap_abs_gap_abs_mass_frac": 0.54,
            "math/token_grad/top100_loss_abs_loss_abs_mass_frac": 0.62,
            "math/token_grad/top100_loss_abs_score_mass_frac": 0.62,
            "math/token_grad_closure/topp100_loss_abs_mass_selected_all_tokens": 1.0,
            "math/token_grad_closure/topp100_loss_abs_mass_projection_share_error": 0.01,
            "math/token_grad_closure/topp100_loss_abs_mass_cosine_error": 0.02,
            "math/token_grad_closure/topp100_loss_abs_mass_norm_ratio_error": 0.03,
            "math/reward/training_reward_mean": 0.5,
            "math/reward/training_accuracy": 0.5,
            "math/coverage/duplicate_rate": 0.0,
            "math/coverage/new_sample_rate": 1.0,
            "global/cost/gpu_count": 1.0,
            "global/cost/step_seconds": 2.0,
            "global/validation/val-core_AIME2024_reward_mean_1": 0.3,
            "global/validation_gain/val-core_AIME2024_reward_mean_1": 0.05,
            "global/validation_gain_stats/val-core_AIME2024_reward_mean_1/ci_high": 0.1,
            "rollout_corr/kl": 0.02,
            "rollout_corr/rollout_is_mean": 1.0,
            "rollout_corr/training_ppl": 3.0,
            "actor/lr": 1e-5,
            "actor/topk_distill_loss": 0.04,
            "actor/kl_loss": 0.01,
            "training/rollout_log_probs_diff": 0.01,
        }

        filtered = logger.filter_tensorboard_metrics(metrics)

        self.assertIn("math/loss/token_opd_loss_mean", filtered)
        self.assertIn("math/loss/token_opd_loss_std", filtered)
        self.assertIn("math/loss/token_opd_loss_variance", filtered)
        self.assertIn("math/loss/sample_opd_loss_mean", filtered)
        self.assertIn("math/loss/sample_opd_loss_std", filtered)
        self.assertIn("math/loss/sample_opd_loss_variance", filtered)
        self.assertIn("global/loss/token_opd_loss_mean", filtered)
        self.assertIn("global/loss/sample_opd_loss_variance", filtered)
        self.assertNotIn("global/full_grad/total_grad_norm", filtered)
        self.assertIn("global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k", filtered)
        self.assertIn("global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total", filtered)
        self.assertIn("global/full_grad_contribution/math_to_total/signed_projection_share", filtered)
        self.assertIn("global/full_grad_closure/domain_sum_vs_training/cosine", filtered)
        self.assertIn("global/full_grad_closure/domain_sum_vs_training/rel_l2", filtered)
        self.assertIn("global/full_grad_cost/backward_seconds", filtered)
        self.assertIn("global/full_grad_cost/domain_summary_seconds", filtered)
        self.assertIn("global/full_grad_cost/finish_mini_batch_seconds", filtered)
        self.assertIn("global/audit/full_gradient_domain_sequential_available", filtered)
        self.assertIn("global/audit/full_gradient_domain_sequential_unsupported", filtered)
        self.assertIn("global/audit/full_gradient_replicated_all_reduce", filtered)
        self.assertIn("global/audit/full_gradient_replica_count", filtered)
        self.assertIn("global/audit/pre_update_audit_used", filtered)
        self.assertIn("global/audit/full_gradient_execution_timing_pre_update", filtered)
        self.assertIn("global/audit/sample_gradient_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_norm_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_cos_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_distributed_world_size", filtered)
        self.assertIn("math/teacher/teacher_student_gap_mean", filtered)
        self.assertIn("math/token_gap/gap_abs_sum", filtered)
        self.assertIn("math/token_gap/gap_signed_p95", filtered)
        self.assertIn("global/token_gap_vocab_cosine/math_vs_code/gap_abs_sum_cosine", filtered)
        self.assertIn(
            "global/entropy_vocab_cosine/math_vs_code/teacher_student_cross_entropy_sum_cosine",
            filtered,
        )
        self.assertIn("global/entropy_vocab_cosine/math_vs_code/student_entropy_sum_cosine", filtered)
        self.assertIn("math/entropy/sum_teacher_entropy", filtered)
        self.assertIn("math/entropy/teacher_entropy_mean", filtered)
        self.assertIn("math/entropy/teacher_entropy_p95", filtered)
        self.assertIn("math/entropy/student_entropy_mean", filtered)
        self.assertIn("math/entropy/teacher_student_cross_entropy_mean", filtered)
        self.assertIn("math/advantage/positive_frac", filtered)
        self.assertIn("math/length/response_mean", filtered)
        self.assertIn("math/length/response_p95", filtered)
        self.assertIn("math/length/response_clip_ratio", filtered)
        self.assertIn("math/sample_grad/norm_mean", filtered)
        self.assertIn("math/sample_grad/norm_p95", filtered)
        self.assertIn("math/sample_grad_cos/domain_cos_mean", filtered)
        self.assertIn("math/sample_grad_cos/domain_cos_negative_frac", filtered)
        self.assertIn("math/sample_grad_contribution/projection_share_mean", filtered)
        self.assertIn("math/sample_grad_contribution/projection_share_normalized_sum", filtered)
        self.assertIn("math/sample_grad_contribution/top1_abs_share", filtered)
        self.assertIn("math/sample_grad_contribution/top1_abs_share_normalized", filtered)
        self.assertIn("math/sample_grad_contribution/projection_share_trusted", filtered)
        self.assertIn("math/sample_grad_closure/projection_share_sum_error", filtered)
        self.assertIn("math/sample_grad_closure/projection_share_normalized_sum_error", filtered)
        self.assertIn("math/sample_grad_closure/vector_cosine", filtered)
        self.assertIn("math/sample_grad_closure/vector_rel_l2", filtered)
        self.assertIn("math/sample_grad_cost/backward_recompute_count", filtered)
        self.assertIn("math/sample_grad_cost/restore_post_target_rel_l2_max", filtered)
        self.assertIn("global/full_grad_training_parity/audit_total_vs_training_total/cosine", filtered)
        self.assertIn("global/full_grad_training_parity/audit_total_vs_training_total/rel_l2", filtered)
        self.assertIn("global/full_grad_training_parity/sequence_total_vs_training_total/cosine", filtered)
        self.assertIn("global/full_grad_training_parity/sequence_total_vs_training_total/rel_l2", filtered)
        self.assertIn("math/token_grad_cost/seconds_sum", filtered)
        self.assertIn("math/token_grad_cost/backward_fallback_count", filtered)
        self.assertIn("global/token_grad_cost/seconds", filtered)
        self.assertIn("global/token_grad_cost/seconds_per_selected_token", filtered)
        self.assertIn("global/token_grad_cost/backward_fallback_seconds_sum", filtered)
        self.assertIn("global/token_grad_cost/global_candidate_gap_mass", filtered)
        self.assertIn("global/token_grad_cost/global_candidate_loss_abs_mass", filtered)
        self.assertIn("global/token_grad_cost/valid_frac", filtered)
        self.assertIn("global/token_grad_cost/restore_post_target_rel_l2_max", filtered)
        self.assertIn("global/token_grad_cost/restore_original_rel_l2_max", filtered)
        self.assertIn("math/token_grad_cost/restore_post_target_max_abs_max", filtered)
        self.assertIn("math/token_grad_cost/restore_original_max_abs_max", filtered)
        self.assertIn("math/token_grad/global_candidate_gap_mass", filtered)
        self.assertIn("math/token_grad/global_candidate_loss_abs_mass", filtered)
        self.assertIn("math/token_grad/top100_gap_gap_mass_frac", filtered)
        self.assertIn("math/token_grad/top100_gap_abs_gap_abs_mass_frac", filtered)
        self.assertIn("math/token_grad/top100_loss_abs_loss_abs_mass_frac", filtered)
        self.assertIn("math/token_grad/top100_loss_abs_score_mass_frac", filtered)
        self.assertIn("math/token_grad_closure/topp100_loss_abs_mass_selected_all_tokens", filtered)
        self.assertIn("math/token_grad_closure/topp100_loss_abs_mass_projection_share_error", filtered)
        self.assertIn("math/token_grad_closure/topp100_loss_abs_mass_cosine_error", filtered)
        self.assertIn("math/token_grad_closure/topp100_loss_abs_mass_norm_ratio_error", filtered)
        self.assertIn("math/reward/training_reward_mean", filtered)
        self.assertIn("math/reward/training_accuracy", filtered)
        self.assertIn("math/coverage/duplicate_rate", filtered)
        self.assertIn("global/cost/step_seconds", filtered)
        self.assertIn("global/validation_gain/val-core_AIME2024_reward_mean_1", filtered)
        self.assertIn("rollout_corr/kl", filtered)
        self.assertIn("actor/lr", filtered)
        self.assertIn("actor/topk_distill_loss", filtered)
        self.assertNotIn("math/loss/opd_loss_p95", filtered)
        self.assertNotIn("math/loss/kl_spike_rate", filtered)
        self.assertNotIn("global/full_grad_conflict/math_vs_code/full_grad_dot_train_i_k", filtered)
        self.assertNotIn("math/grad_conflict/code/grad_cosine_train_i_k", filtered)
        self.assertNotIn("global/grad_conflict/math_vs_code/grad_cosine_train_i_k", filtered)
        self.assertNotIn("math/grad/grad_norm", filtered)
        self.assertNotIn("math/teacher/teacher_logprob_mean", filtered)
        self.assertNotIn("math/coverage/new_sample_rate", filtered)
        self.assertNotIn("global/cost/gpu_count", filtered)
        self.assertNotIn("global/validation/val-core_AIME2024_reward_mean_1", filtered)
        self.assertNotIn("global/validation_gain_stats/val-core_AIME2024_reward_mean_1/ci_high", filtered)
        self.assertNotIn("rollout_corr/rollout_is_mean", filtered)
        self.assertNotIn("rollout_corr/training_ppl", filtered)
        self.assertNotIn("actor/kl_loss", filtered)
        self.assertNotIn("training/rollout_log_probs_diff", filtered)

    def test_audit_logger_uses_domain_category_tags_for_validation_gain_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": str(Path(temp_dir) / "audit"),
                        "domains": ["math", "code"],
                    }
                }
            )
            metrics = logger.log_validation_metrics({"val/math/score": 0.2, "val/code/pass@k": 0.1}, step=0)
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.25}, step=1))
            metrics.update(
                logger.log_training_cost(
                    {"timing_s/step": 2.0, "perf/total_num_tokens": 7, "perf/max_memory_allocated_gb": 1.5},
                    step=0,
                    n_gpus=1,
                )
            )

            expected_metric_keys = [
                "math/validation_gain/score",
                "math/validation_gain_stats/score/variance",
                "global/cost/gpu_seconds_step",
            ]
            for key in expected_metric_keys:
                self.assertIn(key, metrics)
            self.assertNotIn("math/validation/score", metrics)
            self.assertNotIn("code/validation/pass_k", metrics)
            self.assertNotIn("mopd/validation/val_math_score", metrics)

    def test_sequential_backward_domain_tracker_computes_total_cosine_and_contribution(self) -> None:
        try:
            import math

            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {"enabled": True, "domains": ["math", "code"], "storage_dtype": "float32", "learning_rate": 0.1},
        )

        metrics, _targets = tracker._finish_direct_domain_gradient_metrics(
            {
                "math": ((torch.tensor([1.0, 0.0]),), 1.0),
                "code": ((torch.tensor([0.0, 1.0]),), 1.0),
            }
        )

        expected_total_norm = math.sqrt(2.0)
        expected_domain_total_cosine = 1.0 / expected_total_norm
        self.assertAlmostEqual(metrics["math/full_grad/grad_norm"], 1.0, places=6)
        self.assertAlmostEqual(metrics["code/full_grad/grad_norm"], 1.0, places=6)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)
        self.assertAlmostEqual(
            metrics["global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total"],
            expected_domain_total_cosine,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_alignment/code_vs_total/full_grad_cosine_domain_total"],
            expected_domain_total_cosine,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_contribution/math_to_total/signed_projection_share"],
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_contribution/code_to_total/signed_projection_share"],
            0.5,
            places=6,
        )

    def test_sequential_backward_tracker_snapshots_after_first_domain_block(self) -> None:
        try:
            import math

            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"domain": [domain], "opd_teacher": [domain]}

            def __len__(self) -> int:
                return 1

        actor = ToyActor()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {"enabled": True, "domains": ["math", "code"], "storage_dtype": "float32", "learning_rate": 0.1},
        )
        tracked = tracker.prepare_micro_batches(
            [ToyMicroBatch("code"), ToyMicroBatch("math"), ToyMicroBatch("math"), ToyMicroBatch("code")]
        )
        self.assertEqual([domain for domain, _ in tracked], ["math", "math", "code", "code"])

        inputs = {
            "math": [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 2.0]])],
            "code": [torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 4.0]])],
        }
        offsets = {"math": 0, "code": 0}
        math_grad = sum(inputs["math"])
        code_grad = sum(inputs["code"])
        metrics, _targets = tracker._finish_direct_domain_gradient_metrics(
            {
                "math": ((math_grad.reshape(-1),), float(torch.dot(math_grad.reshape(-1), math_grad.reshape(-1)))),
                "code": ((code_grad.reshape(-1),), float(torch.dot(code_grad.reshape(-1), code_grad.reshape(-1)))),
            }
        )

        self.assertAlmostEqual(metrics["math/full_grad/grad_norm"], math.sqrt(5.0), places=6)
        self.assertAlmostEqual(metrics["code/full_grad/grad_norm"], 5.0, places=6)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)
        self.assertAlmostEqual(
            metrics["global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k"],
            11.0 / (math.sqrt(5.0) * 5.0),
            places=6,
        )

    def test_sequential_tracker_skips_all_ranks_when_any_rank_is_unsupported(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"domain": [domain], "opd_teacher": [domain]}

            def __len__(self) -> int:
                return 1

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {"enabled": True, "domains": ["math", "code"]},
        )
        micro_batches = [ToyMicroBatch("math"), ToyMicroBatch("code")]

        with patch("mopd_verl.full_gradient.tracker._all_ranks_true", return_value=False):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertFalse(tracker._prepared_supported)
        self.assertEqual([domain for domain, _ in tracked], [None, None])
        self.assertEqual([micro_batch for _, micro_batch in tracked], micro_batches)

    def test_sequential_tracker_requires_aligned_domain_boundaries_across_ranks(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"domain": [domain], "opd_teacher": [domain]}

            def __len__(self) -> int:
                return 1

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {"enabled": True, "domains": ["math", "code"]},
        )
        micro_batches = [
            ToyMicroBatch("math"),
            ToyMicroBatch("math"),
            ToyMicroBatch("code"),
        ]

        with (
            patch("mopd_verl.full_gradient.tracker._all_ranks_true", return_value=True),
            patch("mopd_verl.full_gradient.tracker._all_ranks_equal_ints", return_value=False),
        ):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertFalse(tracker._prepared_supported)
        self.assertEqual([domain for domain, _ in tracked], [None, None, None])

    def test_sequential_tracker_accepts_aligned_domain_partition_metadata(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str, sample_count: int) -> None:
                self.non_tensor_batch = {"domain": [domain] * sample_count, "opd_teacher": [domain] * sample_count}
                self._sample_count = sample_count

            def __len__(self) -> int:
                return self._sample_count

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domain_gradient_enabled": True,
                "domains": ["math", "code"],
                "domain_partition": {
                    "aligned": True,
                    "domain_order": ["math", "code"],
                    "domain_block_sample_counts": {"math": 2, "code": 2},
                },
            },
        )
        micro_batches = [ToyMicroBatch("math", 2), ToyMicroBatch("code", 2)]

        with (
            patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=2),
            patch("mopd_verl.full_gradient.tracker._all_ranks_true", return_value=True),
            patch("mopd_verl.full_gradient.tracker._all_ranks_equal_ints", return_value=True),
        ):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertTrue(tracker._prepared_supported)
        self.assertTrue(tracker.domain_gradient_enabled)
        self.assertEqual([domain for domain, _ in tracked], ["math", "code"])

    def test_full_gradient_statistics_sum_rank_shards_without_replica_averaging(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import (
                _current_grad_difference_snapshot,
                _gradient_replica_count,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[1.0, 2.0]])
        first_chunks = (torch.tensor([1.0, 0.0]),)

        with (
            patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=2),
            patch(
                "mopd_verl.full_gradient.tracker._all_reduce_values_sum",
                side_effect=lambda values: [value * 2.0 for value in values],
            ) as reduce_mock,
        ):
            self.assertEqual(_gradient_replica_count(actor), 2)
            snapshot = _current_grad_difference_snapshot(actor, first_chunks)

        self.assertIsNotNone(snapshot)
        reduce_mock.assert_called_once()
        self.assertAlmostEqual(snapshot.first_norm_sq, 2.0, places=6)
        self.assertAlmostEqual(snapshot.second_norm_sq, 8.0, places=6)
        self.assertAlmostEqual(snapshot.total_norm_sq, 10.0, places=6)
        self.assertAlmostEqual(snapshot.first_second_dot, 0.0, places=6)

    def test_full_shard_gradient_statistics_do_not_apply_replica_division(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import _gradient_replica_count
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": -1}}

        with patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=4):
            self.assertEqual(_gradient_replica_count(ToyActor()), 1)

    def test_direct_domain_targets_emit_training_grad_closure_metrics(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {"enabled": True, "domains": ["math", "code"]},
        )
        metrics = tracker._domain_target_closure_metrics(
            {
                "math": ((torch.tensor([1.0, 1.0]),), 2.0),
                "code": ((torch.tensor([1.0, 2.0]),), 5.0),
            },
            reference_chunks=(torch.tensor([2.0, 3.0]),),
        )

        prefix = "global/full_grad_closure/domain_sum_vs_training"
        self.assertAlmostEqual(metrics[f"{prefix}/rel_l2"], 0.0, places=7)
        self.assertAlmostEqual(metrics[f"{prefix}/cosine"], 1.0, places=7)
        self.assertAlmostEqual(metrics[f"{prefix}/norm_ratio"], 1.0, places=7)
        self.assertAlmostEqual(metrics[f"{prefix}/projection_share"], 1.0, places=7)

    def test_sequential_tracker_keeps_sample_norm_for_full_param_replicas(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 1}}

        with patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=2):
            tracker = SequentialBackwardDomainGradientTracker(
                ToyActor(),
                {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "sample_gradient_enabled": True,
                    "sample_gradient_norm_enabled": True,
                    "sample_gradient_cos_enabled": True,
                },
            )

        self.assertTrue(tracker.sample_norm_enabled)
        self.assertTrue(tracker.sample_cos_enabled)
        self.assertTrue(tracker.sample_log_sample_level)
        self.assertFalse(tracker._sample_gradient_distributed_unsupported)
        self.assertFalse(tracker._sample_gradient_cos_distributed_unsupported)

        tracker._sample_records = [
            {"domain": "math", "sample_grad_norm": 1.0},
            {"domain": "code", "sample_grad_norm": 3.0},
        ]
        with patch("mopd_verl.full_gradient.tracker._all_gather_list", side_effect=lambda values: list(values)):
            metrics = tracker._sample_norm_metrics()

        self.assertEqual(metrics["math/sample_grad/norm_mean"], 1.0)
        self.assertEqual(metrics["code/sample_grad/norm_mean"], 3.0)

    def test_full_gradient_meta_carries_domain_partition_metadata(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": True,
                    "sequence_masked_target_enabled": True,
                    "sequence_masked_target_use_as_primary": True,
                    "sequence_replay_skip_non_target_domains": True,
                    "training_gradient_from_domain_sum_enabled": True,
                }
            }
        )

        meta = logger.full_gradient_meta(
            "train",
            3,
            {
                "aligned": True,
                "domain_block_sample_counts": {"math": 8, "code": 8},
            },
        )

        self.assertEqual(
            meta["mopd_full_gradient"]["domain_partition"]["domain_block_sample_counts"],
            {"math": 8, "code": 8},
        )
        self.assertTrue(meta["mopd_full_gradient"]["sequence_masked_target_enabled"])
        self.assertTrue(meta["mopd_full_gradient"]["sequence_masked_target_use_as_primary"])
        self.assertTrue(meta["mopd_full_gradient"]["sequence_replay_skip_non_target_domains"])
        self.assertEqual(meta["mopd_full_gradient"]["sequence_masked_target_closure_rel_l2_threshold"], 0.02)
        self.assertTrue(meta["mopd_full_gradient"]["training_gradient_from_domain_sum_enabled"])

    def test_token_gradient_frequency_controls_worker_meta(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": False,
                    "sample_gradient_enabled": False,
                    "token_gradient_enabled": True,
                    "token_gradient_freq_steps": 4,
                    "token_gradient_backward_recompute_enabled": True,
                    "token_gradient_backward_sync_enabled": True,
                }
            }
        )

        inactive_meta = logger.full_gradient_meta("train", 3)
        active_meta = logger.full_gradient_meta("train", 4)

        self.assertFalse(logger.should_compute_token_gradient(3))
        self.assertTrue(logger.should_compute_token_gradient(4))
        self.assertFalse(inactive_meta["mopd_full_gradient"]["token_gradient_enabled"])
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_enabled"])
        self.assertEqual(active_meta["mopd_full_gradient"]["token_gradient_freq_steps"], 4)
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_gap_selection_enabled"])
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_gap_abs_selection_enabled"])
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_loss_abs_selection_enabled"])
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_backward_recompute_enabled"])
        self.assertTrue(active_meta["mopd_full_gradient"]["token_gradient_backward_sync_enabled"])
        self.assertFalse(logger.should_compute_domain_gradient(3))
        self.assertTrue(logger.should_compute_domain_gradient(4))

    def test_full_grad_training_parity_frequency_controls_worker_meta(self) -> None:
        logger = MOPDAuditLogger(
            {
                "mopd_audit": {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "full_gradient_enabled": True,
                    "full_grad_training_parity_freq_steps": -1,
                }
            }
        )

        meta = logger.full_gradient_meta("train", 1)

        self.assertFalse(logger.should_log_full_grad_training_parity(1))
        self.assertEqual(meta["mopd_full_gradient"]["full_grad_training_parity_freq_steps"], -1)

    def test_tracker_skips_full_grad_training_parity_when_frequency_is_negative(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[1.0, 1.0]])
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "enabled": True,
                "domains": ["math", "code"],
                "full_grad_training_parity_freq_steps": -1,
            },
        )
        tracker._last_audit_total_chunks = (torch.tensor([1.0, 1.0]),)
        tracker._last_sequence_total_chunks = (torch.tensor([2.0, 2.0]),)

        self.assertEqual(tracker.full_grad_training_parity_metrics(), {})
        self.assertEqual(tracker._last_audit_total_chunks, tuple())
        self.assertEqual(tracker._last_sequence_total_chunks, tuple())

    def test_finish_mini_batch_does_not_retain_parity_chunks_when_disabled(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        sequence_targets = {"math": ((torch.tensor([1.0]),), 1.0)}
        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "full_grad_training_parity_freq_steps": -1,
            },
        )
        tracker._schedule_candidates = [{"micro_batch": object()}]

        with patch.object(
            tracker,
            "_recompute_sequence_domain_targets",
            return_value=({}, sequence_targets, (torch.tensor([1.0]),), 1.0),
        ), patch.object(
            tracker,
            "_finish_direct_domain_gradient_metrics",
            return_value=({}, sequence_targets),
        ), patch.object(
            tracker,
            "_summed_domain_target_reference_chunks",
            return_value=(torch.tensor([1.0]),),
        ), patch.object(
            tracker,
            "_domain_target_closure_metrics",
            return_value={},
        ), patch.object(
            tracker,
            "_sample_norm_metrics",
            return_value={},
        ), patch(
            "mopd_verl.full_gradient.tracker._all_reduce_sum",
            side_effect=lambda value: float(value),
        ):
            tracker.finish_mini_batch()

        self.assertEqual(tracker._last_audit_total_chunks, tuple())
        self.assertEqual(tracker._last_sequence_total_chunks, tuple())
        self.assertEqual(tracker._schedule_candidates, [])
        self.assertEqual(tracker._sample_records, [])
        self.assertEqual(tracker._token_gradient_candidates, {})

    def test_finish_mini_batch_keeps_parity_chunks_until_parity_check(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        next(actor.actor_module.parameters()).grad = torch.tensor([[1.0]])
        sequence_targets = {"math": ((torch.tensor([1.0]),), 1.0)}
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "enabled": True,
                "domains": ["math"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "full_grad_training_parity_freq_steps": 1,
            },
        )
        tracker._schedule_candidates = [{"micro_batch": object()}]

        with patch.object(
            tracker,
            "_recompute_sequence_domain_targets",
            return_value=({}, sequence_targets, (torch.tensor([1.0]),), 1.0),
        ), patch.object(
            tracker,
            "_finish_direct_domain_gradient_metrics",
            return_value=({}, sequence_targets),
        ), patch.object(
            tracker,
            "_summed_domain_target_reference_chunks",
            return_value=(torch.tensor([1.0]),),
        ), patch.object(
            tracker,
            "_domain_target_closure_metrics",
            return_value={},
        ), patch.object(
            tracker,
            "_sample_norm_metrics",
            return_value={},
        ), patch(
            "mopd_verl.full_gradient.tracker._all_reduce_sum",
            side_effect=lambda value: float(value),
        ):
            tracker.finish_mini_batch()

        self.assertTrue(tracker._last_audit_total_chunks)
        self.assertTrue(tracker._last_sequence_total_chunks)
        self.assertEqual(tracker._schedule_candidates, [])

        with patch(
            "mopd_verl.full_gradient.tracker._snapshot_current_grad_chunks",
            return_value=(torch.tensor([1.0]),),
        ), patch(
            "mopd_verl.full_gradient.tracker._gradient_chunk_pair_stats",
            return_value=({"rel_l2": 0.0}, []),
        ):
            metrics = tracker.full_grad_training_parity_metrics()

        self.assertIn("global/full_grad_training_parity/audit_total_vs_training_total/rel_l2", metrics)
        self.assertEqual(tracker._last_audit_total_chunks, tuple())
        self.assertEqual(tracker._last_sequence_total_chunks, tuple())

    def test_finish_mini_batch_clears_cpu_refs_on_error(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 1}}

        tracker = SequentialBackwardDomainGradientTracker(ToyActor(), {"domains": ["math"]})
        tracker._schedule_candidates = [{"micro_batch": object()}]
        tracker._sample_records = [{"sample_id": "sample-0"}]
        tracker._token_gradient_candidates = {"math": [{"micro_batch": object()}]}

        with patch.object(tracker, "_finish_mini_batch_impl", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                tracker.finish_mini_batch()

        self.assertEqual(tracker._schedule_candidates, [])
        self.assertEqual(tracker._sample_records, [])
        self.assertEqual(tracker._token_gradient_candidates, {})

    def test_token_gradient_top_p_controls_gap_abs_mass_selection(self) -> None:
        try:
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.config = ActorConfig()

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_top_p": 0.8,
            },
        )
        records = [
            {"gap": 6.0, "gap_abs": 60.0, "loss_abs": 5.0, "token_id": 1},
            {"gap": 3.0, "gap_abs": 30.0, "loss_abs": 90.0, "token_id": 2},
            {"gap": -10.0, "gap_abs": 10.0, "loss_abs": 5.0, "token_id": 3},
        ]

        selections = dict(tracker._gap_abs_token_selections(records))
        score_selections = {
            (selection, score_key): rows
            for selection, score_key, rows in tracker._token_score_selections(records)
        }

        self.assertIn("topp80_gap_abs_mass", selections)
        self.assertEqual(len(selections["topp80_gap_abs_mass"]), 2)
        self.assertEqual(len(selections["top100_gap_abs"]), 3)
        self.assertIn(("topp80_gap_mass", "gap"), score_selections)
        self.assertEqual(len(score_selections[("topp80_gap_mass", "gap")]), 2)
        self.assertEqual(len(score_selections[("top100_gap", "gap")]), 3)
        self.assertIn(("topp80_loss_abs_mass", "loss_abs"), score_selections)
        self.assertEqual(len(score_selections[("topp80_loss_abs_mass", "loss_abs")]), 1)
        self.assertEqual(score_selections[("topp80_loss_abs_mass", "loss_abs")][0]["token_id"], 2)
        self.assertEqual(len(score_selections[("top100_loss_abs", "loss_abs")]), 3)

    def test_token_gradient_top_p_one_selects_all_scored_tokens(self) -> None:
        try:
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.config = ActorConfig()

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_top_p": 1.0,
            },
        )
        records = [
            {"gap": 6.0, "gap_abs": 60.0, "loss_abs": 5.0, "token_id": 1},
            {"gap": -10.0, "gap_abs": 10.0, "loss_abs": 5.0, "token_id": 2},
            {"gap": 0.0, "gap_abs": 0.0, "loss_abs": 0.0, "token_id": 3},
        ]

        score_selections = {
            (selection, score_key): rows
            for selection, score_key, rows in tracker._token_score_selections(records)
        }

        self.assertEqual(len(score_selections[("topp100_gap_mass", "gap")]), 3)
        self.assertEqual(len(score_selections[("topp100_gap_abs_mass", "gap_abs")]), 3)
        self.assertEqual(len(score_selections[("topp100_loss_abs_mass", "loss_abs")]), 3)

    def test_token_gradient_top_p_one_emits_domain_closure_metrics(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_gap_selection_enabled": False,
                "token_gradient_gap_abs_selection_enabled": True,
                "token_gradient_loss_abs_selection_enabled": False,
                "token_gradient_top_p": 1.0,
            },
        )
        tracker._token_gradient_candidates = {
            "math": [
                {
                    "context": {"loss_scale_factor": 1.0, "on_policy": True},
                    "micro_batch": object(),
                    "tokens": [
                        {"sample_id": "sample-0", "sample_index": 0, "position": 0, "gap_abs": 2.0},
                        {"sample_id": "sample-0", "sample_index": 0, "position": 1, "gap_abs": 1.0},
                    ],
                }
            ]
        }

        def fake_recompute(
            selected_tokens: list[dict[str, object]],
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "token_grad_available": 1.0,
                "token_grad_norm": 2.0,
                "token_grad_non_none_grad_count": 1.0,
                "token_grad_param_count": 1.0,
                "token_grad_none_grad_count": 0.0,
                "token_grad_seconds": 0.1,
                "token_grad_autograd_seconds": 0.1,
                "math_cos": 1.0,
                "math_projection_share": 1.0 if len(selected_tokens) == 2 else 0.5,
            }

        target_chunks = (torch.tensor([2.0, 0.0]),)
        with patch.object(
            tracker,
            "_recompute_token_selection_gradient_stats",
            side_effect=fake_recompute,
        ):
            metrics = tracker._token_gradient_metrics({"math": (target_chunks, 4.0)})

        prefix = "math/token_grad_closure/topp100_gap_abs_mass"
        self.assertEqual(metrics[f"{prefix}_candidate_token_frac"], 1.0)
        self.assertEqual(metrics[f"{prefix}_candidate_sample_frac"], 1.0)
        self.assertEqual(metrics[f"{prefix}_selected_all_tokens"], 1.0)
        self.assertEqual(metrics[f"{prefix}_selected_all_samples"], 1.0)
        self.assertEqual(metrics[f"{prefix}_projection_share_error"], 0.0)
        self.assertEqual(metrics[f"{prefix}_cosine_error"], 0.0)
        self.assertEqual(metrics[f"{prefix}_norm_ratio"], 1.0)
        self.assertEqual(metrics[f"{prefix}_norm_ratio_error"], 0.0)

    def test_token_gradient_selection_switches_control_score_families(self) -> None:
        try:
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.config = ActorConfig()

        records = [
            {"gap": 10.0, "gap_abs": 10.0, "loss_abs": 1.0},
            {"gap": -10.0, "gap_abs": 1.0, "loss_abs": 10.0},
        ]
        signed_gap_only = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_gap_selection_enabled": True,
                "token_gradient_gap_abs_selection_enabled": False,
                "token_gradient_loss_abs_selection_enabled": False,
            },
        )
        gap_abs_only = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_gap_selection_enabled": False,
                "token_gradient_gap_abs_selection_enabled": True,
                "token_gradient_loss_abs_selection_enabled": False,
            },
        )
        loss_only = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_gap_selection_enabled": False,
                "token_gradient_gap_abs_selection_enabled": False,
                "token_gradient_loss_abs_selection_enabled": True,
            },
        )

        signed_gap_scores = {
            score_key for _selection, score_key, _rows in signed_gap_only._token_score_selections(records)
        }
        gap_abs_scores = {score_key for _selection, score_key, _rows in gap_abs_only._token_score_selections(records)}
        loss_scores = {score_key for _selection, score_key, _rows in loss_only._token_score_selections(records)}

        self.assertEqual(signed_gap_scores, {"gap"})
        self.assertEqual(gap_abs_scores, {"gap_abs"})
        self.assertEqual(loss_scores, {"loss_abs"})

    def test_token_gradient_uses_global_gap_abs_selection_counts(self) -> None:
        try:
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.config = ActorConfig()

        with patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=2):
            tracker = SequentialBackwardDomainGradientTracker(
                ToyActor(),
                {
                    "domains": ["math"],
                    "token_gradient_enabled": True,
                    "token_gradient_gap_selection_enabled": False,
                    "token_gradient_top_p": 0.5,
                },
            )
        tracker._token_gradient_candidates = {
            "math": [
                {
                    "context": {"loss_scale_factor": 1.0, "on_policy": True},
                    "micro_batch": object(),
                    "tokens": [
                        {"sample_id": "local-0", "sample_index": 0, "position": 1, "gap_abs": 90.0},
                        {"sample_id": "local-1", "sample_index": 0, "position": 2, "gap_abs": 80.0},
                    ],
                }
            ]
        }
        remote_metadata = {
            "domain": "math",
            "sample_id": "remote-0",
            "sample_index": 0,
            "position": 3,
            "gap_abs": 100.0,
            "owner_rank": 1,
            "token_candidate_id": 0,
        }

        def fake_all_gather(values: list[object]) -> list[object]:
            if values and isinstance(values[0], str):
                return list(values) + ["math"]
            if values and isinstance(values[0], dict):
                return list(values) + [remote_metadata]
            return list(values)

        recompute_calls: list[list[float]] = []

        def fake_recompute(selected_tokens: list[dict[str, object]], **_kwargs: object) -> dict[str, object]:
            recompute_calls.append([float(token["gap_abs"]) for token in selected_tokens])
            return {
                "token_grad_available": 1.0,
                "token_grad_norm": 1.0,
                "token_grad_seconds": 0.1,
                "token_grad_autograd_seconds": 0.1,
                "math_cos": 0.5,
                "math_projection_share": 0.25,
            }

        with patch("mopd_verl.full_gradient.tracker._distributed_rank", return_value=0), patch(
            "mopd_verl.full_gradient.tracker._all_gather_list",
            side_effect=fake_all_gather,
        ), patch.object(
            tracker,
            "_recompute_token_selection_gradient_stats",
            side_effect=fake_recompute,
        ):
            metrics = tracker._token_gradient_metrics({"math": ((), 1.0)})

        self.assertEqual(recompute_calls, [[90.0, 80.0], [90.0]])
        self.assertEqual(metrics["math/token_grad/top100_gap_abs_selected_token_count"], 3.0)
        self.assertEqual(metrics["math/token_grad/top100_gap_abs_selected_sample_count"], 3.0)
        self.assertEqual(metrics["math/token_grad/top100_gap_abs_gap_abs_mass"], 270.0)
        self.assertEqual(metrics["math/token_grad/global_candidate_token_count"], 3.0)
        self.assertEqual(metrics["math/token_grad/global_candidate_sample_count"], 3.0)
        self.assertEqual(metrics["math/token_grad/global_candidate_gap_abs_mass"], 270.0)
        self.assertEqual(metrics["math/token_grad/topp50_gap_abs_mass_selected_token_count"], 2.0)
        self.assertAlmostEqual(metrics["math/token_grad/topp50_gap_abs_mass_gap_abs_mass_frac"], 190.0 / 270.0)

    def test_token_gradient_collects_all_valid_tokens_before_global_selection(self) -> None:
        try:
            import torch

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            config = ActorConfig()

        class FakeMicroBatch:
            def __init__(self) -> None:
                self.batch = {
                    "old_log_probs": torch.zeros((2, 3), dtype=torch.float32),
                    "math_teacher_log_prob": torch.tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        dtype=torch.float32,
                    ),
                    "response_mask": torch.ones((2, 3), dtype=torch.float32),
                    "responses": torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.long),
                }
                self.non_tensor_batch = {
                    "domain": ["math", "math"],
                    "opd_teacher": ["math", "math"],
                    "sample_id": ["sample-0", "sample-1"],
                }

            def __len__(self) -> int:
                return 2

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_top_k": 100,
            },
        )

        rows = tracker._select_token_gradient_candidates(FakeMicroBatch(), domain="math")

        self.assertEqual(len(rows), 6)
        self.assertEqual({row["sample_id"] for row in rows}, {"sample-0", "sample-1"})
        self.assertEqual([row["token_id"] for row in rows if row["sample_id"] == "sample-1"], [23, 22, 21])

    def test_token_gradient_loss_abs_selection_uses_scaled_loss_contribution(self) -> None:
        try:
            import torch

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}
            loss_agg_mode = "token-mean"

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            config = ActorConfig()

        class FakeMicroBatch:
            def __init__(
                self,
                *,
                sample_id: str,
                response_mask: torch.Tensor,
                teacher_logp: torch.Tensor,
                raw_loss_scores: torch.Tensor,
                responses: torch.Tensor,
            ) -> None:
                self.batch = {
                    "old_log_probs": torch.zeros_like(response_mask),
                    "math_teacher_log_prob": teacher_logp,
                    "response_mask": response_mask,
                    "responses": responses,
                }
                self.non_tensor_batch = {
                    "domain": ["math"],
                    "opd_teacher": ["math"],
                    "sample_id": [sample_id],
                }
                self.raw_loss_scores = raw_loss_scores

            def __len__(self) -> int:
                return 1

        def fake_loss_scores(
            _actor: object,
            micro_batch: FakeMicroBatch,
            *,
            on_policy: bool,
        ) -> tuple[torch.Tensor, str]:
            del on_policy
            return micro_batch.raw_loss_scores, "unit_loss"

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
                "token_gradient_gap_selection_enabled": False,
                "token_gradient_gap_abs_selection_enabled": False,
                "token_gradient_loss_abs_selection_enabled": True,
                "token_gradient_top_k": 1,
            },
        )
        long_sample = FakeMicroBatch(
            sample_id="long",
            response_mask=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
            teacher_logp=torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),
            raw_loss_scores=torch.tensor([[6.0, 1.0, 1.0]], dtype=torch.float32),
            responses=torch.tensor([[101, 102, 103]], dtype=torch.long),
        )
        short_sample = FakeMicroBatch(
            sample_id="short",
            response_mask=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            teacher_logp=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            raw_loss_scores=torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float32),
            responses=torch.tensor([[201, 202, 203]], dtype=torch.long),
        )

        with patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_token_loss_scores",
            side_effect=fake_loss_scores,
        ):
            rows = tracker._select_token_gradient_candidates(long_sample, domain="math")
            rows.extend(tracker._select_token_gradient_candidates(short_sample, domain="math"))

        long_token = next(row for row in rows if row["sample_id"] == "long" and row["position"] == 0)
        short_token = next(row for row in rows if row["sample_id"] == "short" and row["position"] == 0)
        self.assertEqual(long_token["loss_raw_abs"], 6.0)
        self.assertAlmostEqual(long_token["loss_abs"], 2.0)
        self.assertEqual(short_token["loss_raw_abs"], 3.0)
        self.assertAlmostEqual(short_token["loss_abs"], 3.0)
        self.assertAlmostEqual(long_token["loss_contribution_scale"], 1.0 / 3.0)
        self.assertEqual(short_token["loss_contribution_scale"], 1.0)
        selections = {
            selection: selected_rows
            for selection, score_key, selected_rows in tracker._token_score_selections(rows)
            if score_key == "loss_abs"
        }
        self.assertEqual(selections["top1_loss_abs"][0]["sample_id"], "short")

    def test_token_gradient_caches_all_selected_sample_slices_for_full_distribution(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            config = ActorConfig()

        class FakeMicroBatch:
            batch = None
            non_tensor_batch = {"domain": ["math", "math", "math"], "opd_teacher": ["math", "math", "math"]}

            def __len__(self) -> int:
                return 3

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "domains": ["math"],
                "token_gradient_enabled": True,
            },
        )
        tracker.start_mini_batch()

        token_rows = [
            {"sample_id": "sample-0", "sample_index": 0, "position": 3, "gap_abs": 9.0},
            {"sample_id": "sample-0", "sample_index": 0, "position": 4, "gap_abs": 8.0},
            {"sample_id": "sample-1", "sample_index": 1, "position": 2, "gap_abs": 7.0},
            {"sample_id": "sample-2", "sample_index": 2, "position": 1, "gap_abs": 6.0},
        ]
        copied_indices: list[list[int]] = []

        def fake_copy(_micro_batch: object, indices: list[int]) -> object:
            copied_indices.append(indices)
            return {"indices": indices}

        with patch.object(
            tracker,
            "_select_token_gradient_candidates",
            return_value=token_rows,
        ), patch(
            "mopd_verl.full_gradient.tracker._copy_data_proto_rows_to_cpu",
            side_effect=fake_copy,
        ):
            tracker.record_pre_update_micro_batch(
                "math",
                FakeMicroBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(copied_indices, [[0, 1, 2]])
        self.assertEqual(len(tracker._token_gradient_candidates["math"]), 1)
        cached_rows = [
            row
            for candidate in tracker._token_gradient_candidates["math"]
            for row in candidate["tokens"]
        ]
        self.assertEqual({row["sample_id"] for row in cached_rows}, {"sample-0", "sample-1", "sample-2"})
        self.assertEqual([row["sample_index"] for row in cached_rows], [0, 0, 1, 2])
        self.assertEqual([row["original_sample_index"] for row in cached_rows], [0, 0, 1, 2])

    def test_sequential_tracker_disables_sample_norm_for_sharded_params(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": -1}}

        with patch("mopd_verl.full_gradient.tracker._distributed_world_size", return_value=2):
            tracker = SequentialBackwardDomainGradientTracker(
                ToyActor(),
                {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "sample_gradient_enabled": True,
                    "sample_gradient_norm_enabled": True,
                    "sample_gradient_cos_enabled": False,
                },
            )

        self.assertFalse(tracker.sample_norm_enabled)
        self.assertFalse(tracker.sample_log_sample_level)
        self.assertTrue(tracker._sample_gradient_distributed_unsupported)
        self.assertTrue(tracker._sample_gradient_norm_distributed_unsupported)

    def test_sequential_tracker_disables_sample_gradient_when_fsdp_size_is_not_one(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": -1}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_norm_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )

        self.assertFalse(tracker.sample_norm_enabled)
        self.assertFalse(tracker.sample_cos_enabled)
        self.assertTrue(tracker._sample_gradient_distributed_unsupported)
        self.assertTrue(tracker._sample_gradient_norm_distributed_unsupported)
        self.assertTrue(tracker._sample_gradient_cos_distributed_unsupported)

    def test_sequential_tracker_allows_token_sequence_replay_for_fsdp_size_two(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 2}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "token_gradient_enabled": True,
            },
        )

        self.assertTrue(tracker.token_gradient_enabled)
        self.assertTrue(tracker._token_gradient_sequence_replay_supported)
        self.assertFalse(tracker._token_gradient_distributed_unsupported)
        self.assertFalse(tracker._sample_gradient_distributed_unsupported)

    def test_sequential_tracker_configures_domain_sum_training_source(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 1}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "storage_dtype": "bfloat16",
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "sequence_replay_skip_non_target_domains": True,
                "training_gradient_from_domain_sum_enabled": True,
            },
        )

        self.assertTrue(tracker.sequence_replay_skip_non_target_domains)
        self.assertEqual(tracker.sequence_masked_target_closure_rel_l2_threshold, 0.02)
        self.assertTrue(tracker.training_gradient_from_domain_sum_enabled)
        self.assertEqual(tracker._domain_target_storage_dtype(), "float32")

    def test_domain_sum_training_refuses_untrusted_sequence_target(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.config = {"fsdp_config": {"fsdp_size": 1}}
                self.scaler = None

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = None
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "enabled": True,
                "domains": ["math"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "training_gradient_from_domain_sum_enabled": True,
            },
        )
        tracker._last_domain_targets_for_training = {"math": ((torch.tensor([3.0]),), 9.0)}
        tracker._last_domain_target_source_for_training = 4.0
        tracker._last_domain_targets_for_training_trusted = False

        applied, metrics = tracker.apply_domain_sum_gradient_for_training()

        self.assertFalse(applied)
        self.assertIsNone(parameter.grad)
        self.assertEqual(metrics["global/audit/training_gradient_from_domain_sum_target_trusted"], 0.0)
        self.assertEqual(metrics["global/audit/training_gradient_from_domain_sum_untrusted_target"], 1.0)

    def test_sequential_tracker_disables_token_gradient_without_sequence_primary_for_shards(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 2}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": False,
                "token_gradient_enabled": True,
            },
        )

        self.assertFalse(tracker.token_gradient_enabled)
        self.assertFalse(tracker._token_gradient_sequence_replay_supported)
        self.assertTrue(tracker._token_gradient_distributed_unsupported)

    def test_token_sequence_replay_caches_candidates_for_fsdp_size_two(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 2}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            config = ActorConfig()

        class FakeMicroBatch:
            batch = None
            non_tensor_batch = {"domain": ["math", "math"], "opd_teacher": ["math", "math"]}

            def __len__(self) -> int:
                return 2

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math"],
                "sequence_masked_target_enabled": True,
                "sequence_masked_target_use_as_primary": True,
                "token_gradient_enabled": True,
            },
        )
        tracker.start_mini_batch()

        token_rows = [
            {"sample_id": "sample-0", "sample_index": 0, "position": 3, "gap_abs": 9.0},
            {"sample_id": "sample-1", "sample_index": 1, "position": 2, "gap_abs": 7.0},
        ]
        copied_indices: list[list[int]] = []

        def fake_copy(_micro_batch: object, indices: list[int]) -> object:
            copied_indices.append(indices)
            return {"indices": indices}

        with patch.object(
            tracker,
            "_select_token_gradient_candidates",
            return_value=token_rows,
        ), patch(
            "mopd_verl.full_gradient.tracker._copy_data_proto_rows_to_cpu",
            side_effect=fake_copy,
        ):
            tracker.record_pre_update_micro_batch(
                "math",
                FakeMicroBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(copied_indices, [[0, 1]])
        self.assertEqual(len(tracker._token_gradient_candidates["math"]), 1)
        cached_rows = tracker._token_gradient_candidates["math"][0]["tokens"]
        self.assertEqual([row["sample_id"] for row in cached_rows], ["sample-0", "sample-1"])

    def test_balance_batch_logs_seqlen_tokens_and_workload_separately(self) -> None:
        trainer_path = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "verl"
            / "verl"
            / "trainer"
            / "ppo"
            / "ray_trainer.py"
        )
        source = trainer_path.read_text(encoding="utf-8")

        self.assertIn("global_workload_lst = calculate_workload(global_seqlen_lst)", source)
        self.assertIn('prefix=logging_prefix.replace("seqlen", "workload")', source)
        self.assertNotIn("global_seqlen_lst = calculate_workload(global_seqlen_lst)", source)

    def test_dp_actor_uses_pre_update_gradient_tracker(self) -> None:
        actor_path = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "verl"
            / "verl"
            / "workers"
            / "actor"
            / "dp_actor.py"
        )
        source = actor_path.read_text(encoding="utf-8")
        patch_source = (
            Path(__file__).resolve().parents[1] / "scripts" / "apply_gopd_audit_patch.py"
        ).read_text(encoding="utf-8")
        fsdp_worker_source = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "verl"
            / "verl"
            / "workers"
            / "fsdp_workers.py"
        ).read_text(encoding="utf-8")
        gradient_worker_source = (
            Path(__file__).resolve().parents[1] / "mopd_verl" / "full_gradient_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn("SequentialBackwardDomainGradientTracker", source)
        self.assertNotIn("mopd_gradient_tracker.capture_micro_batch(", source)
        pre_update_anchor = (
            "self.actor_optimizer.zero_grad()\n"
            "                # MOPD audit: domain-gradient tracker begin\n"
            "                if mopd_gradient_tracker is not None:\n"
            "                    append_to_dict(\n"
            "                        metrics,\n"
            "                        mopd_gradient_tracker.run_pre_update_audit("
        )
        self.assertIn(pre_update_anchor, source)
        self.assertIn("apply_domain_sum_gradient_for_training()", source)
        self.assertIn("training_gradient_from_domain_sum_skipped_backward", source)
        self.assertIn("full_grad_training_parity_metrics()", source)
        self.assertIn("run_pre_update_audit(", patch_source)
        self.assertIn("apply_domain_sum_gradient_for_training()", patch_source)
        self.assertIn("training_gradient_from_domain_sum_skipped_backward", patch_source)
        self.assertIn("full_grad_training_parity_metrics()", patch_source)
        self.assertIn("SequentialBackwardDomainGradientTracker", patch_source)
        self.assertNotIn("mopd_gradient_tracker.before_backward(", source)
        self.assertNotIn("mopd_gradient_tracker.after_backward(", source)
        self.assertNotIn("mopd_gradient_tracker.capture_micro_batch(", patch_source)
        self.assertIn(
            'gradient_checkpointing_kwargs={"use_reentrant": False}',
            fsdp_worker_source,
        )
        self.assertNotIn("_grad_stats_from_true_backward", gradient_worker_source)
        self.assertNotIn("sample_recompute_used_true_backward_fallback", gradient_worker_source)

    def test_sample_gradient_cos_uses_all_candidates(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
                "sample_gradient_backward_sync_enabled": False,
            },
        )
        candidates: list[dict[str, object]] = [
            {
                "row": {
                    "sample_grad_norm": float(index),
                    "loss_scale_factor": 1.0,
                    "on_policy": True,
                    "computed_for_cos": False,
                },
                "micro_batch": object(),
            }
            for index in range(20)
        ]
        tracker._sample_candidates = {"math": candidates}
        processed: list[object] = []

        def fake_recompute(micro_batch: object, **_: object) -> dict[str, float]:
            processed.append(micro_batch)
            return {
                "sample_to_domain_cos": 1.0,
                "sample_projection_share": 0.05,
                "sample_recompute_grad_norm": 1.0,
            }

        tracker._recompute_sample_to_domain_stats = fake_recompute
        metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(processed, [candidate["micro_batch"] for candidate in candidates])
        self.assertEqual(metrics["math/sample_grad_cos/sample_count"], 20.0)
        self.assertEqual(metrics["math/sample_grad_cos/attempted_count"], 20.0)
        self.assertEqual(metrics["math/sample_grad_cos/unavailable_count"], 0.0)
        self.assertEqual(metrics["math/sample_grad_cos/valid_frac"], 1.0)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/projection_share_sum"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["math/sample_grad_contribution/projection_share_normalized_sum"],
            1.0,
            places=6,
        )
        self.assertTrue(
            all(
                abs(candidate["row"]["sample_projection_share_normalized"] - 0.05) < 1e-9
                for candidate in candidates
            )
        )
        self.assertAlmostEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_error"],
            0.0,
            places=6,
        )
        self.assertTrue(all(candidate["row"]["computed_for_cos"] for candidate in candidates))

    def test_sample_gradient_cos_metrics_gather_global_values(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        tracker._sample_candidates = {
            "math": [
                {
                    "row": {
                        "loss_scale_factor": 1.0,
                        "on_policy": True,
                        "computed_for_cos": False,
                    },
                    "micro_batch": object(),
                }
            ]
        }
        tracker._recompute_sample_to_domain_stats = lambda *_args, **_kwargs: {
            "sample_to_domain_cos": 0.25,
            "sample_projection_share": 2.0,
            "sample_recompute_grad_norm": 1.0,
        }

        def gather_values(values: object) -> list[float]:
            gathered = list(values)
            if gathered == [0.25]:
                return [0.25, 0.75]
            if gathered == [2.0]:
                return [2.0, 3.0]
            return gathered

        with patch(
            "mopd_verl.full_gradient.tracker._all_gather_list",
            side_effect=gather_values,
        ):
            metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(metrics["math/sample_grad_cos/sample_count"], 2.0)
        self.assertAlmostEqual(metrics["math/sample_grad_cos/domain_cos_mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/projection_share_sum"], 5.0, places=6)
        self.assertAlmostEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_error"],
            4.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["math/sample_grad_contribution/projection_share_normalized_sum"],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["math/sample_grad_contribution/projection_share_normalized_max"],
            0.6,
            places=6,
        )
        self.assertAlmostEqual(
            tracker._sample_candidates["math"][0]["row"]["sample_projection_share_normalized"],
            0.4,
            places=6,
        )

    def test_sequential_tracker_fallback_sample_id_includes_rank(self) -> None:
        try:
            import torch

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class FakeBatch:
            batch = {"response_mask": torch.ones(1, 2)}
            non_tensor_batch = {"domain": ["math"], "opd_teacher": ["math"]}

            def __len__(self) -> int:
                return 1

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_norm_enabled": True,
            },
        )

        with patch("mopd_verl.full_gradient.tracker._distributed_rank", return_value=7):
            tracker.record_pre_update_micro_batch(
                "math",
                FakeBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
            )

        assert tracker._sample_records
        self.assertEqual(
            tracker._sample_records[0]["sample_id"],
            "step0:rank7:micro0:row0",
        )

    def test_sample_gradient_projection_sum_uses_replica_scaled_share(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 1}}

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        tracker._sample_candidates = {
            "math": [
                {
                    "row": {
                        "loss_scale_factor": 1.0,
                        "on_policy": True,
                        "computed_for_cos": False,
                    },
                    "micro_batch": object(),
                }
            ]
        }
        tracker._recompute_sample_to_domain_stats = lambda *_args, **_kwargs: {
            "sample_to_domain_cos": 1.0,
            "sample_projection_share": 0.5,
            "sample_projection_share_raw": 1.0,
            "sample_projection_share_scale": 0.5,
            "sample_recompute_grad_norm": 1.0,
        }

        with patch(
            "mopd_verl.full_gradient.tracker._all_gather_list",
            side_effect=lambda values: list(values) + list(values),
        ), patch(
            "mopd_verl.full_gradient.tracker._gradient_replica_count",
            return_value=2,
        ):
            metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_across_replicas"],
            2.0,
        )
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_raw"],
            2.0,
        )
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_raw_expected"],
            2.0,
        )
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_raw_error"],
            0.0,
        )
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_replica_count"],
            2.0,
        )
        self.assertEqual(metrics["math/sample_grad_contribution/projection_share_sum"], 1.0)
        self.assertEqual(metrics["math/sample_grad_contribution/projection_share_sum_error"], 0.0)
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_normalized_sum"],
            1.0,
        )
        self.assertEqual(metrics["math/sample_grad_closure/projection_share_sum"], 1.0)
        self.assertEqual(metrics["math/sample_grad_closure/projection_share_normalized_sum"], 1.0)
        self.assertEqual(metrics["math/sample_grad_closure/projection_share_sum_raw"], 2.0)
        self.assertEqual(metrics["math/sample_grad_closure/projection_share_sum_raw_expected"], 2.0)
        self.assertEqual(metrics["math/sample_grad_closure/projection_share_sum_raw_error"], 0.0)

    def test_sample_gradient_cos_caches_structural_autograd_unavailability(self) -> None:
        try:
            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        tracker._sample_candidates = {
            "math": [
                {
                    "row": {
                        "loss_scale_factor": 1.0,
                        "on_policy": True,
                        "computed_for_cos": False,
                    },
                    "micro_batch": object(),
                }
                for _ in range(3)
            ]
        }
        recompute_count = 0

        def disconnected_recompute(*_args: object, **_kwargs: object) -> dict[str, object]:
            nonlocal recompute_count
            recompute_count += 1
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_recompute_grad_norm": 0.0,
                "sample_recompute_non_none_grad_count": 0.0,
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": "all_parameters_disconnected",
            }

        tracker._recompute_sample_to_domain_stats = disconnected_recompute
        metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(recompute_count, 1)
        self.assertEqual(metrics["math/sample_grad_cos/attempted_count"], 3.0)
        self.assertEqual(metrics["math/sample_grad_cos/unavailable_count"], 3.0)
        self.assertEqual(metrics["math/sample_grad_cos/all_parameters_disconnected_count"], 3.0)
        self.assertTrue(
            all(
                candidate["row"]["sample_recompute_autograd_error"]
                == "all_parameters_disconnected"
                for candidate in tracker._sample_candidates["math"]
            )
        )

    def test_low_precision_gradient_storage_uses_fp32_cosine_accumulation(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import (
                SequentialBackwardDomainGradientTracker,
                _current_grad_difference_snapshot,
                _storage_dtype,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(3, 1, bias=False)

        self.assertEqual(_storage_dtype("fp16"), torch.float16)
        self.assertEqual(_storage_dtype("bf16"), torch.bfloat16)
        actor = ToyActor()
        tracker = SequentialBackwardDomainGradientTracker(actor, {"domains": ["math", "code"]})
        gradient = torch.tensor([0.125, -0.25, 0.5], dtype=torch.bfloat16)
        target = torch.tensor([0.5, 0.25, -0.125], dtype=torch.bfloat16)

        norm_sq, dot = tracker._grad_stats_from_tensors((gradient,), (target,))
        expected_gradient = gradient.float()
        expected_target = target.float()

        self.assertAlmostEqual(norm_sq, torch.dot(expected_gradient, expected_gradient).item(), places=7)
        self.assertAlmostEqual(dot, torch.dot(expected_gradient, expected_target).item(), places=7)

        parameter = next(actor.actor_module.parameters())
        parameter.grad = gradient.float().reshape_as(parameter)
        original_grad = parameter.grad.clone()
        snapshot = _current_grad_difference_snapshot(actor, (target,), "bfloat16")

        self.assertIsNotNone(snapshot)
        self.assertIsNotNone(snapshot.second_chunks)
        second_chunk = snapshot.second_chunks[0]
        self.assertEqual(second_chunk.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(parameter.grad, original_grad))
        expected_second = gradient.float() - target.float()
        self.assertTrue(torch.allclose(second_chunk.float(), expected_second, atol=2e-3))
        self.assertAlmostEqual(
            snapshot.second_target_norm_sq,
            torch.dot(second_chunk.float(), second_chunk.float()).item(),
            places=7,
        )

    def test_sequential_bfloat16_tracker_keeps_two_domain_targets_without_touching_grad(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "enabled": True,
                "domains": ["math", "code"],
                "storage_dtype": "bfloat16",
                "sample_gradient_enabled": True,
                "sample_gradient_norm_enabled": False,
                "sample_gradient_cos_enabled": True,
            },
        )
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[3.0, 4.0]])
        total_grad = parameter.grad.clone()
        domain_targets = {
            "math": ((torch.tensor([1.0, 0.0], dtype=torch.bfloat16),), 1.0),
            "code": ((torch.tensor([0.0, 1.0], dtype=torch.bfloat16),), 1.0),
        }

        with patch(
            "mopd_verl.full_gradient.tracker.get_torch_device",
            return_value=SimpleNamespace(max_memory_allocated=lambda: 0),
        ):
            metrics, captured_targets = tracker._finish_direct_domain_gradient_metrics(domain_targets)

        self.assertEqual(set(captured_targets), {"math", "code"})
        self.assertTrue(
            all(chunk.dtype == torch.bfloat16 for chunks, _ in captured_targets.values() for chunk in chunks)
        )
        self.assertIn("global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k", metrics)
        self.assertTrue(torch.equal(next(actor.actor_module.parameters()).grad, total_grad))
        self.assertFalse(hasattr(tracker, "_sample_restore_grad_chunks"))

    def test_zero_sample_autograd_gradient_does_not_replace_training_gradient(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[3.0, 4.0]])
        original_grad = parameter.grad.clone()
        tracker = SequentialBackwardDomainGradientTracker(actor, {"domains": ["math", "code"]})
        zero_loss = (parameter * 0.0).sum()

        with patch("mopd_verl.full_gradient.tracker._actor_micro_batch_loss", return_value=zero_loss):
            stats = tracker._recompute_sample_to_domain_stats(
                object(),
                target_chunks=(torch.ones(parameter.numel(), dtype=torch.bfloat16),),
                target_norm=parameter.numel() ** 0.5,
                target_norm_sq=float(parameter.numel()),
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertTrue(torch.equal(parameter.grad, original_grad))
        self.assertEqual(stats["sample_recompute_grad_norm"], 0.0)
        self.assertIsNone(stats["sample_to_domain_cos"])
        self.assertIsNone(stats["sample_projection_share"])
        self.assertEqual(stats["sample_recompute_available"], 0.0)
        self.assertEqual(tracker._sample_zero_norm_count, 1)

    def test_disconnected_sample_autograd_is_not_reported_as_zero_norm(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        tracker = SequentialBackwardDomainGradientTracker(actor, {"domains": ["math", "code"]})
        disconnected_loss = torch.tensor(1.0, requires_grad=True)

        with patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_loss",
            return_value=disconnected_loss,
        ):
            stats = tracker._recompute_sample_to_domain_stats(
                object(),
                target_chunks=(torch.tensor([1.0]),),
                target_norm=1.0,
                target_norm_sq=1.0,
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(
            stats["sample_recompute_autograd_error"],
            "all_parameters_disconnected",
        )
        self.assertEqual(stats["sample_recompute_non_none_grad_count"], 0.0)
        self.assertEqual(tracker._sample_zero_norm_count, 0)

    def test_sample_recompute_matches_training_on_policy_old_log_prob(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import _actor_micro_batch_loss
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class PolicyLossConfig:
            loss_mode = "vanilla"
            only_reverse_kl_advantages = True

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ActorConfig:
            clip_ratio = 0.2
            clip_ratio_low = None
            clip_ratio_high = None
            entropy_coeff = 0.0
            loss_agg_mode = "token-mean"
            use_kl_loss = False
            policy_loss = PolicyLossConfig()

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_module.weight.data.zero_()
                self.config = ActorConfig()

            def _forward_micro_batch(
                self,
                model_inputs: dict[str, torch.Tensor],
                *,
                temperature: float,
                calculate_entropy: bool,
                **_kwargs: object,
            ) -> tuple[None, torch.Tensor]:
                del temperature, calculate_entropy
                log_prob = self.actor_module.weight.reshape(1, 1).expand_as(model_inputs["response_mask"])
                return None, log_prob

        class FakeBatch:
            def __init__(self) -> None:
                self.batch = {
                    "response_mask": torch.ones((1, 1), dtype=torch.float32),
                    "old_log_probs": torch.full((1, 1), 0.1, dtype=torch.float32),
                    "math_teacher_log_prob": torch.full((1, 1), -0.5, dtype=torch.float32),
                    "advantages": torch.zeros((1, 1), dtype=torch.float32),
                }
                self.non_tensor_batch = {}
                self.meta_info = {"temperature": 1.0}

            def to(self, device: object) -> "FakeBatch":
                del device
                return self

        actor = ToyActor()
        with patch("mopd_verl.full_gradient.actor_loss.get_device_id", return_value="cpu"):
            loss = _actor_micro_batch_loss(actor, FakeBatch(), loss_scale_factor=1.0, on_policy=True)
        gradient = torch.autograd.grad(loss, tuple(actor.actor_module.parameters()))[0]

        self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertGreater(float(gradient.item()), 0.0)

        actor.actor_module.weight.data.zero_()
        with patch("mopd_verl.full_gradient.actor_loss.get_device_id", return_value="cpu"):
            off_policy_loss = _actor_micro_batch_loss(
                actor,
                FakeBatch(),
                loss_scale_factor=1.0,
                on_policy=False,
            )
        off_policy_gradient = torch.autograd.grad(off_policy_loss, tuple(actor.actor_module.parameters()))[0]

        self.assertGreater(abs(float(off_policy_gradient.item()) - float(gradient.item())), 1e-4)

    def test_sample_recompute_disables_inplace_logprob_backward(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import _actor_micro_batch_loss
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class PolicyLossConfig:
            loss_mode = "vanilla"
            only_reverse_kl_advantages = True

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ActorConfig:
            clip_ratio = 0.2
            clip_ratio_low = None
            clip_ratio_high = None
            entropy_coeff = 0.0
            loss_agg_mode = "token-mean"
            use_kl_loss = False
            policy_loss = PolicyLossConfig()

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_module.weight.data.zero_()
                self.config = ActorConfig()
                self.inplace_backward_flags: list[object] = []

            def _forward_micro_batch(
                self,
                model_inputs: dict[str, torch.Tensor],
                *,
                temperature: float,
                calculate_entropy: bool,
                inplace_backward: object = None,
                **_kwargs: object,
            ) -> tuple[None, torch.Tensor]:
                del temperature, calculate_entropy
                self.inplace_backward_flags.append(inplace_backward)
                log_prob = self.actor_module.weight.reshape(1, 1).expand_as(model_inputs["response_mask"])
                return None, log_prob

        class FakeBatch:
            def __init__(self) -> None:
                self.batch = {
                    "response_mask": torch.ones((1, 1), dtype=torch.float32),
                    "old_log_probs": torch.full((1, 1), 0.1, dtype=torch.float32),
                    "math_teacher_log_prob": torch.full((1, 1), -0.5, dtype=torch.float32),
                    "advantages": torch.zeros((1, 1), dtype=torch.float32),
                }
                self.non_tensor_batch = {}
                self.meta_info = {"temperature": 1.0}

            def to(self, device: object) -> "FakeBatch":
                del device
                return self

        actor = ToyActor()
        with patch("mopd_verl.full_gradient.actor_loss.get_device_id", return_value="cpu"):
            loss = _actor_micro_batch_loss(
                actor,
                FakeBatch(),
                loss_scale_factor=1.0,
                on_policy=True,
                safe_logprob_backward=True,
            )
        gradient = torch.autograd.grad(loss, tuple(actor.actor_module.parameters()))[0]

        self.assertEqual(actor.inplace_backward_flags, [False])
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_sample_recompute_projection_shares_reconstruct_domain_gradient(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import (
                SequentialBackwardDomainGradientTracker,
                _actor_micro_batch_loss,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class PolicyLossConfig:
            loss_mode = "vanilla"
            only_reverse_kl_advantages = True

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ActorConfig:
            clip_ratio = 0.2
            clip_ratio_low = None
            clip_ratio_high = None
            entropy_coeff = 0.0
            loss_agg_mode = "token-mean"
            use_kl_loss = False
            policy_loss = PolicyLossConfig()
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_module.weight.data.zero_()
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()

            def _forward_micro_batch(
                self,
                model_inputs: dict[str, torch.Tensor],
                *,
                temperature: float,
                calculate_entropy: bool,
                inplace_backward: object = None,
                **_kwargs: object,
            ) -> tuple[None, torch.Tensor]:
                del temperature, calculate_entropy, inplace_backward
                return None, self.actor_module(model_inputs["features"])

        class FakeBatch:
            def __init__(self, features: list[float], teacher_log_prob: float) -> None:
                self.batch = {
                    "features": torch.tensor([features], dtype=torch.float32),
                    "response_mask": torch.ones((1, 1), dtype=torch.float32),
                    "old_log_probs": torch.zeros((1, 1), dtype=torch.float32),
                    "math_teacher_log_prob": torch.full(
                        (1, 1),
                        teacher_log_prob,
                        dtype=torch.float32,
                    ),
                    "advantages": torch.zeros((1, 1), dtype=torch.float32),
                }
                self.non_tensor_batch = {}
                self.meta_info = {"temperature": 1.0}

            def to(self, device: object) -> "FakeBatch":
                del device
                return self

        actor = ToyActor()
        batches = [
            FakeBatch([1.0, 0.0], -0.5),
            FakeBatch([0.0, 1.0], -1.0),
        ]
        parameter = next(actor.actor_module.parameters())
        actor.actor_optimizer.zero_grad()
        with patch("mopd_verl.full_gradient.actor_loss.get_device_id", return_value="cpu"):
            for batch in batches:
                loss = _actor_micro_batch_loss(
                    actor,
                    batch,
                    loss_scale_factor=0.5,
                    on_policy=True,
                )
                loss.backward()
        domain_gradient = parameter.grad.detach().reshape(-1).clone()
        target_norm_sq = float(torch.dot(domain_gradient, domain_gradient).item())
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"), patch(
            "mopd_verl.full_gradient.actor_loss.get_device_id",
            return_value="cpu",
        ):
            stats = [
                tracker._recompute_sample_to_domain_stats(
                    batch,
                    target_chunks=(domain_gradient,),
                    target_norm=target_norm_sq**0.5,
                    target_norm_sq=target_norm_sq,
                    loss_scale_factor=0.5,
                    on_policy=True,
                )
                for batch in batches
            ]

        projection_sum = sum(float(item["sample_projection_share"]) for item in stats)
        self.assertAlmostEqual(projection_sum, 1.0, places=6)
        self.assertTrue(all(item["sample_recompute_available"] == 1.0 for item in stats))
        self.assertTrue(torch.equal(parameter.grad.reshape(-1), domain_gradient))

    def test_sample_recompute_scales_local_projection_share_by_replica_count(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            loss_agg_mode = "token-mean"
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_module.weight.data.fill_(1.0)
                self.config = ActorConfig()

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        target_gradient = torch.tensor([1.0])

        with patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_loss",
            return_value=parameter.sum(),
        ), patch(
            "mopd_verl.full_gradient.tracker._gradient_replica_count",
            return_value=2,
        ):
            stats = tracker._recompute_sample_to_domain_stats(
                object(),
                target_chunks=(target_gradient,),
                target_norm=2.0**0.5,
                target_norm_sq=2.0,
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(stats["sample_recompute_backward_sync_used"], 0.0)
        self.assertEqual(stats["sample_recompute_replica_count"], 2.0)
        self.assertAlmostEqual(stats["sample_to_domain_cos"], 1.0, places=6)
        self.assertAlmostEqual(stats["sample_projection_share_raw"], 1.0, places=6)
        self.assertAlmostEqual(stats["sample_projection_share"], 0.5, places=6)
        self.assertAlmostEqual(stats["sample_projection_share_scale"], 0.5, places=6)
        self.assertAlmostEqual(stats["sample_recompute_grad_norm_raw"], 1.0, places=6)
        self.assertAlmostEqual(stats["sample_recompute_grad_norm"], 0.5, places=6)
        self.assertIsNone(parameter.grad)

    def test_token_gradient_keeps_domain_targets_without_sample_cos(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        first_domain_gradient = torch.tensor([1.0, 0.0], dtype=torch.float32)
        parameter.grad = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": False,
                "token_gradient_enabled": True,
            },
        )

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"):
            _metrics, domain_targets = tracker._finish_domain_gradient_metrics((first_domain_gradient,))

        self.assertIn("math", domain_targets)
        self.assertIn("code", domain_targets)
        code_chunks, code_norm_sq = domain_targets["code"]
        self.assertAlmostEqual(code_norm_sq, 4.0, places=6)
        self.assertTrue(torch.equal(code_chunks[0].float(), torch.tensor([0.0, -2.0])))

    def test_direct_domain_gradient_metrics_use_recomputed_targets(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {"domains": ["math", "code"], "domain_gradient_enabled": True},
        )
        tracker._sample_counts = {"math": 1, "code": 1}
        targets = {
            "math": ((torch.tensor([1.0, 0.0], dtype=torch.float32),), 1.0),
            "code": ((torch.tensor([0.0, 2.0], dtype=torch.float32),), 4.0),
        }

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"):
            metrics, returned_targets = tracker._finish_direct_domain_gradient_metrics(targets)

        self.assertIs(returned_targets, targets)
        self.assertEqual(metrics["global/audit/full_gradient_domain_direct_recompute_used"], 1.0)
        self.assertAlmostEqual(metrics["math/full_grad/grad_norm"], 1.0, places=6)
        self.assertAlmostEqual(metrics["code/full_grad/grad_norm"], 2.0, places=6)
        self.assertAlmostEqual(
            metrics["global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_contribution/math_to_total/signed_projection_share"],
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_contribution/code_to_total/signed_projection_share"],
            0.8,
            places=6,
        )

    def test_token_gradient_metrics_restore_original_training_grad(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[7.0]], dtype=torch.float32)
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {"domains": ["math"], "token_gradient_enabled": True},
        )
        tracker._token_gradient_candidates = {
            "math": [
                {
                    "context": {"loss_scale_factor": 1.0, "on_policy": True},
                    "micro_batch": object(),
                    "tokens": [
                        {
                            "sample_id": "s0",
                            "sample_index": 0,
                            "position": 0,
                            "gap": 1.0,
                            "gap_abs": 1.0,
                            "loss_abs": 1.0,
                        }
                    ],
                }
            ]
        }

        def fake_recompute(*_args: object, **_kwargs: object) -> dict[str, object]:
            parameter.grad = torch.tensor([[99.0]], dtype=torch.float32)
            return {
                "token_grad_available": 1.0,
                "token_grad_norm": 1.0,
                "token_grad_non_none_grad_count": 1.0,
                "token_grad_param_count": 1.0,
                "token_grad_none_grad_count": 0.0,
                "token_grad_autograd_error": None,
                "math_cos": 1.0,
                "math_projection_share": 1.0,
            }

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"), patch(
            "mopd_verl.full_gradient.tracker._write_jsonl_rows"
        ), patch.object(tracker, "_recompute_token_selection_gradient_stats", side_effect=fake_recompute):
            tracker._token_gradient_metrics(
                {"math": ((torch.tensor([1.0], dtype=torch.float32),), 1.0)}
            )

        self.assertTrue(torch.equal(parameter.grad, torch.tensor([[7.0]], dtype=torch.float32)))

    def test_token_gradient_respects_storage_dtype_and_restores_grad_dtype(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import (
                SequentialBackwardDomainGradientTracker,
                _restore_parameter_grads_from_targets,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False).to(dtype=torch.bfloat16)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        try:
            parameter.grad = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            expected_grad_dtype = torch.float32
            grad_dtypes = (parameter.grad.dtype,)
        except RuntimeError:
            parameter.grad = torch.tensor([[1.0, 2.0]], dtype=parameter.dtype)
            expected_grad_dtype = parameter.dtype
            grad_dtypes = (torch.float32,)
        target_map = {
            "math": ((torch.tensor([0.25, 0.50], dtype=torch.float32),), 0.3125),
            "code": ((torch.tensor([0.75, 1.50], dtype=torch.float32),), 2.8125),
        }
        _restore_parameter_grads_from_targets((parameter,), target_map, grad_dtypes=grad_dtypes)

        self.assertEqual(parameter.grad.dtype, expected_grad_dtype)
        self.assertTrue(torch.allclose(parameter.grad.float(), torch.tensor([[1.0, 2.0]], dtype=torch.float32)))

        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "storage_dtype": "bfloat16",
                "token_gradient_enabled": True,
            },
        )
        self.assertEqual(tracker._domain_target_storage_dtype(), "bfloat16")

    def test_token_gradient_strict_restore_uses_original_grad_snapshot(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import (
                SequentialBackwardDomainGradientTracker,
                _parameter_grad_snapshot_diff_stats,
                _restore_parameter_grads_from_snapshot,
                _restore_parameter_grads_from_targets,
                _snapshot_parameter_grads,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[1.25, -2.50]], dtype=torch.float32)
        grad_snapshot = _snapshot_parameter_grads((parameter,))
        target_map = {
            "math": ((torch.tensor([10.0, 20.0], dtype=torch.float32),), 500.0),
        }

        _restore_parameter_grads_from_targets((parameter,), target_map)
        self.assertFalse(torch.equal(parameter.grad, grad_snapshot[0]))

        _restore_parameter_grads_from_snapshot((parameter,), grad_snapshot)
        diff_stats = _parameter_grad_snapshot_diff_stats((parameter,), grad_snapshot)
        self.assertEqual(parameter.grad.dtype, torch.float32)
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([[1.25, -2.50]], dtype=torch.float32)))
        self.assertEqual(diff_stats["rel_l2"], 0.0)
        self.assertEqual(diff_stats["max_abs"], 0.0)

        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "token_gradient_enabled": True,
                "token_gradient_strict_grad_restore": True,
            },
        )
        self.assertTrue(tracker.token_gradient_strict_grad_restore)

    def test_sample_recompute_uses_backward_and_restores_training_grad_when_zero(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[7.0]])
        training_grad = parameter.grad.detach().clone()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        backward_calls = 0
        original_backward = torch.Tensor.backward

        def checked_backward(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
            nonlocal backward_calls
            backward_calls += 1
            return original_backward(tensor, *args, **kwargs)

        with patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_loss",
            side_effect=lambda *_args, **_kwargs: actor.actor_module(torch.tensor([[2.0]])).sum() * 0.0,
        ), patch.object(torch.Tensor, "backward", checked_backward):
            stats = tracker._recompute_sample_to_domain_stats(
                object(),
                target_chunks=(torch.tensor([1.0], dtype=torch.float32),),
                target_norm=1.0,
                target_norm_sq=1.0,
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(backward_calls, 1)
        self.assertEqual(float(stats["sample_recompute_grad_norm"]), 0.0)
        self.assertEqual(stats["sample_recompute_non_none_grad_count"], 1.0)
        self.assertEqual(stats["sample_recompute_available"], 0.0)
        self.assertIsNone(stats["sample_projection_share"])
        self.assertTrue(torch.equal(parameter.grad, training_grad))

    def test_token_selection_recompute_does_not_call_backward(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = {"fsdp_config": {"fsdp_size": 1}}

        class FakeBatch:
            def __init__(self) -> None:
                self.batch = {"response_mask": torch.ones((1, 1), dtype=torch.float32)}
                self.non_tensor_batch = {}
                self.meta_info = {"temperature": 1.0}

            def to(self, device: object) -> "FakeBatch":
                del device
                return self

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.tensor([[7.0]])
        training_grad = parameter.grad.detach().clone()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "token_gradient_enabled": True,
            },
        )
        selected_tokens = [
            {
                "candidate_index": 0,
                "micro_batch": FakeBatch(),
                "sample_index": 0,
                "position": 0,
                "loss_scale_factor": 1.0,
                "on_policy": True,
            }
        ]

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"), patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_loss",
            side_effect=lambda *_args, **_kwargs: actor.actor_module(torch.tensor([[2.0]])).sum(),
        ), patch.object(
            torch.Tensor,
            "backward",
            side_effect=AssertionError("token selection recompute must never call loss.backward()"),
        ), patch(
            "mopd_verl.full_gradient.tracker._finalize_fsdp_after_auxiliary_backward"
        ) as finalize_fsdp:
            stats = tracker._recompute_token_selection_gradient_stats(
                selected_tokens,
                target_map={"math": ((torch.tensor([1.0], dtype=torch.float32),), 1.0)},
                restore_grads=False,
            )

        self.assertEqual(stats["token_grad_available"], 1.0)
        self.assertEqual(stats["token_grad_non_none_grad_count"], 1.0)
        self.assertEqual(stats["token_grad_param_count"], 1.0)
        self.assertEqual(stats["token_grad_none_grad_count"], 0.0)
        self.assertIsNone(stats["token_grad_autograd_error"])
        self.assertTrue(torch.equal(parameter.grad, training_grad))
        finalize_fsdp.assert_called_once_with(actor)

        with patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"), patch(
            "mopd_verl.full_gradient.tracker._actor_micro_batch_loss",
            side_effect=lambda *_args, **_kwargs: actor.actor_module(torch.tensor([[2.0]])).sum(),
        ), patch(
            "torch.autograd.grad",
            return_value=(None,),
        ), patch(
            "mopd_verl.full_gradient.tracker._finalize_fsdp_after_auxiliary_backward"
        ) as finalize_fsdp_none:
            disconnected_stats = tracker._recompute_token_selection_gradient_stats(
                selected_tokens,
                target_map={"math": ((torch.tensor([1.0], dtype=torch.float32),), 1.0)},
                restore_grads=False,
            )

        self.assertEqual(disconnected_stats["token_grad_available"], 0.0)
        self.assertEqual(disconnected_stats["token_grad_non_none_grad_count"], 0.0)
        self.assertEqual(disconnected_stats["token_grad_param_count"], 1.0)
        self.assertEqual(disconnected_stats["token_grad_none_grad_count"], 1.0)
        self.assertEqual(disconnected_stats["token_grad_autograd_error"], "all_parameters_disconnected")
        self.assertTrue(torch.equal(parameter.grad, training_grad))
        finalize_fsdp_none.assert_called_once_with(actor)

    def test_auxiliary_recompute_avoids_selected_topk_context(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient.tracker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class PolicyLossConfig:
            topk_distill_enabled = True

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ActorConfig:
            fsdp_config = {"fsdp_size": 1}
            policy_loss = PolicyLossConfig()
            use_kl_loss = False
            kl_loss_coef = 0.0
            entropy_coeff = 0.0

            def get(self, key: str, default: object = None) -> object:
                return getattr(self, key, default)

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)
                self.config = ActorConfig()
                self.context_depth = 0

        class FakeBatch:
            def __init__(self) -> None:
                self.batch = {"response_mask": torch.ones((1, 1), dtype=torch.float32)}
                self.non_tensor_batch = {}
                self.meta_info = {"temperature": 1.0}

            def to(self, device: object) -> "FakeBatch":
                del device
                return self

        actor = ToyActor()
        parameter = next(actor.actor_module.parameters())
        preserved_grad = torch.full_like(parameter, 7.0)
        parameter.grad = preserved_grad.clone()
        tracker = SequentialBackwardDomainGradientTracker(
            actor,
            {
                "domains": ["math", "code"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
                "token_gradient_enabled": True,
                "token_gradient_strict_grad_restore": True,
            },
        )

        def fake_loss(_actor: object, _batch: object, **_kwargs: object) -> torch.Tensor:
            self.assertEqual(actor.context_depth, 0)
            return actor.actor_module(torch.tensor([[2.0]])).sum()

        original_backward = torch.Tensor.backward
        backward_calls = 0

        def checked_backward(tensor: torch.Tensor, *args: object, **kwargs: object) -> object:
            nonlocal backward_calls
            self.assertEqual(actor.context_depth, 0)
            backward_calls += 1
            return original_backward(tensor, *args, **kwargs)

        common_patches = (
            patch("mopd_verl.full_gradient.tracker.get_device_id", return_value="cpu"),
            patch("mopd_verl.full_gradient.tracker._actor_micro_batch_loss", side_effect=fake_loss),
            patch(
                "mopd_verl.full_gradient.tracker._selected_topk_support_from_inputs",
                return_value=(
                    torch.ones((1, 1), dtype=torch.long),
                    torch.zeros((1, 1)),
                ),
            ),
            patch.object(torch.Tensor, "backward", checked_backward),
        )
        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
        ):
            sample_stats = tracker._recompute_sample_to_domain_stats(
                FakeBatch(),
                target_chunks=(torch.tensor([1.0], dtype=torch.float32),),
                target_norm=1.0,
                target_norm_sq=1.0,
                loss_scale_factor=1.0,
                on_policy=True,
            )
            token_stats = tracker._recompute_token_selection_gradient_stats(
                [
                    {
                        "candidate_index": 0,
                        "micro_batch": FakeBatch(),
                        "sample_index": 0,
                        "position": 0,
                        "loss_scale_factor": 1.0,
                        "on_policy": True,
                    }
                ],
                target_map={"math": ((torch.tensor([1.0], dtype=torch.float32),), 1.0)},
                restore_grads=True,
            )

        self.assertEqual(sample_stats["sample_recompute_available"], 1.0)
        self.assertEqual(token_stats["token_grad_available"], 1.0)
        self.assertEqual(backward_calls, 2)
        self.assertEqual(actor.context_depth, 0)
        self.assertTrue(torch.equal(parameter.grad, preserved_grad))

    def test_audit_logger_emits_domain_category_metrics_on_synthetic_batch(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"torch is not installed: {exc}")

        class SyntheticBatch:
            def __init__(self, output_dir: Path) -> None:
                self.batch = {
                    "response_mask": torch.tensor(
                        [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32
                    ),
                    "old_log_probs": torch.tensor(
                        [[-0.2, -0.4, -0.3], [-0.5, -0.3, 0.0], [-0.6, -0.8, -0.4], [-0.7, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "math_teacher_log_prob": torch.tensor(
                        [[-0.3, -0.5, -0.35], [-0.45, -0.4, 0.0], [-0.9, -0.7, -0.5], [-0.8, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "code_teacher_log_prob": torch.tensor(
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, 0.0], [-0.5, -0.6, -0.3], [-0.6, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "base_log_prob": torch.tensor(
                        [[-0.25, -0.45, -0.4], [-0.55, -0.35, 0.0], [-0.65, -0.85, -0.45], [-0.75, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "advantages": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0], [0.4, 0.2, 0.1], [0.5, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "token_level_scores": torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        dtype=torch.float32,
                    ),
                    "responses": torch.tensor(
                        [[101, 102, 103], [101, 104, 0], [102, 105, 105], [102, 0, 0]],
                        dtype=torch.long,
                    ),
                }
                self.non_tensor_batch = {
                    "domain": ["math", "math", "code", "code"],
                    "opd_teacher": ["math", "math", "code", "code"],
                    "ability": ["math", "math", "code", "code"],
                    "data_source": ["AIME2024", "AIME2024", "code", "code"],
                    "sample_id": [f"sample-{idx}" for idx in range(4)],
                    "output_dir": str(output_dir),
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "audit"
            logger = MOPDAuditLogger(
                {
                    "mopd_audit": {
                        "enabled": True,
                        "output_dir": str(output_dir),
                        "domains": ["math", "code"],
                        "tier2_window_size": 3,
                        "calibration_bins": 3,
                    },
                    "actor_rollout_ref": {"actor": {"policy_loss": {"lambda_vals": 1.0}}},
                }
            )
            batch = SyntheticBatch(output_dir)
            metrics = logger.log_training_step(batch=batch, step=0, lr="1e-5")
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.2, "val/code/score": 0.1}, step=0))
            metrics.update(logger.log_validation_metrics({"val/math/score": 0.25, "val/code/score": 0.05}, step=1))
            metrics.update(
                logger.log_training_cost(
                    {"timing_s/step": 2.0, "perf/total_num_tokens": 7, "perf/max_memory_allocated_gb": 1.5},
                    step=0,
                    n_gpus=1,
                )
            )

            expected_metric_keys = [
                "math/loss/token_opd_loss_mean",
                "math/loss/token_opd_loss_std",
                "math/loss/token_opd_loss_variance",
                "math/loss/sample_opd_loss_mean",
                "math/loss/sample_opd_loss_std",
                "math/loss/sample_opd_loss_variance",
                "global/loss/token_opd_loss_mean",
                "global/loss/sample_opd_loss_variance",
                "math/advantage/positive_frac",
                "math/length/response_mean",
                "math/length/response_p95",
                "math/length/response_clip_ratio",
                "math/calibration/calibration_error",
                "math/token_conflict/proxy_mass",
                "math/token_conflict/proxy_mean",
                "math/token_conflict/teacher_disagreement_mean",
                "math/token_conflict/teacher_teacher_diff_p95",
                "math/token_conflict/student_teacher_diff_mean",
                "math/token_conflict/combined_diff_p95",
                "math/token_conflict/top1_token_share",
                "global/cost/gpu_seconds_step",
                "math/validation_gain_stats/score/variance",
            ]
            for key in expected_metric_keys:
                self.assertIn(key, metrics)
            self.assertAlmostEqual(metrics["math/loss/token_opd_loss_mean"], 0.06, places=6)
            self.assertAlmostEqual(metrics["math/loss/token_opd_loss_variance"], 0.0034, places=6)
            self.assertAlmostEqual(metrics["math/loss/token_opd_loss_std"], 0.0583095, places=6)
            self.assertAlmostEqual(metrics["math/loss/sample_opd_loss_mean"], 0.15, places=6)
            self.assertAlmostEqual(metrics["math/loss/sample_opd_loss_variance"], 0.01, places=6)
            self.assertAlmostEqual(metrics["math/loss/sample_opd_loss_std"], 0.1, places=6)
            self.assertAlmostEqual(metrics["math/advantage/positive_frac"], 1.0, places=6)
            self.assertAlmostEqual(metrics["math/length/response_mean"], 2.5, places=6)
            self.assertAlmostEqual(metrics["math/length/response_p95"], 2.95, places=6)
            self.assertAlmostEqual(metrics["math/length/response_clip_ratio"], 0.5, places=6)
            self.assertAlmostEqual(metrics["global/loss/sample_opd_loss_mean"], -0.05, places=6)
            self.assertAlmostEqual(metrics["global/loss/sample_opd_loss_variance"], 0.05625, places=6)
            self.assertAlmostEqual(metrics["math/reward/training_reward_mean"], 0.5, places=6)
            self.assertAlmostEqual(metrics["math/reward/training_accuracy"], 0.5, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/proxy_mass"], 0.065, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/proxy_mean"], 0.013, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/proxy_mass_frac"], 0.065 / 0.165, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/teacher_disagreement_mean"], 0.14, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/teacher_teacher_diff_mass"], 0.7, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/teacher_teacher_diff_mean"], 0.14, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/teacher_teacher_diff_p95"], 0.28, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/student_teacher_diff_mass"], 0.4, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/student_teacher_diff_mean"], 0.08, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/combined_diff_mass"], 0.065, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/combined_diff_mean"], 0.013, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/combined_diff_p95"], 0.028, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/token_abs_opd_loss_mean"], 0.08, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/top1_token_share"], 0.03 / 0.065, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/top1_teacher_diff_share"], 0.3 / 0.7, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/top10_token_share"], 1.0, places=6)
            self.assertAlmostEqual(metrics["math/token_conflict/unique_token_count"], 4.0, places=6)
            removed_metric_keys = [
                "math/grad_conflict/code/grad_cosine_train_i_k",
                "math/grad/gradient_signal",
                "math/grad/gradient_noise",
                "math/grad/grad_norm",
                "math/grad_anchor/code/anchor_grad_cosine_i_j",
                "global/grad_conflict/math_vs_code/grad_cosine_train_i_k",
                "math/loss/kl_spike_rate",
                "math/coverage/new_sample_rate",
                "global/reliability/rank_stability_across_windows",
            ]
            for key in removed_metric_keys:
                self.assertNotIn(key, metrics)

            expected_files = [
                "domain_step_metrics.jsonl",
                "loss_variance_domain_step.jsonl",
                "loss_variance_sample.jsonl",
                "token_conflict_attribution.jsonl",
                "validation_probe.jsonl",
                "validation_gain_variance.jsonl",
                "training_cost.jsonl",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)
            token_conflict_rows = [
                json.loads(line)
                for line in (output_dir / "token_conflict_attribution.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            math_top = next(row for row in token_conflict_rows if row["domain"] == "math" and row["rank"] == 1)
            code_top = next(row for row in token_conflict_rows if row["domain"] == "code" and row["rank"] == 1)
            self.assertEqual(math_top["token_id"], 102)
            self.assertAlmostEqual(math_top["conflict_proxy_sum"], 0.03, places=6)
            self.assertAlmostEqual(math_top["teacher_teacher_diff_sum"], 0.3, places=6)
            self.assertAlmostEqual(math_top["student_teacher_diff_mean"], 0.1, places=6)
            self.assertEqual(code_top["token_id"], 102)
            self.assertAlmostEqual(code_top["conflict_proxy_sum"], 0.06, places=6)
            removed_files = [
                "trend_stability.jsonl",
                "gradient_noise.jsonl",
                "rank_stability.jsonl",
                "teacher_logits_reliability.jsonl",
                "calibration.jsonl",
                "sample_influence.jsonl",
                "coverage_diversity.jsonl",
                "shadow_probe.jsonl",
                "domain_conflict.jsonl",
            ]
            for filename in removed_files:
                self.assertFalse((output_dir / filename).exists(), filename)

    def test_merge_teacher_data_adds_extra_info_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            math_path = temp_path / "math.parquet"
            code_path = temp_path / "code.parquet"
            output_path = temp_path / "merged.parquet"

            pd.DataFrame(
                [
                    {"data_source": "math", "prompt": "1+1", "extra_info": {"index": 0}},
                    {"data_source": "math", "prompt": "2+2", "extra_info": None},
                ]
            ).to_parquet(math_path, index=False)
            pd.DataFrame(
                [
                    {"data_source": "code", "prompt": "write add", "extra_info": {"index": 2}},
                ]
            ).to_parquet(code_path, index=False)

            merge_teacher_data(math_path, code_path, output_path)
            self.assertEqual(teacher_counts(output_path), {"code": 1, "math": 2, "reasoning": 0, "search": 0, "tool": 0})
            sample_validation = validate_sample_ids(output_path)
            self.assertTrue(sample_validation.is_valid)

    def test_searchqa_to_verl_parquet_adds_search_teacher_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "searchqa.jsonl"
            parquet_path = temp_path / "train.parquet"
            record = {
                "id": "nq0",
                "data_source": "nq",
                "question": "Who wrote the first computer program?",
                "golden_answers": ["Ada Lovelace"],
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            row_count = searchqa_to_verl_parquet(jsonl_path, parquet_path, split="train")
            frame = pd.read_parquet(parquet_path)
            extra_info = frame.iloc[0]["extra_info"]

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "searchR1_nq")
            self.assertEqual(frame.iloc[0]["ability"], "searchqa")
            self.assertEqual(frame.iloc[0]["reward_model"]["ground_truth"], {"target": ["Ada Lovelace"]})
            self.assertEqual(extra_info["opd_teacher"], "search")
            self.assertEqual(extra_info["domain"], "search")
            self.assertEqual(extra_info["source_domain"], "search")
            self.assertTrue(extra_info["need_tools_kwargs"])
            self.assertEqual(extra_info["tools_kwargs"]["search"]["create_kwargs"]["data_source"], "searchR1_nq")
            self.assertIn("<tool_call>", frame.iloc[0]["prompt"][1]["content"])
            self.assertEqual(teacher_counts(parquet_path), {"code": 0, "math": 0, "reasoning": 0, "search": 1, "tool": 0})

    def test_general_reasoner_to_verl_parquet_adds_reasoning_teacher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "general_reasoner.jsonl"
            parquet_path = temp_path / "test.parquet"
            record = {
                "id": "g0",
                "question": "What is 1+1?",
                "answer": "2",
                "difficulty": "easy",
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            row_count = general_reasoner_to_verl_parquet(jsonl_path, parquet_path, split="test")
            frame = pd.read_parquet(parquet_path)
            extra_info = frame.iloc[0]["extra_info"]

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "general-reasoner")
            self.assertEqual(frame.iloc[0]["ability"], "reasoning")
            self.assertEqual(frame.iloc[0]["reward_model"]["ground_truth"], "2")
            self.assertEqual(extra_info["opd_teacher"], "reasoning")
            self.assertEqual(extra_info["domain"], "reasoning")
            self.assertEqual(extra_info["validation_dataset"], "general-reasoner")
            self.assertIn("Please reason step by step", frame.iloc[0]["prompt"][0]["content"])
            self.assertEqual(teacher_counts(parquet_path), {"code": 0, "math": 0, "reasoning": 1, "search": 0, "tool": 0})

    def test_search_retrieval_service_formats_verl_tool_response(self) -> None:
        class FakeBackend:
            def search(self, query: str, topk: int) -> list[SearchResult]:
                return [SearchResult(title="Ada Lovelace", snippet=f"{query} result", url="https://example.com")]

        service = RetrievalService(backend=FakeBackend())
        payload = service.search_batch(["first programmer"], topk=1)
        document = payload["result"][0][0]["document"]["contents"]

        self.assertIn("Ada Lovelace", document)
        self.assertIn("first programmer result", document)
        self.assertIn("https://example.com", document)

    def test_searchqa_reward_dispatch_uses_em(self) -> None:
        try:
            from verl.utils.reward_score import default_compute_score
        except ModuleNotFoundError as exc:
            self.skipTest(f"vendored verl is not importable: {exc}")

        score = default_compute_score(
            "searchR1_nq",
            "<answer>Ada Lovelace</answer>",
            {"target": ["Ada Lovelace"]},
        )

        self.assertEqual(score, 1.0)

    def test_general_reasoner_reward_dispatch_uses_math_verify(self) -> None:
        try:
            import math_verify  # noqa: F401
            from verl.utils.reward_score import default_compute_score
        except ModuleNotFoundError as exc:
            self.skipTest(f"General-Reasoner reward dependencies are not importable: {exc}")

        score = default_compute_score(
            "general-reasoner",
            "We compute the result and obtain \\boxed{2}.",
            "2",
        )

        self.assertEqual(score, 1.0)

    def test_math_eval_jsonl_to_verl_parquet_adds_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "test.jsonl"
            parquet_path = temp_path / "test.parquet"
            jsonl_path.write_text('{"id":"p0","problem":"Find 1+1.","answer":"2"}\n', encoding="utf-8")

            row_count = math_eval_jsonl_to_verl_parquet(jsonl_path, parquet_path, "HMMT25Feb")
            frame = pd.read_parquet(parquet_path)

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "HMMT25Feb")
            self.assertEqual(frame.iloc[0]["ability"], "math")
            self.assertEqual(frame.iloc[0]["reward_model"]["ground_truth"], "2")
            self.assertIn("\\boxed{}", frame.iloc[0]["prompt"][0]["content"])
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "HMMT25Feb")
            self.assertEqual(frame.iloc[0]["extra_info"]["sample_id"], "validation:HMMT25Feb:p0")

    def test_evalplus_jsonl_to_verl_parquet_adds_code_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "HumanEvalPlus.jsonl"
            parquet_path = temp_path / "test.parquet"
            evalplus_record = {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2]],
                "plus_input": [[3, 4]],
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            }
            jsonl_path.write_text(json.dumps(evalplus_record) + "\n", encoding="utf-8")

            row_count = evalplus_jsonl_to_verl_parquet(jsonl_path, parquet_path, "HumanEvalPlus")
            frame = pd.read_parquet(parquet_path)
            ground_truth = frame.iloc[0]["reward_model"]["ground_truth"]

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "HumanEvalPlus")
            self.assertEqual(frame.iloc[0]["ability"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["opd_teacher"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "HumanEvalPlus")
            self.assertEqual(frame.iloc[0]["extra_info"]["prompt_template"], "paper_evalplus_qwen_chat")
            self.assertIn("Present the code in", frame.iloc[0]["prompt"][0]["content"])
            self.assertIn("You need to think first then write the Python code.", frame.iloc[0]["prompt"][0]["content"])
            self.assertNotIn("Return only the code, without explanations.", frame.iloc[0]["prompt"][0]["content"])
            self.assertIn('"entry_point": "add"', ground_truth)
            self.assertIn('"assert_case"', ground_truth)
            self.assertIn("check(add)", ground_truth)

    def test_lcb_jsonl_to_verl_parquet_adds_code_validation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            jsonl_path = temp_path / "test.jsonl"
            parquet_path = temp_path / "test.parquet"
            record = {
                "question_title": "Add numbers",
                "question_content": "Implement add.",
                "platform": "leetcode",
                "question_id": "q0",
                "contest_id": "c0",
                "contest_date": "2025-01-01T00:00:00",
                "starter_code": "def add(a, b):",
                "difficulty": "easy",
                "public_test_cases": '[{"input":"1\\n2","output":"3","testtype":"stdin"}]',
                "private_test_cases": '[{"input":"3\\n4","output":"7","testtype":"stdin"}]',
                "metadata": "{}",
            }
            jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            row_count = lcb_jsonl_to_verl_parquet([jsonl_path], parquet_path)
            frame = pd.read_parquet(parquet_path)

            self.assertEqual(row_count, 1)
            self.assertEqual(frame.iloc[0]["data_source"], "LiveCodeBench")
            self.assertEqual(frame.iloc[0]["ability"], "code")
            self.assertEqual(frame.iloc[0]["extra_info"]["validation_dataset"], "LiveCodeBench")
            self.assertEqual(frame.iloc[0]["extra_info"]["prompt_template"], "paper_lcb_qwen3_non_thinking")
            self.assertIn("Question:\nImplement add.", frame.iloc[0]["prompt"][0]["content"])
            self.assertIn("Present the code in", frame.iloc[0]["prompt"][0]["content"])
            self.assertIn('"outputs": ["3"]', frame.iloc[0]["reward_model"]["ground_truth"])

    def test_prepare_paper_eval_data_writes_all_validation_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "G-OPD"
            for relative in ["aime24", "aime25", "hmmt25_feb", "hmmt25_nov"]:
                data_dir = root / "data" / relative
                data_dir.mkdir(parents=True, exist_ok=True)
                (data_dir / "test.jsonl").write_text(
                    '{"id":"0","problem":"Compute 1+1.","answer":"2"}\n',
                    encoding="utf-8",
                )
            code_data_dir = root / "code_eval" / "data"
            code_data_dir.mkdir(parents=True, exist_ok=True)
            evalplus_record = {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
                "base_input": [[1, 2]],
                "plus_input": [[3, 4]],
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            }
            evalplus_row = json.dumps(evalplus_record) + "\n"
            (code_data_dir / "HumanEvalPlus.jsonl").write_text(evalplus_row, encoding="utf-8")
            mbpp_record = {
                **evalplus_record,
                "task_id": "Mbpp/0",
                "assertion": "assert add(1, 2) == 3",
            }
            (code_data_dir / "MbppPlus.jsonl").write_text(json.dumps(mbpp_record) + "\n", encoding="utf-8")
            lcb_dir = root / "code_eval" / "coding" / "LiveCodeBench" / "code_generation_lite"
            lcb_dir.mkdir(parents=True, exist_ok=True)
            lcb_record = {
                "question_title": "Add numbers",
                "question_content": "Implement add.",
                "platform": "leetcode",
                "question_id": "q0",
                "contest_id": "c0",
                "contest_date": "2025-01-01T00:00:00",
                "starter_code": "def add(a, b):",
                "difficulty": "easy",
                "public_test_cases": '[{"input":"1\\n2","output":"3","testtype":"stdin"}]',
                "private_test_cases": "[]",
                "metadata": "{}",
            }
            (lcb_dir / "test.jsonl").write_text(json.dumps(lcb_record) + "\n", encoding="utf-8")

            counts = prepare_paper_eval_data(root)
            output_root = root / "eval" / "domains"

            self.assertEqual(
                counts,
                {
                    "aime24": 1,
                    "aime25": 1,
                    "hmmt25_feb": 1,
                    "hmmt25_nov": 1,
                    "humaneval_plus": 1,
                    "mbpp_plus": 1,
                    "lcb": 1,
                },
            )
            self.assertTrue((output_root / "math/data/AIME24/test.parquet").exists())
            self.assertTrue((output_root / "math/data/AIME25/test.parquet").exists())
            self.assertTrue((output_root / "math/data/HMMT25Feb/test.parquet").exists())
            self.assertTrue((output_root / "math/data/HMMT25Nov/test.parquet").exists())
            self.assertTrue((output_root / "code/data/HumanEvalPlus/test.parquet").exists())
            self.assertTrue((output_root / "code/data/MBPPPlus/test.parquet").exists())
            self.assertTrue((output_root / "code/data/LiveCodeBench/test.parquet").exists())

    def test_validate_teacher_labels_rejects_missing_or_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.parquet"
            pd.DataFrame(
                [
                    {"prompt": "valid", "extra_info": {"opd_teacher": "math"}},
                    {"prompt": "missing", "extra_info": {"index": 1}},
                    {"prompt": "bad", "extra_info": {"opd_teacher": "science"}},
                ]
            ).to_parquet(path, index=False)

            validation = validate_teacher_labels(path)
            self.assertFalse(validation.is_valid)
            self.assertEqual(validation.counts, {"code": 0, "math": 1, "reasoning": 0, "search": 0, "tool": 0})
            self.assertEqual(len(validation.invalid_rows), 2)

    def test_validate_sample_ids_rejects_missing_duplicate_or_mismatched_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_sample_ids.parquet"
            pd.DataFrame(
                [
                    {"prompt": "valid", "extra_info": {"opd_teacher": "math", "domain": "math", "sample_id": "m0"}},
                    {"prompt": "missing", "extra_info": {"opd_teacher": "math", "domain": "math"}},
                    {"prompt": "duplicate", "extra_info": {"opd_teacher": "math", "domain": "math", "sample_id": "m0"}},
                    {"prompt": "mismatch", "extra_info": {"opd_teacher": "code", "domain": "math", "sample_id": "c0"}},
                ]
            ).to_parquet(path, index=False)

            validation = validate_sample_ids(path)
            self.assertFalse(validation.is_valid)
            self.assertEqual(validation.duplicate_count, 1)
            self.assertEqual(len(validation.invalid_rows), 3)

    def test_smoke_data_contains_both_teacher_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_smoke_data(temp_dir)

            self.assertEqual(teacher_counts(paths["train"]), {"code": 1, "math": 1, "reasoning": 0, "search": 0, "tool": 0})
            self.assertEqual(teacher_counts(paths["val"]), {"code": 1, "math": 1, "reasoning": 0, "search": 0, "tool": 0})
            self.assertTrue(validate_sample_ids(paths["train"]).is_valid)
            self.assertTrue(validate_sample_ids(paths["val"]).is_valid)


if __name__ == "__main__":
    unittest.main()
