from __future__ import annotations

import json
import tempfile
import unittest
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
from mopd_verl.settings import load_config
from mopd_verl.smoke_data import write_smoke_data
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

    def test_default_command_contains_multi_teacher_setting(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_math_code.yaml"
        config = load_config(config_path)
        command = build_command(config)
        rendered = format_command(command)

        self.assertIn("actor_rollout_ref.model.path=Qwen/Qwen3-4B", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-Math", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.base_model_path=Qwen3-4B-Non-Thinking-RL-Code", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true", rendered)
        self.assertIn("actor_rollout_ref.actor.policy_loss.lambda_vals=1.25", rendered)
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
        self.assertIn("eval/domains/code/data/LiveCodeBench/test.parquet", rendered)

    def test_searchqa_command_enables_tool_rollout(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_searchqa.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("actor_rollout_ref.model.path=Qwen/Qwen3-4B", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-SearchQA", rendered)
        self.assertIn("+actor_rollout_ref.ref.model.base_model_path=Qwen/Qwen3-4B", rendered)
        self.assertIn("+data.domain_sampling_weights={search: 1}", rendered)
        self.assertIn("+data.domain_train_files=", rendered)
        self.assertIn("data/SearchQA/train.parquet", rendered)
        self.assertIn("actor_rollout_ref.rollout.name=sglang", rendered)
        self.assertIn("actor_rollout_ref.rollout.mode=sync", rendered)
        self.assertIn("actor_rollout_ref.rollout.max_model_len=15000", rendered)
        self.assertIn("actor_rollout_ref.rollout.multi_turn.enable=true", rendered)
        self.assertIn("actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/tool_config/search_tool_config.yaml", rendered)
        self.assertIn("+data.need_tools_kwargs=True", rendered)

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

    def test_general_reasoner_command_uses_reasoning_teacher(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_general_reasoner.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("actor_rollout_ref.model.path=../models/Qwen3-4B", rendered)
        self.assertIn("+actor_rollout_ref.model.base_model_path=../models/Qwen3-4B", rendered)
        self.assertEqual(config.model.reasoning_teacher_path, "../models/General-Reasoner-Qwen3-14B")
        self.assertIn("+actor_rollout_ref.ref.model.path=../models/General-Reasoner-Qwen3-14B", rendered)
        self.assertIsNone(config.model.secondary_teacher_path)
        self.assertNotIn("+actor_rollout_ref.ref.model.base_model_path=Qwen/Qwen3-14B", rendered)
        self.assertIn("+data.domain_sampling_weights={reasoning: 1}", rendered)
        self.assertIn("data/GeneralReasoner/WebInstructVerified/train.parquet", rendered)
        self.assertIn("eval/domains/greasoner/data/WebInstructVerified/test.parquet", rendered)
        self.assertIn("+data.apply_chat_template_kwargs.enable_thinking=True", rendered)
        self.assertIn("+data.need_tools_kwargs=False", rendered)
        self.assertIn("trainer.experiment_name=Qwen3-4B-GeneralReasoner-Qwen3-14B-MOPD", rendered)
        self.assertNotIn("actor_rollout_ref.rollout.multi_turn.tool_config_path", rendered)

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

    def test_audit_smoke_command_contains_tensorboard_and_audit_settings(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_audit_smoke.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn('trainer.logger=["console","tensorboard"]', rendered)
        self.assertIn("data.validation_shuffle=False", rendered)
        self.assertNotIn("actor_rollout_ref.rollout.seed", rendered)
        self.assertIn("actor_rollout_ref.rollout.enable_chunked_prefill=False", rendered)
        self.assertIn("actor_rollout_ref.rollout.val_kwargs.do_sample=False", rendered)
        self.assertIn("+mopd_audit.enabled=true", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/smoke", rendered)
        self.assertIn("+mopd_audit.tensorboard_layout=domain_category", rendered)
        self.assertIn("+mopd_audit.tensorboard_prune_mode=core", rendered)
        self.assertIn("+mopd_audit.max_samples_per_domain=8", rendered)
        self.assertIn("+data.domain_sampling_weights={math: 0.5, code: 0.5}", rendered)
        self.assertIn("+mopd_audit.full_gradient_enabled=false", rendered)
        self.assertIn("+mopd_audit.full_gradient_train_max_samples_per_domain=null", rendered)

    def test_formal_command_enables_full_parameter_gradient_audit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_single_a800.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertIn("+mopd_audit.full_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.full_gradient_freq_steps=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_train_max_samples_per_domain=null", rendered)
        self.assertIn("+mopd_audit.full_gradient_micro_batch_size_per_gpu=1", rendered)
        self.assertIn("+mopd_audit.full_gradient_storage_dtype=bfloat16", rendered)
        self.assertIn("+mopd_audit.sample_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_norm_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_freq_steps=1", rendered)
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
        self.assertIn("trainer.default_local_dir=checkpoints/formal_single_a800", rendered)
        self.assertNotIn("/root/autodl-tmp/opd_mopd/G-OPD/G-OPD-Training-Data", rendered)
        self.assertNotIn("/root/autodl-tmp/opd_mopd/OPD-code", rendered)
        self.assertNotIn("+paper_eval.enabled=true", rendered)
        self.assertNotIn("run_paper_eval_suite.sh", rendered)

    def test_single_h200_profile_preserves_math_code_semantics(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_single_h200.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 1024)
        self.assertEqual(config.actor.ppo_mini_batch_size, 1024)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertFalse(config.actor.optimizer_offload)
        self.assertEqual(config.rollout.log_prob_micro_batch_size_per_gpu, 2)
        self.assertFalse(config.rollout.enforce_eager)
        self.assertEqual(config.rollout.max_num_batched_tokens, 65536)
        self.assertEqual(config.rollout.max_num_seqs, 16)
        self.assertEqual(config.trainer.n_gpus_per_node, 1)
        self.assertEqual(config.trainer.save_freq, 50)
        self.assertEqual(config.ray_kwargs.ray_init.num_cpus, 8)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.8", rendered)
        self.assertIn("eval/domains/math/data/AIME25/test.parquet", rendered)
        self.assertIn("eval/domains/code/data/HumanEvalPlus/test.parquet", rendered)
        self.assertNotIn(
            "data.val_files=['data/G-OPD-Training-Data/Eurus/code_train.parquet']",
            rendered,
        )
        self.assertIn("trainer.default_local_dir=checkpoints/formal_single_h200", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_single_h200", rendered)
        self.assertIn("+mopd_audit.full_gradient_storage_dtype=bfloat16", rendered)

    def test_dual_a800_profile_uses_replicated_two_gpu_gradient_audit(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mopd_formal_dual_a800.yaml"
        config = load_config(config_path)
        rendered = format_command(build_command(config))

        self.assertEqual(config.data.train_batch_size, 256)
        self.assertEqual(config.data.max_response_length, 8192)
        self.assertEqual(config.actor.ppo_mini_batch_size, 256)
        self.assertEqual(config.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertTrue(config.actor.gradient_checkpointing)
        self.assertEqual(config.rollout.tensor_model_parallel_size, 2)
        self.assertEqual(config.rollout.gpu_memory_utilization, 0.8)
        self.assertEqual(config.trainer.n_gpus_per_node, 2)
        self.assertTrue(config.audit.enabled)
        self.assertEqual(config.actor.fsdp_size, 1)
        self.assertTrue(config.audit.full_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_enabled)
        self.assertTrue(config.audit.sample_gradient_norm_enabled)
        self.assertFalse(config.audit.sample_gradient_cos_enabled)
        self.assertEqual(config.trainer.total_training_steps, 10)
        self.assertIn("trainer.n_gpus_per_node=2", rendered)
        self.assertIn("data.train_batch_size=256", rendered)
        self.assertIn("data.max_response_length=8192", rendered)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=256", rendered)
        self.assertIn("actor_rollout_ref.model.enable_gradient_checkpointing=True", rendered)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=2", rendered)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.8", rendered)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.fsdp_size=1", rendered)
        self.assertIn("+mopd_audit.output_dir=audit/formal_dual_a800", rendered)
        self.assertIn("+mopd_audit.full_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_norm_enabled=true", rendered)
        self.assertIn("+mopd_audit.sample_gradient_cos_enabled=false", rendered)
        self.assertIn("trainer.default_local_dir=checkpoints/formal_dual_a800", rendered)

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
            "global/audit/full_gradient_domain_sequential_available": 1.0,
            "global/audit/full_gradient_domain_sequential_unsupported": 0.0,
            "global/audit/full_gradient_replicated_all_reduce": 1.0,
            "global/audit/full_gradient_replica_count": 2.0,
            "global/audit/sample_gradient_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_norm_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_cos_distributed_unsupported": 1.0,
            "global/audit/sample_gradient_distributed_world_size": 2.0,
            "math/grad_conflict/code/grad_cosine_train_i_k": -0.4,
            "global/grad_conflict/math_vs_code/grad_cosine_train_i_k": -0.4,
            "math/grad/grad_norm": 1.0,
            "math/teacher/teacher_logprob_mean": -0.5,
            "math/teacher/teacher_student_gap_mean": 0.1,
            "math/advantage/positive_frac": 0.5,
            "math/length/response_mean": 1024.0,
            "math/length/response_p95": 2048.0,
            "math/length/response_clip_ratio": 0.25,
            "math/sample_grad/norm_mean": 1.2,
            "math/sample_grad/norm_p95": 2.4,
            "math/sample_grad_cos/domain_cos_mean": 0.3,
            "math/sample_grad_cos/domain_cos_negative_frac": 0.25,
            "math/sample_grad_contribution/projection_share_mean": 0.05,
            "math/sample_grad_contribution/top1_abs_share": 0.2,
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
        self.assertIn("global/audit/full_gradient_domain_sequential_available", filtered)
        self.assertIn("global/audit/full_gradient_domain_sequential_unsupported", filtered)
        self.assertIn("global/audit/full_gradient_replicated_all_reduce", filtered)
        self.assertIn("global/audit/full_gradient_replica_count", filtered)
        self.assertIn("global/audit/sample_gradient_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_norm_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_cos_distributed_unsupported", filtered)
        self.assertIn("global/audit/sample_gradient_distributed_world_size", filtered)
        self.assertIn("math/teacher/teacher_student_gap_mean", filtered)
        self.assertIn("math/advantage/positive_frac", filtered)
        self.assertIn("math/length/response_mean", filtered)
        self.assertIn("math/length/response_p95", filtered)
        self.assertIn("math/length/response_clip_ratio", filtered)
        self.assertIn("math/sample_grad/norm_mean", filtered)
        self.assertIn("math/sample_grad/norm_p95", filtered)
        self.assertIn("math/sample_grad_cos/domain_cos_mean", filtered)
        self.assertIn("math/sample_grad_cos/domain_cos_negative_frac", filtered)
        self.assertIn("math/sample_grad_contribution/projection_share_mean", filtered)
        self.assertIn("math/sample_grad_contribution/top1_abs_share", filtered)
        self.assertIn("math/reward/training_reward_mean", filtered)
        self.assertIn("math/reward/training_accuracy", filtered)
        self.assertIn("math/coverage/duplicate_rate", filtered)
        self.assertIn("global/cost/step_seconds", filtered)
        self.assertIn("global/validation_gain/val-core_AIME2024_reward_mean_1", filtered)
        self.assertIn("rollout_corr/kl", filtered)
        self.assertIn("actor/lr", filtered)
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

    def test_same_forward_domain_probe_sums_to_full_gradient(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(1, 1, bias=False)
                self.actor_module.weight.data.fill_(2.0)
                self.config = {"entropy_coeff": 0.0, "use_kl_loss": False}

        def policy_loss_fn(
            old_log_prob: torch.Tensor,
            log_prob: torch.Tensor,
            advantages: torch.Tensor,
            response_mask: torch.Tensor,
            loss_agg_mode: str,
            config: object,
            rollout_is_weights: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            del old_log_prob, loss_agg_mode, config, rollout_is_weights
            loss_mat = -advantages * log_prob
            return (loss_mat * response_mask).sum() / response_mask.sum(), {}

        actor = ToyActor()
        inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        log_prob = actor.actor_module(inputs)
        old_log_prob = torch.zeros_like(log_prob)
        advantages = torch.tensor([[1.0], [2.0], [-1.0], [3.0]])
        response_mask = torch.ones_like(log_prob)
        model_inputs = {"opd_teacher": ["math", "math", "code", "code"]}
        probe = SameForwardDomainGradientProbe(
            actor,
            {
                "enabled": True,
                "domains": ["math", "code"],
                "storage_dtype": "float32",
                "learning_rate": 1e-5,
                "offload_domain_gradients": False,
            },
        )

        probe.start_mini_batch()
        probe.capture_micro_batch(
            model_inputs=model_inputs,
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            entropy=None,
            policy_loss_fn=policy_loss_fn,
            loss_agg_mode="token-mean",
            loss_scale_factor=1.0,
            rollout_is_weights=None,
        )
        full_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=actor.config,
        )
        full_grad = torch.autograd.grad(full_loss, tuple(actor.actor_module.parameters()))[0].detach().flatten()
        summed_domain_grad = probe._gpu_vectors["math"] + probe._gpu_vectors["code"]
        self.assertTrue(torch.allclose(summed_domain_grad.cpu(), full_grad.cpu(), atol=1e-6))

        metrics = probe.finish_mini_batch()
        self.assertIn("math/full_grad/grad_norm", metrics)
        self.assertIn("code/full_grad/grad_norm", metrics)
        self.assertIn("global/full_grad_conflict/code_vs_math/full_grad_cosine_train_i_k", metrics)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)
        self.assertIn("global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total", metrics)
        self.assertIn("global/full_grad_contribution/code_to_total/signed_projection_share", metrics)

    def test_full_gradient_vector_reductions_are_chunked(self) -> None:
        try:
            import math

            import torch

            from mopd_verl.full_gradient_worker import (
                _local_vector_dot,
                _local_vector_norm,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        left = torch.arange(1.0, 11.0)
        right = torch.arange(10.0, 0.0, -1.0)
        with patch("mopd_verl.full_gradient_worker._VECTOR_REDUCTION_CHUNK_SIZE", 3):
            dot = _local_vector_dot(left, right)
            norm = _local_vector_norm(left)

        self.assertAlmostEqual(dot, 220.0, places=6)
        self.assertAlmostEqual(norm, math.sqrt(385.0), places=6)

    def test_same_forward_domain_probe_reports_offloaded_full_grad_metrics(self) -> None:
        try:
            import math

            import torch
            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            pass

        probe = SameForwardDomainGradientProbe(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math", "code"],
                "storage_dtype": "float32",
                "offload_domain_gradients": True,
            },
        )

        probe.start_mini_batch()
        probe._cpu_vectors = {
            "math": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "code": torch.tensor([0.0, 2.0], dtype=torch.float32),
        }
        probe._sample_counts = {"math": 3, "code": 5}

        metrics = probe.finish_mini_batch()

        self.assertAlmostEqual(metrics["math/full_grad/grad_norm"], 1.0, places=6)
        self.assertAlmostEqual(metrics["code/full_grad/grad_norm"], 2.0, places=6)
        self.assertEqual(metrics["math/full_grad/sample_count"], 3.0)
        self.assertEqual(metrics["code/full_grad/sample_count"], 5.0)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)
        self.assertAlmostEqual(
            metrics["global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total"],
            1.0 / math.sqrt(5.0),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["global/full_grad_contribution/code_to_total/signed_projection_share"],
            4.0 / 5.0,
            places=6,
        )

    def test_sequential_backward_domain_tracker_computes_total_cosine_and_contribution(self) -> None:
        try:
            import math

            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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

        actor.actor_optimizer.zero_grad()
        tracker.start_mini_batch()
        actor.actor_module(torch.tensor([[1.0, 0.0]])).sum().backward()
        tracker.after_backward("math", 1)
        actor.actor_module(torch.tensor([[0.0, 1.0]])).sum().backward()
        tracker.after_backward("code", 1)
        metrics = tracker.finish_mini_batch()

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

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)
                self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=0.1)

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"opd_teacher": [domain]}

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
        actor.actor_optimizer.zero_grad()
        tracker.start_mini_batch()
        for domain, _ in tracked:
            actor.actor_module(inputs[domain][offsets[domain]]).sum().backward()
            offsets[domain] += 1
            tracker.after_backward(domain, 1)

        metrics = tracker.finish_mini_batch()

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
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"opd_teacher": [domain]}

            def __len__(self) -> int:
                return 1

        tracker = SequentialBackwardDomainGradientTracker(
            ToyActor(),
            {"enabled": True, "domains": ["math", "code"]},
        )
        micro_batches = [ToyMicroBatch("math"), ToyMicroBatch("code")]

        with patch("mopd_verl.full_gradient_worker._all_ranks_true", return_value=False):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertFalse(tracker._prepared_supported)
        self.assertEqual([domain for domain, _ in tracked], [None, None])
        self.assertEqual([micro_batch for _, micro_batch in tracked], micro_batches)

    def test_sequential_tracker_requires_aligned_domain_boundaries_across_ranks(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str) -> None:
                self.non_tensor_batch = {"opd_teacher": [domain]}

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
            patch("mopd_verl.full_gradient_worker._all_ranks_true", return_value=True),
            patch("mopd_verl.full_gradient_worker._all_ranks_equal_ints", return_value=False),
        ):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertFalse(tracker._prepared_supported)
        self.assertEqual([domain for domain, _ in tracked], [None, None, None])

    def test_sequential_tracker_accepts_aligned_domain_partition_metadata(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            pass

        class ToyMicroBatch:
            def __init__(self, domain: str, sample_count: int) -> None:
                self.non_tensor_batch = {"opd_teacher": [domain] * sample_count}
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
            patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2),
            patch("mopd_verl.full_gradient_worker._all_ranks_true", return_value=True),
            patch("mopd_verl.full_gradient_worker._all_ranks_equal_ints", return_value=True),
        ):
            tracked = tracker.prepare_micro_batches(micro_batches)

        self.assertTrue(tracker._prepared_supported)
        self.assertTrue(tracker.domain_gradient_enabled)
        self.assertEqual([domain for domain, _ in tracked], ["math", "code"])

    def test_full_gradient_statistics_sum_rank_shards_without_replica_averaging(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import (
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
            patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2),
            patch(
                "mopd_verl.full_gradient_worker._all_reduce_values_sum",
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
            from mopd_verl.full_gradient_worker import _gradient_replica_count
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": -1}}

        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=4):
            self.assertEqual(_gradient_replica_count(ToyActor()), 1)

    def test_sequential_tracker_keeps_sample_norm_for_full_param_replicas(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": 1}}

        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2):
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
        with patch("mopd_verl.full_gradient_worker._all_gather_list", side_effect=lambda values: list(values)):
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

    def test_sequential_tracker_disables_sample_norm_for_sharded_params(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient tracker dependencies are not installed: {exc}")

        class ToyActor:
            config = {"fsdp_config": {"fsdp_size": -1}}

        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2):
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
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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

    def test_dp_actor_uses_real_backward_sequential_gradient_tracker(self) -> None:
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
        self.assertIn("SequentialBackwardDomainGradientTracker", patch_source)
        self.assertIn("mopd_gradient_tracker.before_backward(", patch_source)
        self.assertNotIn("mopd_gradient_tracker.capture_micro_batch(", patch_source)
        self.assertIn(
            'gradient_checkpointing_kwargs={"use_reentrant": False}',
            fsdp_worker_source,
        )
        self.assertNotIn("_grad_stats_from_true_backward", gradient_worker_source)
        self.assertNotIn("sample_recompute_used_true_backward_fallback", gradient_worker_source)

    def test_sample_gradient_cos_uses_all_candidates(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
            metrics["math/sample_grad_contribution/projection_share_sum_error"],
            0.0,
            places=6,
        )
        self.assertTrue(all(candidate["row"]["computed_for_cos"] for candidate in candidates))

    def test_sample_gradient_cos_metrics_gather_global_values(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
            "sample_projection_share": 0.4,
            "sample_recompute_grad_norm": 1.0,
        }

        with patch(
            "mopd_verl.full_gradient_worker._all_gather_list",
            side_effect=lambda values: list(values) + [0.75 if values and values[0] == 0.25 else 0.6],
        ):
            metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(metrics["math/sample_grad_cos/sample_count"], 2.0)
        self.assertAlmostEqual(metrics["math/sample_grad_cos/domain_cos_mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/projection_share_sum"], 1.0, places=6)

    def test_sample_gradient_projection_sum_is_normalized_across_replicas(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
            "sample_projection_share": 1.0,
            "sample_recompute_grad_norm": 1.0,
        }

        with patch(
            "mopd_verl.full_gradient_worker._all_gather_list",
            side_effect=lambda values: list(values) + list(values),
        ), patch(
            "mopd_verl.full_gradient_worker._gradient_replica_count",
            return_value=2,
        ):
            metrics = tracker._sample_cos_metrics({"math": ((), 1.0)})

        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_sum_across_replicas"],
            2.0,
        )
        self.assertEqual(
            metrics["math/sample_grad_contribution/projection_share_replica_count"],
            2.0,
        )
        self.assertEqual(metrics["math/sample_grad_contribution/projection_share_sum"], 1.0)
        self.assertEqual(metrics["math/sample_grad_contribution/projection_share_sum_error"], 0.0)

    def test_sample_gradient_cos_caches_structural_autograd_unavailability(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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

    def test_same_forward_sample_cos_metrics_use_core_tensorboard_tags_and_signed_projection(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            pass

        tracker = SameForwardDomainGradientProbe(
            ToyActor(),
            {
                "enabled": True,
                "domains": ["math"],
                "sample_gradient_enabled": True,
                "sample_gradient_cos_enabled": True,
            },
        )
        candidates = [
            {"row": {"loss_scale_factor": 1.0, "on_policy": True}, "micro_batch": object()},
            {"row": {"loss_scale_factor": 1.0, "on_policy": True}, "micro_batch": object()},
        ]
        tracker._sample_candidates = {"math": candidates}
        tracker._domain_target_chunks = {"math": ()}
        tracker._domain_norms = {"math": 1.0}
        tracker._domain_norm_sqs = {"math": 1.0}
        values = iter(
            [
                {"cosine": -0.25, "projection_share": -0.5},
                {"cosine": 0.75, "projection_share": 0.25},
            ]
        )

        tracker._recompute_sample_to_domain_stats = lambda *_args, **_kwargs: next(values)
        metrics = tracker._sample_cos_metrics()

        self.assertAlmostEqual(metrics["math/sample_grad_cos/domain_cos_mean"], 0.25, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_cos/domain_cos_negative_frac"], 0.5, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/projection_share_mean"], -0.125, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/projection_share_negative_frac"], 0.5, places=6)
        self.assertAlmostEqual(metrics["math/sample_grad_contribution/top1_abs_share"], 0.5, places=6)

    def test_same_forward_keeps_sample_gradient_metrics_under_replicated_distributed(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            pass

        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2):
            probe = SameForwardDomainGradientProbe(
                ToyActor(),
                {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "sample_gradient_enabled": True,
                    "sample_gradient_norm_enabled": True,
                    "sample_gradient_cos_enabled": True,
                    "domain_gradient_enabled": False,
                },
            )

        self.assertTrue(probe.sample_norm_enabled)
        self.assertTrue(probe.sample_cos_enabled)
        self.assertTrue(probe.sample_log_sample_level)
        self.assertTrue(probe.domain_gradient_enabled)

        metrics = probe.finish_mini_batch()

        self.assertEqual(metrics["global/audit/full_gradient_replicated_all_reduce"], 1.0)
        self.assertNotIn("global/audit/sample_gradient_distributed_unsupported", metrics)
        self.assertNotIn("global/audit/sample_gradient_distributed_world_size", metrics)
        self.assertNotIn("math/sample_grad/norm_mean", metrics)
        self.assertNotIn("math/sample_grad_cos/domain_cos_mean", metrics)

    def test_same_forward_sample_norm_only_does_not_compute_replicated_domain_gradient(self) -> None:
        try:
            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            pass

        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2):
            probe = SameForwardDomainGradientProbe(
                ToyActor(),
                {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "sample_gradient_enabled": True,
                    "sample_gradient_norm_enabled": True,
                    "sample_gradient_cos_enabled": False,
                    "domain_gradient_enabled": False,
                },
            )

        self.assertTrue(probe.sample_norm_enabled)
        self.assertFalse(probe.sample_cos_enabled)
        self.assertFalse(probe.domain_gradient_enabled)

        metrics = probe.finish_mini_batch()

        self.assertNotIn("global/audit/full_gradient_replicated_all_reduce", metrics)
        self.assertNotIn("global/audit/sample_gradient_distributed_unsupported", metrics)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)

    def test_same_forward_replicated_distributed_all_reduces_domain_targets(self) -> None:
        try:
            import math

            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import SameForwardDomainGradientProbe
        except ModuleNotFoundError as exc:  # pragma: no cover - local lightweight env
            self.skipTest(f"gradient probe dependencies are not installed: {exc}")

        class ToyActor:
            def __init__(self) -> None:
                self.actor_module = nn.Linear(2, 1, bias=False)

        def fake_all_reduce_vector_sum(vector: torch.Tensor) -> torch.Tensor:
            if torch.allclose(vector.cpu(), torch.tensor([1.0, 0.0])):
                vector.add_(torch.tensor([3.0, 0.0], dtype=vector.dtype, device=vector.device))
            elif torch.allclose(vector.cpu(), torch.tensor([0.0, 2.0])):
                vector.add_(torch.tensor([0.0, 5.0], dtype=vector.dtype, device=vector.device))
            return vector

        actor = ToyActor()
        with patch("mopd_verl.full_gradient_worker._distributed_world_size", return_value=2):
            probe = SameForwardDomainGradientProbe(
                actor,
                {
                    "enabled": True,
                    "domains": ["math", "code"],
                    "offload_domain_gradients": False,
                    "sample_gradient_enabled": True,
                    "sample_gradient_cos_enabled": True,
                },
            )

        probe._gpu_vectors = {
            "math": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "code": torch.tensor([0.0, 2.0], dtype=torch.float32),
        }
        probe._sample_counts = {"math": 1, "code": 1}

        with patch("mopd_verl.full_gradient_worker._all_reduce_vector_sum", fake_all_reduce_vector_sum):
            metrics = probe._compute_domain_summary_metrics()

        self.assertAlmostEqual(metrics["math/full_grad/grad_norm"], 4.0, places=6)
        self.assertAlmostEqual(metrics["code/full_grad/grad_norm"], 7.0, places=6)
        self.assertNotIn("global/full_grad/total_grad_norm", metrics)
        self.assertAlmostEqual(
            metrics["global/full_grad_conflict/code_vs_math/full_grad_cosine_train_i_k"],
            0.0,
            places=6,
        )
        self.assertTrue(torch.allclose(probe._gpu_vectors["math"], torch.tensor([4.0, 0.0])))
        self.assertTrue(torch.allclose(probe._gpu_vectors["code"], torch.tensor([0.0, 7.0])))
        self.assertTrue(torch.allclose(probe._domain_target_chunks["math"][0], torch.tensor([4.0, 0.0])))

    def test_low_precision_gradient_storage_uses_fp32_cosine_accumulation(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import (
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

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
        captured_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}

        def capture_targets(
            domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        ) -> dict[str, float]:
            captured_targets.update(domain_targets)
            return {}

        tracker._sample_cos_metrics = capture_targets
        actor.actor_optimizer.zero_grad()
        tracker.start_mini_batch()
        actor.actor_module(torch.tensor([[1.0, 0.0]])).sum().backward()
        tracker.after_backward("math", 1)
        actor.actor_module(torch.tensor([[0.0, 1.0]])).sum().backward()
        tracker.after_backward("code", 1)
        total_grad = next(actor.actor_module.parameters()).grad.clone()

        with patch(
            "mopd_verl.full_gradient_worker.get_torch_device",
            return_value=SimpleNamespace(max_memory_allocated=lambda: 0),
        ):
            tracker.finish_mini_batch()

        self.assertEqual(set(captured_targets), {"math", "code"})
        self.assertTrue(
            all(chunk.dtype == torch.bfloat16 for chunks, _ in captured_targets.values() for chunk in chunks)
        )
        self.assertTrue(torch.equal(next(actor.actor_module.parameters()).grad, total_grad))
        self.assertFalse(hasattr(tracker, "_sample_restore_grad_chunks"))

    def test_zero_sample_autograd_gradient_does_not_replace_training_gradient(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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

        with patch("mopd_verl.full_gradient_worker._actor_micro_batch_loss", return_value=zero_loss):
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

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
            "mopd_verl.full_gradient_worker._actor_micro_batch_loss",
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

            from mopd_verl.full_gradient_worker import _actor_micro_batch_loss
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
        with patch("mopd_verl.full_gradient_worker.get_device_id", return_value="cpu"):
            loss = _actor_micro_batch_loss(actor, FakeBatch(), loss_scale_factor=1.0, on_policy=True)
        gradient = torch.autograd.grad(loss, tuple(actor.actor_module.parameters()))[0]

        self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertGreater(float(gradient.item()), 0.0)

        actor.actor_module.weight.data.zero_()
        with patch("mopd_verl.full_gradient_worker.get_device_id", return_value="cpu"):
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

            from mopd_verl.full_gradient_worker import _actor_micro_batch_loss
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
        with patch("mopd_verl.full_gradient_worker.get_device_id", return_value="cpu"):
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

            from mopd_verl.full_gradient_worker import (
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
        with patch("mopd_verl.full_gradient_worker.get_device_id", return_value="cpu"):
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

        with patch("mopd_verl.full_gradient_worker.get_device_id", return_value="cpu"):
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

    def test_sample_recompute_does_not_backward_when_autograd_grad_is_zero(self) -> None:
        try:
            import torch
            from torch import nn

            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker
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
        with patch(
            "mopd_verl.full_gradient_worker._actor_micro_batch_loss",
            side_effect=lambda *_args, **_kwargs: actor.actor_module(torch.tensor([[2.0]])).sum(),
        ), patch(
            "torch.autograd.grad",
            return_value=(torch.zeros_like(parameter),),
        ), patch.object(
            torch.Tensor,
            "backward",
            side_effect=AssertionError("sample recompute must never call loss.backward()"),
        ):
            stats = tracker._recompute_sample_to_domain_stats(
                object(),
                target_chunks=(torch.tensor([1.0], dtype=torch.float32),),
                target_norm=1.0,
                target_norm_sq=1.0,
                loss_scale_factor=1.0,
                on_policy=True,
            )

        self.assertEqual(float(stats["sample_recompute_grad_norm"]), 0.0)
        self.assertEqual(stats["sample_recompute_non_none_grad_count"], 1.0)
        self.assertEqual(stats["sample_recompute_available"], 0.0)
        self.assertIsNone(stats["sample_projection_share"])
        self.assertTrue(torch.equal(parameter.grad, training_grad))

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
                "validation_probe.jsonl",
                "validation_gain_variance.jsonl",
                "training_cost.jsonl",
            ]
            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)
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
