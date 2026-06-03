#!/usr/bin/env python3
"""Patch a G-OPD checkout so verl calls the local MOPD audit helpers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _replace_once(path: Path, old: str, new: str, required: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        if not required:
            return False
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _migrate_old_audit_logger_name(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    new_text = re.sub(r'(?<![."])\bmopd_audit_logger\b', "self.mopd_audit_logger", text)
    while "self.self." in new_text:
        new_text = new_text.replace("self.self.", "self.")
    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def _strip_trainer_audit_blocks(text: str) -> str:
    text = re.sub(
        r"\n {20}paper_eval_default_model_path = self\.config\.actor_rollout_ref\.model\.path\n"
        r".*?"
        r"\n {20}val_metrics\.update\(\n"
        r" {24}run_paper_eval_from_config\(\n"
        r" {28}self\.config,\n"
        r" {28}step=self\.global_steps,\n"
        r" {28}default_model_path=paper_eval_default_model_path,\n"
        r" {24}\)\n"
        r" {20}\)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\n {12}paper_eval_default_model_path = self\.config\.actor_rollout_ref\.model\.path\n"
        r".*?"
        r"\n {12}val_metrics\.update\(\n"
        r" {16}run_paper_eval_from_config\(\n"
        r" {20}self\.config,\n"
        r" {20}step=self\.global_steps,\n"
        r" {20}default_model_path=paper_eval_default_model_path,\n"
        r" {16}\)\n"
        r" {12}\)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\n {20}val_metrics\.update\(\n"
        r" {24}run_paper_eval_from_config\(\n"
        r" {28}self\.config,\n"
        r" {28}step=self\.global_steps,\n"
        r" {28}default_model_path=self\.config\.actor_rollout_ref\.model\.path,\n"
        r" {24}\)\n"
        r" {20}\)",
        "",
        text,
    )
    text = re.sub(
        r"\n {12}val_metrics\.update\(\n"
        r" {16}run_paper_eval_from_config\(\n"
        r" {20}self\.config,\n"
        r" {20}step=self\.global_steps,\n"
        r" {20}default_model_path=self\.config\.actor_rollout_ref\.model\.path,\n"
        r" {16}\)\n"
        r" {12}\)",
        "",
        text,
    )
    text = re.sub(
        r"\n {20}if self\.mopd_audit_logger\.enabled:\n"
        r" {24}audit_lr = self\.config\.actor_rollout_ref\.actor\.optim\.lr\n"
        r" {24}audit_metrics = self\.mopd_audit_logger\.log_training_step\(\n"
        r" {28}batch=batch,\n"
        r" {28}step=self\.global_steps,\n"
        r" {28}lr=audit_lr,\n"
        r" {24}\)\n"
        r" {24}metrics\.update\(audit_metrics\)\n"
        r"(?: {24}if self\.mopd_audit_logger\.should_compute_full_gradient\(self\.global_steps\):\n"
        r" {28}batch\.meta_info\.update\(self\.mopd_audit_logger\.full_gradient_meta\(\"train\", self\.global_steps\)\)\n"
        r" {28}full_gradient_output = self\.actor_rollout_wg\.compute_mopd_full_gradient_metrics\(batch\)\n"
        r" {28}metrics\.update\(reduce_metrics\(full_gradient_output\.meta_info\[\"metrics\"\]\)\)\n)?",
        "",
        text,
    )
    text = re.sub(
        r"\n {20}metrics\.update\(self\.mopd_audit_logger\.log_validation_metrics\(val_metrics, self\.global_steps\)\)",
        "",
        text,
    )
    text = re.sub(
        r"\n {12}audit_val_metrics = self\.mopd_audit_logger\.log_validation_metrics\(val_metrics, self\.global_steps\)\n"
        r" {12}logger\.log\(data=(?:self\.mopd_audit_logger\.filter_tensorboard_metrics\()?(\{\*\*val_metrics, \*\*audit_val_metrics\})(?:\))?, step=self\.global_steps\)",
        "\n            logger.log(data=val_metrics, step=self.global_steps)",
        text,
    )
    text = re.sub(
        r"\n {16}logger\.log\(data=self\.mopd_audit_logger\.filter_tensorboard_metrics\(metrics\), step=self\.global_steps\)",
        "\n                logger.log(data=metrics, step=self.global_steps)",
        text,
    )
    text = re.sub(
        r"\n {12}# MOPD audit: validation anchor begin\n.*?\n {12}# MOPD audit: validation anchor end\n",
        "\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\n {12}if \(\n"
        r" {16}getattr\(self, \"mopd_audit_logger\", None\) is not None\n"
        r" {16}and self\.mopd_audit_logger\.should_update_validation_anchor\(self\.global_steps\)\n"
        r" {12}\):\n.*?\n {12}# evaluate using reward_function",
        "\n            # evaluate using reward_function",
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        'test_batch.meta_info["validate"] = True\n\n\n            # evaluate using reward_function',
        'test_batch.meta_info["validate"] = True\n\n            # evaluate using reward_function',
    )
    return text


def _strip_fsdp_worker_audit_blocks(text: str) -> str:
    return re.sub(
        r"\n {4}# MOPD audit: full-parameter gradient helpers begin\n.*?"
        r"\n {4}# MOPD audit: full-parameter gradient helpers end\n",
        "\n",
        text,
        flags=re.DOTALL,
    )


def patch_dataset(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "utils" / "dataset" / "rl_dataset.py"
    old = '''        if "opd_teacher" in row_dict.get("extra_info", {}):
            row_dict["opd_teacher"] = row_dict.get("extra_info", {}).get("opd_teacher")
            
        return row_dict
'''
    new = '''        if "opd_teacher" in row_dict.get("extra_info", {}):
            row_dict["opd_teacher"] = row_dict.get("extra_info", {}).get("opd_teacher")
        if "domain" in row_dict.get("extra_info", {}):
            row_dict["domain"] = row_dict.get("extra_info", {}).get("domain")
        if "sample_id" in row_dict.get("extra_info", {}):
            row_dict["sample_id"] = row_dict.get("extra_info", {}).get("sample_id")
        if "source_domain" in row_dict.get("extra_info", {}):
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
        if "validation_dataset" in row_dict.get("extra_info", {}):
            row_dict["validation_dataset"] = row_dict.get("extra_info", {}).get("validation_dataset")
            
        return row_dict
'''
    changed = _replace_once(path, old, new)
    changed |= _replace_once(
        path,
        '''        if "source_domain" in row_dict.get("extra_info", {}):
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
            
        return row_dict
''',
        '''        if "source_domain" in row_dict.get("extra_info", {}):
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
        if "validation_dataset" in row_dict.get("extra_info", {}):
            row_dict["validation_dataset"] = row_dict.get("extra_info", {}).get("validation_dataset")
            
        return row_dict
''',
    )
    return changed


def patch_reward_score(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "utils" / "reward_score" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    if (
        "HMMT25Feb" in text
        and "HMMT25Nov" in text
        and "HumanEvalPlus" in text
        and "MBPPPlus" in text
        and "LiveCodeBench" in text
        and "code_reward.compute_score" in text
    ):
        return False
    changed = False
    old = '"AIME2024", "AIME2025", "MMLUPro"'
    new = '"AIME2024", "AIME2025", "HMMT25Feb", "HMMT25Nov", "HMMT", "MMLUPro"'
    if old in text:
        text = text.replace(old, new, 1)
        changed = True
    elif "HMMT25Feb" not in text or "HMMT25Nov" not in text:
        raise RuntimeError(f"Reward score patch anchor not found in {path}: {old!r}")

    old_extended = (
        'elif data_source in ["codecontests", "apps", "codeforces", "taco", '
        '"HumanEvalPlus", "MBPPPlus", "LiveCodeBench"]:'
    )
    old = 'elif data_source in ["codecontests", "apps", "codeforces", "taco"]:'
    new = '''elif data_source in ["HumanEvalPlus", "MBPPPlus", "LiveCodeBench"]:
        from mopd_verl import code_reward

        res = code_reward.compute_score(data_source, solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:'''
    if old_extended in text:
        text = text.replace(old_extended, new, 1)
        changed = True
    elif old in text:
        text = text.replace(old, new, 1)
        changed = True
    elif "code_reward.compute_score" not in text:
        raise RuntimeError(f"Reward score patch anchor not found in {path}: {old!r}")
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_trainer(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    changed = False
    original = path.read_text(encoding="utf-8")
    normalized = _strip_trainer_audit_blocks(original)
    normalized = re.sub(r'(?<![."])\bmopd_audit_logger\b', "self.mopd_audit_logger", normalized)
    while "self.self." in normalized:
        normalized = normalized.replace("self.self.", "self.")
    if normalized != original:
        path.write_text(normalized, encoding="utf-8")
        changed = True
    changed |= _replace_once(
        path,
        "from verl import DataProto\nfrom mopd_verl.verl_audit import MOPDAuditLogger\n",
        "from verl import DataProto\nfrom mopd_verl.paper_eval import run_paper_eval_from_config\nfrom mopd_verl.verl_audit import MOPDAuditLogger\n",
    )
    changed |= _replace_once(
        path,
        "from verl import DataProto\n",
        "from verl import DataProto\nfrom mopd_verl.paper_eval import run_paper_eval_from_config\nfrom mopd_verl.verl_audit import MOPDAuditLogger\n",
    )
    changed |= _replace_once(
        path,
        '''        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
''',
        '''        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.mopd_audit_logger = MOPDAuditLogger(self.config)

        self.global_steps = 0
''',
    )
    changed |= _replace_once(
        path,
        '''        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
''',
        '''        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.mopd_validation_anchor_dataloader = None
        mopd_audit_cfg = self.config.get("mopd_audit", {})
        anchor_files = mopd_audit_cfg.get("full_gradient_validation_files", [])
        if anchor_files:
            anchor_data_config = deepcopy(self.config.data)
            with open_dict(anchor_data_config):
                anchor_data_config.shuffle = False
                anchor_data_config.validation_shuffle = False
            anchor_dataset = create_rl_dataset(
                anchor_files,
                anchor_data_config,
                self.tokenizer,
                self.processor,
                max_samples=-1,
            )
            anchor_batch_size = mopd_audit_cfg.get("full_gradient_validation_batch_size", None)
            if anchor_batch_size is None:
                anchor_batch_size = 1
            self.mopd_validation_anchor_dataloader = StatefulDataLoader(
                dataset=anchor_dataset,
                batch_size=int(anchor_batch_size),
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            assert len(self.mopd_validation_anchor_dataloader) >= 1, "MOPD validation anchor dataloader is empty!"
            print(
                f"Size of MOPD validation anchor dataloader: {len(self.mopd_validation_anchor_dataloader)}, "
                f"batch size: {anchor_batch_size}"
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
''',
    )
    changed |= _replace_once(
        path,
        '''    def _validate(self):
''',
        '''    def _mopd_add_validation_anchor_log_probs(self, anchor_batch, size_divisor):
        anchor_batch, anchor_pad_size = pad_dataproto_to_divisor(anchor_batch, size_divisor)
        old_log_prob = self.actor_rollout_wg.compute_log_prob(anchor_batch)
        if "entropys" in old_log_prob.batch:
            old_log_prob.batch.pop("entropys")
        anchor_batch = anchor_batch.union(old_log_prob)
        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(anchor_batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(anchor_batch)
            anchor_batch = anchor_batch.union(ref_log_prob)
        if self.use_base_models:
            if self.ref_base_model_path is not None:
                if not self.ref_in_actor:
                    base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(anchor_batch)
                else:
                    base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(anchor_batch)
                anchor_batch = anchor_batch.union(base_ref_log_prob)
            if self.base_model_path is not None:
                ref_input_tensors = {}
                for tensor_key in ("ref_input_ids", "ref_attention_mask", "ref_position_ids"):
                    if tensor_key in anchor_batch.batch:
                        ref_input_tensors[tensor_key] = anchor_batch.batch.pop(tensor_key)
                base_log_prob = self.actor_rollout_wg.compute_base_log_prob(anchor_batch)
                anchor_batch = anchor_batch.union(base_log_prob)
                for tensor_key, tensor_value in ref_input_tensors.items():
                    anchor_batch.batch[tensor_key] = tensor_value
        return unpad_dataproto(anchor_batch, pad_size=anchor_pad_size)

    def _compute_mopd_validation_anchor_metrics(self):
        if (
            getattr(self, "mopd_audit_logger", None) is None
            or not self.mopd_audit_logger.should_update_validation_anchor(self.global_steps)
            or getattr(self, "mopd_validation_anchor_dataloader", None) is None
        ):
            return {}

        audit_validation_metrics = {}
        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        for anchor_data in self.mopd_validation_anchor_dataloader:
            anchor_batch = DataProto.from_single_dict(anchor_data)
            if "uid" not in anchor_batch.non_tensor_batch:
                anchor_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(anchor_batch.batch))], dtype=object
                )
            anchor_batch = anchor_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            anchor_gen_batch = self._get_gen_batch(anchor_batch)
            anchor_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            anchor_gen_batch_padded, pad_size = pad_dataproto_to_divisor(anchor_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                anchor_output_padded = self.actor_rollout_wg.generate_sequences(anchor_gen_batch_padded)
            else:
                anchor_output_padded = self.async_rollout_manager.generate_sequences(anchor_gen_batch_padded)
            anchor_output = unpad_dataproto(anchor_output_padded, pad_size=pad_size)
            anchor_batch = anchor_batch.union(anchor_output)
            anchor_batch.meta_info["validate"] = True
            anchor_batch = self._mopd_add_validation_anchor_log_probs(anchor_batch, size_divisor)
            audit_validation_metrics.update(
                self.mopd_audit_logger.log_validation_anchor_batch(anchor_batch, self.global_steps)
            )
            if self.mopd_audit_logger.should_compute_full_gradient(self.global_steps):
                anchor_batch.meta_info.update(
                    self.mopd_audit_logger.full_gradient_meta("validation_anchor", self.global_steps)
                )
                full_gradient_output = self.actor_rollout_wg.compute_mopd_full_gradient_metrics(anchor_batch)
                audit_validation_metrics.update(reduce_metrics(full_gradient_output.meta_info["metrics"]))
        return audit_validation_metrics

    def _validate(self):
''',
    )
    changed |= _replace_once(
        path,
        '''                    # update critic
                    if self.use_critic:
''',
        '''                    if self.mopd_audit_logger.enabled:
                        audit_lr = self.config.actor_rollout_ref.actor.optim.lr
                        audit_metrics = self.mopd_audit_logger.log_training_step(
                            batch=batch,
                            step=self.global_steps,
                            lr=audit_lr,
                        )
                        metrics.update(audit_metrics)
                        if self.mopd_audit_logger.should_compute_full_gradient(self.global_steps):
                            batch.meta_info.update(self.mopd_audit_logger.full_gradient_meta("train", self.global_steps))
                            full_gradient_output = self.actor_rollout_wg.compute_mopd_full_gradient_metrics(batch)
                            metrics.update(reduce_metrics(full_gradient_output.meta_info["metrics"]))

                    # update critic
                    if self.use_critic:
''',
    )
    changed |= _replace_once(
        path,
        '''                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
''',
        '''                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    paper_eval_default_model_path = self.config.actor_rollout_ref.model.path
                    paper_eval_cfg = self.config.get("paper_eval", {})
                    if (
                        paper_eval_cfg.get("enabled", False)
                        and paper_eval_cfg.get("evaluate_current_checkpoint", True)
                        and self.global_steps > 0
                    ):
                        paper_eval_checkpoint_root = self.config.trainer.default_local_dir
                        if not os.path.isabs(paper_eval_checkpoint_root):
                            paper_eval_checkpoint_root = os.path.join(os.getcwd(), paper_eval_checkpoint_root)
                        self._save_checkpoint()
                        paper_eval_default_model_path = os.path.join(
                            paper_eval_checkpoint_root, f"global_step_{self.global_steps}", "actor"
                        )
                    val_metrics.update(
                        run_paper_eval_from_config(
                            self.config,
                            step=self.global_steps,
                            default_model_path=paper_eval_default_model_path,
                        )
                    )
                    metrics.update(val_metrics)
                    metrics.update(self.mopd_audit_logger.log_validation_metrics(val_metrics, self.global_steps))
''',
    )
    changed |= _replace_once(
        path,
        '''            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
''',
        '''            paper_eval_default_model_path = self.config.actor_rollout_ref.model.path
            val_metrics.update(
                run_paper_eval_from_config(
                    self.config,
                    step=self.global_steps,
                    default_model_path=paper_eval_default_model_path,
                )
            )
            pprint(f"Initial validation metrics: {val_metrics}")
            audit_val_metrics = self.mopd_audit_logger.log_validation_metrics(val_metrics, self.global_steps)
            logger.log(
                data=self.mopd_audit_logger.filter_tensorboard_metrics({**val_metrics, **audit_val_metrics}),
                step=self.global_steps,
            )
            if self.config.trainer.get("val_only", False):
                return
''',
    )
    changed |= _replace_once(
        path,
        '''            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            audit_val_metrics = self.mopd_audit_logger.log_validation_metrics(val_metrics, self.global_steps)
            logger.log(
                data=self.mopd_audit_logger.filter_tensorboard_metrics({**val_metrics, **audit_val_metrics}),
                step=self.global_steps,
            )
            if self.config.trainer.get("val_only", False):
                return
''',
        '''            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            paper_eval_default_model_path = self.config.actor_rollout_ref.model.path
            val_metrics.update(
                run_paper_eval_from_config(
                    self.config,
                    step=self.global_steps,
                    default_model_path=paper_eval_default_model_path,
                )
            )
            pprint(f"Initial validation metrics: {val_metrics}")
            audit_val_metrics = self.mopd_audit_logger.log_validation_metrics(val_metrics, self.global_steps)
            logger.log(
                data=self.mopd_audit_logger.filter_tensorboard_metrics({**val_metrics, **audit_val_metrics}),
                step=self.global_steps,
            )
            if self.config.trainer.get("val_only", False):
                return
''',
    )
    changed |= _replace_once(
        path,
        '''                logger.log(data=metrics, step=self.global_steps)
''',
        '''                logger.log(data=self.mopd_audit_logger.filter_tensorboard_metrics(metrics), step=self.global_steps)
''',
    )
    changed |= _replace_once(
        path,
        '''        sample_uids = []
''',
        '''        sample_uids = []
        audit_validation_metrics = {}
''',
    )
    changed |= _replace_once(
        path,
        '''                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation
''',
        '''                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                if self.mopd_audit_logger.enabled:
                    metrics.update(
                        self.mopd_audit_logger.log_training_cost(metrics=metrics, step=self.global_steps, n_gpus=n_gpus)
                    )
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation
''',
    )
    changed |= _replace_once(
        path,
        '''            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
''',
        '''            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # MOPD audit: validation anchor begin
            if (
                getattr(self, "mopd_audit_logger", None) is not None
                and self.mopd_audit_logger.should_update_validation_anchor(self.global_steps)
                and getattr(self, "mopd_validation_anchor_dataloader", None) is None
            ):
                anchor_batch = test_batch
                anchor_batch, anchor_pad_size = pad_dataproto_to_divisor(anchor_batch, size_divisor)
                old_log_prob = self.actor_rollout_wg.compute_log_prob(anchor_batch)
                if "entropys" in old_log_prob.batch:
                    old_log_prob.batch.pop("entropys")
                anchor_batch = anchor_batch.union(old_log_prob)
                if self.use_reference_policy:
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(anchor_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(anchor_batch)
                    anchor_batch = anchor_batch.union(ref_log_prob)
                if self.use_base_models:
                    if self.ref_base_model_path is not None:
                        if not self.ref_in_actor:
                            base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(anchor_batch)
                        else:
                            base_ref_log_prob = self.actor_rollout_wg.compute_base_ref_log_prob(anchor_batch)
                        anchor_batch = anchor_batch.union(base_ref_log_prob)
                    if self.base_model_path is not None:
                        ref_input_tensors = {}
                        for tensor_key in ("ref_input_ids", "ref_attention_mask", "ref_position_ids"):
                            if tensor_key in anchor_batch.batch:
                                ref_input_tensors[tensor_key] = anchor_batch.batch.pop(tensor_key)
                        base_log_prob = self.actor_rollout_wg.compute_base_log_prob(anchor_batch)
                        anchor_batch = anchor_batch.union(base_log_prob)
                        for tensor_key, tensor_value in ref_input_tensors.items():
                            anchor_batch.batch[tensor_key] = tensor_value
                anchor_batch = unpad_dataproto(anchor_batch, pad_size=anchor_pad_size)
                audit_validation_metrics.update(
                    self.mopd_audit_logger.log_validation_anchor_batch(anchor_batch, self.global_steps)
                )
                if self.mopd_audit_logger.should_compute_full_gradient(self.global_steps):
                    anchor_batch.meta_info.update(
                        self.mopd_audit_logger.full_gradient_meta("validation_anchor", self.global_steps)
                    )
                    full_gradient_output = self.actor_rollout_wg.compute_mopd_full_gradient_metrics(anchor_batch)
                    audit_validation_metrics.update(reduce_metrics(full_gradient_output.meta_info["metrics"]))
            # MOPD audit: validation anchor end

            # evaluate using reward_function
''',
    )
    changed |= _replace_once(
        path,
        '''        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict
''',
        '''        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if getattr(self, "mopd_validation_anchor_dataloader", None) is not None:
            audit_validation_metrics.update(self._compute_mopd_validation_anchor_metrics())

        metric_dict.update(audit_validation_metrics)
        return metric_dict
''',
    )
    changed |= _migrate_old_audit_logger_name(path)
    return changed


def patch_fsdp_worker(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "workers" / "fsdp_workers.py"
    changed = False
    original = path.read_text(encoding="utf-8")
    normalized = _strip_fsdp_worker_audit_blocks(original)
    if normalized != original:
        path.write_text(normalized, encoding="utf-8")
        changed = True
    changed |= _replace_once(
        path,
        '''    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
''',
        '''    # MOPD audit: full-parameter gradient helpers begin
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="cyan", role="mopd_full_gradient")
    def compute_mopd_full_gradient_metrics(self, data: DataProto):
        from mopd_verl.full_gradient_worker import compute_mopd_full_gradient_metrics

        return compute_mopd_full_gradient_metrics(self, data)
    # MOPD audit: full-parameter gradient helpers end

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
''',
    )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gopd_dir", help="Path to the G-OPD checkout root.")
    args = parser.parse_args()

    gopd_dir = Path(args.gopd_dir).resolve()
    changed = {
        "dataset": patch_dataset(gopd_dir),
        "reward_score": patch_reward_score(gopd_dir),
        "trainer": patch_trainer(gopd_dir),
        "fsdp_worker": patch_fsdp_worker(gopd_dir),
    }
    for name, was_changed in changed.items():
        print(f"{name}: {'patched' if was_changed else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
