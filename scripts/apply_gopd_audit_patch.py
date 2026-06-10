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
        r"(?: {28}full_gradient_output = self\.actor_rollout_wg\.compute_mopd_full_gradient_metrics\(batch\)\n"
        r" {28}metrics\.update\(reduce_metrics\(full_gradient_output\.meta_info\[\"metrics\"\]\)\)\n)?)?",
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


def _strip_dp_actor_audit_blocks(text: str) -> str:
    text = re.sub(
        r"\n {8,24}# MOPD audit: (?:same-forward domain-gradient probe|domain-gradient tracker) begin\n.*?"
        r"\n {8,24}# MOPD audit: (?:same-forward domain-gradient probe|domain-gradient tracker) end\n",
        "\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\n {16}micro_batches = list\(micro_batches\)\n"
        r" {16}if mopd_gradient_tracker is not None:\n"
        r" {20}tracked_micro_batches = mopd_gradient_tracker\.prepare_micro_batches\(micro_batches\)\n"
        r" {16}else:\n"
        r" {20}tracked_micro_batches = \[\(None, micro_batch\) for micro_batch in micro_batches\]\n",
        "\n",
        text,
    )
    return text.replace(
        "                for mopd_domain, micro_batch in tracked_micro_batches:\n",
        "                for micro_batch in micro_batches:\n",
    )


def _strip_main_ppo_domain_sampler_blocks(text: str) -> str:
    text = re.sub(
        r"\n {4}# MOPD audit: domain weighted sampler begin\n"
        r" {4}from mopd_verl\.domain_sampling import "
        r"create_domain_weighted_sampler as create_mopd_domain_weighted_sampler\n\n"
        r" {4}domain_sampler = create_mopd_domain_weighted_sampler\(data_config, dataset\)\n"
        r" {4}if domain_sampler is not None:\n"
        r" {8}sampler = domain_sampler\n"
        r" {4}# MOPD audit: domain weighted sampler end\n"
        r" {4}elif data_config\.sampler",
        "\n    if data_config.sampler",
        text,
    )
    return re.sub(
        r"\n {4}# MOPD audit: domain sampler begin\n.*?"
        r"\n {4}# MOPD audit: domain sampler end\n"
        r" {4}elif data_config\.sampler",
        "\n    if data_config.sampler",
        text,
        flags=re.DOTALL,
    )


def patch_dataset(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "utils" / "dataset" / "rl_dataset.py"
    changed = False
    changed |= _replace_once(
        path,
        '''        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
''',
        '''        for file_idx, parquet_file in enumerate(self.data_files):
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            original_file = self.original_data_files[file_idx] if file_idx < len(self.original_data_files) else parquet_file
            try:
                from mopd_verl.domain_sampling import annotate_hf_dataset_domain, domain_for_data_file

                domain = domain_for_data_file(self.config, original_file)
                if domain is not None:
                    dataframe = annotate_hf_dataset_domain(dataframe, domain)
            except ImportError:
                pass
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
''',
    )
    old = '''        if "opd_teacher" in row_dict.get("extra_info", {}):
            row_dict["opd_teacher"] = row_dict.get("extra_info", {}).get("opd_teacher")
            
        return row_dict
'''
    new = '''        if "opd_teacher" in row_dict.get("extra_info", {}) and "opd_teacher" not in row_dict:
            row_dict["opd_teacher"] = row_dict.get("extra_info", {}).get("opd_teacher")
        if "domain" in row_dict.get("extra_info", {}) and "domain" not in row_dict:
            row_dict["domain"] = row_dict.get("extra_info", {}).get("domain")
        if "sample_id" in row_dict.get("extra_info", {}):
            row_dict["sample_id"] = row_dict.get("extra_info", {}).get("sample_id")
        if "source_domain" in row_dict.get("extra_info", {}) and "source_domain" not in row_dict:
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
        if "validation_dataset" in row_dict.get("extra_info", {}):
            row_dict["validation_dataset"] = row_dict.get("extra_info", {}).get("validation_dataset")
            
        return row_dict
'''
    changed |= _replace_once(path, old, new)
    changed |= _replace_once(
        path,
        '''        if "source_domain" in row_dict.get("extra_info", {}):
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
            
        return row_dict
''',
        '''        if "source_domain" in row_dict.get("extra_info", {}) and "source_domain" not in row_dict:
            row_dict["source_domain"] = row_dict.get("extra_info", {}).get("source_domain")
        if "validation_dataset" in row_dict.get("extra_info", {}):
            row_dict["validation_dataset"] = row_dict.get("extra_info", {}).get("validation_dataset")
            
        return row_dict
''',
    )
    return changed


def patch_main_ppo(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "trainer" / "main_ppo.py"
    changed = False
    original = path.read_text(encoding="utf-8")
    normalized = _strip_main_ppo_domain_sampler_blocks(original)
    if normalized != original:
        path.write_text(normalized, encoding="utf-8")
        changed = True
    changed |= _replace_once(
        path,
        '''    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
''',
        '''    # MOPD audit: domain sampler begin
    if data_config.get("domain_train_files", None):
        return None

    from mopd_verl.domain_sampling import create_domain_weighted_sampler as create_mopd_domain_weighted_sampler

    domain_sampler = create_mopd_domain_weighted_sampler(data_config, dataset)
    if domain_sampler is not None:
        sampler = domain_sampler
    # MOPD audit: domain sampler end
    elif data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
''',
        required=True,
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
        '''        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
''',
        '''        train_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        # MOPD audit: exact domain batch sampler begin
        train_batch_sampler = None
        if train_sampler is None:
            from mopd_verl.domain_sampling import create_domain_batch_sampler as create_mopd_domain_batch_sampler

            train_batch_sampler = create_mopd_domain_batch_sampler(
                self.config.data,
                self.train_dataset,
                int(train_batch_size),
            )
            if train_batch_sampler is None:
                train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        # MOPD audit: exact domain batch sampler end
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        if train_batch_sampler is not None:
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        else:
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=train_batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=train_sampler,
            )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
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
    return changed


def patch_dp_actor(gopd_dir: Path) -> bool:
    path = gopd_dir / "verl" / "verl" / "workers" / "actor" / "dp_actor.py"
    changed = False
    original = path.read_text(encoding="utf-8")
    normalized = _strip_dp_actor_audit_blocks(original)
    if normalized != original:
        path.write_text(normalized, encoding="utf-8")
        changed = True
    changed |= _replace_once(
        path,
        '''        metrics = {}
''',
        '''        metrics = {}
        # MOPD audit: domain-gradient tracker begin
        mopd_gradient_tracker = None
        mopd_full_gradient_cfg = data.meta_info.get("mopd_full_gradient", {})
        if isinstance(mopd_full_gradient_cfg, dict) and mopd_full_gradient_cfg.get("enabled", False):
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker

            mopd_gradient_tracker = SequentialBackwardDomainGradientTracker(self, mopd_full_gradient_cfg)
        # MOPD audit: domain-gradient tracker end
''',
    )
    changed |= _replace_once(
        path,
        '''                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
''',
        '''                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                micro_batches = list(micro_batches)
                if mopd_gradient_tracker is not None:
                    tracked_micro_batches = mopd_gradient_tracker.prepare_micro_batches(micro_batches)
                else:
                    tracked_micro_batches = [(None, micro_batch) for micro_batch in micro_batches]

                self.actor_optimizer.zero_grad()
                # MOPD audit: domain-gradient tracker begin
                if mopd_gradient_tracker is not None:
                    mopd_gradient_tracker.start_mini_batch()
                # MOPD audit: domain-gradient tracker end

                for mopd_domain, micro_batch in tracked_micro_batches:
''',
    )
    changed |= _replace_once(
        path,
        '''                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
''',
        '''                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # MOPD audit: domain-gradient tracker begin
                    if mopd_gradient_tracker is not None:
                        mopd_gradient_tracker.after_backward(mopd_domain, len(micro_batch), micro_batch)
                    # MOPD audit: domain-gradient tracker end

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
''',
    )
    changed |= _replace_once(
        path,
        '''                grad_norm = self._optimizer_step()
''',
        '''                # MOPD audit: domain-gradient tracker begin
                if mopd_gradient_tracker is not None:
                    append_to_dict(metrics, mopd_gradient_tracker.finish_mini_batch())
                # MOPD audit: domain-gradient tracker end
                grad_norm = self._optimizer_step()
''',
    )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gopd_dir", help="Path to the G-OPD checkout root.")
    args = parser.parse_args()

    gopd_dir = Path(args.gopd_dir).resolve()
    changed = {
        "main_ppo": patch_main_ppo(gopd_dir),
        "dataset": patch_dataset(gopd_dir),
        "reward_score": patch_reward_score(gopd_dir),
        "trainer": patch_trainer(gopd_dir),
        "fsdp_worker": patch_fsdp_worker(gopd_dir),
        "dp_actor": patch_dp_actor(gopd_dir),
    }
    for name, was_changed in changed.items():
        print(f"{name}: {'patched' if was_changed else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
