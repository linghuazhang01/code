"""Minimal verl driver for the dedicated paper-native TIP update."""

from __future__ import annotations

import time
import uuid
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class TIPNativeTrainer(RayPPOTrainer):
    """Reuse verl rollout, validation, and checkpointing around native TIP."""

    def fit(self) -> None:
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train",
            True,
        ):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="TIP-native training",
        )
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    progress.close()
                    return
                self.global_steps += 1
                step_start = time.time()
                metrics: dict[str, Any] = {}

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["do_sample"] = (
                    self.config.actor_rollout_ref.rollout.do_sample
                )
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                generation_start = time.time()
                generation, generation_metrics = self._generate_training_sequences(
                    gen_batch
                )
                metrics.update(generation_metrics)
                metrics["timing/gen"] = time.time() - generation_start
                generation.meta_info.pop("timing", None)

                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch = batch.union(generation)
                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"],
                    dim=-1,
                ).tolist()
                batch.meta_info["multi_turn"] = False

                update_start = time.time()
                update_output = self.actor_rollout_wg.update_tip_native(batch)
                metrics.update(reduce_metrics(update_output.meta_info["metrics"]))
                metrics["timing/update_tip_native"] = time.time() - update_start

                is_last = self.global_steps >= self.total_training_steps
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last
                        or self.global_steps % self.config.trainer.test_freq == 0
                    )
                ):
                    metrics.update(self._validate())
                if self.config.trainer.save_freq > 0 and (
                    is_last
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "timing/step": time.time() - step_start,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)
                progress.update(1)
                if is_last:
                    progress.close()
                    return
        progress.close()
