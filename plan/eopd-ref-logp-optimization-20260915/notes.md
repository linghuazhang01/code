# Investigation Notes

## Live baseline evidence

- Step time: about 1464 s.
- Teacher `ref/log P`: about 752 s.
- Generation: about 347 s.
- Actor update: about 272 s.
- Dedicated teacher GPUs show substantially lower utilization than the actor/rollout GPUs and retain memory headroom.

## Current configuration

- `actor_rollout_ref.worker_placement.separate_ref_policy: true`
- 6 actor/rollout GPUs and 2 teacher GPUs.
- Teacher top-k: 16; teacher entropy enabled.
- `teacher_performance.enabled: true`
- `teacher_performance.logprob_chunk_size: 1024`
- `teacher_performance.max_micro_batch_size: 32`
- `teacher_performance.max_tokens_per_micro_batch: 57344`
- Ref log-prob microbatch defaults to 1 before the performance wrapper groups examples.
- Ref FSDP parameter offload is enabled by the launcher default.

## Findings

1. The ref `param_offload=True` launcher override is not on this dedicated GPU teacher's hot path. `teacher_model_device=gpu` builds without FSDP CPUOffload, and `compute_ref_log_prob` performs no manual ref load/offload.
2. The live run uses Transformers 4.51.3. The prior stable-sort allowlist contained only 4.57.6, so the configured optimization fell back to stock dispatch. The official 4.51.3 wheel source hash was independently verified as `c4ae8784c667994091405761b588d9d73f8b4290f5bd4d43dcd5ea3842dc63cb`.
3. Prior fixed-input H200 evidence measured stable-sort at 22.18% faster for complete teacher compute. Applying that percentage to 752 s gives about 585 s ref and about 1297 s/step, but this is an estimate, not an L20Y measurement.
4. EOPD computes chosen logP, entropy, and Top-K over the same full-vocabulary logits. The old remove-padding path also computes these statistics for prompt positions and discards them after padding.
5. Entropy must retain the legacy dtype/runtime function: a local BF16 check found FP32 entropy can differ by about 0.1, enough to change the EOPD 0.8 gate.
6. Reducing the Top-K chunk from 1024 to 256 frees about 1.74 GiB of modeled workspace, equivalent to 1536 tokens of batching capacity under the current guard. The adaptive policy uses 256 only when it admits more rows; ties retain 1024.
7. Compact int32 Top-K IDs save up to 528 MiB for a padded `[528, 16384, 16]` support tensor. The student path converts IDs back to long before `torch.gather`.
8. A replicated two-teacher layout can remove forward all-gathers, and the existing FSDP1 `fsdp_size=1` contract provides WORLD `NO_SHARD`. It is higher risk because a full 30B BF16 replica approaches the L20Y memory limit.

## Verification boundary

- The current running job was not interrupted or changed.
- Local tests verify routing, config composition, adaptive grouping, response indexing, source guards, and CPU numerical contracts.
- Real L20Y speed, peak memory, chosen-logP error, and entropy gate decisions still require the documented fixed-batch GPU A/B.
