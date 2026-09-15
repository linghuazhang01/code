# EOPD Teacher ref/log P Optimization

## Goal

Reduce the `timing_s/ref` bottleneck of the running EOPD configuration without changing the training objective, teacher outputs, batch semantics, or the currently running job.

## Baseline

- Run: `qwen1p7b-30b-eopd-native-8gpu-b528`
- `timing_s/ref`: about 752 s (about 51% of one step)
- Placement: 6 actor/rollout GPUs + 2 dedicated teacher GPUs
- Teacher GPU utilization: low duty cycle; memory headroom is visible
- Teacher post-processing: top-k 16 + entropy, chunked path enabled

## Phases

- [complete] 1. Audit the exact ref/log-prob execution path and earlier tuning evidence.
- [complete] 2. Select reversible, semantics-preserving optimization knobs.
- [complete] 3. Implement separate candidates and narrowly scoped runtime improvements.
- [complete] 4. Run focused unit/config tests and static checks.
- [complete] 5. Independent code review and GPU A/B validation protocol.

## Constraints

- Do not interrupt or restart the current run.
- Do not edit source code directly on the remote server.
- Preserve numerical semantics: same teacher checkpoint, top-k K, entropy calculation, sequence set, and global batch.
- Keep the tuned launch separate from the baseline for clean A/B comparison.

## Risks to verify

- Disabling teacher parameter offload can increase persistent VRAM use.
- Enlarging teacher microbatches can increase activation/logit peak memory.
- Distributed FSDP ranks must execute the same number and order of collectives.
- Chunk-size changes must not alter top-k or entropy results.

## Selected order

1. P0: compare an explicit stock-dispatch control against the exact-source-guarded stable-sort MoE path for Transformers 4.51.3.
2. P1: opt-in response-only teacher statistics, chosen/Top-K normalizer reuse, compact int32 support IDs, and adaptive 1024/256 chunk selection.
3. P1 follow-up: FSDP forward prefetch as an isolated A/B.
4. P2: full teacher replicas (`ref fsdp_size=1`) only after a bounded L20Y memory smoke.
