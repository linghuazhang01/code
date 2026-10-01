# GPU performance configuration

Training profiles request the following rollout and reference-teacher settings:

```yaml
rollout:
  enforce_eager: false
  max_num_seqs: 64

teacher_performance:
  enabled: true
  topk_logprob_chunk_enabled: true
  topk_logprob_chunk_size: 1024
  max_micro_batch_size: 32
  max_tokens: 57344
  moe_dispatch: stable_sort
  memory_margin_gib: 12.0
```

CUDA Graph changes execution scheduling; the configured Flash Attention backend
is unchanged. Existing engines must restart before changed rollout settings take
effect. `max_num_seqs` controls concurrent sequences rather than the global
training batch size. The configured KV-cache memory fraction, generation length,
sampling policy and actor optimizer batch sizes remain independent.

## Teacher behavior and applicability

`settings.py` validates the teacher configuration and `launch.py` forwards all
fields under `actor_rollout_ref.ref.teacher_performance`. Reference-worker
initialization installs the wrappers in `teacher_performance.py`.

- The TopK chunk controls how many token positions are postprocessed together;
  it does not change the number of returned support tokens (`topk_distill_k=32`).
  Memory-guarded chunking defaults to 1024 with the independent
  `topk_logprob_chunk_enabled: true`, including colocated multi-rank teachers
  and configurations with `enabled: false`. It preserves forward microbatches
  and collective call order. Explicit call-site chunk overrides take priority.
- With the batching bundle enabled, teacher input rows are grouped contiguously,
  up to 32 sequences and 57,344
  non-padding input tokens. Output row order and optional output tensors are
  preserved. Each group is reduced further when the memory estimate requires it.
- Optimization requires a CUDA-resident BF16 reference model, remove-padding,
  sequence parallel size one, no optimizer and unfused execution. A dedicated
  multi-rank FSDP reference worker is supported: it synchronizes rank-local
  batch boundaries using the minimum available capacity before entering the
  model, preserving collective call order. Colocated multi-rank teachers retain
  their original batching and MoE path while independently using the chunk
  wrapper. Unsupported execution paths retain their original methods and log
  the reason; chunking does not bypass these model/device guards.
- The memory budget estimates logits and chunk workspace using free and cached
  memory while retaining at least 12 GiB headroom. It is a heuristic, not an OOM
  guarantee: cached bytes may not all be immediately reusable contiguous memory.
  An oversized single row falls back to chunk16 with a warning. The chunk
  wrapper records the last configured/effective chunk and fallback reason in
  `_teacher_chunk_stats`; INFO logs report the first selection and any change.
  The module enables INFO by default even with verl's WARNING root logger,
  while preserving an explicitly configured module log level.
- Stable-sort MoE dispatch is enabled only for the verified 48-block Qwen3 MoE
  implementation in Transformers 4.57.6 with its matching original forward hash.
  Other model implementations retain stock dispatch. Expert accumulation order
  is preserved to limit floating-point differences.
- Expandable CUDA allocator segments are enabled only for a pure reference
  worker with `worker_placement.separate_ref_policy=true`. The setting is
  process-wide, so a ref child in a shared actor/rollout process must not enable
  it merely because its own role is `ref`.

Caps are intentionally limited to the tested upper bounds: chunk 1024,
32 sequences, 57,344 tokens; the memory margin must be finite and at least 12 GiB.
For a dedicated multi-rank reference worker, the effective batch capacity is the
minimum across ranks, so adding reference GPUs does not silently create uneven
collective schedules.
`teacher_performance.enabled: true` is the base setting for every training
profile, including colocated templates; colocated multi-rank teachers fall back
to their original batching and MoE path at runtime, while separate reference
workers (dedicated or `share_ref_policy_gpus`) use the bundle. On 2026-10-01 the
shared 4-student/3-teacher V6 run spent 1254–1279 s per step in the reference
phase with the bundle disabled (steps 1–4), versus 51–62 s for the bundled
6-student/2-teacher `retry1` run at similar token counts (different nodes:
H200 NVL versus L20Y, so this is not a matched benchmark).
Use `teacher_performance.enabled: false` to disable the batching/MoE/statistics
bundle while retaining the independent chunk default. To disable automatic
chunking explicitly, set `topk_logprob_chunk_enabled: false`; with batching
still enabled, its automatic chunk remains 16 and adaptive chunk selection is
disabled. Both switches must be false to retain the entire original teacher
path. `moe_dispatch: stock` disables only the MoE dispatch replacement.
Expandable allocator segments remain a separate dedicated-worker setting.
Changed teacher settings take effect when workers are initialized again.

## Validation evidence and limits

A fixed-input H200 experiment with the matching external-launcher backend
reduced rollout time from 178.24 s (eager, 24 sequences) to 130.98 s (Graph, 64
sequences), with no preemptions. Generation-period NVML utilization increased
from about 83% to 99.9%; this is not whole-training-step average utilization.

For the same 64-row teacher workload, chunk 256 to 1024 reduced mean time from
24.54 s to 23.67 s (3.54%), while allocated memory rose by about 0.87 GiB.
Chosen-token logP and Top32 IDs matched exactly; maximum Top32 logP difference
was 3.82e-6. This chunk increase did not improve average NVML utilization.

The resumed 4-actor/1-teacher H200 training run loaded a complete step25
checkpoint, captured CUDA Graphs, synchronized actor weights, and completed
step26/27. Teacher runtime readback confirmed all requested caps, 48 optimized
MoE blocks and expandable segments; its first allocated-memory peak was
75.42 GiB. Successful initial steps do not establish long-run OOM immunity or
performance for all model sizes and distributed layouts.

The profile-coverage test checks inheritance and matrix entries. Historical
server-only deployment snapshots are maintained separately from the reviewed
repository configuration set. In particular, old paper-native EOPD snapshots
with non-null `rollout_correction.rollout_is` are rejected by current validation;
performance tuning does not silently rewrite their objective configuration.

The independent chunk regression checks colocated world-size4 execution with
the batching bundle disabled, preserving forward count/order and input metadata.
Fixed CPU FP32/BF16 logits compare chunk16/256/1024 for identical Top32 IDs,
logP and reverse-KL loss. These checks do not measure H200 performance or peak
memory for the current colocated run.

## Timing interpretation

A typical step executes rollout, reward/student logP recomputation, teacher,
advantage preparation, and actor update mostly sequentially. Four actor workers
run in parallel; their wall times must not be added as if they were sequential.
`timing_s/step` excludes periodic validation and checkpoint saving. Core logging
retains generation/update/step timing but prunes teacher and old-logP timing;
retained Ray task events can provide worker execution durations without adding
instrumentation. Worker timing and driver timing have different RPC boundaries.
