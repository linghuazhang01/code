# GPU performance configuration

Training profiles request the following rollout and reference-teacher settings:

```yaml
rollout:
  enforce_eager: false
  max_num_seqs: 64

teacher_performance:
  enabled: true
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

`settings.py` validates the teacher configuration and `launch.py` forwards its six
fields under `actor_rollout_ref.ref.teacher_performance`. Reference-worker
initialization installs the wrappers in `teacher_performance.py`.

- The TopK chunk controls how many token positions are postprocessed together;
  it does not change the number of returned support tokens (`topk_distill_k=32`).
- Teacher input rows are grouped contiguously, up to 32 sequences and 57,344
  non-padding input tokens. Output row order and optional output tensors are
  preserved. Each group is reduced further when the memory estimate requires it.
- Optimization requires a single-rank CUDA-resident BF16 reference model,
  remove-padding, sequence parallel size one, no optimizer and unfused execution.
  Unsupported layouts retain the original computation path and log the reason.
- The memory budget estimates logits and chunk workspace using free and cached
  memory while retaining at least 12 GiB headroom. It is a heuristic, not an OOM
  guarantee: cached bytes may not all be immediately reusable contiguous memory.
  An oversized single row retains the original chunk behavior.
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
Use `teacher_performance.enabled: false` to retain the original teacher path,
or `moe_dispatch: stock` to disable only the MoE dispatch replacement.

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

## Timing interpretation

A typical step executes rollout, reward/student logP recomputation, teacher,
advantage preparation, and actor update mostly sequentially. Four actor workers
run in parallel; their wall times must not be added as if they were sequential.
`timing_s/step` excludes periodic validation and checkpoint saving. Core logging
retains generation/update/step timing but prunes teacher and old-logP timing;
retained Ray task events can provide worker execution durations without adding
instrumentation. Worker timing and driver timing have different RPC boundaries.
