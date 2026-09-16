# Teacher log-probability efficiency

The EOPD 8-GPU configuration uses a teacher microbatch cap of 32 and a
top-k postprocessing chunk size of 1024. Each rank proposes a group under
its current token and memory budget, then all ranks use the minimum proposed
group size. Sharded teachers also synchronize chunk safety for each group;
with one chunk candidate this uses two collectives per group after initial
validation. Replicated teachers with FSDP size 1 skip this synchronization.
Row order is preserved and memory limits remain rank-local heuristics.

Stable-sort MoE dispatch accepts the pinned forward-source hashes for
Transformers 4.51.3 and 4.57.6. Unknown versions, modified sources and
unsupported model layouts retain stock execution.

The memory guard remains a heuristic, with a 12 GiB safety margin and a
57344-token cap. Validate latency, throughput and peak memory for new workloads.
The two-teacher-GPU run measured approximately 37–41 seconds per teacher
reference phase in its first two steps. Active-only GPU samples averaged
58.8%; this excludes idle samples and is not a full-phase utilization average.

The logits fallback processes 1024-token chunks instead of individual rows.
When the FlashAttention cross-entropy implementation is available, it remains
the preferred path; the fallback change does not explain that path's speedup.
