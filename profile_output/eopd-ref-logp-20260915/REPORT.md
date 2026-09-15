# EOPD ref/logP performance report

The live 8×L20Y run spends about 752 s (51% of a roughly 1464 s step) in teacher ref/logP. The dominant confirmed configuration defect is that Transformers 4.51.3 was excluded from the exact-source stable-sort allowlist, so the requested MoE optimization fell back to stock dispatch.

The local patch adds a verified 4.51.3 source hash while retaining safe fallback, plus opt-in response-only and memory-aware teacher post-processing. Historical H200 fixed-input evidence measured a 22.18% complete-teacher reduction from stable-sort alone; an extrapolation would be about 752→585 s for ref and 1464→1297 s per step. This is a planning estimate, not a measured L20Y result.

## Instrumentation changelog

| Area | Change |
|---|---|
| MoE dispatch | Version→exact-source-SHA compatibility gate now includes Transformers 4.51.3. |
| Teacher statistics | Optional response-only joint chosen-logP/entropy/Top-K path; entropy uses the existing runtime implementation per chunk. |
| Batching | Optional adaptive 1024/256 chunk choice based on synchronized admitted row count, with a second free-memory check immediately before forward. |
| Transfer | Optional int32 Top-K support IDs; worst-case padded global tensor saves 528 MiB. |
| Telemetry | The teacher batching wrapper records rows, effective tokens, microbatch count, per-group size, effective chunk, and fallback state. |
| Topology | Dedicated `fsdp_size=1` replicas batch independently; sharded teachers preserve synchronized forward ordering. |
| Activation guard | Fused statistics and compact IDs run only after the policy-local performance configuration passes all topology/dtype guards. |

## Next measurement

Run the explicit stock-control versus stable-sort fixed-input ABBA first. Then test the opt-in kernel candidate, followed by forward prefetch. Adaptive chunk promotion still requires measured L20Y peak-memory headroom even though the selected chunk is revalidated immediately before forward. Do not begin with co-location or full replication on L20Y; replication remains a bounded memory smoke after the lower-risk wins are measured.
