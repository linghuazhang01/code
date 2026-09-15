# Implementation Report

Status: implementation complete; GPU A/B pending.

## Implemented

- Added a version-to-exact-source-SHA allowlist so Transformers 4.51.3 can use the already benchmarked stable-sort Qwen3 MoE dispatch. Unknown or modified sources still fall back.
- Added an opt-in teacher-only statistics path. It processes response prediction positions only, reuses the FP32 normalizer for chosen logP and Top-K, and calls the existing runtime entropy function per chunk to retain the EOPD gate semantics.
- Added opt-in int32 teacher Top-K IDs; the downstream gather path restores int64.
- Added adaptive chunk selection: keep 1024 on a tie, use 256 only if lower workspace increases the synchronized forward group size, revalidate free memory immediately before forward, and use 16 if the selected chunk is no longer safe.
- Added per-call batch-plan telemetry on the wrapper (`rows`, `tokens`, microbatch groups, effective chunk, guard fallback).
- Added correct independent batching for a dedicated `fsdp_size=1` replicated teacher; sharded ranks retain synchronized forward counts.
- Added five isolated configs: explicit stock control, stable-sort only, kernel tuned, FSDP forward-prefetch, and an experimental full-replica memory smoke.
- Gated all fused/int32 execution on the successfully activated policy-local performance state, so unsupported topology/dtype fallbacks cannot accidentally retain raw opt-in flags.

## GPU A/B protocol

Use the same checkpoint and captured rank-local input tensors. Keep `entropy=True`, Top-K 16, global batch, sequence order, temperature, and all objective settings fixed.

1. Warm up once, then run the explicit stock-control and stable-sort configs as stock → stable → stable → stock. Accept only if both repeated stable cases improve and all outputs are exact.
2. Compare stable-only against the kernel candidate with the same ABBA ordering. Require Top-K IDs exact, all outputs finite, report max/mean chosen-logP and Top-K-logP error, entropy max/mean error, and the count of `entropy > 0.8` decision flips. A non-zero flip blocks adoption.
3. Record both ranks' maximum wall time, model-forward time, post-processing time, output-to-CPU/Ray time, microbatch group plan, NCCL time, and CUDA allocated/reserved peaks.
4. Test forward prefetch separately. Accept only with at least 10% median ref-stage improvement and no memory regression beyond the safety envelope.
5. Run the replica candidate only as a one-batch bounded smoke. Reject immediately on less than 8 GiB measured headroom, allocator retries, OOM, or any output mismatch.
6. Promote a winner only after three complete training steps show the same direction in `timing_s/ref` and total step time.

## Local verification

- Focused teacher/config/FSDP/EOPD suite: 105 passed, 1 skipped, 1 deselected.
- Python compile checks passed.
- `git diff --check` passed for the scoped files.
- One unrelated repository test remains unavailable because `configs/matrices/eopd_baseline_matrix.yaml` is missing; broad collection also finds a pre-existing missing `scripts.prepare_wildsci_data` module.

## Operational status

- No remote source synchronization.
- No new GPU task was launched.
- The running 752-second-ref job was not restarted or interrupted; all changes require a fresh worker process.
