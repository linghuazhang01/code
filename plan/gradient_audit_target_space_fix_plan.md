# Gradient Audit Target Space Fix Plan

## Goal

Unify domain, sample, and token gradient statistics into the same validated gradient target space, so projection share, cosine, and closure metrics can be trusted.

## Current Diagnosis

The current experiments show:

- Single micro-batch training loss/backward matches `_actor_micro_batch_loss(...)` exactly.
- Full-mask and zero-mask identity tests pass.
- Batch-size 4 can nearly close after removing the selected top-k head fast path.
- Batch-size 16 still does not close:
  - `audit_total_vs_training_total/rel_l2 ~= 0.4047`
  - `cosine ~= 0.9176`
  - `projection_share ~= 0.7719`
- Token top-p=1 selects all valid response tokens but does not close at batch-size 16.
- Sample raw projection share is near 1.9, while scaled debug share is near 0.95, but vector cosine is still poor.

The most likely root cause is target-space mismatch:

1. Token backward recompute reads `.grad` before FSDP finalize.
2. Compact per-domain direct replay is not a validated proxy for training final `.grad`.
3. Sample/token projection is currently computed against a domain target that may itself be wrong.

## Phase 0: Preserve Current State

- [x] Check git status and list uncommitted changes.
- [x] Do not overwrite user changes.
- [x] Keep current b16 smoke config as the main validation profile.
- [x] Record baseline metrics from latest run:
  - `codex_grad_b16_1step_20260627_060925`
  - local artifact directory: `temp/remote_codex_grad_b16_1step_20260627_060925`

## Phase 1: Fix Token Finalize Timing

### Target Files

- `mopd_verl/full_gradient/tracker.py`

### Change

In `_recompute_token_selection_gradient_stats(...)`, when `token_gradient_backward_recompute_enabled=True`:

- run token masked `loss.backward()`;
- call `_finalize_fsdp_after_auxiliary_backward(self.actor)`;
- then read `parameter.grad`;
- compute dot, norm, cosine, projection.

This should match sample/domain timing:

```text
loss.backward()
FSDP finalize
read .grad
compute stats
```

Status: implemented. Token backward recompute now snapshots pre-finalize,
finalizes FSDP auxiliary backward state, then reads post-finalize `.grad`.
The pre/post comparison is logged for diagnosing FSDP finalize effects.

### Add Debug Metrics

If feasible, record:

```text
token_grad_debug/pre_finalize_vs_post_finalize/rel_l2
token_grad_debug/pre_finalize_vs_post_finalize/cosine
token_grad_debug/pre_finalize_vs_post_finalize/norm_ratio
```

### Validation

Run b16 1-step smoke and inspect:

```text
math/token_grad/top100p_loss_abs_cos_to_domain
math/token_grad_contribution/top100p_loss_abs_projection_share
math/token_grad_closure/top100p_loss_abs_norm_ratio

code/token_grad/top100p_loss_abs_cos_to_domain
code/token_grad_contribution/top100p_loss_abs_projection_share
code/token_grad_closure/top100p_loss_abs_norm_ratio
```

Expected:

- token top-p=1 metrics should improve.
- If still not close, the remaining issue is likely the domain reference target.

## Phase 2: Add Training Micro-Batch Accumulation Target

### Target Files

- `mopd_verl/full_gradient/tracker.py`
- `third_party/verl/verl/workers/actor/dp_actor.py` only if extra call-site metadata is needed.

### Change

Extend `training_micro_batch_backward_parity_debug(...)` to accumulate post-finalize deltas:

```text
G_train_domain[domain] += actual_delta_chunks
G_recompute_domain[domain] += recompute_delta_chunks
```

The existing method already computes:

```text
pre_chunks
actual_total_chunks
actual_delta_chunks = actual_total_chunks - pre_chunks
recompute_delta_chunks = recompute_total_chunks - pre_chunks
```

We should store these per domain and compare at the end of training backward.

### Add Metrics

Add closure metrics:

```text
global/full_grad_closure/training_micro_sum_vs_training_total/*
global/full_grad_closure/recompute_micro_sum_vs_training_micro_sum/*
global/full_grad_closure/recompute_micro_sum_vs_direct_domain_sum/*
global/full_grad_closure/training_micro_domain_sum_vs_training_total/*
global/full_grad_closure/recompute_micro_domain_sum_vs_training_total/*
```

Also add per-domain comparisons:

```text
math/full_grad_closure/training_micro_sum_vs_direct_domain/*
code/full_grad_closure/training_micro_sum_vs_direct_domain/*
math/full_grad_closure/recompute_micro_sum_vs_direct_domain/*
code/full_grad_closure/recompute_micro_sum_vs_direct_domain/*
```

### Purpose

This answers:

```text
Do true training micro-batch deltas add up to training final .grad?
Do recomputed micro-batch deltas add up to direct domain targets?
Is direct domain replay the broken construction?
```

Status: implemented as a diagnostic target, not as the primary audit target.
The accumulated true-training micro deltas and recomputed micro deltas are
compared against the final training gradient, compact direct target, and
sequence masked target.

### Validation

Run b16 1-step smoke.

Interpretation:

- If `training_micro_sum_vs_training_total` closes but `direct_domain_sum_vs_training_total` does not:
  - direct compact replay is wrong.
- If `training_micro_sum_vs_training_total` does not close:
  - the training final `.grad` construction differs from per-micro debug deltas; inspect restore/finalize or optimizer-space grad handling.
- If `recompute_micro_sum_vs_training_micro_sum` does not close:
  - micro parity storage or restore logic is wrong despite row-level equality.

## Phase 3: Add Sequence-Preserving Masked Domain Replay

### Target Files

- `mopd_verl/full_gradient/tracker.py`

### Change

Add a new domain target construction mode:

```text
for target_domain in domains:
  clear grads
  replay all tracked micro-batches in the original training order
  if micro_batch.domain == target_domain:
    response_mask_override = response_mask
  else:
    response_mask_override = zero_mask
  use the same loss_scale_factor as training
  use the same sync/no_sync convention as the chosen smoke profile
  backward
  finalize
  snapshot target_domain grad
```

Status: implemented. The sequence replay uses the stored training micro-batch
order, the same loss builder, the same `loss_scale_factor`, and the configured
training sync/no-sync convention. The b16/b32 smoke configs enable it as the
primary domain target.

This relies on already verified identities:

```text
response_mask_override=response_mask == default full loss
response_mask_override=zero_mask == zero contribution
```

### Add Metrics

```text
global/full_grad_closure/sequence_masked_sum_vs_training_total/*
global/full_grad_closure/compact_direct_sum_vs_training_total/*
global/full_grad_closure/sequence_masked_sum_vs_compact_direct_sum/*

math/full_grad_closure/sequence_masked_vs_compact_direct/*
code/full_grad_closure/sequence_masked_vs_compact_direct/*
```

### Target Selection Policy

Temporarily keep compact direct target for reporting, but mark it as untrusted if it does not close.

Current implementation policy:

1. `sequence_masked_replay`, when enabled and available, is the primary trusted target.
2. `compact_direct_replay` remains a comparison/debug target.
3. `training_micro_accumulation` remains a diagnostic reference, because it is collected
   from parity probes inside the real training backward and should not become the
   default audit target until remote smoke confirms it closes robustly.

### Add Source Metrics

```text
global/audit/full_gradient_domain_target_source_sequence_masked_replay
global/audit/full_gradient_domain_target_source_compact_direct
global/audit/full_gradient_domain_target_trusted
```

## Phase 4: Rebase Sample/Token Onto Trusted Target

### Target Files

- `mopd_verl/full_gradient/tracker.py`

### Change

Make sample/token projection use only the chosen trusted target:

```text
trusted_domain_targets = sequence_masked_replay when available
```

If no trusted target is available:

- still log sample/token debug metrics;
- set `projection_share_trusted = 0`;
- avoid treating raw share as contribution ratio.

Status: mostly implemented. `finish_mini_batch()` now replaces the domain target
map with sequence masked replay targets when
`sequence_masked_target_use_as_primary=true`, so sample/token projection uses
the selected primary target. The explicit `projection_share_trusted` aliases are
still pending cleanup after the next smoke run confirms closure.

### Sample Gradient Validation

Expected for full sample coverage:

```text
sample vector cosine -> 1
sample vector projection_share -> 1
sample vector norm_ratio -> 1
```

For per-sample raw share:

- the sum should approach 1 only after target and scale conventions are unified.
- normalized share remains a ranking metric, not a contribution ratio, until closure passes.

### Token Gradient Validation

For top-p=1:

```text
closure_selected_all_tokens = 1
cos_to_domain ~= 1
projection_share ~= 1
norm_ratio ~= 1
```

## Phase 5: Clean Up Temporary Debug Code

After closure is confirmed:

- [ ] Remove obsolete replica-scale hypothesis metrics.
- [ ] Remove noisy historical compatibility metrics.
- [ ] Keep only:
  - target source
  - target trust flag
  - domain closure
  - sample closure
  - token closure
  - failure reason counters

## Smoke Validation Matrix

### Required First Run

```text
test_grad_configs/mopd_dynamic_weight_qwen4b_8b_aw2_fsdpsize2_tail_topp1_b16_4step_smoke.yaml
```

Expected after Phase 1:

- token top-p=1 improves.

Expected after Phase 2/3:

- at least one trusted target closes against training total.

Expected after Phase 4:

- sample/token closure uses trusted target.

### Follow-Up Run

```text
configs/mopd_formal_audit_grad_consistency_2gpu_b32_2step_smoke.yaml
```

Purpose:

- stress multiple steps;
- catch state leakage across steps;
- verify target source remains stable.

## Acceptance Criteria

### Domain Target Closure

```text
trusted_domain_sum_vs_training_total/rel_l2 <= 1e-3 initially
trusted_domain_sum_vs_training_total/cosine >= 0.999
trusted_domain_sum_vs_training_total/projection_share ~= 1
trusted_domain_sum_vs_training_total/norm_ratio ~= 1
```

### Micro-Batch Accumulation Closure

```text
training_micro_sum_vs_training_total/rel_l2 <= 1e-3
recompute_micro_sum_vs_training_micro_sum/rel_l2 <= 1e-6
```

### Token Top-p=1 Closure

```text
selected_all_tokens = 1
cos_to_domain >= 0.999
projection_share ~= 1
norm_ratio ~= 1
```

### Sample Gradient Closure

```text
sample vector cosine >= 0.999
sample vector projection_share ~= 1
sample vector norm_ratio ~= 1
```

## Risk Notes

- Training micro-batch parity debug currently consumes and restores backward state. It must not alter the actual training gradient.
- Any new target construction must restore original `.grad` and FSDP state before returning.
- Sequence-masked replay can be expensive; keep it enabled only in smoke/debug configs first.
- Do not interpret projection share as final contribution ratio until `domain_target_trusted=1`.

## Status

Planned. No code changes applied yet beyond this plan document.
