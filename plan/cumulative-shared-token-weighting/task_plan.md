# Task Plan: Cumulative Shared-Token Weighting

## Goal

Implement cumulative absolute configured-loss shared-token selection, expose
backward amplification metrics, and provide a runnable Top-500 smoke config
without changing the default training behavior.

## Phases

- [x] Phase 1: Plan and inspect the existing two-pass weighting path
- [x] Phase 2: Add configuration and cumulative state/selection helpers
- [x] Phase 3: Integrate selection, checkpoint persistence, and metrics
- [x] Phase 4: Add smoke config, documentation, and regression tests
- [x] Phase 5: Run focused/full verification and code review
- [x] Phase 6: Align Control-enabled profiles to the report-derived 44-ID union
- [x] Phase 7: Re-audit composition and add a Control × cumulative-shared
  amplification-metrics smoke profile

## Key Questions

1. How can cumulative state resume exactly from optimizer checkpoints?
2. How can the implementation avoid counting the same global step twice?
3. Which metrics prove that selected tokens receive the configured backward
   multiplier while raw forward loss remains unchanged?

## Decisions Made

- Keep `per_step_mean_abs_loss` as the default selection mode.
- Add `cumulative_abs_loss` as an opt-in mode.
- Interpret `top_k: null` as all token types observed in every configured
  domain within the selected time horizon.
- Include the current step in the cumulative score before selecting tokens for
  that step's production backward.
- Persist cumulative sums/counts in optimizer state and fail fast on
  incompatible restored metadata.
- Measure amplification with detached raw/effective loss-mass and token-weight
  metrics; do not alter existing forward loss metrics.

## Errors Encountered

- The repository root is `OPD/code`, not `OPD`; all implementation and
  verification commands use the actual repository root.
- Initial metric integration did not add `token_weight` to the TensorBoard
  `core` whitelist; added the category, explicit metric allowlist, and a
  regression test.
- The first metric draft would have issued one distributed all-reduce per
  scalar; replaced it with one packed float64 collective per step.
- The first distributed token-statistics draft used `all_gather_object` for
  normal vocabulary IDs; replaced it with packed sum/count tensors and
  `all_reduce`, retaining object gather only as a defensive fallback for
  pathological token IDs above one million.
- Selected ID tensors were initially recreated for every micro-batch; added
  per-device/dtype caching and shared-selection cache invalidation.

## Status

**Complete** - implementation, regression tests, static checks, config
validation, and independent review are complete.
