# Gradient Audit Sync Backward Plan

## Goal
Make domain, token, and sample gradient audit use one comparable sync-backward gradient space, then verify audit gradients against the real training gradient.

## Phases
- [x] Phase 1: Inspect current gradient paths and config knobs
- [x] Phase 2: Implement sync-backward sample gradient protocol
- [x] Phase 3: Fix token projection scaling to use sync optimizer-space metrics
- [x] Phase 4: Add training-gradient parity diagnostics
- [x] Phase 5: Add sample vector closure diagnostics
- [x] Phase 6: Update config, TensorBoard filter, and tests
- [x] Phase 7: Local verification

## Key Decisions
- Use sync backward for all official domain/token/sample gradient metrics.
- Treat direct domain recompute as the audit target, then separately compare its summed gradient to real training `.grad`.
- Keep normalized sample share only as a fallback relative ranking when vector closure fails.

## Status
Local implementation is complete. Waiting for remote smoke validation.
