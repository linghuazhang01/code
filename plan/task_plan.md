# Task Plan: Dynamic Domain Budgeting for MOPD

## Goal

Implement a config-driven MOPD domain-budget controller that uses a
time-series-normalized teacher-student capability gap to define target domain
contributions and a smoothed sequence-level OPD-loss variance to split those
contributions into sampling probabilities and per-domain loss scales.

## Phases

- [x] Phase 1: Create an isolated working copy without ignored secrets or artifacts
- [x] Phase 2: Audit existing config, sampler, validation, loss, and logging paths
- [x] Phase 3: Freeze the controller state, update equations, and runtime interfaces
- [x] Phase 4: Implement configuration, controller, dynamic sampling, and loss scaling
- [x] Phase 5: Add a runnable training profile, documentation, and focused tests
- [x] Phase 6: Run verification, code review, and cleanup

## Key Questions

1. Where can a runtime-updatable domain probability vector be applied without
   recreating the full trainer or dataloader?
2. Where can per-domain sequence-level OPD loss mean and variance be collected
   without changing the production objective?
3. Which existing validation metrics can supply current student scores, and how
   should fixed teacher and initial-gap values enter through configuration?
4. Does the production loss reduction preserve the invariant
   `target_contribution = sample_share * loss_scale`?

## Decisions Made

- Work only in `code_dynamic_domain_budgeting`; leave `code` untouched.
- Do not modify the optimizer or introduce gradient surgery.
- Use evaluation-window updates rather than per-step updates.
- Use exact `loss_scale = target_contribution / actual_sample_share` after
  floors and normalization; never add epsilon to the denominator.
- Treat scalar OPD-loss variance as a configurable proxy, not as a proved
  gradient-variance estimator.
- Preserve all copied user changes and avoid unrelated cleanup.

## Errors Encountered

- The source repository has no session hook script; no emulation command was run.
- The host Python lacks `torch` and `PyYAML`; final tests may require the
  repository runtime or dependency installation.

## Status

**Complete** - implementation, config profile, documentation, tests, dry-run,
and independent read-only review are finished.
