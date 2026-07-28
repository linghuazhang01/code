# Notes: Cumulative Shared-Token Weighting

## Existing Path

- `DomainGradientAudit.run_before_training()` already performs a no-grad loss
  collection pass before the production actor forward.
- `DomainGradientAudit.training_gradient_mask()` composes domain, control, and
  shared-token multipliers.
- `gate_tensor_gradient()` preserves forward values and scales backward
  derivatives.
- Shared-token mode already requires one optimizer mini-batch and one PPO
  epoch, so one cumulative update can represent one complete global step.
- Dynamic domain-weight state demonstrates optimizer-state persistence.

## Intended Cumulative Score

For domain `d`, step `t`, and token type `v`:

```text
C[d, t, v] = C[d, t-1, v] + sum(abs(configured_token_loss))
```

Each domain is independently ranked by `C`; the final selected set is the
intersection of all configured domain Top-K sets.

## Metrics Needed

- Selected-token occurrence count and fraction.
- Raw configured-loss absolute mass.
- Effective weighted configured-loss absolute mass.
- Effective/raw mass ratio.
- Mean and maximum applied token gradient multiplier.
- Cumulative observed token-type count and cumulative absolute-loss mass per
  domain.

## Implemented

- Added `all_domain_shared_token_selection_mode` with
  `per_step_mean_abs_loss` and `cumulative_abs_loss`.
- Added immutable cumulative state with deterministic serialization,
  per-domain absolute-loss sums/counts, and `last_updated_step`.
- Added optimizer-state persistence and strict domain/mode restoration checks.
- Added actual production-mask comparison through
  `gradient_multiplier_mean_abs_error`.
- Added raw, token-weighted, and full effective configured-loss mass metrics.
- Added core TensorBoard routing for the `token_weight` category.
- Added a three-GPU cumulative Top-500 smoke profile.
- Packed ordinary vocabulary statistics as `[domain, sum/count, token_id]`
  float64 tensors for distributed `all_reduce`; this avoids replicating every
  rank's Python dictionaries on every worker.
- Cached immutable Control/shared selected-token tensors for repeated mask
  construction across amplification, production, and parity passes.
