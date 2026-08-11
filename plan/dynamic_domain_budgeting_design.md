# Dynamic Domain Budgeting Design

## Contract

For domain `d`, the driver maintains target objective coefficient `q[d]`,
desired sample allocation `p[d]`, and current-batch scale `lambda[d]`.

1. Validation updates the positive teacher-student capability gap.
2. A causal EMA divided by `max(initial_gap, gap_floor)` provides stable
   time-series normalization, including when the initial gap is zero.
3. Normalized gap and an optional prior define `q`.
4. Unweighted per-sequence token-mean OPD losses define the variance proxy.
5. `p[d]` is proportional to `q[d] * sqrt(smoothed_variance[d])`.
6. The sampler converts `p` to integer domain counts with a true lower-bound
   projection and residual-quota rounding.
7. The actor uses `lambda[d] = q[d] / p_active[d]`, where `p_active` is the
   domain share among non-empty sequences used by the optimizer reduction.

The exact per-batch invariant is therefore:

```text
p_active[d] * lambda[d] = q[d]
```

No epsilon is added to the lambda denominator. Positivity is guaranteed by the
exploration/sampling floors and by requiring at least one active sequence for
every domain. `p_active` equals `p_observed` when every sampled response is
non-empty.

## Runtime boundaries

- The actor objective must use `seq-mean-token-mean`; token-global averaging
  would make the contribution depend on token share rather than sample share.
- Variance is a scalar-loss proxy, not a gradient-variance estimator.
- Capability gap measures remaining need, not marginal learnability.
- Teacher probe scores are fixed config inputs. The existing validation path
  supplies student aggregate metrics but does not run teacher generation.
  Formal launch is blocked until `teacher_scores_calibrated=true`.
- Controller observations are causal: data from step `t` changes later batches.
- The controller is mutually exclusive with the older audit gradient-based
  dynamic loss weighting.

## Persistence

The latest state and an append-only history are written under the configured
output directory. The same state is embedded in each trainer checkpoint and is
restored together with the dataloader and sampler allocation. Controller
restore rejects semantic config changes, while sampler restore includes its
RNG state so a resumed run does not replay earlier batches.
