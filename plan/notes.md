# Notes: Dynamic Domain Budgeting

## Mathematical Invariants

- Target domain contribution: `q[d]`.
- Realized sample share: `p[d]`.
- Per-domain loss scale: `lambda[d]`.
- Required closure: `p[d] * lambda[d] = q[d]`.
- For fixed `q` and equal per-sample costs, the sequence-loss estimator
  variance is minimized by `p[d] proportional to q[d] * sqrt(V[d])`.

## Proposed Window Update

1. Paired capability gap:
   `C[d,k] = mean(max(teacher_reward - student_reward, 0))`.
2. Causal smoothing and initial-headroom normalization:
   `G[d,k] = clip(EMA(C[d,k]) / (C[d,0] + eps), 0, gap_max)`.
3. Target contribution:
   `q[d,k] proportional to prior[d] * (G[d,k] + eps)^alpha`.
4. Sequence OPD-loss variance:
   first average valid token loss per sequence, then compute cross-sequence
   variance for each domain.
5. Positive time smoothing:
   `V_tilde[d,k] = exp(EMA(log(V_hat[d,k] + eps)))`.
6. Allocation and scaling:
   `p[d,k] proportional to q[d,k] * sqrt(V_tilde[d,k] + eps)` and
   `lambda[d,k] = q[d,k] / p[d,k]`.

## Boundaries

- `q` is an objective coefficient, not a guaranteed gradient contribution.
- Loss variance is a cheap proxy unless correlation with gradient dispersion
  is measured separately.
- Per-domain loss scaling must match the production sample/token reduction.
- Floors, caps, and integer batch quotas can change realized `q`; log both
  target and realized values.

## Repository Findings

- The controller belongs on the Ray driver and is independent of the audit
  subsystem.
- The current sampler stores a static integer allocation. It needs a runtime
  update method and the trainer must retain its reference.
- Every actor batch must use its observed integer domain shares when computing
  `lambda[d] = q[d] / p_observed[d]`; this protects the invariant from rounding
  and stale prefetched batches.
- Strict v1 enables replacement sampling, keeps at least one row per domain,
  and requires `data.dataloader_num_workers: 0`.
- The production loss defaults to `token-mean`. Dynamic domain budgeting must
  require `seq-mean-token-mean` so `p * lambda` has sample-level semantics.
- The existing audit dynamic domain loss weighting is mutually exclusive with
  this controller.
- Validation exposes student aggregate metrics only. Fixed teacher probe scores
  and candidate student metric keys therefore enter through config.
- Raw configured token losses are already returned from the actor on request;
  their per-sequence token means can update the variance proxy without an extra
  model forward.

## Verification so far

- Focused feature and related regression suite: 96 passed, 34 subtests passed.
- All 44 ordinary/named config references load successfully.
- Dynamic profile renders the required Hydra overrides in dry-run mode.
- Python syntax compilation passed for every modified Python integration file.
- `git diff --check` passed.
- Independent incremental review found no remaining P0, P1, or P2 issues.
- Broad test suite: 198 passed; four launcher assertions remain outside this
  feature. Three reproduce in the original repository because `start.sh`
  contains pre-existing debug output. The fourth only affects the isolated
  copy because the requested code copy intentionally did not duplicate 3.4 GB
  of ignored training data.

## Four-Domain Audit Update

- The dynamic profile now covers `math`, `code`, `science`, and `if`; an equal
  initial 504-sample allocation produces exactly 126 rows per domain.
- IF training has 16,575 rows and uses `m2rl_ifbench`; the loader overwrites
  `domain`, `opd_teacher`, and `source_domain` with `if` based on the configured
  file-to-domain mapping.
- The fixed IF capability probe is deliberately IFBench-only in v1, using
  `val-core/m2rl_ifbench/reward/mean@1`. IFEval is not included because the
  current M2RL training reward router does not expose a separate IFEval path.
- Time normalization is explicitly relative remaining gap:
  `EMA(gap[d]) / initial_gap[d]`. Consequently, the first update is uniform
  under uniform priors even when absolute initial gaps differ; later q values
  prioritize domains retaining more of their initial gap.
- Current verification supersedes the older counts above: 211 broad tests
  pass, all 57 config/profile references load, compilation and diff checks
  pass, and the four-domain launcher dry-run succeeds.
