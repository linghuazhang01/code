# EOPD Baseline Integration Plan

## Goal

Add Entropy-Aware On-Policy Distillation (EOPD) as an explicit, optional
baseline without changing the default OPD, Top-k KL, or MOPD objectives.

## Method Contract

For each valid response token, EOPD keeps the existing clipped on-policy
reverse-KL policy-gradient term and adds

`alpha * 1[teacher_entropy >= tau] * truncated_topk_forward_kl`.

The truncated forward KL follows the paper and official artifact:

- teacher entropy is full-vocabulary entropy in nats;
- the support is the teacher's top-k tokens;
- teacher probabilities are renormalized over that support;
- student log-probabilities retain their full-vocabulary normalization;
- gated forward-KL values are divided by all valid response tokens;
- paper defaults are `tau=0.8`, `alpha=1.0`, and `k=16`.

## Compatibility Boundary

- `distill_loss_builder: eopd` is the only activation switch.
- Existing builders keep their current control flow and defaults.
- EOPD always uses teacher top-k support and does not reuse the existing
  symmetric top-k-renormalized KL implementation.
- Dynamic domain budgeting remains restricted to the existing exclusive
  Top-k OPD objective.
- Multi-domain batches select entropy and top-k tensors with the same
  per-sample teacher routing already used for teacher log-probabilities.

## Verification

- [x] Unit-test the truncated forward KL against a manual calculation.
- [x] Test the entropy threshold boundary and all-valid-token denominator.
- [x] Test additive clipped-OPD plus EOPD behavior through the unified actor loss.
- [x] Test multi-domain teacher tensor selection.
- [x] Test that default and existing Top-k builders resolve unchanged.
- [x] Add and dry-render paired OPD/EOPD baseline profiles.
- [x] Compile changed Python modules and run the project test suite.
- [x] Complete an independent read-only code review with no P0/P1 findings.

Project tests: `265 passed, 1 skipped, 3 failed`. The three failures predate
this integration and are caused by an empty user config, a missing science
profile YAML, and a Slurm CPU-count expectation mismatch. The EOPD-specific
suite passes `6/6`.
