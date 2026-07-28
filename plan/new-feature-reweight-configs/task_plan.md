# Task Plan: New Feature Reweight Configs

## Goal

Create one standalone 8-GPU Qwen4B/Qwen30B Top-K32 training config for each
new gradient-reweighting feature, using the existing non-reweighted
math+code+science Top-K32 config as the only baseline.

## Phases

- [x] Phase 1: Plan and define the candidate feature matrix
- [x] Phase 2: Inspect the baseline, config contracts, and existing profiles
- [x] Phase 3: Create standalone configs with exactly one reweight mechanism
- [x] Phase 4: Validate parsing, invariants, diffs, and focused tests
- [x] Phase 5: Independent review and delivery

## Candidate Feature Matrix

1. Dynamic domain weighting with `gradient_norm`
2. Dynamic domain weighting with `domain_gradient_projection_share`
3. Static report-aligned Control-44 token weighting
4. Per-step all-domain shared high-loss token weighting
5. Cumulative all-domain shared absolute-loss token weighting

## Key Questions

1. Which baseline fields must change for each mechanism to be operational?
2. Which settings must remain identical to the non-reweighted baseline?
3. What unique output/checkpoint paths prevent experiment collisions?
4. Which batch-shape constraints are required by shared-token selection?

## Decisions Made

- Use full standalone YAML files because production configs are independently
  runnable and do not currently use an inheritance contract.
- Keep each config mechanism-pure so ablations have a single treatment.
- Preserve the baseline file unchanged.
- Use `freq=4`, signal EMA `0.70`, weight EMA `0.50`, `alpha=0.75`,
  and `[1/3, 3]` for both dynamic-domain variants.
- Use the 44 Qwen3 token IDs labeled `discourse/control` in the prior
  Rising/Stable Mean-Gap Top-100 report at `2.0x`.
- Use per-step shared Top-100 at `1.5x` and cumulative shared Top-500
  at `2.0x`.
- Add `actor_rollout_ref.actor.ppo_epochs=1` explicitly to both shared-token
  configs.
- Follow-up: disable tail token-gradient selection and enable Top-p
  configured-loss-mass selection with `top_p=0.10` in all five configs.

## Errors Encountered

- The system `python3` used by the mechanical patch generator does not provide
  PyYAML (`ModuleNotFoundError: yaml`). No config file was written. Replaced
  YAML parsing with standard-library text extraction from the existing
  Control-44 profile.

## Status

**Complete** - Five configs created, updated to Top-p=0.10 token-gradient
observation, validated, and independently reviewed.
