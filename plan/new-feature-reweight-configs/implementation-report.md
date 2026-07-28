# Implementation Report: New Feature Reweight Configs

## Delivered

- Added a standalone 8-GPU Top-K32 gradient-norm domain-reweight config.
- Added a standalone 8-GPU Top-K32 projection-share domain-reweight config.
- Added a standalone 8-GPU Top-K32 static report-aligned Control-44
  token-reweight config.
- Added a standalone 8-GPU Top-K32 per-step shared Top-100 token-reweight
  config.
- Added a standalone 8-GPU Top-K32 cumulative shared Top-500 token-reweight
  config.
- Preserved the non-reweighted baseline and kept each treatment isolated.
- Updated all five configs to disable tail token-gradient observation and
  enable Top-p configured-loss-mass token gradients at `top_p=0.10`.

## Validation

- Parsed all five files with PyYAML and the project `load_config`.
- Rendered all five Hydra launch commands.
- Verified recursive baseline parity outside an explicit allowlist.
- Verified exactly one reweight family is enabled in every config.
- Verified unique output/checkpoint paths and Control-44 uniqueness.
- Ran 53 focused unit tests successfully.
- `git diff --check` passed.
- Independent planner, architecture, and config reviews found no blocking
  issues or treatment contamination.
- Repeated config loading, command rendering, 53 focused tests, and review
  after the Top-p follow-up; all passed.

## Remaining Boundaries

- No GPU training job was launched.
- `tensorboard_prune_mode: core` does not currently retain the new
  projection-source diagnostic scalars; this is a logger whitelist concern,
  not a config-loading failure.
