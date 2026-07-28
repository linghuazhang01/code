# Implementation Report: Cumulative Shared-Token Weighting

## Delivered

- Added opt-in `cumulative_abs_loss` selection while preserving
  `per_step_mean_abs_loss` as the default.
- Added current-step-inclusive cumulative absolute configured-loss sums and
  occurrence counts for every configured domain/token ID.
- Added deterministic per-domain Top-K ranking and all-domain intersection.
- Added same-step deduplication through `last_updated_step`.
- Added optimizer-checkpoint persistence and strict restore validation.
- Added raw, token-weighted, and full effective configured-loss mass metrics.
- Added exact production-mask validation through
  `gradient_multiplier_mean_abs_error`.
- Added core TensorBoard routing for all token-weight monitoring metrics.
- Added packed distributed sum/count reduction for normal tokenizer vocabulary
  IDs and cached selected-ID device tensors across repeated mask construction.
- Added a Qwen3-0.6B student / Qwen3-8B teacher, 2+1 GPU, Top-500,
  three-step smoke config.
- Added a second three-step integration profile that composes the report-aligned
  44-ID Control set at 2x with cumulative shared Top-500 at 3x, explicitly
  exposing the 6x overlap case to the amplification metrics.
- Consolidated the GPU integration profile into
  `mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::control44_cumulative`;
  the resolved `MOPDConfig` and rendered Hydra command remain identical to the
  former expanded YAML.
- Expanded both Control-enabled smoke configs from 31 to the 44 unique IDs
  counted as `discourse/control` in the prior six Rising/Stable Mean-Gap
  Top-100 groups. Eight additional curated IDs from the original experiment
  design were removed so the treatment exactly matches the report.

## Configuration

```yaml
audit:
  all_domain_shared_token_loss_weighting_enabled: true
  all_domain_shared_token_loss_weight: 2.0
  all_domain_shared_token_selection_mode: cumulative_abs_loss
  all_domain_shared_token_top_k: 500
```

`all_domain_shared_token_top_k: null` uses all token types observed in every
domain from the start of the run through the current step.

## Verification

- Focused domain-gradient/config tests passed.
- Full `unittest` discovery passed with 149 tests.
- TensorBoard token-weight core-filter assertion passed.
- Cumulative Top-500 config load and Hydra command rendering passed.
- New token-weighting modules pass focused Ruff `F`, `B`, and `I` checks.
- `git diff --check` passed.

No GPU training job was launched as part of local verification. Optimizer
`state_dict` round-trip behavior was tested, but a real multi-rank FSDP
save/restart integration remains a GPU-environment validation boundary.
