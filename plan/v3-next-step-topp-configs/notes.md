# Notes: ExpandedPruned-V3 Next-Step Top-P Configs

## Existing Work Preserved

- Four untracked 8-GPU overlays already exist for Top-P 0.01, 0.02, 0.05,
  and 0.10.
- `tests/test_v3_math_selector_configs.py` already contains a four-case 8-GPU
  contract test.
- `configs/token_selection/math/README.md` already documents those four
  8-GPU overlays.

## Contract Findings

- The active candidate pool remains the sorted, unique 115-ID Math V3 pool
  with SHA256
  `ab3b17d65320ef778dbe5c6f6f475012658735711c41314ced618f78b80a3bb0`.
- All profiles retain Metric A (`top_logp_diff`), `execution_timing=pre_update`,
  cadence `i1/w1`, fixed selected-token weighting, and the strict occurrence
  gate at 20 mean occurrences per step.
- `control_token_online_budget_mode=top_p` makes Top-P the active budget.
  Runtime selection accumulates ranked token-type occurrences until the
  requested fraction of all valid response tokens is reached.
- The loader still requires a positive `control_token_online_top_k` in every
  mode, so the overlays inherit K25 from the base config. Runtime Top-P
  selection does not use K25 as a cap.
- `control_token_online_top_k_per_group` must remain null because the V3 pool
  is unified and Top-P does not support grouped Top-K quotas.

## Implementation Decision

- Final matrix: 4 Top-P values x 5 topology variants = 20 profiles.
- New files required: 16 (the four 8-GPU profiles already exist).
- Every overlay extends the topology-matched V3 next-step K25 standalone base
  so GPU placement, batch size, and teacher FSDP settings remain canonical.
