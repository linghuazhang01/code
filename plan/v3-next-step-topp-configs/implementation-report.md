# ExpandedPruned-V3 Next-Step Top-P Implementation Report

## Outcome

Completed the Math-only next-step Top-P matrix for occurrence targets 1%, 2%,
5%, and 10% across the existing 4--8 GPU resource topologies.

- Final matrix: 20 overlay configs.
- Existing user work preserved: four 8-GPU overlays plus their initial tests
  and README documentation.
- New configs added: sixteen overlays for the missing 4--7 GPU combinations.

## Config Contract

Each overlay:

- extends the same-topology V3 `i1/w1/K25` standalone base;
- preserves the 115-ID unified Math candidate pool and Metric A
  (`top_logp_diff`);
- sets `control_token_online_budget_mode: top_p`;
- sets `control_token_online_top_p` to 0.01, 0.02, 0.05, or 0.10;
- leaves `control_token_online_top_k_per_group` null;
- assigns a unique WandB run ID, audit directory, paper-eval directory,
  Hugging Face path prefix, and local checkpoint directory.

The inherited positive K25 value is retained because the current schema and
checkpoint state require it. The runtime Top-P branch does not use it as a
selection cap.

Top-P is the requested share of all valid response-token occurrences covered
by selected candidate token types. It is not a percentage of the 115 candidate
types and not cumulative selection-score mass. A complete token type may cause
overshoot; insufficient eligible coverage is logged as a shortfall.

## Verification

- Parsed and recursively resolved all 20 YAML overlays.
- Verified the exact 4-by-5 percentage/topology Cartesian product.
- Verified every overlay extends the corresponding topology base.
- Verified the candidate pool size, sort/uniqueness, and frozen SHA256.
- Verified `pre_update`, Metric A, Top-P values, inherited K25 compatibility,
  i1/w1 cadence, batch sizes, GPU placement, and conditional ref FSDP.
- Verified all artifact namespaces are unique and every run ID is at most 64
  characters.
- Compiled `tests/test_v3_math_selector_configs.py` with `py_compile`.
- Passed `git diff --check`, focused whitespace checks, and a focused secret
  scan.
- Independent code review reported no findings.

The standard `pytest`, `ruff`, and `pyright` commands were unavailable on the
current host because those tools and `uv` are not installed. No dependency or
environment changes were made merely to run them.
