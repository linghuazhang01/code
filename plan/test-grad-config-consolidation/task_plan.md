# Test Gradient Config Consolidation

## Goal

Replace duplicated expanded smoke YAML files with named-profile matrices while
preserving ordinary single-config compatibility, resolved training semantics,
launch-time GPU/resource checks, and reproducible profile identities.

## Target Layout

- One gradient-reliability matrix with five named profiles.
- One domain/token-weighting matrix with four named profiles.
- Three standalone specialty integration configs.
- Reference syntax: `path/to/matrix.yaml::profile_name`.

## Phases

- [x] Add immutable config-reference parsing and deterministic deep merge.
- [x] Add resolver contract tests and regular-YAML compatibility tests.
- [x] Build the two profile matrices and prove resolved-config equivalence.
- [x] Update local, direct, and Slurm launch paths to accept profile references.
- [x] Remove expanded configs and update profile tests/documentation/hashes.
- [x] Run focused and full verification plus independent code review.

## Compatibility Decisions

- Existing `load_config(path)` behavior for ordinary YAML remains unchanged.
- Matrix YAML requires an explicit profile; silent default selection is
  forbidden.
- Dictionary values deep-merge; lists and scalars replace the base value.
- Unknown profiles fail with a deterministic list of available profiles.
- Launch logs and generated run IDs include the selected profile.
- Resolved configuration semantics, not raw YAML shape, define equivalence.

## Verification Boundary

CPU tests and dry-run launch checks can verify parsing, merging, rendered Hydra
commands, resource calculations, and provenance strings. A real multi-rank GPU
run remains a separate environment-level smoke test.
