# Test Gradient Config Consolidation Report

## Result

The test-gradient configuration surface now contains five physical YAML files
representing twelve runnable experiment profiles:

- five gradient-reliability profiles;
- four domain/loss-weighting profiles;
- three standalone specialty integration profiles.

This replaces fifteen expanded YAML files while preserving the resolved
configuration and rendered launch command of every retained profile.

## Profile Matrices

Gradient reliability:

```text
test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw2_fsdp1_audit_on
test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw2_fsdp1_audit_off
test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw2_fsdp2_audit_on
test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw4_fsdp2_audit_on
test_grad_configs/mopd_grad_reliability_qwen0p6b_8b_matrix.yaml::aw4_fsdp2_audit_off
```

Domain and token-loss weighting:

```text
test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::gradnorm
test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::projection
test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::projection_control_perstep
test_grad_configs/mopd_domain_weighting_qwen0p6b_8b_matrix.yaml::control44_cumulative
```

The weighting matrix deliberately avoids `_token` in its filename because the
repository treats `*_token*` as a secret-like filename pattern.

## Resolution Contract

- Ordinary YAML remains loadable without a selector.
- A matrix requires an explicit `::profile` selector.
- Mappings deep-merge recursively.
- Lists and scalar values replace the base value.
- Invalid or unknown profile names fail before training starts.
- Local, direct, and Slurm launch paths preserve the complete config reference.
- Run identifiers include the selected profile.
- Slurm resource requests apply supported Hydra GPU/CPU placement overrides,
  including the runtime `actor_rollout_ref.worker_placement.*` alias.

## Removed Redundancy

The standalone control-only, shared-per-step-only, and
shared-cumulative-only GPU smoke profiles were removed. Unit and contract tests
continue to isolate these mechanisms, while the combined weighting profiles
cover their training integration.

## Verification

- All retained expanded profiles were compared against their matrix-resolved
  replacements at both `MOPDConfig` and rendered command levels.
- The full unit-test suite passed: 159 tests.
- Bash syntax, Python compilation, diff whitespace, and all five documented
  SHA-256 checks passed.
- Dry-run launch rendering passed for a weighting profile, a reliability
  profile, and an ordinary YAML config.
- Static type checking reported zero errors; its two warnings concern missing
  third-party PyYAML source/stubs.

## Boundary

No real multi-rank GPU training was launched. The remaining environment-level
acceptance check is one short GPU smoke run per matrix family.
