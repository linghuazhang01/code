# Notes: New Feature Reweight Configs

## Baseline

- `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml`
- Qwen3-4B student, shared Qwen3-30B-A3B-Instruct-2507 teacher.
- Six actor/rollout GPUs plus two ref/teacher GPUs.
- Math, code, and science; train batch 504.
- `actor.ppo_mini_batch_size=504`, `ppo_micro_batch_size_per_gpu=1`.
- Top-K32 renormalized reverse-KL distillation.
- Existing full-gradient and tail-token-gradient observations remain enabled.
- All production reweight mechanisms are disabled.

## Configuration Invariants

- Preserve every baseline field except:
  - header comments describing the active mechanism;
  - `audit.output_dir`;
  - explicit reweight fields;
  - `paper_eval.output_dir`;
  - `trainer.experiment_name`;
  - `trainer.default_local_dir`;
  - shared configs add explicit `actor_rollout_ref.actor.ppo_epochs=1`.
- Give every config a unique audit, paper-eval, experiment, and checkpoint path.
- Exactly one production reweight family may be enabled per config.
- Both dynamic variants use the same controller hyperparameters so only the
  signal source changes.
- Shared selection requires one optimizer mini-batch and one PPO epoch.

## Selected Treatments

| Config | Treatment |
|---|---|
| gradnorm | dynamic domain, gradient norm, freq 4, 0.70/0.50 EMA, alpha 0.75 |
| projection | dynamic domain, absolute projection-share controller |
| control44 | static report-aligned Qwen3 Control-44 token IDs at 2.0x |
| shared per-step | per-step mean absolute configured loss, Top-100, 1.5x |
| shared cumulative | cumulative absolute configured-loss mass, Top-500, 2.0x |

## Validation Findings

- The system Python lacks PyYAML; repository config validation must use the
  configured project environment or loaders after file creation.
- Reused cached PyYAML 6.0.2 with the bundled Python 3.12 runtime.
- All five configs load through `mopd_verl.settings.load_config`.
- All five configs render through `build_command` and `format_command`.
- Recursive raw-YAML comparison found no differences outside the allowlist.
- Every config enables exactly one reweight family.
- Every config keeps token-gradient audit enabled, disables tail selection,
  enables Top-p selection, uses `top_p=0.10`, and keeps fixed Top-k disabled.
- Control config contains the report-aligned 44 unique IDs at 2.0x.
- Shared configs explicitly set PPO epochs to one and retain
  `train_batch_size == ppo_mini_batch_size == 504`.
- All audit, paper-eval, experiment, and checkpoint paths are unique.
- `git diff --check` passed.
- Focused unit tests passed: 53/53.
- The follow-up token-gradient change was reloaded, command-rendered, and
  independently reviewed with no blocking issues.
