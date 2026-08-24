# OPD Baseline Configs

The source-of-truth launch interface is `canonical/`: one baseline corresponds
to one directly launchable YAML file. For example:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/canonical/tip_native_full_vocab_rho50.yaml
```

The small canonical YAMLs use relative `extends` references to avoid copying
the shared budget. `opd_baselines.yaml` remains the underlying comparison
matrix and can still be launched explicitly for profile sweeps:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/opd_baselines.yaml::tip_topk32_rho50
```

All canonical configs target the same project setting: a Qwen3-1.7B student,
one Qwen3-30B-A3B-Instruct-2507 teacher, and the repository's math/code/science
training data. Paper-specific model checkpoints and datasets are intentionally
not part of the algorithm-parity contract.

## Fair-comparison profiles

These profiles keep the teacher-selected, renormalized Top-32 reverse-KL
objective fixed and change only the detached token scoring/weighting rule:

- `topk32_uniform`
- `entropy_topk32_rho50`
- `entropy_sample_topk32_rho50`
- `tip_topk32_rho50`
- `fire_token_topk32`

`tip_topk32_rho50` clips student entropy at the complete rollout batch's 98th
percentile, applies batch-global min-max normalization to entropy and
disagreement, combines them with TIP's Soft-OR score, and keeps the Top-rho
tokens within each response. Its disagreement and training loss remain the
configured Top-32 reverse KL, so the profile is deliberately named
`tip_topk32` rather than native full-vocabulary TIP.

## Native-objective profiles

- `tip_native_full_vocab_rho50`: normalized full-vocabulary student entropy,
  batch q98 clipping/min-max, full-vocabulary reverse-KL disagreement,
  Soft-OR, per-rollout Top-rho, and selected-token full-vocabulary reverse-KL
  training. Teacher and student run sequentially on colocated workers.
- `opd_native`: chosen-token OPD policy gradient.
- `gopd_native_lambda0p5`: the G-OPD interpolation regime (`0 < lambda < 1`),
  with a frozen step-zero student reference.
- `exopd_native_lambda1p25`: the ExOPD (`lambda > 1`) instance of G-OPD,
  with a frozen initial-student reference.
- `eopd_native`: OPD plus strict `H_teacher > 0.8` gated Top-16 forward KL;
  rollout IS is disabled so it cannot reweight only the OPD summand.
- `fire_opd_native`: FiRe batch-global entropy normalization, per-trajectory
  mean-one token weights, and exact bottom-20% trajectory filtering by mean
  teacher chosen-token log-probability on chosen-token OPD.

Before launching, adjust model/data paths and worker placement for the target
machine. G-OPD uses `distill_loss_builder: gopd`, accepts `lambda >= 0`, and
requires `model.gopd_reference_path`. ExOPD uses
`distill_loss_builder: exopd`, enforces `lambda > 1`, and normally points that
reference to the frozen step-zero student. `model.student_base_path` remains
a backward-compatible alias.

These are method-native, project-adapted baselines: scoring, selection, loss,
reduction, and update semantics follow the named methods, while model, data,
batch size, learning rate, and training steps follow this project's experiment
design. The canonical matrix uses batch 504, learning rate `5e-6`, and 200
steps. Its three domain teacher paths alias the same 30B checkpoint, so it is a
multi-domain single-teacher setup.

Full-vocabulary TIP uses a smaller, memory-safe batch of 24, one rollout per
prompt, 8192 response tokens, and the same `5e-6` project learning rate. It
caches teacher logits in bf16 host memory and uses a colocated sequential
teacher/student topology on 8 GPUs. Those resource adaptations do not change
TIP's scoring or training objective.
