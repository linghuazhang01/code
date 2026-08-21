# OPD Baseline Configs

All canonical comparison runs live in `opd_baselines.yaml` as named profiles.
List profiles with:

```bash
python -c "from mopd_verl.config_profiles import list_config_profiles; print(*list_config_profiles('configs/baselines/opd_baselines.yaml'), sep='\\n')"
```

Run one profile with:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/opd_baselines.yaml::tip_topk32_rho50
```

## Fair-comparison profiles

These profiles keep the teacher-selected, renormalized Top-32 reverse-KL
objective fixed and change only the detached token scoring/weighting rule:

- `topk32_uniform`
- `entropy_topk32_rho50`
- `entropy_sample_topk32_rho50`
- `tip_topk32_rho50`
- `fire_token_topk32`

`tip_topk32_rho50` implements TIP's entropy/disagreement min-max
normalization, Soft-OR score, and Top-rho token selection. Its disagreement
signal is the configured Top-32 KL, so the profile is deliberately named
`tip_topk32` rather than native full-vocabulary TIP.

## Native-objective profiles

- `opd_native`: chosen-token OPD policy gradient.
- `exopd_native_lambda1p25`: ExOPD with a frozen initial-student reference.
- `eopd_native`: OPD plus teacher-entropy-gated Top-16 forward KL.
- `fire_opd_native`: FiRe token weighting plus global-batch bottom-20%
  trajectory filtering on chosen-token OPD.

Before launching, adjust model/data paths and worker placement for the target
machine. ExOPD requires `model.student_base_path`; it should point to the
frozen initial student, normally the same checkpoint as `model.student_path`
at step zero.

## Standalone Qwen3-1.7B and Qwen3-4B / Qwen3-30B training configs

The standalone configs are directly launchable without a profile suffix.
Each baseline has matching Qwen3-1.7B and Qwen3-4B student variants under the
`qwen1p7b_30b_*` and `qwen4b_30b_*` prefixes, with both requested worker
topologies:

- `*_4gpu_b525.yaml`: 3 actor/student GPUs, 1 ref/teacher GPU, global batch 525.
- `*_8gpu_b528.yaml`: 6 actor/student GPUs, 2 ref/teacher GPUs, global batch 528.

The fair token-scoring comparison uses the same Top-32 reverse-KL objective:

- `qwen1p7b_30b_entropy_topk32_rho50_*`
- `qwen1p7b_30b_tip_topk32_rho50_*`
- `qwen1p7b_30b_fire_topk32_matched_*`
- `qwen4b_30b_entropy_topk32_rho50_*`
- `qwen4b_30b_tip_topk32_rho50_*`
- `qwen4b_30b_fire_topk32_matched_*`

The native-objective comparison uses each paper's training objective:

- `qwen1p7b_30b_exopd_native_lambda1p25_*`
- `qwen1p7b_30b_eopd_native_*`
- `qwen1p7b_30b_fire_opd_native_*`
- `qwen4b_30b_exopd_native_lambda1p25_*`
- `qwen4b_30b_eopd_native_*`
- `qwen4b_30b_fire_opd_native_*`

For example:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/qwen1p7b_30b_tip_topk32_rho50_4gpu_b525.yaml

python -m mopd_verl.launch \
  --config configs/baselines/qwen1p7b_30b_tip_topk32_rho50_8gpu_b528.yaml
```
