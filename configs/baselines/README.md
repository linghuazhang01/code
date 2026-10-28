# OPD Baseline Configs

> **2026-10-18:** the Math+Code+Science configs described in this file (`opd_baselines.yaml`, `canonical/`, `qwen1p7b_30b_*`, `qwen4b_30b_*`) were removed. The baselines that remain are in `math/` and `math_code/` (see their READMEs). The text below is kept as history of the removed set.

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
design. The canonical matrix uses batch 512, learning rate `5e-6`, and 200
steps. Its three domain teacher paths alias the same 30B checkpoint, so it is a
multi-domain single-teacher setup.

Full-vocabulary TIP uses a smaller, memory-safe batch of 24, one rollout per
prompt, 8192 response tokens, and the same `5e-6` project learning rate. It
caches teacher logits in bf16 host memory and uses a colocated sequential
teacher/student topology on 8 GPUs. Those resource adaptations do not change
TIP's scoring or training objective.

## Math-only 4-8 GPU suite

`math/` contains the directly launchable Math-only comparison suite for OPD,
FiRE-OPD, scalable TIP-TopK32, and EOPD. Its 4/5/6/7/8-GPU profiles use global
batches 255/256/255/258/259 respectively, with one H200 teacher GPU and the
remaining GPUs assigned to the actor. See `math/README.md` for the complete
matrix and launch commands.

## Four-GPU co-located native suite

The following profiles provide a matched three-way comparison for the
Qwen3-1.7B student and Qwen3-30B-A3B-Instruct-2507 teacher:

- `qwen1p7b_30b_opd_native_4gpu_b512_colocated.yaml`
- `qwen1p7b_30b_eopd_native_4gpu_b512_colocated.yaml`
- `qwen1p7b_30b_fire_opd_native_4gpu_b512_colocated.yaml`

All three use four actor/rollout workers and initialize the teacher as a
four-rank FSDP reference model on those same workers (`separate_ref_policy:
false`). The global batch is `512 = 4 * 128`. The EOPD profile also disables rollout IS, as
required by the native EOPD contract.

Launch one profile with:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/qwen1p7b_30b_eopd_native_4gpu_b512_colocated.yaml
```

## Eight-GPU 6+2 native suite

The following profiles provide the same three-way comparison with six
actor/student GPUs and two dedicated ref/teacher GPUs:

- `qwen1p7b_30b_opd_native_8gpu_b512.yaml`
- `qwen1p7b_30b_eopd_native_8gpu_b512.yaml`
- `qwen1p7b_30b_fire_opd_native_8gpu_b512.yaml`

All three use global batch `512`. The dedicated two-GPU teacher uses
`teacher_performance.topk_logprob_chunk_size: 1024`, `max_micro_batch_size: 32`,
and `max_tokens: 57344`. EOPD disables rollout IS as required by its native
objective contract.

Launch one profile with:

```bash
python -m mopd_verl.launch \
  --config configs/baselines/qwen1p7b_30b_opd_native_8gpu_b512.yaml
```

## 2026-10-15 speed/robustness update (all 8-GPU profiles)

Every 8-GPU launchable config under `configs/` (baselines, canonical, Math-only
`*_8gpu_b256`, Math+Code `*_8gpu_8c`, token-selection 8-GPU profiles) now sets
`teacher_performance.fused_statistics: true`, `teacher_performance.compact_topk_ids: true`
and routes rewards through `custom_reward_function.name=compute_score_batched` +
`reward_model.reward_manager=batch` (spawn-pool grader with a 120 s hard timeout;
timed-out samples score 0, which only affects logged training accuracy, never the
distillation loss). The 1.7B paper runs were trained without these flags; resolved
configs differ from those runs only in these keys. Measured context: the host CPU
is no longer throttled since the 2026-10-14 reboots (4.1 GHz vs 605 MHz), which is
the largest single speed-up.

## 2026-10-15 8-GPU colocated (author request)

Every 8-GPU profile that used a dedicated teacher (6 student + 2 teacher, or 7 + 1 for the Math-only `*_8gpu_b256.yaml` (formerly `*_8gpu_b259.yaml`)) now runs colocated on all eight GPUs: `worker_placement.separate_ref_policy: false`, actor/rollout on 8 GPUs, teacher FSDP size 8, `trainer.n_gpus_per_node: 8`. Teacher batching (`teacher_performance.enabled`), fused statistics, compact Top-K ids and the batched reward grader stay on. vLLM `gpu_memory_utilization` is 0.75 for Qwen3-1.7B and 0.4 for Qwen3-4B (the colocated profiles that already completed). Batches that do not divide by 8 changed: Math 259 -> 256, Math+Code 516 -> 512; 504 and 528 were set to 512 on 2026-10-18 (255 on 3 GPUs). File names keep their old topology tags; run names (`experiment_name`, `wandb_run_id`, checkpoint/audit/eval/HF paths) now say `8c` and the new batch, so no new run writes into an old run's directory. 4/6/7-GPU profiles that inherit from these files keep their previous values.
