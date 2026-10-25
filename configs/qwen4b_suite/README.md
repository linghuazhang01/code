# Qwen3-4B student suite (Math-only and Math+Code)

Second-student replication of the paper's comparison: the same six methods, the
same Qwen3-30B-A3B-Instruct-2507 teacher, data, optimizer, rollout and
evaluation protocol as the Qwen3-1.7B runs, with only the student (and the
ExOPD reference) changed to `../mopd/models/Qwen3-4B`.
Generated on 2026-10-14 by a script; every user-facing file is
`{method}_{topology}.yaml` and extends `_methods/{method}.yaml`, which extends
`_common.yaml`. The fragments are not experiment cells.

## Methods (`_methods/`)

| File | Method | Objective / rule | Mirrors (1.7B) |
|---|---|---|---|
| `opd` | **OPD** (paper row) | equal-weight Top-32 renormalized reverse KL, the objective SAPD reweights | `token_selection/math/taxonomy/uniform_top32kl_*`, `baselines/math_code/topk32_uniform_*` |
| `opd_native` | OPD Native (extra) | chosen-token OPD policy gradient | `baselines/math/_methods/opd`, `baselines/math_code/opd_native_*` |
| `tip_topk32` | TIP-TopK32, rho=0.5 | Soft-OR of q98-clipped student entropy and Top-32 RKL disagreement, top half per response, seq-mean-token-mean | `baselines/*/tip_topk32*` |
| `eopd` | EOPD | chosen-token OPD + teacher-entropy>0.8 gated Top-16 forward KL, alpha=1, rollout IS off | `baselines/*/eopd*` |
| `exopd` | ExOPD, lambda=1.25 | G-OPD extrapolation, frozen initial Qwen3-4B as reference | `baselines/*/exopd*` |
| `fire_opd` | FiRe-OPD | drop bottom-20% trajectories by mean teacher chosen-token log-prob, mean-one teacher-confidence x student-confusion weights | `baselines/*/fire_opd*` |
| `sapd` | **SAPD** (paper.md 1.3) | Top-32 RKL + fixed pools (Math Connective 124 / Code Programming 163), next-step ID-mean TopLoss ranking, >20 occurrence gate, whole-ID budget Math 5% / Code 1%, gamma=4, per-domain mean-one | `token_selection/math_code/taxonomy/mopd_math_code_toploss_m05_c01_math_control_code_progvocab_fixed4_*` |

The SAPD pools are the paper lists (SHA256 in the 1.7B config header):
124 Connective IDs = `pool == Control` rows and 163 Programming IDs =
`taxonomy == code_lexical_structure` rows of
`analysis/c01-c02-selection-stepdiff-20260930/pruned_pool/`. The Math-only
SAPD uses only the 124 list with a 5% budget.

### Position-budget arms (Math+Code only, added 2026-10-14)

`math_code/sapd_pos_{topology}.yaml` (S9) and `math_code/sapd_csid_{topology}.yaml` (S9-ID)
extend `math_code/sapd_{topology}.yaml` and change only
`control_token_online_selection_timing: current_step` plus
`control_token_online_selection_unit: occurrence` (S9, one candidate position at a time,
budget filled exactly) or `token_id` (S9-ID, whole IDs), and the run names
`q4b-mc-{sapdpos|sapdcsid}-{3g3c|4g3s1t|8g8c}-b528-s42`. Both run one extra
no-grad student prepass per step. Contract test: `tests/test_current_step_position_budget.py`.

## Topologies

| Suffix | GPUs | Layout | Math batch | Math+Code batch | ref FSDP | vLLM mem |
|---|---:|---|---:|---:|---:|---:|
| `3gpu_3c` | 3 | actor + rollout + teacher colocated on all three GPUs | 255 = 3 x 85 | 528 = 3 x 176 | 3 | 0.60 |
| `8gpu_8c` | 8 | actor + rollout + FSDP-8 teacher colocated on all eight GPUs (formerly `8gpu_6s2t`) | 256 = 8 x 32 | 528 = 8 x 66 | 8 | 0.60 |

Notes on the budget:

- Math-only batches follow `configs/baselines/math` (nearest integer to 256
  divisible by the actor count); 60 steps, HF checkpoint at 60.
- Math+Code uses 528 for all three layouts (the batch of the 1.7B SAPD
  3s1t and 8g6s2t runs; the 1.7B `baselines/math_code` suite used 516). 60
  steps, HF checkpoint at step 60. `multi_teacher_distill: true` as in the
  SAPD runs; all domain teacher paths alias the same 30B checkpoint.
- vLLM `gpu_memory_utilization` is lowered to 0.5 / 0.6 because the 4B student
  shares its GPUs with rollout (and, on `3gpu_3c`, with the teacher). The
  colocated 3-GPU layout has not been smoke-tested with a 4B student; run one
  step with `--dry-run` and then a short real step before queueing it.
- Heavy per-step vector logging (token gap / logp / entropy vocab vectors) is
  off. To reproduce the paper's Section 3 gap analysis on the 4B runs, add
  to the chosen OPD file:
  `audit: {token_gap_enabled: true, token_gap_freq_steps: 1, token_gap_vocab_vector_enabled: true, token_gap_vocab_vector_freq_steps: 1, entropy_enabled: true, entropy_vocab_vector_enabled: true}`.
- `runtime.cuda_visible_devices` is null; set it when the physical GPU ids are
  not contiguous (the 1.7B 3s1t runs used `"0,1,2,4"`).

## Launch

```bash
# resolve and print the command without starting anything
bash scripts/run_mopd.sh configs/qwen4b_suite/math/sapd_3gpu_3c.yaml --dry-run
```

```bash
# local launch on an allocated set of GPUs
GPU_IDS=0,1,2 bash start.sh --local --config configs/qwen4b_suite/math_code/sapd_3gpu_3c.yaml
```

```bash
# Slurm
bash scripts/run_mopd.sh configs/qwen4b_suite/math_code/opd_8gpu_8c.yaml --slurm --slurm-args "--mem=800G"
```

Run names follow `q4b-{math|mc}-{method}-{3g3c|4g3s1t|8g8c}-b{batch}-s42`
and are unique across audit, checkpoint, HF and W&B paths. All 42 files resolve
with `mopd_verl.config_profiles.load_raw_config`; the resolved SAPD/EOPD/ExOPD
files differ from their 1.7B counterparts only in student path, batch (Math+Code),
memory fraction, names, and defaults made explicit.

## 2026-10-15 8-GPU files are colocated

At the author's request every `*_8gpu_6s2t.yaml` here now runs colocated on eight GPUs (actor, vLLM rollout and the FSDP-8 teacher share all GPUs; vLLM memory 0.4 as in `3gpu_3c`). Math and Code batches changed from 258 to 256 (divisible by 8); Math+Code stays 528. File names keep `6s2t`; run names and paths use `8g8c`.

## 2026-10-17 Math+Code OPD with per-token logging

`math_code/opd_toklog_{3gpu_3c,8gpu_8c}.yaml` extend the plain `opd_*` files of the same topology and only add the per-token audit block (token gap / entropy / logp vectors, response records, per-response loss) and `-toklog-` run names `q4b-mc-opd-toklog-{3g3c,4g3s1t,8g8c}-b528-s42`. Training is identical to the plain run. The 1.7B counterparts at batch 528 are in `../qwen1p7b_suite/math_code/`.

## 2026-10-17 File names and vLLM memory

Every former `*_8gpu_6s2t.yaml` is now `*_8gpu_8c.yaml`, so the name matches the colocated layout set on 2026-10-15; contents, run names and paths are unchanged. `rollout.gpu_memory_utilization` is 0.60 in every file of this suite (`3gpu_3c` was 0.40).

The `4gpu_3s1t` files (3 actor GPUs + 1 dedicated teacher) were removed the same day under the colocated-only rule; see `analysis/config-cleanup-20261017/`. Run names that mention `4g3s1t` above refer to runs trained before.

## 2026-10-25 Math-only and Code-only native OPD entropy top 20% / tail 80%

`{math,code}/opd_native_ent_{top20,tail80}_toklog_{4gpu_4c,8gpu_8c}_b256.yaml` (8 files) are the Qwen3-4B twins of the 1.7B student-entropy baselines in `../token_selection/{math/taxonomy,code}/native_opd_ent_*_b256.yaml`: native OPD that trains only on the top 20% (`topk`, retention 0.2) or only on the bottom 80% (`bottomk`, retention 0.8) of each response's positions ranked by student entropy, weights 0/1 (`token_baseline_normalize_weights: false`), `audit.loss_variance_signal: policy_gradient_distillation_signal+entropy_token_weighting`, per-token logging on, 60 steps, HF checkpoint at step 60. They extend `{math,code}/_opd_native_toklog_{4gpu_4c,8gpu_8c}_b256.yaml`, which are not launchable by themselves: they change only the topology of `opd_native_toklog_3gpu_3c.yaml` (the equal-weight naive runs `q4b-{math,code}-opdnative-toklog-3g3c-b255-s42`) to 4 / 8 colocated GPUs, FSDP-4 / FSDP-8 teacher, batch 256, and keep its run names. Run names `q4b-{math,code}-opdnative-{enttop20,enttail80}-toklog-{4g4c,8g8c}-b256-s42`, HF `checkpoints/qwen4b/{math,code}/<run>`. Contract test: `tests/test_qwen4b_entropy_selection_baselines.py`; the real-batch audit test `tests/test_entropy_audit_real_batch.py` picks these files up by name.
