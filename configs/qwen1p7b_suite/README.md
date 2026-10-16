# Qwen3-1.7B student suite at global batch 528 (Math+Code)

Equal-weight Top-32 renormalized reverse-KL OPD with a Qwen3-1.7B student on the Math+Code mixture at
**global batch 528**, the batch of the 1.7B SAPD Math+Code runs and of `../qwen4b_suite/math_code`.
Qwen3-30B-A3B-Instruct-2507 teacher, 60 steps, HF checkpoint at step 60 only, seed 42.
`configs/baselines/math_code/topk32_uniform_*` (516 / 512, `multi_teacher_distill: false`) is unchanged.

Every file here has the **per-token audit on** (token gap, entropy and logp vectors per position and per
token ID, top-K teacher-student cross entropy per ID, gzip response records, per-response loss / reward /
correctness; every step). Logging only: no switch changes the loss or the gradients, and weighting and
online selection stay off. The audit adds per-step disk and time cost; there is no plain (no-audit) 528
file for the 1.7B student.

| File | GPUs | Layout | Batch | vLLM mem | Run name |
|---|---:|---|---|---:|---|
| `math_code/opd_toklog_3gpu_3c.yaml` | 3 | actor + rollout + FSDP-3 teacher colocated | 528 = 3 x 176 | 0.40 | `q1p7b-mc-opd-toklog-3g3c-b528-s42` |
| `math_code/opd_toklog_4gpu_3s1t.yaml` | 4 | 3 actor/rollout + 1 dedicated teacher | 528 = 3 x 176 | 0.75 | `q1p7b-mc-opd-toklog-4g3s1t-b528-s42` |
| `math_code/opd_toklog_8gpu_8c.yaml` | 8 | actor + rollout + FSDP-8 teacher colocated | 528 = 8 x 66 | 0.75 | `q1p7b-mc-opd-toklog-8g8c-b528-s42` |

The matching Qwen3-4B files are `../qwen4b_suite/math_code/opd_toklog_{3gpu_3c,4gpu_3s1t,8gpu_6s2t}.yaml`
(`8gpu_6s2t` is 8-GPU colocated, run name `…8g8c…`); they extend the plain 4B `opd_*` files and only add
the audit block and `-toklog-` run names.

`_common.yaml` is `../qwen4b_suite/math_code/_common.yaml` with the student path changed to
`../mopd/models/Qwen3-1.7B`; `_methods/opd.yaml` is copied unchanged. Each topology file resolves to
its 4B toklog counterpart except for the student path, run/output names and the vLLM memory fraction
(`tests/test_qwen1p7b_math_code_opd_b528.py`). The 0.40 / 0.75 / 0.75 fractions follow the 1.7B runs
(Math-only `3gpu_3c`, SAPD Math+Code 3s1t and 8g8c). HF upload goes to
`checkpoints/baselines/math_code/<run>` in `icemoon28/opd-checkpoints`.

Nothing here has been launched or smoke-tested on GPUs.

```bash
bash scripts/run_mopd.sh configs/qwen1p7b_suite/math_code/opd_toklog_4gpu_3s1t.yaml --dry-run
```
