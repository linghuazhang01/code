# Loss Vector Compatibility

## Definitions

- `logp_abs_vector`: `abs(teacher_logp - old_student_logp)` on sampled response tokens.
- `topk_teacher_student_cross_entropy_vocab`: per-position top-k teacher/student
  cross entropy, aggregated into a dense vocabulary vector by the observed
  response token ID.
- Both are audit-only signals and do not alter the actor loss builder.

## Compatibility

| Training loss builder | Log-p absolute vector | Top-k CE vocab vector | Notes |
| --- | --- | --- | --- |
| `policy_gradient` | Yes | Yes | CE requests teacher top-k tensors only on audit steps; PG remains the training objective. |
| `chosen_token_reverse_kl` | Yes | Yes | Chosen-token loss is unchanged; CE is an extra no-grad audit forward. |
| `topk_kl` | Yes | Yes | Existing top-k tensors can be reused; explicit audit settings control vector output. |

## Outputs

- `logp_abs_vectors.jsonl`
  - `logp_abs_vector_domain`
- `logp_abs_vocab_vectors.jsonl`
  - `logp_abs_sum_vector_vocab`
  - `logp_abs_mean_vector_vocab`
- `topk_teacher_student_cross_entropy_vocab_vectors.jsonl`
  - `teacher_student_cross_entropy_sum_vector_vocab`
  - `teacher_student_cross_entropy_mean_vector_vocab`

## Verification

- Python syntax compilation passed for every changed Python file.
- `git diff --check` passed.
- Target YAML assertions passed with Ruby/Psych.
- Focused unit tests were added but could not be executed locally because no
  available interpreter includes both PyYAML and Torch.

## Remote four-domain smoke validation

- Server: 3x NVIDIA A800 80GB; functional run used 1 student GPU and 1 teacher GPU.
- Student: Qwen3-4B; teacher: Qwen3-30B-A3B shared by math/code/if/science.
- Objective: `policy_gradient`; `topk_distill_enabled=false`; one training step.
- Run id: `pg4domain_vector_smoke_20260712_061905_r4_1student`.
- Step completed in 35.56 seconds with no traceback or OOM.
- All six JSONL vector products contained exactly math, code, if, and science.
- Every vocabulary vector had fixed length 151,936 and finite values.
- For every sampled token, `logp_abs_vector_domain == abs(gap_signed_vector_domain)`.
- CE metadata matched teacher support, k=32, no tail bucket, temperature=1.0.
- The run emitted `actor/pg_loss` and no runtime `actor/topk_distill_loss` metric.

Remote artifact directory:

`/root/autodl-tmp/opd_mopd/OPD-code/audit/pg4domain_vector_remote_smoke/pg4domain_vector_smoke_20260712_061905_r4_1student`

Limitation: the same server's 2-rank student/FSDP smoke stalled after rollout
while both student GPUs remained at 100% utilization. The successful run proves
the four-domain policy-gradient vector data path, but not the final 6-student +
2-teacher distributed placement.
