# Notes: Audit-only loss vectors

## Current behavior

- Teacher-student cross entropy is computed only when `uses_topk_distill_loss(...)` is true.
- `entropy_vocab_vector_enabled` aggregates available student entropy and cross entropy by observed response token ID.
- `token_gap_vocab_vector_enabled` already aggregates signed and absolute teacher/student chosen-token log-probability gaps.
- Policy-gradient reward is `teacher_logp - old_student_logp`; existing `gap_abs` is its absolute value.

## Required behavior

- Request teacher top-k tensors for audit even when the training builder is policy gradient or chosen-token reverse KL.
- Compute top-k teacher-student CE only on configured audit steps.
- Emit explicitly named `logp_abs` domain and vocabulary vectors without removing `gap_abs` compatibility fields.
