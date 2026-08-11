# EOPD baseline

This project exposes **Entropy-Aware On-Policy Distillation (EOPD)** as an
optional loss builder. Existing OPD, Top-k KL, and MOPD configurations are not
changed or implicitly upgraded.

Sources:

- Paper: [Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)
- Official artifact: [WLS04/EOPD](https://github.com/WLS04/EOPD)

## Objective implemented here

For every valid response token, EOPD retains the existing clipped OPD policy
loss and adds a truncated forward KL only when the teacher has high entropy:

```text
L_EOPD = L_clipped_OPD
       + alpha * mean_valid_tokens(
           1[H(teacher) >= tau]
           * KL(q_teacher_topk || p_student_full_vocab_at_teacher_topk)
         )
```

The implementation follows four details that distinguish EOPD from this
project's existing `topk_kl` baseline:

1. The teacher entropy is computed over the full vocabulary in natural-log
   units (nats).
2. The support is selected by the teacher's Top-k token IDs.
3. Teacher probabilities are renormalized inside that Top-k support, while the
   gathered student probabilities retain the full-vocabulary normalizer.
4. The gated KL numerator is divided by the number of all valid response
   tokens, not only the number of high-entropy tokens.

The official artifact uses the inclusive boundary `entropy >= threshold`; this
repository follows that executable behavior.

## Config switch

The three method-specific parameters are under `actor`:

```yaml
actor:
  distill_loss_builder: eopd
  distill_mode: chosen_token_policy_gradient
  eopd_entropy_threshold: 0.8
  eopd_forward_kl_weight: 1.0
  eopd_topk_k: 16
```

Use the paired baseline matrix to keep all other settings fixed:

```bash
# Existing OPD baseline
bash scripts/run_mopd.sh --dry-run \
  'configs/matrices/eopd_baseline_matrix.yaml::opd'

# EOPD baseline
bash scripts/run_mopd.sh --dry-run \
  'configs/matrices/eopd_baseline_matrix.yaml::eopd'
```

The output directories and experiment names are separate, so selecting EOPD
does not overwrite an OPD run.

## Paper hyperparameters and local baseline

| Setting | Paper | Paired local baseline |
|---|---:|---:|
| Entropy threshold `tau` | 0.8 | 0.8 |
| Forward-KL coefficient `alpha` | 1.0 | 1.0 |
| Teacher Top-k `k` | 16 | 16 |
| Learning rate | 3e-6 | 5e-6 |
| Optimizer | AdamW | inherited from current verl setup |
| LR scheduler | cosine | inherited from current verl setup |
| Training batch size | 128 | 504 |
| PPO mini-batch size | 32 | 504 |
| Samples per prompt | 1 | 1 |
| Rollout temperature | 1.0 | 1.0 |
| Top-p | 1.0 for Qwen, 0.8 for Llama | 1.0 for Qwen |
| Max response length | 4096 | 16384 |
| Epochs | 3 for MATH, 2 for DAPO | 3, capped by 200 steps |

The local matrix deliberately keeps the project's current OPD experimental
settings fixed and changes only the objective. This is the appropriate setup
for a fair OPD-versus-EOPD baseline comparison. To reproduce the paper's
training schedule instead, also set the learning rate, batch sizes, response
length, and epoch/step limit to the paper values.

The paper's current v3 appendix and Top-k ablation specify `k=16`. The official
artifact README still shows `topk_logits=32`; this implementation uses the
paper value but exposes `eopd_topk_k` so either choice is reproducible.

## Compatibility boundaries

- `distill_loss_builder: policy_gradient` remains ordinary OPD.
- `distill_loss_builder: topk_kl` remains the existing replacement-style
  Top-k distillation objective.
- EOPD requires teacher Top-k tensors and full-vocabulary teacher entropy.
  These tensors are requested only when EOPD is selected.
- The paired EOPD profile disables teacher-prefix sampling and dynamic domain
  budgeting. Those extensions should be evaluated as separate ablations.
