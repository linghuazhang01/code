# Multi-Teacher OPD Math + Code Training

Chinese version: [README.zh.md](README.zh.md)

This checkout is the current OPD/MOPD training entrypoint. The training runtime imports `verl` from `third_party/verl`; a separate remote `G-OPD` checkout is not required for the default training path.

## Layout

- Code: `OPD-code/`
- Data: `OPD-code/data/G-OPD-Training-Data/`
- Vendored verl: `OPD-code/third_party/verl/`
- Models: `OPD-code/../models/`
- Logs: `OPD-code/logs/`
- Checkpoints: `OPD-code/checkpoints/`
- Audit output: `OPD-code/audit/`

## Configs

Three formal MOPD variants are kept with 2/4/6/8 GPU profiles, plus metrics smoke profiles:

| Config | Purpose |
| --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2-GPU formal 4B math/code run with every MOPD audit family enabled. |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4-GPU all-audit run, same objective with a larger global batch. |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6-GPU all-audit run, TP=2 with three rollout data-parallel groups. |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8-GPU all-audit run, TP=4 with two rollout data-parallel groups. |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2-GPU all-audit run where token-gradient selection only uses loss magnitude. |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4-GPU loss-only token-gradient audit run. |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6-GPU loss-only token-gradient audit run. |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8-GPU loss-only token-gradient audit run. |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2-GPU run with the same model/data/objective and all audit disabled. |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4-GPU audit-off run. |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6-GPU audit-off run using the memory-safe TP=2 profile. |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8-GPU audit-off run. |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2-GPU one-step metrics smoke run with all audit outputs and full-vocab vectors enabled. |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2-GPU one-step smoke run with loss-only token-gradient selection. |

All profiles use:

- student: `../models/Qwen3-4B`
- math teacher: `../models/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- code teacher: `../models/Qwen3-4B-Non-Thinking-RL-Code-Step300`
- train files under `data/G-OPD-Training-Data/DeepMath-103K/` and `data/G-OPD-Training-Data/Eurus/`
- teacher top-k local-support distillation with `topk_distill_k=32`

GPU scaling:

| GPUs | Config suffix | train/mini batch | rollout TP | Ray CPUs |
| --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 256 | 2 | 8 |
| 4 | `_4gpu` | 512 | 4 | 16 |
| 6 | `_6gpu` | 768 | 2 | 24 |
| 8 | `_8gpu` | 1024 | 4 | 32 |

`mopd_formal_audit_all_*gpu.yaml` additionally enables:

- sample-level and validation audit rows
- full-gradient audit
- sample-gradient norm and sample-to-domain cosine
- token gap occurrence vectors and full-vocab vectors
- teacher/student entropy occurrence vectors and full-vocab vectors
- token conflict attribution
- token-gradient audit with domain-level signed-gap, gap-abs, and loss top-k/top-p selections

`mopd_formal_audit_loss_only_*gpu.yaml` keeps the same audit surface as the all-audit profiles, including full/sample gradients, token gap vectors, entropy vectors, token conflict, and token-gradient logging. The only difference is token-gradient candidate selection: `token_gradient_gap_selection_enabled=false`, `token_gradient_gap_abs_selection_enabled=false`, and `token_gradient_loss_abs_selection_enabled=true`.

`mopd_formal_audit_off_*gpu.yaml` sets `audit.enabled=false` and explicitly turns off all audit subfamilies.

The smoke profiles are tracked as metrics test profiles. They use `data.train_batch_size=32`, `actor.ppo_mini_batch_size=32`, and `trainer.total_training_steps=1`, while keeping the formal `data.max_response_length=16384` and full-vocab token gap and entropy vectors enabled.

## Launch

Run on a synced remote checkout:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

Run the audit-off profile:

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

Run the loss-only token-gradient audit profile:

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_2gpu.yaml \
  --run-id mopd_audit_loss_only_2gpu_$(date +%Y%m%d_%H%M%S)
```

Use the matching GPU list for larger profiles:

```bash
GPU_IDS=0,1,2,3 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_6gpu.yaml \
  --run-id mopd_audit_all_6gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

Local dry-run:

```bash
scripts/run_mopd.sh configs/mopd_formal_audit_all_2gpu.yaml --dry-run
```

Metrics smoke:

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_smoke.yaml \
  --run-id mopd_metrics_loss_only_smoke_$(date +%Y%m%d_%H%M%S)
```

Sync from local without launching:

```bash
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh --sync-only
```

Sync and launch:

```bash
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

## Audit Files

When an `mopd_formal_audit_all_*gpu.yaml` or `mopd_formal_audit_loss_only_*gpu.yaml` config is used, JSONL audit files are written under the matching directory, for example `audit/formal_audit_all_2gpu/` or `audit/formal_audit_loss_only_2gpu/`.

Important files include:

- `domain_step_metrics.jsonl`
- `loss_variance_domain_step.jsonl`
- `loss_variance_sample.jsonl`
- `token_gap_vectors.jsonl`
- `token_gap_vocab_vectors.jsonl`
- `entropy_distribution_vectors.jsonl`
- `entropy_vocab_vectors.jsonl`
- `token_conflict_attribution.jsonl`
- `token_grad_metrics.jsonl`
- `sample_grad_metrics.jsonl`
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

Full-vocab vector files use token-id coordinates: index `v` corresponds to tokenizer token id `v`. `token_gap_vocab_vectors.jsonl` stores signed/absolute log-prob gap sum and mean vectors. `entropy_vocab_vectors.jsonl` stores `student_entropy` and `teacher_student_cross_entropy` sum and mean vectors.

For detailed metric definitions, see [metrics_zh.md](metrics_zh.md). For config field explanations and common overrides, see [CONFIG_GUIDE.zh.md](CONFIG_GUIDE.zh.md).
