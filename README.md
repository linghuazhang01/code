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
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6-GPU loss-only token-gradient audit run using fsdp=2 sequence replay. |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8-GPU loss-only token-gradient audit run. |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2-GPU run with the same model/data/objective and all audit disabled. |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4-GPU audit-off run. |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6-GPU audit-off run using the memory-safe TP=2 profile. |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8-GPU audit-off run. |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2-GPU one-step metrics smoke run with all audit outputs and full-vocab vectors enabled. |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2-GPU one-step smoke run with loss-only token-gradient selection. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml` | 2-GPU Qwen3-0.6B smoke run for full/sample/token gradient closure. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b16_1step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 16 and one training step. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b32_2step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 32 and two training steps. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b64_3step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 64 and three training steps. |
| `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_2gpu_b16_2step_smoke.yaml` | 2-GPU smoke for Qwen3-4B student distilled from Qwen3-30B-A3B teachers over math/code/IF/science. The effective batch is 16 so the four domains sample 4 examples each. |
| `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu.yaml` | 6-GPU Qwen30B split-teacher profile: 4 actor/rollout GPUs and 2 ref/teacher GPUs, equal math/code/IF/science sampling, policy-gradient distillation, domain-gradient audit every 2 steps, and token-gap full-vocab vectors. |
| `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_6gpu_fsdp.yaml` | 6-GPU Qwen30B compatibility profile with the same four domains and split teacher placement, `actor.fsdp_size=1`, and batch size 512. |

Formal 2/4/6/8 GPU profiles use:

- student: `../models/Qwen3-4B`
- math teacher: `../models/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- code teacher: `../models/Qwen3-4B-Non-Thinking-RL-Code-Step300`
- train files under `data/G-OPD-Training-Data/DeepMath-103K/` and `data/G-OPD-Training-Data/Eurus/`
- teacher top-k local-support distillation with `topk_distill_k=32`
- teacher models default to CPU storage through `model.teacher_model_device: cpu`; set it to `gpu` in the YAML only when the run has enough spare GPU memory.

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

`mopd_formal_audit_loss_only_*gpu.yaml` keeps the same loss-only token-gradient selection policy: `token_gradient_gap_selection_enabled=false`, `token_gradient_gap_abs_selection_enabled=false`, and `token_gradient_loss_abs_selection_enabled=true`. The 2/4/8-GPU profiles keep the full all-audit surface, including sample-gradient metrics. The 6-GPU loss-only profile is the memory-safe fsdp=2 profile: it keeps full-gradient and token-gradient audits through sequence replay, caps `data.max_response_length=10240`, sets `rollout.max_model_len=12288`, uses `token_gradient_top_p=0.15`, and disables sample-gradient metrics because each worker owns only a sharded parameter view.

For fsdp=2 token-gradient runs, `sequence_masked_target_enabled=true` and `sequence_masked_target_use_as_primary=true` are required. `token_gradient_top_p=1.0` is a useful closure check: the `topp100_*` token-gradient selection should cover all candidate tokens and match the corresponding domain gradient with cosine/projection/norm-ratio near 1.

`mopd_formal_audit_off_*gpu.yaml` sets `audit.enabled=false` and explicitly turns off all audit subfamilies.

The all-audit and loss-only smoke profiles are tracked as metrics test profiles. They use `data.train_batch_size=32`, `actor.ppo_mini_batch_size=32`, and `trainer.total_training_steps=1`, while keeping the formal `data.max_response_length=16384` and full-vocab token gap and entropy vectors enabled.

The gradient consistency smoke profiles use `../models/Qwen3-0.6B` to keep closure checks cheap. They enable sequence-masked domain targets, sample/token backward recompute, and token `top_p=1.0` closure checks so the full-token gradient can be compared against the domain gradient.

### Qwen30B Split-Teacher Profiles

The Qwen30B profiles use:

- student: `../models/Qwen3-4B`
- math/code/IF/science teachers: `../models/Qwen3-30B-A3B`
- train files:
  - math: `data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet`
  - code: `data/G-OPD-Training-Data/Eurus/code_train.parquet`
  - IF: `data/G-OPD-Training-Data/IF/train.parquet`
  - science: `data/G-OPD-Training-Data/Science/train.parquet`
- reward router: `grpo/rewards/mixed.py`
- `data.load_parquet_direct=true`, which avoids Hugging Face Arrow cache materialization and normalizes mixed parquet schemas before concatenation.

Current reward routing:

| Domain | Current data source | Reward |
| --- | --- | --- |
| `math` | `DeepMath-103K` | Vendored verl default reward; DeepMath rows route to `math_verify.compute_score`. |
| `code` | Eurus code rows such as `taco` | Vendored verl default reward; code contest rows use sandbox-fusion when configured, otherwise the vendored `prime_code` test-case scorer. |
| `if` | `m2rl_ifbench` from Nemotron/Open-Instruct instruction-following data | `grpo.rewards.m2rl.compute_ifbench_reward`: uses `verifiable_instructions` first, then AllenAI IFBench strict checking as fallback. |
| `science` | `m2rl_gpqa` or science/GPQA rows | `grpo.rewards.m2rl.compute_gpqa_reward`: extracts the final option letter and compares it to the metadata/label answer. |

The main 6-GPU Qwen30B profile keeps `domain_sampling_weights` equal across `math`, `code`, `if`, and `science`, with `data.train_batch_size=504`, `actor.ppo_mini_batch_size=504`, and `actor.fsdp_size=2`. The compatibility `6gpu_fsdp` profile uses `data.train_batch_size=512`, `actor.ppo_mini_batch_size=512`, and `actor.fsdp_size=1`; with equal weights this gives 128 samples per domain per step.

The 4-domain 2-GPU smoke profile was verified remotely on 2026-07-07 for 2 steps with `data.train_batch_size=16`; each domain sampled 4 examples per step, domain-gradient closure reached rel_l2=0 and cosine=1.0 on the audited step, and token-gap vocab vectors were emitted for all four domains. The remote data disk was 96% full in the previous run, so long runs should clean space before launch.

Configs default to `logger: '["console","tensorboard","wandb"]'`, `runtime.wandb_entity: lz101-rice-university`, and `runtime.env_file: .env.local`. Put `WANDB_API_KEY` in `.env.local` on the machine that launches training. This file is gitignored and must not be committed. Override `runtime.wandb_mode=disabled` for local dry runs without W&B.

On a fresh remote checkout, run `scripts/setup_remote_training_env.sh` after `git pull`. It installs the M2RL/IF verifier dependencies and, by default, runs `git lfs pull` plus a Parquet sanity check for the four repo-local training files under `data/G-OPD-Training-Data/`. Set `PULL_GIT_LFS_DATA=0` or `CHECK_MOPD_DATA=0` only when the data disk is managed separately.

## Launch

Run on a synced remote checkout:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

`scripts/start_remote_mopd_training.sh` defaults to `/root/miniconda3/envs/mopd-verl` when that environment exists and prints the resolved Python path at launch.

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

Download the 30B teacher used by the Qwen30B distillation smoke config:

```bash
MODEL_ROOT=/root/autodl-tmp/opd_mopd/models \
  scripts/download_qwen30b_teacher.sh
```

Four-domain Qwen30B configs declare all teacher domains:

```yaml
model:
  domain_teacher_paths:
    math: ../models/Qwen3-30B-A3B
    code: ../models/Qwen3-30B-A3B
    if: ../models/Qwen3-30B-A3B
    science: ../models/Qwen3-30B-A3B
```

When `model.domain_teacher_paths` is present, the launcher renders `actor_rollout_ref.ref.model.teacher_paths` and the ref worker returns `{domain}_teacher_log_prob` tensors such as `if_teacher_log_prob` and `science_teacher_log_prob`. Older two-domain configs still use the legacy `ref.model.base_model_path` path for the code teacher.

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
