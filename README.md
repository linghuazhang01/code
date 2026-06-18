# Multi-Teacher OPD Math + Code Training

Chinese version: [README.zh.md](README.zh.md)

This checkout is the current OPD/MOPD training entrypoint. The training runtime imports `verl` from `third_party/verl`; a separate remote `G-OPD` checkout is no longer required for the default training path.

## Portable Layout

- Code: `OPD-code/`
- Data: `OPD-code/data/G-OPD-Training-Data/`
- Vendored verl: `OPD-code/third_party/verl/`
- Models: `OPD-code/../models/`
- Logs: `OPD-code/logs/`
- Checkpoints: `OPD-code/checkpoints/`
- Audit output: `OPD-code/audit/`

The current remote default is `/root/autodl-tmp/opd_mopd/OPD-code`, but that is only a sync-script default. Override `REMOTE_HOST`, `REMOTE_PORT`, and `REMOTE_CODE_DIR` to move to another server.

## Training Entrypoints

- `scripts/run_mopd.sh`: generic local launcher for any YAML config.
- `scripts/run_math_code_mopd.sh`: math+code MOPD wrapper, defaulting to `configs/mopd_math_code.yaml`.
- `scripts/run_general_reasoner_mopd.sh`: General-Reasoner/GReasoner 14B teacher MOPD wrapper, defaulting to `configs/mopd_general_reasoner.yaml`.
- `scripts/start_remote_mopd_training.sh`: remote environment checks plus detached `screen` launch for any config.
- `scripts/start_general_reasoner_mopd_training.sh`: remote shortcut for `configs/mopd_general_reasoner.yaml`.
- `scripts/prepare_general_reasoner_data.sh`: prepares WebInstruct-verified train/eval parquet for the General-Reasoner MOPD path.

For a focused Chinese guide to the YAML profiles, distillation objective
switches, audit knobs, and common overrides, see
[`CONFIG_GUIDE.zh.md`](CONFIG_GUIDE.zh.md).

## From Scratch

### 1. Sync Code And Data

If you cloned this repository from GitHub, install Git LFS and fetch the LFS-managed data/wheel files before running any training script:

```bash
git clone git@github.com:linghuazhang01/code.git OPD-code
cd OPD-code
git lfs install
git lfs pull
```

Without `git lfs pull`, the parquet files under `data/G-OPD-Training-Data/` and the large wheel under `third_party/verl/` remain as small pointer files, so data checks and training startup will fail.

If local data is missing:

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
bash scripts/download_mopd_data.sh
```

Sync to the current remote without launching training:

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh --sync-only
```

The sync script reads the SSH password from local `ssh.sh` when available, does not print it, uses rsync progress output, and syncs both `data/G-OPD-Training-Data` and `third_party/verl`.

### 2. Install Remote Environment

Run on the remote synced checkout:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/setup_remote_training_env.sh
```

Useful overrides:

```bash
CONDA_ROOT=$HOME/miniconda3
ENV_NAME=mopd-verl
INSTALL_VERL_DEPS=1
FORCE_REINSTALL=0
INSTALL_SGLANG=0
USE_MEGATRON=0
```

For Jupyter/Notebook, use the dedicated non-interactive wrapper:

```python
!bash scripts/setup_notebook_training_env.sh
```

It finds or installs Miniconda, creates `mopd-verl` from `conda-forge` with
`--override-channels` to avoid Anaconda ToS prompts, runs the regular training
environment setup, and registers the `MOPD (mopd-verl)` Jupyter kernel. Select
that kernel or restart the current kernel after installation.

### 3. Download Or Verify Models

The current formal config uses the 0.6B student plus two teacher checkpoints. The downloader also prepares the 4B base model so the student can be switched to `Qwen3-4B` later:

```text
../models/Qwen3-0.6B
../models/Qwen3-4B
../models/Qwen3-4B-Non-Thinking-RL-Math-Step500
../models/Qwen3-4B-Non-Thinking-RL-Code-Step300
```

Run:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/download_mopd_models.sh
```

The script downloads the student from `Qwen/Qwen3-0.6B`, the 4B base model from `Qwen/Qwen3-4B`, and the two teacher checkpoints from the default Keven16 hub ids:

- `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500`
- `Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300`

Set `DOWNLOAD_BASE_4B=0` only when you do not need the `../models/Qwen3-4B` base model.
Set `DOWNLOAD_TEACHERS=0` to skip math/code teacher downloads. If those teacher directories already exist and you only want to verify them, use `DOWNLOAD_TEACHERS=0 REQUIRE_MATH_CODE_TEACHERS=1`. You can still override `MATH_TEACHER_MODEL_ID` and `CODE_TEACHER_MODEL_ID` for another hub source.

Use `MODEL_BACKEND=modelscope` for ModelScope.

### 4. Start Training

Run directly on the remote:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml --run-id mopd_manual_$(date +%Y%m%d_%H%M%S)
```

Or sync and start from local:

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
ASSUME_YES=1 bash scripts/sync_and_start_remote_mopd.sh configs/mopd_formal_single_a800.yaml --run-id mopd_manual_$(date +%Y%m%d_%H%M%S)
```

The remote launcher checks the vendored `verl`, Python imports, data files, local model directories, `screen`, GPU idleness, and stale Ray state before creating a detached screen run. It writes latest pointers to:

```text
logs/opd_target_run_id
logs/opd_target_log
logs/opd_target_config
logs/opd_target_gpu_csv
```

Follow the active log:

```bash
tail -f "$(cat logs/opd_target_log)"
```

## Smoke Test

After environment setup:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/run_remote_one_step_smoke.sh
```

The smoke test uses synthetic `smoke_data/*.parquet` and temporarily maps all model roles to `Qwen/Qwen3-0.6B`. It only verifies that one optimizer step can run.

## General-Reasoner 14B Teacher MOPD

This path trains a Qwen3-4B student against `TIGER-Lab/General-Reasoner-Qwen3-14B` on WebInstruct-verified reasoning prompts. In configs and eval code the short domain name is `greasoner`, while the model and data adapter use `General-Reasoner`.

Prepare WebInstruct-verified parquet files on the remote:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/prepare_general_reasoner_data.sh
```

Prepare only the models needed by this MOPD path:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
DOWNLOAD_STUDENT=0 \
REQUIRE_STUDENT=0 \
DOWNLOAD_TEACHERS=0 \
REQUIRE_MATH_CODE_TEACHERS=0 \
DOWNLOAD_REASONING_BASE_14B=0 \
DOWNLOAD_REASONING_TEACHER=1 \
bash scripts/download_mopd_models.sh
```

The command above prepares:

```text
../models/Qwen3-4B
../models/General-Reasoner-Qwen3-14B
```

It does not download `../models/Qwen3-14B` unless `DOWNLOAD_REASONING_BASE_14B=1` is set explicitly.

Dry-run the generated verl command:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
bash scripts/run_general_reasoner_mopd.sh --dry-run -- \
  trainer.total_training_steps=1
```

Start remote training in a detached screen session:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_general_reasoner_mopd_training.sh
```

Equivalent explicit form:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/start_remote_mopd_training.sh configs/mopd_general_reasoner.yaml \
  --run-id greasoner_14b_mopd_$(date +%Y%m%d_%H%M%S)
```

`configs/mopd_general_reasoner.yaml` sets `trainer.n_gpus_per_node=8`, so the launcher requires eight visible GPUs before it starts Ray/verl. A smaller run needs a compatible config/override set for `trainer.n_gpus_per_node`, tensor parallelism, and batch sizes; exposing only `GPU_IDS=0` is not enough.

The config uses:

- student: `../models/Qwen3-4B`
- reasoning teacher: `../models/General-Reasoner-Qwen3-14B`
- train parquet: `data/GeneralReasoner/WebInstructVerified/train.parquet`
- validation parquet: `eval/domains/greasoner/data/WebInstructVerified/test.parquet`
- teacher label: `extra_info.opd_teacher=reasoning`
- thinking mode: `data.enable_thinking=true`

Follow logs after launch:

```bash
tail -f "$(cat logs/opd_target_log)"
```

## Formal Single-A800 Config

`configs/mopd_formal_single_a800.yaml` uses:

- separate domain train files under `data/G-OPD-Training-Data/DeepMath-103K/` and `data/G-OPD-Training-Data/Eurus/`
- local models under `../models/`
- domain sampling weights `{math: 0.5, code: 0.5}`
- `data.train_batch_size=512`
- `data.val_batch_size=1024`
- `data.max_prompt_length=2048`
- `data.max_response_length=16384`
- `actor.ppo_mini_batch_size=512`
- `actor.ppo_micro_batch_size_per_gpu=1`
- `actor.use_dynamic_bsz=false` in base profiles. Turn it on for heavy
  top-k/audit runs so actor update micro-batches are grouped by token count
  instead of fixed sample count.
- `rollout.gpu_memory_utilization=0.8` in multi-GPU base profiles
  (`0.9` for the single-A800 base). Lower it for heavy top-k/audit runs when
  rollout KV cache competes with actor backward memory.
- `trainer.logger=["console","tensorboard"]`
- `audit.log_sample_level_freq_steps=1`
- `audit.log_validation_metrics_freq_steps=1`
- `audit.full_gradient_enabled=true`
- `audit.full_gradient_freq_steps=1`
- `audit.full_gradient_train_max_samples_per_domain=null`
- `audit.full_gradient_micro_batch_size_per_gpu=1`
- `audit.sample_gradient_enabled=true`
- `audit.sample_gradient_freq_steps=1`
- `audit.sample_gradient_norm_enabled=true`
- `audit.sample_gradient_cos_enabled=false`
- `audit.sample_gradient_log_sample_level_freq_steps=1`
- `audit.token_gap_enabled=true` / `audit.token_gap_freq_steps=1`: records per-domain `teacher_logp - student_logp` and `abs(teacher_logp - student_logp)` distribution stats, plus raw per-domain vectors in `token_gap_vectors.jsonl`.
- `audit.entropy_enabled=true` / `audit.entropy_freq_steps=1`: records per-domain teacher entropy, student entropy, and teacher-student cross-entropy sums and distribution stats, plus raw vectors in `entropy_distribution_vectors.jsonl`. When top-k distillation is active, the teacher-student cross entropy uses the same local support selected by `actor.topk_distill_support_source` and the same renormalization as the top-k objective.
- `audit.token_conflict_enabled=true` / `audit.token_conflict_freq_steps=1`: records token-level teacher/student disagreement summaries and top token rows in `token_conflict_attribution.jsonl`.
- `audit.token_gradient_enabled=false` by default. When enabled, the tracker first collects the per-domain global distribution of all valid response tokens in the current step using `gap_abs = abs(teacher_logp - student_logp)`, then runs extra gradient recompute only for `top100_gap_abs` and the configured top-p mass token set. If `autograd.grad()` is disconnected, the tracker falls back to a safe backward diagnostic. When enabled, domain target chunks are temporarily stored in FP32 to reduce restore `.grad` quantization error. For small debug runs, `audit.token_gradient_strict_grad_restore=true` additionally snapshots the original `.grad` and restores that exact snapshot after fallback.
- `audit.token_gradient_freq_steps=10` in the base profile.
- `audit.token_gradient_top_p=0.10`: for token-gradient audit, aggregate both `top100_gap_abs` and the smallest token set covering this fraction of domain `gap_abs` mass.
- `actor.topk_distill_enabled=false` by default. Enabling it uses local-support matching with renormalized KL over `actor.topk_distill_k` selected tokens. `actor.topk_distill_support_source=teacher` uses teacher top-k ids; `student` uses old actor/student top-k ids and gathers teacher/current-student logprobs on that same support. The default KL direction is reverse; set `actor.topk_distill_kl_direction=forward` for the forward-KL ablation, or use `actor.distill_mode=topk_forward_kl_with_tail` / `topk_reverse_kl_with_tail` for the older tail-bucket objective.
- `rollout.teacher_prefix_sampling_enabled=false` by default. When enabled,
  the trainer reads `rollout.teacher_prefix_dataset_key` from each sample,
  tokenizes and truncates it to `rollout.teacher_prefix_length`, then lets the
  student continue from `prompt + teacher_prefix`. The launcher automatically
  enables `actor.teacher_prefix_enabled` so teacher-prefix masks are routed
  correctly. By default `actor.teacher_prefix_loss_region=suffix_only`, so
  teacher-prefix tokens are context only and do not contribute loss; student
  suffix tokens keep the configured OPD/top-k objective. Set
  `actor.teacher_prefix_loss_region=prefix_and_suffix` to also train prefix
  tokens with forward-KL.
- For top-k distillation at `max_response_length=16384`, `actor.use_dynamic_bsz=true`
  reduces mixed-length actor backward peaks, but it cannot split one very long
  response across multiple backward passes. If a debug run samples near-16K
  responses, use a shorter launch override such as `data.max_response_length=8192`
  or `4096` before turning every gradient audit on.
- Formal single-GPU profiles use `audit.full_gradient_storage_dtype=bfloat16`. The sequential tracker keeps only the two domain targets on CPU; dot, norm, and cosine accumulation remain FP32.
- `paper_eval.enabled=false`

On a single-A800 node with 120 GB of CPU RAM, do not combine batch size 512,
optimizer offload, and per-step full/sample-gradient auditing for the first
formal-data run. Use this low-memory launch first:

```bash
bash scripts/start_remote_mopd_training.sh configs/mopd_formal_single_a800.yaml \
  --run-id mopd_a800_lowmem_$(date +%Y%m%d_%H%M%S) \
  -- \
  data.train_batch_size=128 \
  data.val_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  trainer.val_before_train=false
```

This keeps full-parameter and per-sample gradient diagnostics enabled while
reducing the number of samples retained and recomputed in one actor update.
Every sample in the actor mini-batch receives sample-to-domain cosine and
projection metrics. Raising `RAY_memory_usage_threshold` is not a fix when the
process is already close to physical RAM exhaustion.

## Formal Dual-A800 Config

Use `configs/mopd_formal_dual_a800.yaml` for the current two-A800 diagnostic
run. It uses `train_batch_size=ppo_mini_batch_size=256`,
`max_response_length=16384`, replicated actor audit coordinates
(`actor.fsdp_size=1`), rollout TP=2 across the two GPUs,
`gpu_memory_utilization=0.8`, and `total_training_steps=10`.

The standard per-domain data, OPD loss, teacher confidence/gap, reward, and
cost metrics remain enabled. Full-parameter auditing and sample gradient norms
remain enabled. Sample-to-domain cosine is disabled until the two-pass FSDP
implementation is ready.

Three dual-A800 ablation configs are available for comparing the training
objective while keeping the same model/data/rollout profile:

| Config | Objective | Top-k support |
| --- | --- | --- |
| `configs/mopd_formal_dual_a800_pg_loss.yaml` | chosen-token OPD / PG-style loss | off |
| `configs/mopd_formal_dual_a800_teacher_topk.yaml` | top-k local support matching, reverse-KL, `k=5` | teacher top-k |
| `configs/mopd_formal_dual_a800_student_topk.yaml` | top-k local support matching, reverse-KL, `k=5` | old actor/student top-k |

```bash
cd /root/OPD-code
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800.yaml \
  --run-id mopd_dual_a800_$(date +%Y%m%d_%H%M%S)
```

External paper eval remains available through `eval/scripts/run_paper_eval_suite.sh`, but it is legacy for the portable path because it still requires the full G-OPD eval tree.

For the current metric definitions, use [`metrics_zh.md`](metrics_zh.md) as the source of truth. The formal config logs these audit families:

- per-domain data, OPD loss, teacher confidence/gap, calibration, reward, advantage sign, and response length metrics;
- token-level conflict attribution summaries and top-token JSONL rows;
- full-parameter train gradient metrics: per-domain grad norm, math-vs-code cosine/conflict, domain-vs-total cosine, and signed projection share;
- sampled per-example gradient metrics: sample grad norm for every sample in the actor update mini-batch; sample-to-domain cosine and projection share are disabled in the dual-A800 profile;
- cost metrics such as step seconds, tokens/sec, peak memory, and full-gradient backward time.

The Chinese gradient report for this run is in
[`reports/2026-06-13--mopd-full-sample-gradient-report.zh.md`](reports/2026-06-13--mopd-full-sample-gradient-report.zh.md).

## Formal Scaled-A800 Configs

The multi-GPU A800 profiles scale the current two-A800 setup while keeping 16K
responses, per-step full-gradient audit, and sample gradient norm logging.
Sample-to-domain cosine remains disabled until the two-pass FSDP path is ready.

| Config | GPUs | Train/PPO batch | Response | Rollout TP | vLLM util | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `configs/mopd_formal_4gpu_a800.yaml` | 4 | 512 | 16384 | 4 | 0.8 | One TP=4 rollout group, about 128 prompts/GPU. |
| `configs/mopd_formal_8gpu_a800.yaml` | 8 | 1024 | 16384 | 4 | 0.8 | Two TP=4 rollout groups, about 128 prompts/GPU. |

Launch examples:

```bash
GPU_IDS=0,1,2,3 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_4gpu_a800.yaml \
  --run-id mopd_4gpu_a800_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_8gpu_a800.yaml \
  --run-id mopd_8gpu_a800_$(date +%Y%m%d_%H%M%S)
```

## Formal A800 Audit Overrides

The base 1/2/4/8-card A800 configs keep the standard training/audit setup:
gap, entropy, token-conflict, full-gradient, and sample-gradient norm are on;
sample-gradient cosine, token-gradient audit, and top-k distillation are off.
The dedicated `*_audit_all.yaml` / `*_audit_light.yaml` copies have been
removed; use the base config plus Hydra overrides after `--`.

`audit_all` means `log_sample_level`, `log_validation_metrics`,
`full_gradient_enabled`, sample gradient norm and cosine, token-gradient audit
every step, and teacher-top-5 distillation. To run the student-top-k variant,
change `topk_distill_support_source` to `student`:

```bash
GPU_IDS=0,1 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_dual_a800.yaml \
  --run-id mopd_dual_audit_all \
  -- \
  mopd_audit.log_sample_level=true \
  mopd_audit.log_validation_metrics=true \
  mopd_audit.full_gradient_enabled=true \
  mopd_audit.sample_gradient_enabled=true \
  mopd_audit.sample_gradient_norm_enabled=true \
  mopd_audit.sample_gradient_cos_enabled=true \
  mopd_audit.token_gap_enabled=true \
  mopd_audit.entropy_enabled=true \
  mopd_audit.token_conflict_enabled=true \
  mopd_audit.token_gradient_enabled=true \
  mopd_audit.full_gradient_freq_steps=1 \
  mopd_audit.sample_gradient_freq_steps=1 \
  mopd_audit.sample_gradient_cos_freq_steps=1 \
  mopd_audit.sample_gradient_log_sample_level_freq_steps=1 \
  mopd_audit.token_gap_freq_steps=1 \
  mopd_audit.entropy_freq_steps=1 \
  mopd_audit.token_conflict_freq_steps=1 \
  mopd_audit.token_gradient_freq_steps=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.actor.policy_loss.distill_mode=chosen_token_reverse_kl \
  actor_rollout_ref.actor.policy_loss.topk_distill_enabled=true \
  actor_rollout_ref.actor.policy_loss.topk_distill_kl_direction=reverse \
  actor_rollout_ref.actor.policy_loss.topk_distill_k=5 \
  actor_rollout_ref.actor.policy_loss.topk_distill_support_source=teacher \
  actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket=false
```

All heavy audit families are independently throttled by their `*_freq_steps`
fields. Frequency checks use `step % freq_steps == 0`; setting a larger value
keeps the feature enabled but runs/writes it less often. Top-k distillation is a
training objective, so it is controlled by `actor.topk_distill_enabled` rather
than an audit frequency.

`token_gradient_max_samples_per_domain`, `token_gradient_top_k_per_sample`, and `token_gradient_min_teacher_diff` remain in the config for compatibility with older launch files, but token-gradient audit no longer uses them to truncate the candidate pool; candidate rows use `global_candidate_scope=all_valid_response_tokens`.

`audit_light` disables the extra audit outputs requested for ablation:

```bash
scripts/run_mopd.sh configs/mopd_formal_single_a800.yaml -- \
  mopd_audit.log_sample_level=false \
  mopd_audit.log_validation_metrics=false \
  mopd_audit.full_gradient_enabled=false \
  mopd_audit.sample_gradient_enabled=false \
  mopd_audit.sample_gradient_norm_enabled=false \
  mopd_audit.sample_gradient_cos_enabled=false \
  mopd_audit.token_gap_enabled=false \
  mopd_audit.entropy_enabled=false \
  mopd_audit.token_conflict_enabled=false \
  mopd_audit.token_gradient_enabled=false \
  actor_rollout_ref.actor.policy_loss.topk_distill_enabled=false
```

With `audit_light`, audit still keeps the core per-domain rows and cost rows:
data/token counts, OPD loss mean/std/variance, advantage sign, response length,
reward/accuracy when available, calibration, duplicate rate, the legacy
`teacher_student_gap_mean` and `teacher_confidence_mean`, global loss/data
summary, and `training_cost.jsonl`. For a pure no-audit run, additionally set
`audit.enabled=false`.

## Formal Single-H200 Config

Use `configs/mopd_formal_single_h200.yaml` for one NVIDIA H200 141GB. It keeps
the math/code sampling, model paths, sequence lengths, and
`train_batch_size=ppo_mini_batch_size=1024` unchanged, while using the extra HBM
and bandwidth for:

- GPU-resident optimizer state (`optimizer_offload=false`);
- vLLM CUDA graphs (`enforce_eager=false`);
- `log_prob_micro_batch_size_per_gpu=2`;
- `max_num_batched_tokens=65536`;
- `max_num_seqs=16`.

The first full-length run intentionally keeps
`ppo_micro_batch_size_per_gpu=1`. Responses can reach 16K tokens, so increasing
the backward micro-batch before observing peak memory is not a safe default.

Launch:

```bash
cd /root/OPD-code
GPU_IDS=0 bash scripts/start_remote_mopd_training.sh \
  configs/mopd_formal_single_h200.yaml \
  --run-id mopd_h200_$(date +%Y%m%d_%H%M%S)
```

If the first 5-10 steps stay below roughly 120 GiB peak memory, the next tuning
candidate is `actor.ppo_micro_batch_size_per_gpu=2`. Do not increase the global
train batch merely because the GPU is larger, because that changes rollout
freshness and experiment semantics.

## Domain Sampling

Training uses `data.domain_train_files` plus `data.domain_sampling_weights`:

```yaml
data:
  domain_train_files:
    math:
      - data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
    code:
      - data/G-OPD-Training-Data/Eurus/code_train.parquet
  domain_sampling_weights:
    math: 0.5
    code: 0.5
  domain_sampling_replacement: true
```

The launcher still expands these files into `data.train_files` for verl compatibility. The patched dataset loader tags each row by source file, and the patched trainer uses `DomainBatchSampler` to emit exact domain counts per training batch. With `train_batch_size=512` and `{math: 0.5, code: 0.5}`, every train batch contains 256 math rows and 256 code rows.

## TensorBoard And Audit Files

The formal config uses `trainer.logger=["console","tensorboard"]`. TensorBoard event files are written by the verl trainer, while `scripts/start_remote_mopd_training.sh` writes the raw console log to `logs/<run_id>.log`.

MOPD audit scalars are logged into TensorBoard with domain-first tags such as:

```text
math/length/response_mean
math/advantage/positive_frac
math/sample_grad/norm_mean
math/sample_grad_cos/domain_cos_mean
math/sample_grad_contribution/projection_share_mean
math/token_conflict/teacher_teacher_diff_mean
math/token_conflict/combined_diff_mass
math/token_gap/gap_abs_p95
math/entropy/teacher_entropy_p50
math/token_grad_conflict/other_cos_negative_frac
math/token_grad/top100_gap_abs_gap_abs_mass_frac
math/token_grad/topp10_gap_abs_mass_gap_abs_mass_frac
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/full_grad_contribution/math_to_total/signed_projection_share
```

The JSONL audit files are written under:

```text
audit/formal_single_a800/
```

Important files include `domain_step_metrics.jsonl`, `loss_variance_sample.jsonl`, `token_gap_vectors.jsonl`, `entropy_distribution_vectors.jsonl`, `token_conflict_attribution.jsonl`, `token_grad_metrics.jsonl`, `sample_grad_metrics.jsonl`, `validation_probe.jsonl`, `validation_gain_variance.jsonl`, `training_cost.jsonl`, and `audit_errors.jsonl`.
