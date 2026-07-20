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

## Fresh Setup

Use these local scripts for a fresh machine or a new local checkout:

| Purpose | Script |
| --- | --- |
| Create or refresh the Conda/Python environment from `environment.yml` | `scripts/setup_training_env.sh` |
| Download or validate training data only | `scripts/download_mopd_data.sh` |
| Download or validate model assets only | `scripts/download_mopd_models.sh`, `scripts/download_qwen30b_teacher.sh` |
| Download and validate the current data + model bundle | `scripts/download_training_assets.sh` |
| Launch a local training run | `scripts/run_local_mopd_training.sh` |
| Render/check a config without launching training | `scripts/run_mopd.sh --dry-run` |

Install the training environment from this checkout:

```bash
cd /path/to/OPD-code
bash scripts/setup_training_env.sh
source logs/activate_training_env.sh
```

### Blackwell / CUDA 12.8 environment

RTX PRO 6000 Blackwell and other `sm_120` GPUs must use the dedicated
PyTorch 2.8 / CUDA 12.8 profile. CUDA 12.4 PyTorch wheels do not contain
`sm_120` kernels.

```bash
cd /path/to/OPD-code
ENV_NAME=mopd-verl-blackwell \
ENV_FILE=$(pwd)/environment.blackwell.yml \
  bash scripts/setup_training_env.sh
source logs/activate_training_env.sh
```

The Blackwell profile pins `torch==2.8.0+cu128`, `vllm==0.11.0`,
`transformers==4.55.4`, `tensordict==0.10.0`, and the official
`flash-attn==2.8.3.post1` wheel built for Python 3.10, PyTorch 2.8, CUDA 12,
and CXX11 ABI TRUE. This combination supports the default
`flash_attention_2` backend on `sm_120`. For a conservative first correctness
run, keep remove-padding disabled:

```bash
ENV_NAME=mopd-verl-blackwell \
GPU_IDS=0,1,2 \
  bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --run-id blackwell_fsdpsize2_smoke_$(date +%Y%m%d_%H%M%S) -- \
  actor_rollout_ref.model.use_remove_padding=false \
  trainer.save_freq=-1
```

The old implementation failed remote validation on PyTorch 2.8 / CUDA 12.8
for `world_size=2, fsdp_size=1`. It passed the logical `(ddp=2, fsdp=1)` mesh
directly to FSDP1, so `HYBRID_SHARD` was downgraded to `NO_SHARD` on a
singleton shard group. Gradients differed across ranks, and the next
unconditional `reshard(True)` raised `AssertionError: Expects sharded
strategy`.

The current implementation gives this topology explicit semantics. verl keeps
the `(2,1)` logical mesh for dispatch and replica accounting, while the FSDP1
wrapper uses `NO_SHARD` with the WORLD process group. Each rank therefore owns
a full model and optimizer replica, and FSDP all-reduces gradients after
backward. Every manual reshard entry point now checks the effective strategy.
The local PyTorch 2.8/Gloo two-rank oracle reports identical gradients and
parameters across ranks, `2.61e-8` coordinate-wise global-gradient error,
`2.98e-8` micro-batch accumulation error, and `9.31e-9` optimizer-update error.
Its `fsdp_size=2`
`FULL_SHARD` regression also passes. The old 4B/8B failure remains useful as a
negative regression record; use a new experiment log for the fixed PyTorch
2.8 CUDA real-model result.

`rollout.temperature` is also used to score actor/ref log probabilities and
must be finite and strictly positive. For deterministic greedy rollout, set
`rollout.do_sample=false` with `rollout.temperature=1.0`; do not set the latter
to zero, because the scoring path would evaluate `logits / 0` and produce NaNs.

In a 4-step paired Blackwell smoke, audit on/off with `fsdp_size=2` matched
exactly on all 71 non-timing, non-audit training metrics at every step. With
BF16 domain-gradient vectors, gradient-closure relative L2 was
`0.004433`/`0.004472` at steps 2/4, and training-parity relative L2 was
`4.02e-8`/`0`; all checks passed. The audit replay therefore remained
read-only with respect to the production optimizer update.

Download and validate the assets for the Qwen3-30B-A3B-Instruct-2507
math/code/IF/science profiles:

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

For the current Qwen30B teacher configs, `scripts/download_training_assets.sh`
is the recommended entrypoint. It prepares the four-domain training data, the
Qwen3-4B student, and the shared Qwen3-30B-A3B-Instruct-2507 teacher.

To fetch only data with the same defaults:

```bash
DOWNLOAD_MODELS=0 \
REQUIRE_MODELS=0 \
  scripts/download_training_assets.sh
```

To fetch only models with the same defaults:

```bash
DOWNLOAD_DATA=0 \
REQUIRE_MATH_CODE_TRAIN_DATA=0 \
REQUIRE_4DOMAIN_TRAIN_DATA=0 \
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

The lower-level scripts are also available when you need to operate on one
asset family directly:

```bash
# MOPD training parquet data. The current profiles consume all four domains.
scripts/download_mopd_data.sh

# Qwen3-4B student/base helper used by the current asset bundle.
MODEL_ROOT=$(pwd)/../models \
DOWNLOAD_STUDENT=0 \
DOWNLOAD_BASE_4B=1 \
  scripts/download_mopd_models.sh

# Qwen3-30B-A3B-Instruct-2507 teacher.
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_qwen30b_teacher.sh
```

IF/science validation parquet files are optional unless the config enables
them. To prepare and require those files in the same pass:

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
REQUIRE_M2RL_EVAL_DATA=1 \
  scripts/download_training_assets.sh
```

For a one-command bootstrap that runs the asset script inside the new conda
environment, set `DOWNLOAD_ASSETS=1` on `scripts/setup_training_env.sh`;
asset-specific variables are forwarded.

## Configs

Three formal MOPD variants are kept with 2/4/6/8 GPU profiles, plus metrics smoke profiles:

> **FSDP1 `fsdp_size=1` semantics:** every actor rank stores a complete model,
> gradient, and optimizer replica, while FSDP synchronizes gradients over the
> WORLD process group. Multi-rank execution is supported, but its memory and
> checkpoint footprint are much larger than `FULL_SHARD`; use it only when one
> GPU can hold the complete student. `fsdp_size=-1` or the actor world size
> continues to select `FULL_SHARD` across all actor ranks.

| Config | Purpose |
| --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2-GPU formal 4B math/code run with domain-gradient and observation metrics enabled. |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4-GPU domain-gradient audit run, same objective with a larger global batch. |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6-GPU domain-gradient audit run, TP=2 with three rollout data-parallel groups. |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8-GPU domain-gradient audit run, TP=4 with two rollout data-parallel groups. |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2-GPU compatibility profile with domain-gradient audit and loss observations. |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4-GPU compatibility profile with domain-gradient audit. |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6-GPU domain-gradient profile using `fsdp_size=2`. |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8-GPU compatibility profile with domain-gradient audit. |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2-GPU run with the same model/data/objective and all audit disabled. |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4-GPU audit-off run. |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6-GPU audit-off run using the memory-safe TP=2 profile. |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8-GPU audit-off run. |
| `configs/mopd_formal_audit_all_smoke.yaml` | 2-GPU one-step metrics smoke run with all audit outputs and full-vocab vectors enabled. |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 2-GPU one-step domain-gradient and loss-metric smoke run. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b16_1step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 16 and one training step. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b32_2step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 32 and two training steps. |
| `configs/mopd_formal_audit_grad_consistency_2gpu_b64_3step_smoke.yaml` | 2-GPU gradient consistency smoke with batch size 64 and three training steps. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml` | Original math-only training: 4 actor/rollout GPUs + 2 teacher/ref GPUs. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml` | Original code-only training with the same 6-GPU topology. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml` | Original IF-only training with IFBench validation. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml` | Original science-only training with GPQA validation. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml` | Original equal-weight math+code training. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml` | Original equal-weight math+code+science training. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math.yaml` | Math-only training: 6 actor/rollout GPUs + 2 teacher/ref GPUs, batch 504. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_code.yaml` | Code-only training with the same topology, batch, and audit surface. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_if.yaml` | IF-only training with IFBench validation. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_science.yaml` | Science-only training with GPQA validation. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code.yaml` | Equal-weight math+code training with the same topology and audit surface. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science.yaml` | Equal-weight math+code+science training with the same topology and batch 504. |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml` | Top-32 math+code+science distillation with the same 8-GPU topology, `fsdp_size=1`, and batch 504. |

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
- full/domain-gradient norm, pair cosine, domain-to-total projection, closure, and training parity
- token gap occurrence vectors and full-vocab vectors
- teacher/student entropy occurrence vectors and full-vocab vectors
- token conflict attribution
- optional teacher/student top-k cross-entropy and log-probability vectors

The clean rebuild deliberately retires the old nested sample/token backward replays. All formal profiles set `sample_gradient_enabled=false` and `token_gradient_enabled=false`; re-enabling either key fails fast instead of falling back to FSDP private-state manipulation. The `loss_only` files keep their historical selection fields for config compatibility, but those fields do not trigger gradient replay. The 6-GPU profile remains the memory-safe `fsdp_size=2` variant and caps `data.max_response_length=10240` with `rollout.max_model_len=12288`.

Domain decomposition uses a value-preserving gradient gate, so it does not change the production loss denominator. FSDP owns every backward collective; the audit does not call private finalize hooks or add a second synchronization step.
The audit stores the synchronized local `g_total` and `g_domain` shards on CPU and computes cosine/dot products directly. This requires `D + 1` backward replays for `D` domains. Both `g_total` and domain vectors use `full_gradient_storage_dtype`; BF16 vectors are converted by chunk to FP64 for norm/dot accumulation.

`mopd_formal_audit_off_*gpu.yaml` sets `audit.enabled=false` and explicitly turns off all audit subfamilies.

The all-audit and loss-only smoke profiles are tracked as metrics test profiles. They use `data.train_batch_size=32`, `actor.ppo_mini_batch_size=32`, and `trainer.total_training_steps=1`, while keeping the formal `data.max_response_length=16384` and full-vocab token gap and entropy vectors enabled.

The gradient consistency smoke profiles use `../models/Qwen3-0.6B` to keep checks cheap. They compare the domain-gradient sum with the unmasked audit gradient and compare that audit gradient with the real training gradient before the optimizer step.

### Qwen3-30B-A3B-Instruct-2507 Teacher Profiles

The six original 6-GPU profiles remain unchanged. They share:

- student: `../models/Qwen3-4B`
- teacher: `../models/Qwen3-30B-A3B-Instruct-2507`
- 6 visible GPUs: 4 actor/rollout + 2 teacher/ref
- actor `fsdp_size=2`, giving two HYBRID_SHARD replicas of two shards
- rollout TP=2 and micro-batch size 1
- batch/mini-batch size 512 for the one/two-domain profiles and 504 for the
  three-domain profile
- the unified chosen-token policy-gradient objective
- domain-gradient audit every two steps, BF16 CPU vectors, training parity,
  and token-gap vectors; nested sample/token backward is disabled

The six new 8-GPU base profiles mirror the same domains, data, objective, and
audit surface while using:

- 8 visible GPUs: 6 actor/rollout + 2 teacher/ref
- actor `fsdp_size=1`, using synchronized full-model actor replicas
- rollout TP=2 and batch/mini-batch size 504 for every domain combination
- distinct 8-GPU experiment, audit, checkpoint, and paper-eval output paths

The separate Top-32 profile uses the same `fsdp_size=1` topology and batch 504,
with its own distillation objective and audit cadence.

Math uses `DeepMath-103K/train_filtered_level6.parquet`; code uses
`Eurus/code_train.parquet`; IF uses `IF/train.parquet`; science uses
`Science/train.parquet`; the mixed profiles sample math/code or
math/code/science with equal weight. Both base sets use
`data.load_parquet_direct=true` and
`mopd_verl/mixed_reward.py`.

Configs default to `logger: '["console","tensorboard","wandb"]'`,
`runtime.wandb_entity: null`, and `runtime.env_file: .env.local`. Set the W&B
entity through an override or environment when needed. Put `WANDB_API_KEY` in
`.env.local`; this file is gitignored and must not be committed. Override
`runtime.wandb_mode=disabled` for local dry runs without W&B.

Run `scripts/setup_training_env.sh` to create or update the local Conda
environment. `environment.yml` is the CUDA 12.4 compatibility definition and
`environment.blackwell.yml` is the PyTorch 2.8 / CUDA 12.8 Blackwell
definition; the setup script does not maintain a separate pip requirements
or dependency installer.
Use `scripts/prepare_m2rl_eval_data.sh` separately to prepare IF/science
validation parquet files.

## Launch

Run from the local checkout:

```bash
cd /path/to/OPD-code
GPU_IDS=0,1,2 bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --run-id mopd_fsdpsize2_smoke_$(date +%Y%m%d_%H%M%S)
```

`scripts/run_local_mopd_training.sh` uses `CONDA_ROOT=$HOME/miniconda3` and
`ENV_NAME=mopd-verl` by default. It prints the resolved Python path in the
launch log.

Multi-rank `fsdp_size=1` in historical `mopd_formal_*` profiles now uses
synchronized `NO_SHARD` replication. Before launch, verify that each actor GPU
can hold the complete student, gradients, and optimizer state.

Local dry-run:

```bash
scripts/run_mopd.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --dry-run
```

Download the Qwen3-30B-A3B-Instruct-2507 four-domain training assets:

```bash
MODEL_ROOT=$(pwd)/../models \
MIN_FREE_GB=300 \
  scripts/download_training_assets.sh
```

The mixed profile declares the shared teacher for both domains; the IF and
science profiles use the same model through their corresponding
`domain_teacher_paths` entry:

```yaml
model:
  domain_teacher_paths:
    math: ../models/Qwen3-30B-A3B-Instruct-2507
    code: ../models/Qwen3-30B-A3B-Instruct-2507
```

Because all domain paths resolve to the same teacher, the launcher uses one
primary teacher model rather than loading multiple 30B copies.

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
- `validation_probe.jsonl`
- `validation_gain_variance.jsonl`
- `training_cost.jsonl`
- `audit_errors.jsonl`

Full-vocab vector files use token-id coordinates: index `v` corresponds to tokenizer token id `v`. `token_gap_vocab_vectors.jsonl` stores signed/absolute log-prob gap sum and mean vectors. `entropy_vocab_vectors.jsonl` stores `student_entropy` and `teacher_student_cross_entropy` sum and mean vectors.

For detailed metric definitions, see [metrics_zh.md](metrics_zh.md). For config field explanations and common overrides, see [CONFIG_GUIDE.zh.md](CONFIG_GUIDE.zh.md).
