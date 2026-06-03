# Multi-Teacher OPD Math + Code Training

中文版本: [README.zh.md](README.zh.md)

This directory is a small wrapper around the official `RUCBM/G-OPD` verl fork.
It keeps the local research code minimal and delegates actual PPO/OPD execution
to `verl.trainer.main_ppo`.

## What The Code Does

- `configs/mopd_math_code.yaml` stores the paper-style two-teacher MOPD setting.
- `configs/mopd_formal_single_a800.yaml` stores the current single-A800 formal
  run setting: 0.6B student, two 4B teachers, train batch size 1024, and
  `max_response_length=1024`.
- `mopd_verl/settings.py` loads that YAML into typed dataclasses.
- `mopd_verl/launch.py` converts the typed config into Hydra overrides for
  `python -m verl.trainer.main_ppo`.
- `mopd_verl/prepare_data.py` merges math/code parquet files, validates
  `extra_info.opd_teacher`, and converts paper math-eval JSONL files into verl
  validation parquet files.
- `mopd_verl/paper_eval.py` is injected into the G-OPD/verl trainer so each
  validation can trigger the external seven-benchmark paper eval suite.
- `mopd_verl/smoke_data.py` creates tiny synthetic parquet files for remote
  one-step smoke tests.
- `mopd_verl/verl_audit.py` is injected into the G-OPD/verl trainer and writes
  MOPD audit JSONL plus TensorBoard scalars.
- `configs/mopd_audit_smoke.yaml` enables audit + TensorBoard for one-step
  smoke tests.
- `scripts/apply_gopd_audit_patch.py` idempotently patches the G-OPD checkout
  so the verl dataset/trainer forwards audit fields.
- `scripts/run_math_code_mopd.sh` is the generic launcher.
- `scripts/setup_remote_training_env.sh` bootstraps the remote conda + G-OPD
  environment.
- `scripts/prepare_paper_eval_data.sh` prepares AIME/HMMT validation parquets
  and downloads/checks the HumanEval+, MBPP+, and LiveCodeBench data.
- `scripts/run_paper_eval_suite.sh` runs AIME24, AIME25, HMMT25 Feb., HMMT25
  Nov., HumanEval+, MBPP+, and LCB after validation.
- `scripts/run_remote_one_step_smoke.sh` runs a one-step training smoke test on
  the remote machine after setup and writes a completion marker to its log.

## Training Stack

The default stack follows the official G-OPD / ExOPD recipe:

- Codebase: `RUCBM/G-OPD`, whose training code is based on verl v0.6.1.
- Entrypoint: `verl.trainer.main_ppo`.
- Objective: reverse-KL OPD with ExOPD reward scaling `lambda_vals=1.25`.
- Multi-teacher switch:
  `actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true`.
- Teacher routing field: `extra_info.opd_teacher`. Audit also reads
  `extra_info.domain`, `extra_info.source_domain`, and `extra_info.sample_id`.

## Formal Single-A800 Run Setting

The formal single-GPU setting is `configs/mopd_formal_single_a800.yaml`. It is
not the paper-exact 8-GPU recipe; it is a practical run profile for the current
remote A800 80GB machine:

- Student/training model: `Qwen3-0.6B`.
- Math teacher: `Qwen3-4B-Non-Thinking-RL-Math-Step500`.
- Code teacher: `Qwen3-4B-Non-Thinking-RL-Code-Step300`.
- Training batch: `data.train_batch_size=1024`.
- PPO minibatch: `actor.ppo_mini_batch_size=16`.
- PPO microbatch per GPU: `actor.ppo_micro_batch_size_per_gpu=1`.
- Prompt length: `data.max_prompt_length=2048`.
- Response length: `data.max_response_length=1024`.
- vLLM cache budget: `rollout.gpu_memory_utilization=0.5`.
- Logger: `["console","tensorboard"]`.
- Training-time validation parquets: AIME24, AIME25, HMMT25 Feb., HMMT25 Nov.,
  and Eurus code validation.
- External paper eval: AIME24, AIME25, HMMT25 Feb., HMMT25 Nov., HumanEval+,
  MBPP+, and LCB.

The `1024` response cap is important for throughput interpretation. When
`response_length/mean` and `response_length/clip_ratio` approach 1024 and 1.0,
respectively, the step is generation/validation-bound and can be much slower
even if the training update itself is stable.

`paper_eval.enabled=true` launches the full paper eval suite after every
`_validate()` call. This matches the seven datasets reported in the paper, but
it will substantially lengthen each validation on a single A800. Disable it with
`+paper_eval.enabled=false` for smoke-only runs.

## Start Training

Run the formal single-A800 training from the remote `OPD-code` checkout:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh
```

To print the generated Hydra command without starting training:

```bash
cd /root/autodl-tmp/opd_mopd/OPD-code
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

DRY_RUN=1 \
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh
```

To run in a detached `screen` and save logs:

```bash
mkdir -p /root/autodl-tmp/opd_mopd/logs
screen -dmS mopd_formal bash -lc '
cd /root/autodl-tmp/opd_mopd/OPD-code &&
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate mopd-verl &&
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh 2>&1 | tee /root/autodl-tmp/opd_mopd/logs/formal_$(date +%Y%m%d_%H%M%S).log
'
```

Temporary Hydra overrides can be appended after `--`. For example, this keeps the
formal config but disables external paper eval for a faster sanity run:

```bash
MOPD_CONFIG=configs/mopd_formal_single_a800.yaml \
PYTHON_BIN=python \
bash scripts/run_math_code_mopd.sh -- +paper_eval.enabled=false
```

For a one-step smoke test, use `scripts/run_remote_one_step_smoke.sh` instead of
the formal config.

## Training Data

Production data:

- Train: `G-OPD-Training-Data/math_and_code/train.parquet`
- Validation: `G-OPD-Training-Data/PaperEval/AIME24/test.parquet`
- Validation: `G-OPD-Training-Data/PaperEval/AIME25/test.parquet`
- Validation: `G-OPD-Training-Data/PaperEval/HMMT25Feb/test.parquet`
- Validation: `G-OPD-Training-Data/PaperEval/HMMT25Nov/test.parquet`
- Validation: `G-OPD-Training-Data/Eurus/code_validation.parquet`

Each training row must follow the verl RL dataset shape: `data_source`,
`prompt`, `ability`, `reward_model`, and `extra_info`. For MOPD, `extra_info`
must contain either `{"opd_teacher": "math"}` or `{"opd_teacher": "code"}`.

Smoke-test data:

- Generated by `python -m mopd_verl.smoke_data <output_dir>`.
- Contains two tiny rows, one routed to the math teacher and one routed to the
  code teacher.
- Used only to verify that the remote verl stack can start and execute one
  optimizer step. It is not evidence for model quality.

External paper eval data:

- Math: `data/aime24/test.jsonl`, `data/aime25/test.jsonl`,
  `data/hmmt25_feb/test.jsonl`, `data/hmmt25_nov/test.jsonl`
- EvalPlus: `code_eval/data/HumanEvalPlus.jsonl`,
  `code_eval/data/MbppPlus.jsonl`
- LiveCodeBench: `code_eval/coding/LiveCodeBench/code_generation_lite/test*.jsonl`

Prepare the remote data before the first formal run:

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/prepare_paper_eval_data.sh
```

Full-suite outputs are written under:

```text
/root/autodl-tmp/opd_mopd/eval_outputs/paper_suite/formal_single_a800/step_XXXXXXXX/
```

## Training Models

Paper-style default:

- Student/reference: `Qwen/Qwen3-4B`
- Math teacher: `Qwen3-4B-Non-Thinking-RL-Math`
- Code teacher: `Qwen3-4B-Non-Thinking-RL-Code`

Remote smoke-test default:

- Student/reference/math-teacher/code-teacher:
  `Qwen/Qwen3-0.6B`

The smoke test intentionally uses one small public model for all roles. That
keeps the first remote check cheap while still exercising the multi-teacher
routing and OPD launcher path. For real runs, override the model paths back to
the paper-style Qwen3-4B student and the two domain teachers.

## Remote Setup

The remote server described by `code/ssh.sh` currently has `/root/miniconda3`,
Ubuntu 22.04, CUDA 12.8, and one A800 80GB GPU. The setup script assumes it is
run on that server:

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/setup_remote_training_env.sh
```

Important environment knobs: `REMOTE_ROOT=/root/autodl-tmp/opd_mopd`,
`CONDA_ROOT=/root/miniconda3`, `ENV_NAME=mopd-verl`, `INSTALL_SGLANG=0`, and
`FORCE_REINSTALL=0`.

The script creates:

- `${REMOTE_ROOT}/G-OPD`
- `${REMOTE_ROOT}/smoke_data/train.parquet`
- `${REMOTE_ROOT}/smoke_data/val.parquet`
- `${REMOTE_ROOT}/env.sh`
- `${REMOTE_ROOT}/logs/`

The current setup script also pins the remote stack pieces that were needed on
the A800 container:

- `transformers[hf_xet]==4.51.3`
- `ray[default]==2.46.0`
- `click<8.2`
- `numpy<2.0.0`
- OpenTelemetry packages at `1.26.0`

`HF_ENDPOINT` defaults to `https://hf-mirror.com`, because the tested remote
container could not reach `huggingface.co` directly. FlashInfer installation is
skipped by default to avoid a slow GitHub wheel download; vLLM falls back to the
PyTorch-native sampler, which is sufficient for the smoke test.

## One-Step Smoke Training

After setup:

```bash
bash /root/autodl-tmp/opd_mopd/OPD-code/scripts/run_remote_one_step_smoke.sh
```

The smoke script overrides the production config to:

- use one GPU,
- use batch size 1,
- disable W&B,
- enable TensorBoard,
- enable step 0/step 1 validation to record validation gain,
- disable checkpoint saving,
- stop after `trainer.total_training_steps=1`,
- write logs under `${REMOTE_ROOT}/logs/`,
- write audit JSONL under `${REMOTE_ROOT}/audit/smoke/`,
- write TensorBoard events under `${TENSORBOARD_DIR:-${REMOTE_ROOT}/tensorboard}`,
- clean stale Ray state before startup and stop Ray on script exit.

Verified remote result:

- Server: `autodl-container-857546be50-cbac1eda`
- GPU: `NVIDIA A800 80GB PCIe`
- Env: `/root/miniconda3/envs/mopd-verl`
- Code: `/root/autodl-tmp/opd_mopd/OPD-code`
- G-OPD/verl: `/root/autodl-tmp/opd_mopd/G-OPD`
- Log: `/root/autodl-tmp/opd_mopd/logs/one_step_smoke_20260601_003357.log`
- Result: completed `step:1` with `training/global_step:1`
- Step time: `27.805183600634336` seconds
- Peak allocated GPU memory: `31.825812339782715` GB
- Throughput: `1.5105097165782364` tokens/s
- Post-run status: no active training/Ray processes; GPU memory returned to
  `0 MiB`

TensorBoard layout:

- Event dir: `/root/autodl-tmp/opd_mopd/tensorboard/audit_smoke`
- Scalar tag count: `134`
- Audit tags use the `domain/category/metric` layout by default. The first
  TensorBoard level is the domain name, for example `math/loss/token_opd_loss_variance`,
  `math/full_grad/grad_norm`, `math/full_grad_anchor/AIME2024/full_grad_cosine_i_j`,
  `math/validation/score`, and `math/validation_gain/score`.
- Non-domain metrics use `global/category/metric`, for example
  `global/data/domain_mix_entropy`, `global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k`,
  and `global/cost/gpu_seconds_step`.

Audit JSONL files:

- `domain_step_metrics.jsonl`: per-step, per-domain core metrics.
- `loss_variance_domain_step.jsonl`: per-step, per-domain OPD loss variance summaries.
- `loss_variance_sample.jsonl`: sample-level `opd_loss`, `sample_token_opd_loss_mean`, and `sample_token_opd_loss_variance`; these are not emitted as TensorBoard scalars to avoid tag explosion.
- `validation_probe.jsonl`: raw validation values and adjacent validation gain.
- `validation_gain_variance.jsonl`: rolling validation-gain mean and variance.
- `training_cost.jsonl`: step time, GPU seconds, throughput, and memory.
- `audit_errors.jsonl`: fail-soft audit errors.

Current boundary: low-cost gradient sketches have been removed. `grad`, `grad_anchor`, and `grad_conflict` are no longer computed or logged. When `full_gradient_enabled=true`, `full_grad`, `full_grad_anchor`, and `full_grad_conflict` run real actor backward and read complete actor parameter gradients. The full-gradient path now uses verl's PPO policy-loss function with MOPD `advantages=-reverse_kl`, and validation anchors are token-weighted running mean gradients across validation batches. In the formal config, `full_gradient_train_max_samples_per_domain=null` and `full_gradient_validation_max_samples_per_domain=null`, so full-gradient audit does not truncate domains within the current train or validation batch stream. This is still the current on-policy batch/pass, not a per-step sweep over the entire train parquet, and the predicted delta is a first-order OPD/PPO-surrogate estimate rather than an executed Adam optimizer step. Full metric formulas and the latest retained TensorBoard tags are documented in [`metrics_zh.md`](metrics_zh.md). Removed optional metrics such as rank stability, shadow probe, sample influence, extra teacher-logprob diagnostics, CI, dot-only fields, tail percentiles, and all low-cost gradient proxies are no longer computed or logged.

Validation-gain TensorBoard tags use a domain root only when the raw validation metric key contains a configured domain such as `math` or `code`. Benchmark keys such as `val-core/AIME2024/reward/mean@1` are logged under `global/validation_gain/...`, with the benchmark name folded into the metric segment.

## Metric Semantics

For the authoritative Chinese reference of current TensorBoard tags, JSONL files, and formulas, see [`metrics_zh.md`](metrics_zh.md).

## Implementation Principles

The audit implementation is intentionally low-intrusion:

1. `settings.py` defines typed `AuditConfig` fields such as output directory,
   domains, TensorBoard layout, TensorBoard pruning mode, logging frequencies,
   validation-anchor scheduling, full-gradient settings, calibration bins, and
   feature flags.
2. `launch.py` converts these fields into `+mopd_audit.*` Hydra overrides for
   the G-OPD/verl trainer.
3. `apply_gopd_audit_patch.py` patches the remote G-OPD checkout in two places:
   the dataset forwards `domain`, `sample_id`, and `source_domain`; the trainer
   instantiates `MOPDAuditLogger`, calls it after rollout/ref-logprob data are
   available, adds validation-gain metrics after `_validate()`, and records cost
   after timing/throughput metrics are computed.
4. `MOPDAuditLogger.log_training_step()` reads a `DataProto` batch, computes
   masked token-level OPD loss statistics, aggregates per-domain and per-sample
   rows, and returns scalar metrics for the normal verl logger. Full-parameter
   gradient metrics are computed separately inside patched actor workers via
   `compute_mopd_full_gradient_metrics()`, using verl's PPO policy-loss path
   and token-weighted validation-anchor accumulation. Scalar tags are built by
   `MOPDAuditLogger._tag()` as `domain/category/metric`, so TensorBoard is
   grouped by domain first and by metric family second.
5. Before scalar metrics reach TensorBoard, `filter_tensorboard_metrics()` uses
   `tensorboard_filter.py` to keep only the `core` subset: validation/gain,
   full-gradient alignment, essential loss signals, teacher/calibration, cost,
   and training health. Structured JSONL files in
   `mopd_audit.output_dir` still keep detailed per-sample and per-domain
   diagnostics that should not become high-cardinality TensorBoard tags.
6. Defensive behavior is fail-soft: if audit computation raises an exception,
   the error is written to `audit_errors.jsonl` and training receives only
   `global/audit/error=1.0` instead of crashing.

The design goal is observability for weight-initialization research. The current
gradient diagnostics are full-parameter backward measurements rather than
token-level sketches, so they are more faithful but substantially more
expensive. They should be used with batch-size, validation frequency, and memory
settings that leave enough room for the extra actor backward passes.

## Local Validation

- Run wrapper tests: `PYTHONPATH=code python3 -m unittest discover -s code/tests`
- Build the Hydra command without launching training:
  `DRY_RUN=1 code/scripts/run_math_code_mopd.sh`
- Inspect teacher labels:
  `PYTHONPATH=code python3 -m mopd_verl.prepare_data inspect /path/to/train.parquet`
