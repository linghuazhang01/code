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

## From Scratch

### 1. Sync Code And Data

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
Set `DOWNLOAD_TEACHERS=0` only when those teacher directories already exist and you want to verify them without downloading. You can still override `MATH_TEACHER_MODEL_ID` and `CODE_TEACHER_MODEL_ID` for another hub source.

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

## Formal Single-A800 Config

`configs/mopd_formal_single_a800.yaml` uses:

- separate domain train files under `data/G-OPD-Training-Data/DeepMath-103K/` and `data/G-OPD-Training-Data/Eurus/`
- local models under `../models/`
- domain sampling weights `{math: 0.5, code: 0.5}`
- `data.train_batch_size=128`
- `data.val_batch_size=1024`
- `data.max_prompt_length=2048`
- `data.max_response_length=16384`
- `actor.ppo_mini_batch_size=128`
- `actor.ppo_micro_batch_size_per_gpu=1`
- `rollout.gpu_memory_utilization=0.8`
- `trainer.logger=["console","tensorboard"]`
- `audit.full_gradient_enabled=true`
- `audit.full_gradient_freq_steps=1`
- `audit.full_gradient_train_max_samples_per_domain=null`
- `audit.full_gradient_micro_batch_size_per_gpu=1`
- `audit.sample_gradient_enabled=true`
- `audit.sample_gradient_norm_enabled=true`
- `audit.sample_gradient_cos_enabled=true`
- `audit.sample_gradient_cos_max_samples_per_domain=8`
- `paper_eval.enabled=false`

External paper eval remains available through `scripts/run_paper_eval_suite.sh`, but it is legacy for the portable path because it still requires the full G-OPD eval tree.

For the current metric definitions, use [`metrics_zh.md`](metrics_zh.md) as the source of truth. The formal config logs these audit families:

- per-domain data, OPD loss, teacher confidence/gap, calibration, reward, advantage sign, and response length metrics;
- full-parameter train gradient metrics: per-domain grad norm, math-vs-code cosine/conflict, domain-vs-total cosine, and signed projection share;
- sampled per-example gradient metrics: sample grad norm distribution for every sample in the actor update mini-batch, plus sample-to-domain cosine and projection share for up to 8 selected samples per domain;
- cost metrics such as step seconds, tokens/sec, peak memory, and full-gradient backward time.

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

The launcher still expands these files into `data.train_files` for verl compatibility. The patched dataset loader tags each row by source file, and the patched trainer uses `DomainBatchSampler` to emit exact domain counts per training batch. With `train_batch_size=128` and `{math: 0.5, code: 0.5}`, every train batch contains 64 math rows and 64 code rows.

## TensorBoard And Audit Files

The formal config uses `trainer.logger=["console","tensorboard"]`. TensorBoard event files are written by the verl trainer, while `scripts/start_remote_mopd_training.sh` writes the raw console log to `logs/<run_id>.log`.

MOPD audit scalars are logged into TensorBoard with domain-first tags such as:

```text
math/length/response_mean
math/advantage/positive_frac
math/sample_grad/norm_mean
math/sample_grad_cos/domain_cos_mean
math/sample_grad_contribution/projection_share_mean
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/full_grad_contribution/math_to_total/signed_projection_share
```

The JSONL audit files are written under:

```text
audit/formal_single_a800/
```

Important files include `domain_step_metrics.jsonl`, `loss_variance_sample.jsonl`, `sample_grad_metrics.jsonl`, `validation_probe.jsonl`, `validation_gain_variance.jsonl`, `training_cost.jsonl`, and `audit_errors.jsonl`.
