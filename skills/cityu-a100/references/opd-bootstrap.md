# OPD bootstrap on Burgundy

## Outcome and remote layout

The project wrapper is `scripts/setup_burgundy.sh`. It orchestrates repository,
environment, data, model, and verification phases while keeping large assets
under scratch.

| Asset | Default remote path |
|---|---|
| OPD code | `~/scratch/opd/mopd_code` |
| Miniforge | `~/scratch/opd/miniforge3` |
| Conda env | `~/scratch/opd/miniforge3/envs/mopd-verl-a100` |
| Activation helper | `~/scratch/opd/logs/activate_training_env.sh` |
| Training data | `~/scratch/opd/mopd_code/data/G-OPD-Training-Data` |
| Evaluation data | `~/scratch/opd/mopd_code/data/eval_data` |
| Models | `~/scratch/opd/models` |
| Hugging Face cache | `~/scratch/opd/hf_home` |
| Pip cache | `~/scratch/opd/pip_cache` |
| Verification records | `~/scratch/opd/logs/burgundy-setup-*` |

Do not place the 30B model or Conda environment in `$HOME`.

## Local verification before deployment

From the local OPD repository:

```bash
bash -n \
  scripts/setup_burgundy.sh \
  scripts/setup_training_env.sh \
  scripts/download_mopd_data.sh
python3 -m unittest -v \
  tests.test_setup_burgundy \
  tests.test_download_mopd_data
git diff --check -- \
  scripts/setup_burgundy.sh \
  scripts/setup_training_env.sh \
  scripts/download_mopd_data.sh \
  tests/test_setup_burgundy.py \
  tests/test_download_mopd_data.py
```

Preview every remote sync. Transfer only required source files, exclude
`ssh*.sh`, `.env`, data, models, logs, checkpoints, and caches, and never use
`rsync --delete`. If a clean Git commit contains the required scripts, cloning
or pulling that commit is preferable to a broad dirty-tree sync.

## Run the complete bootstrap

Request an A100 allocation using the allocation reference. Inside that
allocation:

```bash
cd "$HOME/scratch/opd/mopd_code"
bash scripts/setup_burgundy.sh --phases all
```

The wrapper validates that the job runs in `gpu_a100`, the assigned GPU name
contains A100, and reported memory is at least 80,000 MiB before heavy work.
Authentication comes only from standard Git/Hugging Face mechanisms; never
place credentials in the wrapper.

## Phase control and resumption

Valid phases are `repo`, `env`, `data`, `models`, and `verify`:

```bash
# Print configuration without writes.
bash scripts/setup_burgundy.sh --dry-run

# Continue after environment creation.
PHASES=data,models,verify bash scripts/setup_burgundy.sh

# Re-run only the GPU witness and manifest capture.
PHASES=verify bash scripts/setup_burgundy.sh
```

Hugging Face downloads are resumable. Do not remove partial model directories
merely because an allocation expires. The repository's G-OPD source uses HTTPS
because outbound GitHub port 80 timed out on Burgundy.

## Installed environment contract

The current build uses pinned Miniforge `26.5.3-0` with installer SHA-256
verification. The observed environment included:

- Python 3.10.21
- PyTorch 2.6.0+cu124
- vLLM 0.8.5.post1
- Ray 2.47.1
- verl 0.6.1
- Transformers 4.51.3
- FlashAttention 2.7.4.post1

The wrapper writes `pip freeze`, Python version, GPU information, job metadata,
and the seeded CUDA witness into a timestamped verification record.

## Installed data and models

Expected training-data checks from the completed download:

| Dataset | Rows |
|---|---:|
| DeepMath-103K filtered level 6 | 57,046 |
| Eurus code | 25,276 |
| IF | 16,575 |
| Science | 19,670 |

Expected evaluation data includes AIME24, AIME25, HMMT25Feb, HMMT25Nov,
HumanEvalPlus, and MBPPPlus. The default model directories are:

- `~/scratch/opd/models/Qwen3-4B`
- `~/scratch/opd/models/Qwen3-30B-A3B-Instruct-2507`

On 2026-08-28 the complete model directory was approximately 65 GiB, the
Miniforge tree approximately 11 GiB, and the main training dataset directory
approximately 1.6 GiB.

## Current deployment status

The environment, four-domain training data, evaluation data, Qwen3-4B, and
Qwen3-30B-A3B-Instruct-2507 downloads completed. Final seeded CUDA verification
remained blocked because the scheduler-free cards tested on `gpu-a100-07`,
`gpu-a100-11`, and `gpu-a100-05` all reported
`GPU Recovery Action: Reset`.

This is a staged deployment, not a fully GPU-validated environment. Re-run
`PHASES=verify` on a healthy A100 after CSC maintenance before launching
training.

## Known Burgundy-specific failures

- The latest Miniconda route required Anaconda defaults Terms of Service.
  Do not accept legal terms on the user's behalf; use the pinned Miniforge
  installer already encoded in the wrapper.
- GitHub HTTP port 80 timed out. Use HTTPS URLs.
- A previous pip run populated several GiB under the home cache. Future runs
  set `PIP_CACHE_DIR=~/scratch/opd/pip_cache`; do not delete the old cache
  without the user's approval.
- A model/config/import success does not prove kernel dispatch. Always retain
  the seeded witness as the final gate.
