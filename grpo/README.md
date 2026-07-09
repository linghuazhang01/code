# M2RL-style GRPO for Qwen 4B Non-Thinking

This directory is a clean replacement for the earlier ToolRL/General-Reasoner GRPO adapter. The old implementation was moved to `../temp/grpo_legacy_backup_*/grpo` from the project root.

The goal is to reuse the relevant M2RL recipe for two GRPO teacher domains:

- Instruction following: M2RL `run-qwen3-4B-if.sh`, `rm-type=ifbench`.
- Science QA: M2RL `run-qwen3-4B-science.sh`, `rm-type=gpqa`.

The local runtime still uses the vendored `verl` launcher through `scripts/run_mopd.sh`; we do not vendor the full M2RL/slime stack here.

## M2RL Parameters Mirrored

M2RL's single-domain IF and Science RL scripts use:

```text
rollout-max-prompt-len     2048
rollout-max-response-len   32768
n-samples-per-prompt       16
global-batch-size          2048
context-parallel-size      2
max-tokens-per-gpu         17600
```

The local configs map this to:

```text
data.max_prompt_length              2048
data.max_response_length            32768
actor_rollout_ref.rollout.n         16
actor_rollout_ref.rollout.max_model_len 34816
```

Because the local `verl` path does not expose M2RL/slime context parallelism, the configs conservatively set `actor.ppo_max_token_len_per_gpu=34816`. If you run the original M2RL/slime stack with CP=2, the M2RL value around `17600` per GPU is the closer match.

## Reward Functions

Main adapter:

```text
grpo/rewards/m2rl.py
```

It supports:

- `ifbench`: official IFBench strict instruction-following verifier.
- `gpqa`: M2RL-style multiple-choice science reward using option-letter extraction.

Important non-thinking adaptation: M2RL's original reward hub returns `0.0` when the response has no `</think>` tag. This local adapter strips `</think>` if present, but does not require it, because this experiment is explicitly for non-thinking Qwen 4B behavior.

Mixed MOPD math/code/IF/science training should use:

```text
grpo/rewards/mixed.py
```

That router sends `ifbench` and `gpqa` rows to the M2RL adapter, while preserving the vendored `verl` default rewards for existing math/code data sources. Do not point a mixed-domain MOPD config directly at `grpo/rewards/m2rl.py`, because it intentionally rejects non-M2RL math/code rows.

The current Qwen30B MOPD mixed config supports four active training domains:

| Domain | Training parquet | Reward path |
| --- | --- | --- |
| `math` | `data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet` | `mixed.py` -> vendored `verl.utils.reward_score.default_compute_score`; `DeepMath-103K` routes to `math_verify.compute_score`. |
| `code` | `data/G-OPD-Training-Data/Eurus/code_train.parquet` | `mixed.py` -> vendored default reward; rows such as `taco` use sandbox-fusion if configured, otherwise vendored `prime_code.compute_score`. |
| `if` | `data/G-OPD-Training-Data/IF/train.parquet` | `mixed.py` -> `m2rl.py` -> `compute_ifbench_reward`, with `verifiable_instructions` preferred and AllenAI IFBench strict checking as fallback. |
| `science` | `data/G-OPD-Training-Data/Science/train.parquet` | `mixed.py` -> `m2rl.py` -> `compute_gpqa_reward`; rows use `rm_type=gpqa`, `data_source=m2rl_gpqa` or another science/GPQA marker, and metadata with `correct_letter` or an equivalent label. |

For Nemotron/Open-Instruct instruction-following rows, install the lightweight verifier:

```bash
python -m pip install --no-cache-dir git+https://github.com/abukharin-nv/verifiable-instructions.git
```

The adapter uses `verifiable_instructions` first for Open-Instruct/IFEval-style ids such as `length_constraints:number_words`, and falls back to AllenAI IFBench only for older IFBench-style ids.

## Data Requirements

M2RL scripts assume preprocessed parquet paths:

```text
$DATA_DIR/rl_train/if.parquet
$DATA_DIR/rl_train/science.parquet
$DATA_DIR/val/IFBench_test.parquet
$DATA_DIR/val/gpqa.parquet
```

The M2RL repo does not directly ship those parquet files. The public source is the NVIDIA Nemotron RL blend, plus benchmark validation files. Therefore data is not plug-and-play until it is filtered and converted.

This repo's MOPD configs expect the prepared training data under `data/`:

```text
data/G-OPD-Training-Data/IF/train.parquet
data/G-OPD-Training-Data/Science/train.parquet
```

Validation uses the same paths as the sibling GRPO workspace:

```text
eval/domains/ifbench/data/IFBench_test.parquet
eval/domains/science/data/gpqa.parquet
```

During verl validation, generations are scored by the configured reward function. The mixed Qwen30B configs use `grpo/rewards/mixed.py`, which routes `m2rl_ifbench` rows to IFBench/verifiable-instructions strict scoring and `m2rl_gpqa` rows to GPQA option-letter scoring. Validation metrics are grouped by `data_source`, so these appear as `val-core/m2rl_ifbench/...` and `val-core/m2rl_gpqa/...`.

The current local copies were prepared from the sibling GRPO workspace's Nemotron/Open-Instruct source data. Keep launch configs pointed at the repo-local `data/G-OPD-Training-Data/...` paths; the sibling workspace is only a regeneration source.

Regenerate the repo-local data with:

```bash
python scripts/prepare_nemotron_rl_data.py \
  --input /Users/linghuazhang/Desktop/Project/GRPO/data/raw/nemotron-rl-instruction_following/instruction_following.jsonl \
  --if-output data/G-OPD-Training-Data/IF/train.parquet \
  --science-output data/G-OPD-Training-Data/Science/train.parquet \
  --manifest data/nemotron_rl/manifest.json
```

For a tiny local smoke dataset:

```bash
python scripts/prepare_nemotron_rl_data.py \
  --input /Users/linghuazhang/Desktop/Project/GRPO/data/raw/nemotron-rl-instruction_following/instruction_following.jsonl \
  --if-output data/G-OPD-Training-Data/IF/train.parquet \
  --if-max-samples 32
```

Prepare IF/science validation files from M2RL/GRPO-style raw sources:

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

Or prepare a capped validation subset from the Nemotron RL JSONL blend:

```bash
NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl \
M2RL_EVAL_MAX_SAMPLES=512 \
  scripts/prepare_m2rl_eval_data.sh
```

The script validates both outputs with `grpo.data.m2rl validate`. Set `REQUIRE_M2RL_EVAL_DATA=1` when a missing eval parquet should fail setup instead of printing a readiness warning.

For local training, convert any M2RL-style parquet/json/jsonl with:

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"

python -m grpo.data.m2rl prepare \
  --input /path/to/if.parquet \
  --output data/G-OPD-Training-Data/IF/train.parquet \
  --rm-type ifbench \
  --split train \
  --domain if

python -m grpo.data.m2rl prepare \
  --input /path/to/science.parquet \
  --output data/G-OPD-Training-Data/Science/train.parquet \
  --rm-type gpqa \
  --split train \
  --domain science
```

The same converter can prepare validation files directly:

```bash
python -m grpo.data.m2rl prepare \
  --input /path/to/if_val.parquet \
  --output eval/domains/ifbench/data/IFBench_test.parquet \
  --rm-type ifbench \
  --split validation \
  --domain if

python -m grpo.data.m2rl prepare \
  --input /path/to/science_val.parquet \
  --output eval/domains/science/data/gpqa.parquet \
  --rm-type gpqa \
  --split validation \
  --domain science
```

Validate before launching:

```bash
python -m grpo.data.m2rl validate --input /path/to/if.parquet --rm-type ifbench
python -m grpo.data.m2rl validate --input /path/to/science.parquet --rm-type gpqa
```

## Schema Checks

`ifbench` rows must include:

```text
prompt or messages
metadata.instruction_id_list
metadata.kwargs
metadata.prompt_text
```

The IFBench reward also requires a local clone of `allenai/IFBench`:

```bash
git clone https://github.com/allenai/IFBench.git ../IFBench
export IFBENCH_REPO=/Users/linghuazhang/Desktop/Project/OPD/IFBench
```

Alternatively:

```bash
export M2RL_ALLOW_IFBENCH_AUTO_CLONE=1
```

`gpqa` rows must include:

```text
prompt or messages
label, answer, correct_letter, or correct_answer
metadata.choices when the label is not already a letter
```

## Launch

Dry-run first:

```bash
scripts/run_mopd.sh grpo/configs/m2rl_if_smoke.yaml --dry-run
DRY_RUN=1 scripts/run_m2rl_if_grpo.sh
DRY_RUN=1 scripts/run_m2rl_science_grpo.sh
DRY_RUN=1 scripts/run_m2rl_if_science_grpo.sh
```

Actual runs:

```bash
scripts/run_m2rl_if_grpo.sh -- \
  actor_rollout_ref.model.path=/path/to/qwen-4b-non-thinking \
  actor_rollout_ref.model.base_model_path=/path/to/qwen-4b-non-thinking

scripts/run_m2rl_science_grpo.sh -- \
  actor_rollout_ref.model.path=/path/to/qwen-4b-non-thinking \
  actor_rollout_ref.model.base_model_path=/path/to/qwen-4b-non-thinking
```

The default config uses `Qwen/Qwen3-4B` with `enable_thinking=false`, because no local `Qwen4B-Non-Thinking` base checkpoint is present in this workspace. Override model paths for the exact checkpoint.

## Current Fitness Assessment

Reward side:

- IFBench reward is strong if official IFBench metadata is present.
- GPQA reward is self-contained and suitable for multiple-choice science QA.
- Non-thinking outputs are supported.

Data side:

- Training parquets are tracked under `data/G-OPD-Training-Data/`.
- IF/science validation parquets are generated under `eval/domains/...` and are intentionally not committed.
- Training with IF/science validation enabled should not start until `grpo.data.m2rl validate` passes for both train and validation parquet files.
