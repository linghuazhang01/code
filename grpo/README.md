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

## Data Requirements

M2RL scripts assume preprocessed parquet paths:

```text
$DATA_DIR/rl_train/if.parquet
$DATA_DIR/rl_train/science.parquet
$DATA_DIR/val/IFBench_test.parquet
$DATA_DIR/val/gpqa.parquet
```

The M2RL repo does not directly ship those parquet files. The public source is the NVIDIA Nemotron RL blend, plus benchmark validation files. Therefore data is not plug-and-play until it is filtered and converted.

For local training, convert any M2RL-style parquet/json/jsonl with:

```bash
cd /Users/linghuazhang/Desktop/Project/OPD/code
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"

python -m grpo.data.m2rl prepare \
  --input /path/to/if.parquet \
  --output data/M2RL/if/train.parquet \
  --rm-type ifbench \
  --split train \
  --domain if

python -m grpo.data.m2rl prepare \
  --input /path/to/science.parquet \
  --output data/M2RL/science/train.parquet \
  --rm-type gpqa \
  --split train \
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

- M2RL's expected per-domain parquet files are not directly present in this workspace.
- The public Nemotron blend is available, but requires filtering into IF and Science splits.
- Training should not start until `grpo.data.m2rl validate` passes for both train and validation parquet files.
