# OPD Evaluation

This directory contains OPD evaluation implementations, data, and run artifacts.
The only user-facing evaluation launch entrypoint is:

```bash
scripts/run_local_eval.sh --model-path /path/to/model [options]
```

Run it from `code/`. Do not invoke `python -m eval.runner` or model-evaluation
scripts under `eval/scripts/` directly; they are internal implementation details.
Data-preparation utilities under `eval/scripts/` may still be run separately.

## Layout

- `runner.py`: thinking/non-thinking model evaluator.
- `common.py`: shared parquet loading, token accounting, and summarization.
- `report.py`: JSON/Markdown report generation for completed or live runs.
- `paper_eval.py`: runtime hook used by patched verl validation.
- `data_prep/`: JSONL-to-parquet conversion code for paper-eval datasets.
- `domains/`: domain-specific metadata, preparation scripts, and eval data.
- `scripts/`: internal and legacy evaluation helpers; not public launch entrypoints.
- `../data/eval_data/results/`: outputs from the public local-eval entrypoint.

## Domains

| Domain | Code | Eval data | Status |
|---|---|---|---|
| Math | `domains/math/` | `../data/eval_data/math/{AIME24,AIME25,HMMT25Feb,HMMT25Nov}/test.parquet` | Ready |
| Code | `domains/code/` | `../data/eval_data/code/{HumanEvalPlus,MBPPPlus,LiveCodeBench}/test.parquet` | HumanEvalPlus/MBPPPlus ready; generate LiveCodeBench with `prepare_paper_eval_data.sh` |
| IF | `mopd_verl/m2rl_reward.py` | `../data/eval_data/if/{IFBench,IFEval}/test.parquet` | Generate the full bundle with `python -m eval.data_prep.m2rl_eval`; the shell helper prepares IFBench only |
| Science | `domains/science/` | `../data/eval_data/science/{GPQA,HLE,MMLU-Pro,SuperGPQA}/test.parquet` | Generate GPQA/HLE with `python -m eval.data_prep.m2rl_eval`; MMLU-Pro/SuperGPQA include official evaluators |
| Training ceiling | Existing domain scorers | `../data/eval_training_data/{math,code,if,science}/test.parquet` | 10,000 overlapping training samples per domain; use only for training-performance diagnostics |
| ToolRL | `domains/toolrl/` | `../data/eval_data/toolrl/{BFCL,API-Bank,Bamboogle}/test.parquet` | Data/internal evaluators exist; ToolRL datasets are not yet exposed by `run_local_eval.sh` |

SearchQA support remains in `domains/search/` because the thinking evaluator can
still include `data/SearchQA/test.parquet`, but SearchQA is not one of the four
domains requested for this eval layout.

## Preparing Data

Math/code paper-eval data from a G-OPD checkout:

```bash
eval/scripts/prepare_paper_eval_data.sh
```

This pins LiveCodeBench `v6` (`test6.jsonl`, 175 incremental problems), not the
1,055-problem cumulative `release_v6`. For G-OPD's official public+private test
protocol, use `eval/scripts/run_paper_eval_suite.sh`. The generated LiveCodeBench
parquet is intentionally ignored by Git because it contains the full private
test payload; `manifest.json` records its pinned revision and source checksum.

Create deterministic four-domain training-performance ceiling samples without
modifying the original parquet files:

```bash
python scripts/split_domain_eval_training_data.py \
  --eval-size 10000 \
  --seed 42 \
  --overwrite
```

The four outputs are written to
`data/eval_training_data/<domain>/test.parquet`; their hashes and selected
prompt groups are recorded in `data/eval_training_data/manifest.json`. These
samples intentionally overlap the original training files and estimate
training-data performance rather than leakage-free generalization. Existing
files under `data/training_data_split/` are not consumed or refreshed by this
workflow.

Official MMLU-Pro and SuperGPQA data:

```bash
python -m eval.domains.science.download_official_data --force
```

Rebuild their reproducible paper subsets with:

```bash
python -m eval.domains.science.prepare_subsets
```

ToolRL local JSONL staging:

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output data/eval_data/toolrl/BFCL/test.parquet
```

Prepare the complete M2RL paper-evaluation bundle (IFBench, IFEval, GPQA, and
HLE) at the canonical dataset paths:

```bash
python -m eval.data_prep.m2rl_eval
```

To prepare only the training-validation pair (IFBench and GPQA) from explicit
local sources:

```bash
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
  scripts/prepare_m2rl_eval_data.sh
```

or prepare the same IFBench/GPQA pair from the Nemotron RL JSONL blend:

```bash
NEMOTRON_RL_SOURCE=/path/to/instruction_following.jsonl \
M2RL_EVAL_MAX_SAMPLES=512 \
  scripts/prepare_m2rl_eval_data.sh
```

## Running Evaluation

Run all local evaluations through the single public entrypoint:

```bash
scripts/run_local_eval.sh \
  --model-path ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500 \
  --datasets aime24,humaneval_plus,ifeval,gpqa_diamond \
  --modes non_thinking \
  --max-samples 8 \
  --save-completions
```

Important options:

- `--datasets`: comma-separated dataset keys.
- `--modes`: `non_thinking`, `thinking`, or both as a comma-separated list.
- `--max-samples`: maximum examples per dataset.
- `--max-new-tokens`: generation limit for every selected mode.
- `--num-samples`, `--temperature`, `--top-p`, `--seed`: sampling controls.
- `--backend transformers|vllm`: inference backend.
- `--tensor-parallel-size`, `--batch-size`, `--gpu-memory`: vLLM controls.
- `--score-code`: execute generated code for Code scoring; use only in isolation.
- `--save-completions`: retain full completions.
- `--dry-run`: validate inputs and print the resolved command without running it.

Supported dataset keys are `aime24`, `aime25`, `hmmt25feb`, `hmmt25nov`,
`humaneval_plus`, `mbpp_plus`, `livecodebench`, `ifeval`, `ifbench`, and
`gpqa_diamond`. The training-performance keys are `training_ceiling` for all
four domains, plus `training_math`, `training_code`, `training_if`, and
`training_science` for individual domains. Raw full-training routes are
`training_full`, `training_full_math`, `training_full_code`, `training_full_if`,
and `training_full_science`.

## Two-model evaluation launchers

These launchers evaluate `Qwen3-1.7B` and
`Nemotron-Research-GooseReason-4B-Instruct` sequentially on one GPU with
`TP=1`:

The held-out diagnostic and the two training-data launchers use the GooseReason
training profile's validation-inference settings for both models:
`non_thinking`, `max_new_tokens=16384`, greedy `n=1`,
`temperature=0`, `top_p=1`, `seed=42`, `max_model_len=18432`,
`max_num_batched_tokens=32768`, `max_num_seqs=24`, eager execution, and
chunked prefill disabled. The only deliberate topology change is `TP=1` for
the 141 GiB single GPU. Request batching defaults to 24 and vLLM memory
utilization to 0.9.

```bash
# Held-out OOD benchmark suite
scripts/run_two_model_ood_eval.sh

# Standard OOD benchmark for one model
MODEL_PATH=/path/to/model eval/scripts/run_standard_ood_eval.sh

# Deterministic 10,000-row-per-domain training ceiling
scripts/run_two_model_training_ceiling_eval.sh

# Every row in the four raw training parquets
CONFIRM_FULL_TRAINING=1 scripts/run_two_model_full_training_eval.sh
```

The standard OOD benchmark is the pinned `MMLU-Pro-500` subset. It uses 500
questions and four sampled rollouts per question with `non_thinking`,
`temperature=1`, `top_p=1`, `seed=42`, `max_tokens=16384`,
`max_model_len=18432`, `TP=1`, and `gpu_memory_utilization=0.85`. Its artifact
identity is fixed by data SHA-256
`9db4fb82f4fc59ab4514b2f3a2fe54928b3fc9d11a483bf678958261b8f6a4a6`
and ordered selected-ID SHA-256
`ea1c19950afe4ac82a3b32c8afb39b50fa032a64e096fed61364e1d0c1c81760`.
The launcher validates both hashes before inference, writes all 2,000
prompt/response records, records the protocol in `standard_ood_manifest.json`,
and creates `SUCCESS` only after scoring completes. This is a reproducible
OpenPRM-style sample; OpenPRM did not publish the exact sampled IDs or seed.

`scripts/run_two_model_ood_eval.sh` runs this standard MMLU-Pro-500 component
after the existing held-out diagnostic for each model. Set
`INCLUDE_STANDARD_MMLUPRO_500=0` only when intentionally reproducing the
legacy nine-dataset diagnostic without the standard OOD benchmark.

The three held-out/training launchers enable `--save-completions`. Both the raw
`thinking_eval_samples.jsonl` and analysis-facing
`prompt_response_records.jsonl` retain the prompt, response, dataset, sample
ID, model path, run ID, ground truth, rollout index, generation seed, and
`sample_metadata` with the source file, original row position, parquet
`extra_info`, reward config, and any additional source columns.

The OOD launcher's original nine held-out datasets use one greedy rollout per
prompt as a uniform cross-domain diagnostic. This component is not the
dataset-specific G-OPD paper protocol (for example, Math paper evaluation uses
sampled K=32). The standard MMLU-Pro-500 component is reported separately with
its K=4 protocol and must not be averaged into the greedy diagnostic.
LiveCodeBench is excluded
because its official protocol uses four sampled rollouts at temperature 1.0
and a 16,384-token limit; opt in with `OOD_DATASETS=...` only when matching
that protocol. “OOD” here means the held-out benchmark split, not a formal
prompt-hash leakage audit against every training source.

The full-training launcher processes stable 10,000-row shards by default so it
does not hold every training prompt and completion in one process. Override
`SHARD_SIZE` when needed. To continue an interrupted run, reuse `RUN_TAG` and
set `RESUME=1 CONFIRM_FULL_TRAINING=1`; completed shards have a `SUCCESS`
marker and are skipped.
Real runs require at least 50 GiB free in the output filesystem by default;
adjust `MIN_FREE_GB` after estimating completion length and rollout count.
The root `suite_manifest.json` records every source range, model, shard seed,
expected record count, output directory, source SHA-256, and current `SUCCESS`
status. Resume is rejected before any shard is skipped when this immutable
suite signature differs from the original run.
For all launchers, `DRY_RUN=1` validates and prints the plan without
creating output directories, logs, shard markers, or manifests. A non-resume
run refuses to reuse a non-empty output directory instead of deleting prior
rollouts.

### W&B upload

The three launchers upload scalar metrics and a summary table to the
`mopd-eval` W&B project by default. Each model run (and each full-training
shard) also uploads an `evaluation-results` artifact containing the report,
generation config, compact records, full prompt/response rollout JSONL, and
summary files. Runs from the same launcher invocation share one W&B group.

Evaluation reuses the training launcher's `.env.local` parser, including its
support for `export KEY=value`. The existing legacy key name is accepted and
mapped in memory to the standard W&B variable:

```bash
export Wandb_Key=<your-wandb-api-key>
```

`WANDB_API_KEY` is also supported and takes precedence when already present in
the shell or `.env.local`. The key is never copied into reports, status files,
or artifacts. Override the env-file path with `EVAL_WANDB_ENV_FILE` when
needed; the default is `<code-root>/.env.local`.

For full-training, each shard is marked `SUCCESS` and queued locally; W&B
artifacts are uploaded sequentially only after all local generation and the
final suite manifest are complete. A slow network therefore cannot delay the
next generation shard. All three launchers give each upload a 1,800-second
timeout by default; override it with `EVAL_WANDB_TIMEOUT_SECONDS` or set it to
`0` for no timeout.

Optional controls are `EVAL_WANDB_ENTITY`,
`EVAL_WANDB_MODE=online|offline|disabled`, `EVAL_WANDB_UPLOAD_RAW=0`, and
`EVAL_WANDB_ENABLED=0`. The project can be overridden with
`EVAL_WANDB_PROJECT`, whose default is `mopd-eval`. Offline W&B state is stored
below each result directory in `wandb/`, and its exact local run directory is
recorded in `wandb_upload_status.json`.

Raw artifacts contain prompts, responses, ground truth, and source metadata.
Confirm that the target W&B project's visibility and data policy are suitable
before launching; use `EVAL_WANDB_UPLOAD_RAW=0` when only metrics should leave
the machine.

Local evaluation files remain the source of truth. If authentication or the
network fails, the evaluation still completes and writes
`wandb_upload_status.json` with `upload_pending`. Retry one result directory,
or every completed shard below an output root, with:

```bash
python -m eval.wandb_upload \
  --output-dir data/eval_data/results/<suite>/<RUN_TAG>/<model> \
  --project mopd-eval \
  --group <suite>_<RUN_TAG> \
  --env-file .env.local

python -m eval.wandb_upload \
  --output-root data/eval_data/results/two_model_full_training/<RUN_TAG> \
  --project mopd-eval \
  --group two_model_full_training_<RUN_TAG> \
  --env-file .env.local
```

For example, run a bounded four-domain training ceiling evaluation with:

```bash
scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets training_ceiling \
  --max-samples 100
```

`training_ceiling` is not an official benchmark. Results must be labeled as
training-data performance, and Code scoring remains disabled unless
`--score-code` is explicitly enabled in an isolated environment with the full
vendored verl reward dependencies. The launcher never falls back to a Math
scorer for training Code data.

IF scoring requires either `verifiable_instructions`, which is installed by
`scripts/setup_training_env.sh`, or a local official evaluator prepared with:

```bash
scripts/prepare_ifbench_runtime.sh
```

For example, compare thinking and non-thinking with vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/run_local_eval.sh \
  --model-path /path/to/model \
  --datasets aime24,gpqa_diamond \
  --modes non_thinking,thinking \
  --backend vllm \
  --tensor-parallel-size 2 \
  --batch-size 8
```

Outputs are written to `data/eval_data/results/<RUN_ID>/`:

- `thinking_eval_samples.jsonl`
- `thinking_eval_summary.json`
- `thinking_eval_summary.csv`
- `records.jsonl`
- `README.md`

## Scoring

- Math uses boxed-answer style scoring through the project reward router when
  available. MMLU-Pro and SuperGPQA use the science official evaluator.
- Code uses `mopd_verl/code_reward.py` through the vendored verl reward router.
- IF/science validation uses the same verl reward path as training:
  `mopd_verl/mixed_reward.py` routes `m2rl_ifbench` to IFBench/verifiable-instructions
  strict scoring and `m2rl_gpqa` to GPQA option-letter scoring.
- ToolRL parquet data is loadable for cost/token reports.
- ToolRL official benchmark wrappers support API-Bank local scoring, the BFCL
  handler launcher, and Bamboogle search + judge scoring.

## Training Config References

MOPD configs now point validation paths to this directory:

- `configs/mopd_qwen30b_pg_split_teacher_gpu_audit_domain_vocabvec_*.yaml`

Training data remains under `data/G-OPD-Training-Data/` and is intentionally not
mixed with eval data.

## Internal Evaluators

`eval/runner.py`, `eval/official_runner.py`, and model-evaluation scripts under
`eval/scripts/` remain implementation modules for development and compatibility.
They are not supported as independent user launch entrypoints. Data-preparation
utilities are not launch entrypoints and remain directly usable. Extend
`scripts/run_local_eval.sh` first when exposing another benchmark or evaluation
behavior.
