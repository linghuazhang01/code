# ToolRL Eval Domain

This domain tracks ToolRL-style tool-use evaluation data.

## Scope

ToolRL is an external tool-use RL setting. Its reported evaluation families are
not committed in this repo by default. The local structure reserves separate
slots for:

- `data/BFCL/test.parquet`
- `data/API-Bank/test.parquet`
- `data/Bamboogle/test.parquet`

## Prepare Local Data

If you have local JSONL files with `question`/`instruction`/`prompt` and
`answer`/`ground_truth` fields, convert them with:

```bash
python -m eval.domains.toolrl.prepare_data \
  --dataset BFCL \
  --input /path/to/bfcl.jsonl \
  --output data/eval_data/toolrl/BFCL/test.parquet
```

Repeat with `--dataset API-Bank` or `--dataset Bamboogle` for the other slots.

## Scoring

The generic thinking evaluator can load ToolRL parquet files and report token
cost. That parquet path still marks examples with
`requires_external_tool_eval=true` and does not fabricate a reward.

For official benchmark scoring, use the standalone wrappers:

- `api_bank`: local ToolRL-style exact/soft tool-call scoring, no external API.
- `bfcl`: configured handler plus launcher for an external BFCL harness.
- `bamboogle`: optional paid/live search eval with configurable search and judge APIs.

## Code

- `__init__.py`: ToolRL eval dataset metadata.
- `prepare_data.py`: generic JSONL-to-parquet staging helper.
- `api_bank.py`: standalone API-Bank official benchmark wrapper.
- `bfcl_handler.py`: ToolRL/RLLA BFCL handler adapted from the upstream repo.
- `bfcl.py`: external BFCL harness launcher.
- `bamboogle.py`: Bamboogle wrapper with configurable Serper and judge APIs.

## Official Benchmark Wrappers

Run through the unified entrypoint:

```bash
eval/scripts/run_official_eval.sh \
  --domains toolrl \
  --datasets api_bank bfcl bamboogle \
  --model-path /path/to/model
```

`api_bank` is local and does not require an external API.

`bfcl` requires an external BFCL harness command through `--bfcl-command`; the
launcher passes model, output, handler, and optional API endpoint settings via
environment variables.

`bamboogle` requires search and judge services. Configure them with
`--serper-base-url`, `--serper-api-key`, `--judge-base-url`, `--judge-api-key`,
and `--judge-model`.
