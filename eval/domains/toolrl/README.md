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

Internal official benchmark implementations include:

- `api_bank`: local ToolRL-style exact/soft tool-call scoring, no external API.
- `bfcl`: configured handler plus launcher for an external BFCL harness.
- `bamboogle`: optional paid/live search eval with configurable search and judge APIs.

## Code

- `__init__.py`: ToolRL eval dataset metadata.
- `prepare_data.py`: generic JSONL-to-parquet staging helper.
- `api_bank.py`: internal API-Bank official benchmark implementation.
- `bfcl_handler.py`: ToolRL/RLLA BFCL handler adapted from the upstream repo.
- `bfcl.py`: internal external-BFCL-harness adapter.
- `bamboogle.py`: internal Bamboogle implementation with configurable services.

## Launch Policy

The only user-facing evaluation entrypoint is `scripts/run_local_eval.sh`.
The ToolRL official benchmark modules are internal implementations and are not
launched directly. Add ToolRL benchmark routing to `run_local_eval.sh` before
exposing it as a supported public evaluation mode.

Internally, `api_bank` is local and does not require an external API.

The internal `bfcl` implementation requires an external BFCL harness. Its
launcher passes model, output, handler, and optional API endpoint settings via
environment variables.

The internal `bamboogle` implementation requires search and judge services.
