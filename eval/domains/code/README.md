# Code Eval Domain

Code evaluation data lives here, separate from MOPD code training data.

## Data

- `data/HumanEvalPlus/test.parquet`
- `data/MBPPPlus/test.parquet`
- `data/LiveCodeBench-v5/test.parquet`
- `data/LiveCodeBench/test.parquet` (v6)

`HumanEvalPlus` and `MBPPPlus` are the compact paper-eval code validation
sets. LiveCodeBench uses two independent incremental sets: official `v5/`
(167 problems) and `test6.jsonl` / `v6` (175 problems), rather than overlapping
cumulative releases. The pinned dataset does not contain an official
`test5.jsonl`; data preparation derives one from the two pinned v5 parquet shards
only because the G-OPD loader requires that filename.

## Code

- `__init__.py`: code dataset metadata.
- `prompting.py`: paper-aligned EvalPlus and LiveCodeBench prompt builders.
- `eval/data_prep/paper_eval.py`: EvalPlus and LiveCodeBench JSONL-to-parquet
  conversion helpers.
- `mopd_verl/code_reward.py`: project reward implementation used by verl
  reward dispatch.

## Prompt Alignment

`HumanEvalPlus` and `MBPPPlus` reproduce the original G-OPD EvalPlus path,
including the three newline characters between the stripped task prompt and the
markdown Python-code-block / "think first" suffix.

LiveCodeBench uses the original G-OPD `Qwen3NonThinking` user content, including
`You will NOT return anything except for the program.`, and runs with
`enable_thinking=false`. This is an exact user-content contract. G-OPD's LCB
formatter applies the fixed `Qwen/Qwen3-4B` tokenizer/chat template (not the
target checkpoint's chat template), so token-level provenance must pin that
formatter tokenizer revision and chat-template hash rather than infer it from the
parquet.

The validation reward follows the training extractor exactly: it executes the
last `python` block. A response without a `python` marker follows the same
fallback split behavior as the training reward.

The active MOPD standard samples eight completions per problem with temperature
1.0, top-p 1.0, 16,384 max tokens, and requires all public+private tests to
pass. Both LCB releases must use the G-OPD official runner: the generic Docker
Code scorer intentionally rejects LCB until its input/output scorer is isolated.
Historical G-OPD evaluation used four samples and must remain labeled as legacy.
