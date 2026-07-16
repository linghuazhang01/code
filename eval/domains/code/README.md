# Code Eval Domain

Code evaluation data lives here, separate from MOPD code training data.

## Data

- `data/HumanEvalPlus/test.parquet`
- `data/MBPPPlus/test.parquet`
- `data/LiveCodeBench/test.parquet`

`HumanEvalPlus` and `MBPPPlus` are the compact paper-eval code validation
sets. `LiveCodeBench` is the G-OPD-aligned incremental `v6` set: 175 problems
from `test6.jsonl`, rather than cumulative `release_v6` with 1,055 problems.
Some single-GPU configs leave it disabled because public+private test execution
is substantially heavier.

## Code

- `__init__.py`: code dataset metadata.
- `prompting.py`: paper-aligned EvalPlus and LiveCodeBench prompt builders.
- `eval/data_prep/paper_eval.py`: EvalPlus and LiveCodeBench JSONL-to-parquet
  conversion helpers.
- `mopd_verl/code_reward.py`: project reward implementation used by verl
  reward dispatch.

## Prompt Alignment

`HumanEvalPlus` and `MBPPPlus` use the original EvalPlus Qwen/chat instruction:
append the markdown Python-code-block requirement and the paper's "think first"
sentence before applying the model chat template at generation time.

`LiveCodeBench` uses the paper code's `Qwen3NonThinking` prompt content by
default. The runner still controls `enable_thinking` for OPD thinking vs.
non-thinking comparisons, but the user-facing problem instruction is no longer
the simplified generic code prompt.

G-OPD samples four completions per problem with temperature 1.0, top-p 1.0,
16,384 max tokens, and requires all public+private tests to pass. Use
`eval/scripts/run_paper_eval_suite.sh` for that official protocol.
