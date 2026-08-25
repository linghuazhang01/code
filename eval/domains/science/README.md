# Science Eval Domain

This domain owns the science evaluation datasets and their official evaluators.

## Canonical Data Layout

```text
data/eval_data/science/
  GPQA/test.parquet
  HLE/test.parquet
  MMLU-Pro/test.parquet
  SuperGPQA/test.parquet
```

Every runnable dataset uses `science/<Dataset>/test.parquet`. GPQA is the
verl-ready Diamond split. HLE contains text-only inference rows and requires an
official judge for comparable scoring.

## Official Science Data

MMLU-Pro and SuperGPQA can be downloaded with:

```bash
python -m eval.domains.science.download_official_data --force
```

## Code

- `download_official_data.py`: downloads MMLU-Pro and SuperGPQA.
- `official_eval.py`: runs their official multiple-choice evaluation.
- `prepare_subsets.py`: builds the reproducible OpenPRM-style MMLU-Pro subset
  and the exact RSA SuperGPQA subset.

## Launch Policy

Use `python -m eval.official_runner --domains science ...` for MMLU-Pro and
SuperGPQA. Use `scripts/run_local_eval.sh` for the verl-ready GPQA path.

The project-standard OOD benchmark is the pinned MMLU-Pro-500 subset:

```bash
MODEL_PATH=/path/to/model eval/scripts/run_standard_ood_eval.sh
```

Its immutable protocol is 500 questions, K=4, `temperature=1`, `top_p=1`,
`seed=42`, `max_tokens=16384`, non-thinking mode, TP=1, and vLLM memory
utilization 0.85. The launcher verifies the subset and ordered selected-ID
hashes from `subsets/openprm_style_500_seed42/manifest.json` before running.
The subset is an OpenPRM-style reproducible sample rather than OpenPRM's exact
unpublished sample.
