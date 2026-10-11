# Eager vs CUDA-graph evaluation of three baselines

These are the jobs the author chose on 2026-10-11. All checkpoints are step 60. The CityU pool scores below were run with CUDA graphs:

| job | model | metric | CityU pool score |
|---|---|---|---|
| `math4:exopd:2042` / `1042` / `42` | Math-only ExOPD | Math4 | 25.73 / 25.42 / 25.31 |
| `mc8:eopd:2042` / `1042` | Math+Code EOPD | Macro8 | 31.12 / 30.50 |
| `mc8:tip:42` | Math+Code TIP-TopK32 | Macro8 | 31.64 (2026-09-26 eager suite: 30.35) |

Math4 covers AIME24, AIME25, HMMT25Feb and HMMT25Nov. mc8 is Math4 plus HumanEval+, MBPP+, LiveCodeBench v5 and LiveCodeBench v6. Every run uses K=8.

Each job runs twice. The `cg` run uses vLLM CUDA graphs (`enforce_eager=False`). The `eager` run uses `enforce_eager=True`. Nothing else differs between the two:
- the job's base seed with the CityU seed map (16 shards per dataset, shard seed = base + sequence × 1000003)
- T=1.0, top_p=1.0, max_new_tokens 16384, batch 24
- gpu_memory 0.75, max_num_seqs 192
- one TP=1 vLLM worker per GPU

Before any GPU work, `eval_modes.py check-plan` checks every planned suite's shard and seed layout against `reference/`.

## Prerequisites

1. This repo with the mopd-verl environment. The CityU references used vllm 0.23.0, torch 2.11.0+cu130 and transformers 4.57.6 on H200 NVL. Other versions and GPUs work, but the rollouts are then new draws, not reproductions.
2. Checkpoints. Each one is verified by the SHA-256 of `model.safetensors`.
   - ExOPD Math-only and TIP Math+Code are downloaded from HF `icemoon28/opd-checkpoints@767a7139` into `MODEL_ROOT`. To use local copies instead, set `EXOPD_MATH_DIR` / `TIP_MC_DIR`.
   - **EOPD Math+Code is not on the Hub.** Copy the `actor/huggingface` directory of the CityU run `checkpoints/MOPD/qwen1p7b-30b-math-code-eopd-native-4gpu-colocate-b516-s42/global_step_60` to the new machine. Set `EOPD_MC_DIR` to it (sha256 `1f7b3213…`).
3. For the mc8 jobs only, the code-scoring setup:
   - G-OPD at `37371a4c31ad7947746200d234161769191f4748` (`GOPD_DIR`), with the LiveCodeBench files `code_eval/coding/LiveCodeBench/code_generation_lite/test5.jsonl` and `test6.jsonl`. These can be copied from the CityU G-OPD checkout; their SHA-256 values are pinned.
   - Docker image `verlai/verl:vllm023.dev1`.
   - The pinned EvalPlus runtime, built once with `--setup-evalplus`.

## Run

```bash
cd <repo>/scripts/eval_repro/eager_vs_cg
export PYTHON=/path/to/mopd-verl/bin/python GOPD_DIR=/path/to/G-OPD EOPD_MC_DIR=/path/to/eopd_mc/actor/huggingface
bash run_eager_vs_cg.sh --setup-evalplus --dry-run   # checks, downloads, plans; no GPU work
bash run_eager_vs_cg.sh                               # full run; rerun the same command to resume
```

Options:
- `--jobs` takes a subset, e.g. `--jobs math4:exopd:2042,mc8:tip:42`.
- `--modes eager` runs only the eager suites.
- `--download-only` stops after the checkpoint checks.
- `--compare-only` rewrites the comparison from whatever has finished.

Environment variables: `GPU_IDS` (all GPUs), `GPU_MEMORY`, `MAX_NUM_SEQS`, `RUN_ID` (20261011), `OUTPUT_ROOT`, `MODEL_ROOT`.

Suites are written to `OUTPUT_ROOT/evmode_<math4|mc8>_k8_s60_<model>_<cg|eager>_s<seed>_<RUN_ID>`. Order:
1. The three Math4 jobs run first, then mc8.
2. Within a job, `cg` runs before `eager`.
3. While the next suite generates, the previous one merges and scores in the background. Scoring is CPU-only: LiveCodeBench and official EvalPlus.

Eager is several times slower than CUDA graphs, most of all on a slow host CPU.

## Output

`OUTPUT_ROOT/evmode_compare_<RUN_ID>/compare.md` (and `.tsv`) has one table per job. The rows are:
- `eager`: this machine, eager.
- `cg`: this machine, CUDA graphs.
- `cg_pool`: the CityU 3-seed suite of the same model and seed.
- `eager_0926`: TIP seed 42 only.

Below each table it lists paired differences (row − column) with a 95% per-problem bootstrap CI for Math4 / Code4 / Macro8. For Math it also gives the share of byte-identical completions and of rollouts with the same correctness.

What to expect: on CityU, same-seed eager and CUDA-graph suites share 0.0–0.1% identical Math completions and agree on 85–87% of rollout outcomes, about as much as unrelated seeds. One suite's Macro8 also moves by about ±1.3 pp between re-runs. So `eager − cg` mostly measures a fresh draw, and gaps inside the CI are noise.

`reference/*.ref.jsonl.gz` was written on CityU with `python3 eval_modes.py export-ref --suite <suite> --out reference/<tag>.ref.jsonl.gz`. It holds the layout and settings, per-problem 8-rollout correctness, and SHA-1 hashes of the Math completions. It holds no text.
