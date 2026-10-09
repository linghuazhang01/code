# Best Math+Code checkpoint: mc8 eval, eager, 8 GPUs

Model: MOPD Structure-only C01 `q1p7b4g-mc-tl-m05c01-cstruct-8g6s2t-b528-s60`, step 60. HF `icemoon28/opd-checkpoints@19d7f7ff`, sha256 `9b19d234…`.

| reference suite | mode | Math4 | Code4 | Macro8 |
|---|---|---|---|---|
| `mc8_k8_s60_mopd_c01_cstruct_20260926` | eager, DP2, gpu_memory 0.85, seed 42 | 25.10 | 38.21 | **31.66** |
| `mc8_k8_s60_mopd_c01_cstruct_cg_20261003` | CUDA graphs, DP4, seed 42 | 24.79 | 38.11 | 31.45 |
| (s1042 CUDA graphs) | | | | 30.88 |

This script reruns it with **eager vLLM on 8 GPUs (DP8)** and the reference's 128-shard seed layout. Every other setting matches the reference: K=8, T=1.0, top_p=1.0, max_new_tokens 16384, batch 24, max_num_seqs 24, gpu_memory 0.85. Before any GPU work starts, the plan is checked against `reference/reference_manifest_20260926.json`.

## Prerequisites on the new machine

1. This repo with the mopd-verl environment (vllm 0.23.0, torch 2.11.0+cu130, transformers 4.57.6). Other versions or GPUs run fine but will not reproduce the reference bit for bit.
2. A G-OPD checkout at `37371a4c31ad7947746200d234161769191f4748`:
   `git clone https://github.com/RUCBM/G-OPD && git -C G-OPD checkout 37371a4c31ad7947746200d234161769191f4748`
3. The LiveCodeBench sources `G-OPD/code_eval/coding/LiveCodeBench/code_generation_lite/test5.jsonl` (557 MB) and `test6.jsonl` (134 MB). Their SHA-256 values are pinned in `eval/lcb_official.py`. Copy them from the CityU server's G-OPD checkout (`.../G-OPD-Training-Data/.eval-source/G-OPD/code_eval/coding/LiveCodeBench/code_generation_lite/`). `test6.jsonl` is also in HF dataset `livecodebench/code_generation_lite`.
4. Docker with the image `verlai/verl:vllm023.dev1` (`docker pull verlai/verl:vllm023.dev1`).
5. The pinned EvalPlus runtime. Build it once with `--setup-evalplus`.

## Run

```bash
cd <repo>/scripts/eval_repro/best_mc8_eager
GOPD_DIR=/path/to/G-OPD PYTHON=/path/to/env/bin/python bash run_best_mc8_eager_8gpu.sh --setup-evalplus --dry-run
GOPD_DIR=/path/to/G-OPD PYTHON=/path/to/env/bin/python bash run_best_mc8_eager_8gpu.sh
```

Other options: `GPU_IDS`, `GPU_MEMORY`, `MODEL_ROOT`, `OUTPUT_ROOT`, `RUN_TAG`, `--download-only`, and `--resume <RUN_TAG>`. The run first generates (8 GPUs), then merges and runs official LiveCodeBench scoring, then official EvalPlus (CPU). The report goes to `<suite>/repro_report.txt`; it can be rerun with `python3 mc8_tools.py compare --suite <suite>`.

## Reading the report

`mc8_tools.py compare` prints the eight dataset scores, Math4, Code4 and Macro8 next to both references. It then gives per-dataset rollout agreement, computed from fingerprints: a SHA-1 of each response and of its first 200 characters.
- `identical_text ≈ n` against eager_20260926: bitwise reproduction of 31.66.
- High `first200_same` with few identical: the weights and seeds match, but the numerics drifted. For scale, CUDA graphs vs eager on the same checkpoint gives 70–75% first200_same and only 0–13% identical. The score is then an independent draw.
- `first200_same ≈ 0`: the seeds or the weights differ, which is a setup error.

On one evaluation, Macro8 noise is about ±1.3 pp, so rank models on pooled seeds.
