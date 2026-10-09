# sapd_c01 Math4 seed-42 eager reproduction (8 GPUs)

Goal: re-run the evaluation that produced the archived **26.04** (job 196, 2026-09-02) on another 8-GPU machine.

| setting | job 196 (26.04) | this script |
|---|---|---|
| checkpoint | step 60 of training job 193, merged locally (since deleted) | same step-60 save, HF `icemoon28/opd-checkpoints@1fa94a3b`, sha256 `744f2bf3…` |
| vLLM mode | eager | **eager** (no `--cuda-graphs`; asserted after planning) |
| workers | DP1 | **DP8** (`GPU_IDS=0,…,7`) |
| shards / seeds | 16 per dataset, 64 total, seed = 42 + seq·1000003 | identical (checked against `reference/reference_manifest_20260902.json` before launch) |
| K / max_new_tokens / T / top_p | 8 / 16384 / 1.0 / 1.0 | same |
| batch 24, max_num_seqs 24, max_num_batched_tokens 32768, max_model_len 18432, gpu_memory 0.85 | | same |
| env | vllm 0.23.0, torch 2.11.0+cu130, transformers 4.57.6, H200 NVL | printed and flagged if different |

## Run

```bash
cd <repo>/scripts/eval_repro/c01_math4_s42_eager
bash run_c01_s42_eager_8gpu.sh --dry-run    # data/ckpt/env checks + plan only
bash run_c01_s42_eager_8gpu.sh              # full run, then score + compare
```

Overrides: `PYTHON=/path/to/env/bin/python`, `GPU_IDS`, `GPU_MEMORY`, `MODEL_ROOT`, `OUTPUT_ROOT`, `RUN_TAG`.
After a crash, use `--resume <RUN_TAG>`. The report goes to `<suite>/repro_report.txt`.

## Reading the result

`compare_with_reference.py` scores the run (per dataset avg@8, Math4 avg@8 / pass@8). It then compares each of the 960 rollouts with both references: the eager 26.04 run and the CUDA-graph 24.06 run.
- `identical_text` ≈ 960 vs eager_20260902: bitwise reproduction, so the run gives 26.04 again.
- `first200_same` ≈ 700+ with few identical: same weights and seeds, but the numerics drifted (different GPU or library versions). The score is then a fresh draw.
- `first200_same` ≈ 0: the seeds or the weights differ, so something is misconfigured.

The merged job-196 weights no longer exist, so whether the HF file matches them bit for bit cannot be proven. Even an exact 26.04 only shows that one draw can be reproduced. The 3-seed CUDA-graph mean is 24.03.
