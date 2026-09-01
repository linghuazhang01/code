# Candidate Pool Size Ablation — Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| P001 | M0 | 冻结 Small pools | Small-V1/V2 | source tables | IDs/size/SHA256 | MUST | TODO | 验证 39/50/33 与 48/58/40 |
| P002 | M0 | 冻结 Surface pools | Surface-V1/V2 | YAML literal | IDs/size/SHA256 | MUST | TODO | 验证 185/206/188 与 188/207/188 |
| P003 | M0 | overlap/support | all pools | four baselines | overlap, occurrence, eligible size | MUST | TODO | 检查 Small 是否为 Surface 子集 |
| P004 | M1 | fixed-K replay | all pools, A/C/E*/F | four baselines | P/R/F1, K=1…60 | MUST | TODO | interval3, w1…20 |
| P005 | M1 | fixed-ratio replay | all pools, A/C/E*/F | four baselines | PR/F1 vs K/Pool | MUST | TODO | 标记 10/20/30/50% |
| P006 | M2 | surface-only value | Surface minus Small | four baselines | hits, repeatability, delta recall | MUST | TODO | token type×domain |
| P007 | M2 | robustness | shortlist | four baselines | threshold, LOBO, churn | MUST | TODO | common denominator |
| P008 | M3 | exact-D smoke | best Small vs Surface | short train | KL, entropy, D, selected IDs | MUST | TODO | 不以 E* 替代 D |
| P009 | M4 | 1.7B final | best Small vs Surface | full train, 3 seeds | ACC, loss, speed, latency | MUST | BLOCKED | 等 P008 |
| P010 | M5 | 4B confirmation | winner vs control | full train | ACC, loss, speed | MUST | BLOCKED | 等 P009 |
| P011 | M2/M4 | adaptive backfill | Small+Surface | offline then train | F1, coverage, cost | NICE | TODO | 仅在增量稳定时做 |
