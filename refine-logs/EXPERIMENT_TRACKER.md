# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 统计与公式 sanity | offline replay | archived train step | gap/gate/weight parity | MUST | TODO | 不使用 GPU |
| R002 | M1 | A1 smoke | A1 | train 2–5 steps | per-domain gate, weights | MUST | TODO | 不做 validation |
| R003 | M1 | A2 smoke | A2 | train 2–5 steps | gate, control/span weights | MUST | TODO | 不做 validation |
| R004 | M2 | 主对照 | baseline | full train | terminal performance | MUST | TODO | 统一 seed/data order |
| R005 | M2 | 固定权重对照 | fixed control | full train | terminal performance | MUST | TODO | 若已有严格同设置结果可复用 |
| R006 | M2 | stage-aware 对照 | A1 | full train | terminal performance + trajectories | MUST | TODO | span transfer off |
| R007 | M2 | 最终方法 | A2 | full train | terminal performance + trajectories | MUST | TODO | span transfer on |
