---
type: results-report
date: 2026-06-13
experiment_line: mopd-gradient-audit
round: 0
purpose: full-sample-gradient-diagnosis
status: active
source_artifacts:
  - "TensorBoard run: on-policy-distillation/Qwen3-4B-Multi-Teacher-ExOPD-formal-dual-a800"
  - "Remote log: logs/mopd_tp2_ckpt_b256_resp8k_util08_20260612_1102.log"
linked_experiments:
  - mopd_tp2_ckpt_b256_resp8k_util08_20260612_1102
linked_results: []
---

# MOPD Full/Sample Gradient 诊断报告

## Executive Summary

本轮双 A800 诊断实验只完整记录到 step 1-5，step 6 完成 generation 后没有继续产出训练 scalar。日志中没有看到 OOM、Traceback、RuntimeError、ValueError、NaN 或 Inf，因此目前更像是训练进程在 generation 之后停滞或退出，而不是显式异常崩溃。

核心结论是：math 的 sample gradient 相对 code 持续变强，但这不等价于 math 主导了 global/full_grad 更新方向。full_grad 视角下，math 和 code 的梯度方向从 step 2 开始强烈冲突；total update 多数 step 仍更接近 code，只有 step 3 math 的 signed projection share 短暂占主导。特别是在 step 5，math 的单样本梯度均值是 code 的 3.65x，但 math full_grad norm 反而只有 code 的 0.90x；这说明 math 梯度在 domain 内聚合时存在更强的方向分散或抵消。

## Experiment Identity And Setup

- 配置：`configs/mopd_formal_dual_a800.yaml`
- GPU：2 x NVIDIA A800 80GB
- 关键参数：`train_batch_size=256`，`ppo_mini_batch_size=256`，`max_response_length=8192`
- Rollout：vLLM TP=2，`gpu_memory_utilization=0.8`
- Actor：`gradient_checkpointing=true`，`fsdp_size=1`
- Audit：full-gradient 每 step 开启；sample gradient norm 开启；sample-to-domain cosine 关闭
- 可用 TensorBoard step：1-5

## Main Findings

### 1. full_grad 显示 math/code 梯度方向存在强冲突

| Step | math full_grad norm | code full_grad norm | math-vs-code cosine |
| ---: | ---: | ---: | ---: |
| 1 | 0.240204 | 1.890120 | -0.103692 |
| 2 | 0.128565 | 0.165392 | -0.725777 |
| 3 | 0.193730 | 0.187963 | -0.942088 |
| 4 | 0.153246 | 0.157669 | -0.942960 |
| 5 | 0.156139 | 0.173288 | -0.919388 |

从 step 2 开始，math/code full gradient cosine 进入明显负值区间，step 3-5 约为 -0.94、-0.94、-0.92。这说明两个 domain 的更新方向高度相反，不能只看 domain loss 或 sample norm 的大小来判断谁“贡献更大”。

### 2. global/full_grad 的 total update 多数时候更偏 code

| Step | math vs total cosine | code vs total cosine | math signed share | code signed share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.023520 | 0.991900 | 0.0030 | 0.9970 |
| 2 | 0.074770 | 0.631810 | 0.0842 | 0.9158 |
| 3 | 0.255430 | 0.083690 | 0.7589 | 0.2412 |
| 4 | 0.086780 | 0.249900 | 0.2524 | 0.7477 |
| 5 | -0.046570 | 0.435810 | -0.1065 | 1.1066 |

除 step 3 以外，total update 的方向和 signed projection 都更偏 code。step 5 中 math signed share 为负，表示 math 梯度相对 total update 已经是反向投影；code signed share 超过 1，是因为 code 不只是贡献 total update，还抵消了 math 的反向分量。

### 3. sample_grad 解释了“math 比例上涨、code 比例下降”的表观现象

| Step | math sample norm mean | code sample norm mean | math sample norm share | math/code ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.113792 | 0.308501 | 26.9% | 0.37x |
| 2 | 0.100166 | 0.088152 | 53.2% | 1.14x |
| 3 | 0.099863 | 0.038417 | 72.2% | 2.60x |
| 4 | 0.097218 | 0.035304 | 73.4% | 2.75x |
| 5 | 0.073221 | 0.020039 | 78.5% | 3.65x |

math 的 sample gradient norm 在 step 1-5 下降较慢，而 code 从 0.308501 快速降到 0.020039。因此按 sample norm 计算的 math 占比会持续上涨，code 占比自然下降。step 5 的分布也支持这一点：math p50 为 0.06483，code p50 为 0.01420；math p95 为 0.1320，code p95 为 0.0623。

这说明 math 样本的单样本训练压力更稳定、更大；code 样本在前几步被快速压低，后续单样本 gradient 变小。

### 4. 为什么 math sample signal 更强但 full_grad 更小

step 5 中 math 的 token OPD loss 约为 code 的 12.13x，sample OPD loss 约为 5.23x，teacher-student absolute gap 约为 9.99x。也就是说，math 当前仍有更大的 teacher-student mismatch，因此 sample gradient 更大是合理的。

但 full_grad norm 不是 sample norm 的简单平均。sample 指标统计的是 `E_i[||g_i||]`，而 full_grad norm 统计的是 `||sum_i w_i g_i||`。如果一个 domain 内部的 sample gradient 方向更分散，即使每个 sample 的 norm 更大，聚合后的 full_grad vector 仍可能更小。

| Step 5 指标 | math | code | math/code |
| --- | ---: | ---: | ---: |
| sample_grad/norm_mean | 0.073221 | 0.020039 | 3.65x |
| full_grad/grad_norm | 0.156139 | 0.173288 | 0.90x |
| full_grad_norm / sample_grad_norm_mean | 2.13 | 8.65 | 0.25x |
| token fraction | 30.1% | 69.9% | 0.43x |
| response mean tokens | 2653.84 | 6156.77 | 0.43x |
| response_clip_ratio | 1.6% | 50.0% | 0.03x |
| token OPD loss | 0.113738 | 0.009378 | 12.13x |
| sample OPD loss | 301.842 | 57.739 | 5.23x |
| teacher-student absolute gap | 0.124835 | 0.012497 | 9.99x |

这里 `full_grad_norm / sample_grad_norm_mean` 不是严格的数学 coherence，因为 full_grad 还受 sample count、token count、loss normalization 和 domain weighting 影响；但它可以作为一个粗略 aggregation proxy。step 5 中 code 的 proxy 为 8.65，math 只有 2.13，code 约为 math 的 4.06x。这支持一个更细的解释：math 的单样本误差信号更强，但 math sample gradients 可能更不一致，聚合时更容易互相抵消；code 的单样本梯度更小，但 token mass 更大，并且聚合方向可能更一致，所以最终 full_grad norm 仍略大。response mean tokens 也支持 token mass 差异：step 5 中 code 平均 response 长度为 6156.77 tokens，math 为 2653.84 tokens，code 约为 math 的 2.32x。

teacher-student gap 的 TensorBoard 原始 tag 是 `teacher_student_gap_mean`，step 5 中 math/code 分别为 -0.124835 和 -0.012497。上表使用 absolute gap，因此写成正值；其绝对值比例为 9.99x。

需要注意，`response_clip_ratio` 是 response 长度达到上限的比例，不是 PPO ratio clipping。code 的 50.0% 说明 code response 更长、更容易撞到 8K 上限；它不能直接解释为 code gradient 被 PPO clipping 压制。这个指标更适合用来解释 token mass 和长度分布差异，而不是直接解释优化器层面的梯度裁剪。

因此，当前最稳妥的表述是：math 有更强的 per-sample teacher-student mismatch 和 sample gradient pressure；code 有更大的 token mass，并且从 full_grad/sample_grad 的差异看，code 的 sample gradient 可能更方向一致。由于双卡 profile 当前关闭了 sample-to-domain cosine，这个“math 内部方向分散/抵消”的解释还不能被 sample-level cosine 直接证明，只能作为由 full_grad 与 sample_grad 差异支持的机制假设。

## Statistical Validation

这份分析是描述性诊断，不应作为最终统计结论。原因是当前只有 5 个完整训练 step，且 step 6 没有训练 scalar。可以稳定引用的部分是指标方向和相对关系：

- audit health 指标干净：`autograd_unavailable=0`，`true_backward_fallback=0`，`sample_gradient_zero_norm_count=0`
- full-gradient sequential path 可用：`domain_sequential_available=1`
- replicated all-reduce 生效：`replicated_all_reduce=1`，`replica_count=2`
- math/code full_grad cosine 在 step 2-5 持续为负，并在 step 3-5 接近 -0.9
- sample gradient norm 的 math share 从 26.9% 上升到 78.5%
- step 5 中 math sample_grad/norm_mean 是 code 的 3.65x，但 math full_grad/grad_norm 只有 code 的 0.90x，说明 sample-level strength 和 domain-level aggregate norm 不能混为一谈

## What Changed Our Belief

原始观察是“math 的比例一直涨，code 一直降低”。只看 sample gradient norm，确实可以得出这个现象；但 global/full_grad 指标改变了我们对它的解释。

现在更准确的判断是：math 的 per-sample signal 正在变强或衰减更慢，但由于 math/code 梯度方向冲突，以及 domain 内部可能存在方向分散，global update 是否偏向 math 要看 signed projection share 和 domain-vs-total cosine。当前 total update 多数 step 仍由 code 决定，math 主要体现为更大的局部训练压力，而不是稳定主导全局更新方向。

## Limitations

- 只有 step 1-5 的完整训练 scalar，趋势长度不足。
- sample-to-domain cosine 当前在双卡 profile 中关闭，因此无法直接确认每个 sample gradient 与 domain full_grad 的方向一致性。
- `response_clip_ratio` 只表示 response 长度达到上限，不表示 PPO ratio clipping 或梯度裁剪。
- step 6 generation 后缺少后续训练 scalar，需要继续定位停滞原因。
- 当前报告没有使用 validation accuracy 或 downstream paper eval，因此不能推断最终能力变化。

## Next Actions

1. 继续跑更长的稳定训练，至少拿到 20-50 个完整 step，再判断 math/code 占比是否收敛。
2. 优先实现或恢复 two-pass FSDP sample-to-domain cosine，用来区分“sample norm 大”与“sample 方向真正支持 domain update”。
3. 增加 normalized contribution 指标，例如 signed projection share 除以 token fraction 或 sample fraction，避免 token 数差异掩盖 domain pressure。
4. 增加 domain aggregation proxy，例如 `full_grad_norm / sample_grad_norm_mean`、`full_grad_norm / token_fraction`，辅助判断 sample gradient 是否在 domain 内发生抵消。
5. 对 step 6 停滞点单独排查：重点看 generation 完成后的 actor update、full-gradient backward 和 sample-gradient recompute 阶段。
6. 将 4 卡和 8 卡 profile 作为后续 scale 实验入口，保持每 GPU 约 128 prompts，便于和当前双卡诊断结果对齐。

## Artifact And Reproducibility Index

- 配置文件：`configs/mopd_formal_dual_a800.yaml`
- 4 卡 scale 配置：`configs/mopd_formal_4gpu_a800.yaml`
- 8 卡 scale 配置：`configs/mopd_formal_8gpu_a800.yaml`
- TensorBoard run：`on-policy-distillation/Qwen3-4B-Multi-Teacher-ExOPD-formal-dual-a800`
- 远端日志：`logs/mopd_tp2_ckpt_b256_resp8k_util08_20260612_1102.log`
- 关键 TensorBoard tags：`math/full_grad/grad_norm`，`code/full_grad/grad_norm`，`global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k`，`global/full_grad_contribution/*/signed_projection_share`，`math/sample_grad/norm_mean`，`code/sample_grad/norm_mean`
