---
type: results-report
date: 2026-06-27
experiment_line: gradient-audit
round: 0
purpose: training-parity-smoke
status: active
source_artifacts:
  - temp/remote_codex_training_parity_nosync_0p6b_b4_step1_20260627_013540/gradient_smoke_summary.md
  - temp/remote_codex_training_parity_nosync_0p6b_b4_step1_20260627_013540/codex_training_parity_nosync_0p6b_b4_step1_20260627_013540.log
  - temp/remote_codex_training_parity_nosync_0p6b_b4_step1_20260627_013540/audit/
linked_experiments:
  - configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml
linked_results: []
---

# Gradient Audit / Round 0 / Training Parity Smoke / 2026-06-27

## Executive Summary

本轮实验验证的是当前 `gradient audit` 统计逻辑是否能在 pre-audit、sync backward、禁用 selected top-k training fast path 后，让 domain gradient、sample gradient、token gradient 与真实训练梯度对齐。

结论分成两层：

1. audit 内部已经基本闭合：domain target、sample gradient、token top-p=1 gradient 三者在 audit target space 内一致。
2. audit total 与真实 training total 仍未完全闭合：`cosine = 0.9705`，`rel_l2 = 0.2483`，但 `projection_share = 1.00009`，说明主要不是整体尺度错误，而更像某些参数空间参与不一致。

最关键的新证据是 top-param parity 指向 FSDP root flat parameter：

```text
_fsdp_wrapped_module._flat_param
```

真实训练 snapshot 中该参数梯度 norm 为 `0`，但 audit recompute 中非零；普通 layer flat parameter 已经非常接近。这将下一步 debug 范围从“整个 gradient 统计链路”缩小到了“root-level FSDP 参数在训练路径和 audit recompute 路径中的差异”。

## Experiment Identity and Decision Context

实验线：`gradient-audit`

本轮目标：

- 验证 FSDP `no_sync()` accumulation 是否解决真实训练 sequential backward 与 audit recompute 的尺度/同步差异。
- 验证禁用 selected top-k training fast path 后，训练 loss path 与 audit loss path 是否更接近。
- 验证 sample/token/domain gradient 是否在相同 audit target space 内闭合。
- 验证 parity debug 中手动 FSDP finalize 是否会污染真实训练 `.grad`。
- 判断此前 `0.5 * math + code` 或 first-domain replica scaling 假设是否仍成立。

需要支持的决策：

- 当前 sample/token projection share 是否可以继续用于分析。
- domain target 是否可以被认为等价于真实训练 gradient。
- 下一步应该继续查 loss builder、FSDP sync/finalize、还是参数空间差异。

## Setup and Evaluation Protocol

### 4B Smoke Attempt

首先尝试了 4B student 的 2GPU smoke。该 run 没有得到有效 gradient 结论，因为 Ray worker 被 CPU memory monitor 杀掉：

```text
Memory on the node was 230.13GB / 240GB, exceeding threshold 0.95.
```

这次失败是 Ray CPU memory OOM，不是代码 traceback，也不是 token/sample gradient 逻辑直接报错。

### Successful 0.6B Smoke

成功 run：

```text
codex_training_parity_nosync_0p6b_b4_step1_20260627_013540
```

核心配置：

```yaml
student_path: ../models/Qwen3-0.6B
math_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500
code_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Code-Step300
trainer.n_gpus_per_node: 2
data.train_batch_size: 4
trainer.total_training_steps: 1
audit.token_gradient_top_p: 1.0
audit.gradient_training_parity_enabled: true
audit.training_backward_no_sync_enabled: true
audit.selected_topk_head_train_enabled: false
audit.single_backward_smoke: false
```

训练路径实际 flag：

```text
training_backward_no_sync_used = 1
selected_topk_head_train_fast_path_used = 0
selected_topk_head_train_fast_path_requested = 1
single_backward_smoke_enabled = 0
```

这说明：

- 非最后一个 micro-batch 的训练 backward 已进入 FSDP `no_sync()` accumulation。
- selected top-k training fast path 虽然被原逻辑 request，但已被配置禁用。
- 本 run 仍使用正常 sequential micro-batch backward，而不是 single backward smoke。

## Main Findings

### Finding 1: Audit 内部 domain target 闭合

domain gradient 指标：

```text
math/full_grad/grad_norm = 7.825568983952832
code/full_grad/grad_norm = 7.755698115666416
math_vs_code/full_grad_cosine_train_i_k = 0.14733420524946708
math_to_total/signed_projection_share = 0.5039083813865658
code_to_total/signed_projection_share = 0.4960916186134342
```

chosen target closure：

```text
chosen_target/rel_l2 = 0
chosen_target/cosine = 1
chosen_target/projection_share = 1
```

解释：在 audit 自己构造的 target space 内，math + code 的总 target 是闭合的。这说明 domain target 的内部加和逻辑没有明显错误。

### Finding 2: Sample gradient 已经闭合到各自 domain target

Math domain：

```text
sample count = 2
projection share sum = 1.0000278976746682
normalized projection share sum = 1.0
sample-to-domain cosine mean = 0.6759698722817719
sample grad norm mean = 5.289848866614791
restore pre/post target rel_l2 max = 0
```

Code domain：

```text
sample count = 2
projection share sum = 1.0000233803706096
normalized projection share sum = 1.0
sample-to-domain cosine mean = 0.7038868472503299
sample grad norm mean = 5.459133894861759
restore pre/post target rel_l2 max = 0
```

解释：sample gradient 的 raw projection share sum 已经接近 1，normalized share 正好为 1。当前 sample gradient 对 audit domain target 是可信闭合的。

但注意：单个 sample 的 `cos_to_domain` 不需要接近 1，因为一个 sample gradient 只是 domain gradient 的一个组成部分，不应期待单样本方向与 domain 总方向完全一致。

### Finding 3: Token top-p=1 gradient 已经闭合到各自 domain target

Math domain：

```text
top50_loss_abs cosine = 0.7639476574574346
top50_loss_abs projection_share = 0.35808670587748515
topp100_loss_abs_mass cosine = 1.0000139487400506
topp100_loss_abs_mass projection_share = 1.0000278976746682
topp100 selected tokens = 1292
candidate token/sample fraction = 1.0 / 1.0
valid_frac = 1.0
autograd error = none
```

Code domain：

```text
top50_loss_abs cosine = 0.6407567269750122
top50_loss_abs projection_share = 0.2919064099330603
topp100_loss_abs_mass cosine = 1.0000116901169753
topp100_loss_abs_mass projection_share = 1.0000233803706096
topp100 selected tokens = 1128
candidate token/sample fraction = 1.0 / 1.0
valid_frac = 1.0
autograd error = none
```

解释：

- top-p=1 覆盖全部候选 response tokens，因此它应该闭合到 domain target。
- 实验确实观测到 top-p=1 的 cosine 和 projection share 都约等于 1。
- top50 只覆盖高 loss token 子集，因此 projection share 小于 1 是预期结果。

### Finding 4: Loss recompute / full-mask identity 已通过

`gradient_recompute_debug` 共记录 4 个 micro-batch：

```text
gradient_recompute_debug_error_count = 0
full_mask_identity_available_count = 4
full_mask_identity_error_count = 0
default loss == recompute loss
default grad vs full-mask grad cosine mean = 1.0
default grad vs full-mask grad rel_l2 mean = 0.0
default grad vs full-mask grad projection share mean = 1.0
```

解释：

- `_actor_micro_batch_loss(...)` 的默认 mask 与显式 full response mask 一致。
- 至少在 debug 覆盖到的 micro-batch 上，audit recompute loss 与 full-mask loss 没有差异。
- 这削弱了“token top-p=1 不闭合是 mask/loss builder 差异导致”的假设。

### Finding 5: Audit total 与真实 training total 仍未完全闭合

核心 parity 指标：

```text
audit_total_vs_training_total/reference_norm = 11.801422137725373
audit_total_vs_training_total/candidate_norm = 12.1608199495393
audit_total_vs_training_total/diff_norm = 2.930110119117054
audit_total_vs_training_total/rel_l2 = 0.24828449359083432
audit_total_vs_training_total/norm_ratio = 1.0304537713860786
audit_total_vs_training_total/cosine = 0.9705337954643385
audit_total_vs_training_total/projection_share = 1.0000902097939572
```

解释：

- `projection_share ≈ 1` 表明整体投影比例不是主要问题。
- `cosine = 0.9705` 和 `rel_l2 = 0.2483` 表明方向上仍存在非小量差异。
- 这不是“全部梯度都少除/多除一个 replica count”的简单尺度错误。

### Finding 6: parity debug 的 FSDP finalize 没有污染 `.grad`

新增指标：

```text
training_grad_before_vs_after_manual_finalize/rel_l2 = 0
training_grad_before_vs_after_manual_finalize/cosine = 1
training_grad_before_vs_after_manual_finalize/projection_share = 1
```

解释：parity debug 中调用的 manual FSDP finalize 没有改变真实训练 `.grad`。因此当前 mismatch 不能归因于 parity debug 自己污染了训练梯度。

### Finding 7: first-domain replica-scaling 假设被排除

此前怀疑真实训练 total 可能近似：

```text
0.5 * math + code
```

或者存在 first-domain / rest-domain 的 replica average 差异。本轮新指标不支持这个解释：

```text
first_domain_scaled_math_vs_training_total/rel_l2 = 0.5319979832096547
rest_domains_scaled_after_math_vs_training_total/rel_l2 = 0.5267360079799525
```

解释：这两个 hypothesis 的 `rel_l2` 比当前 `audit_total_vs_training_total/rel_l2 = 0.2483` 更差，所以“first domain 被多/少除一次 replica count”不是主因。

### Finding 8: 剩余 mismatch 集中在 FSDP root flat parameter

top-param parity 文件显示：

```text
_fsdp_wrapped_module._flat_param:
  training reference norm = 0
  audit candidate norm = 1.376 on rank 0
  audit candidate norm = 2.584 on rank 1

_fsdp_wrapped_module.model.layers.2._fsdp_wrapped_module._flat_param:
  cosine ≈ 0.99993
  rel_l2 ≈ 0.0114
```

解释：

- 普通 transformer layer 参数已经很接近。
- mismatch 主要来自 root-level FSDP flat parameter：真实训练路径里该参数没有 grad，但 audit recompute 路径里有 grad。
- 下一步应该定位 root flat param 具体映射到哪些原始参数，例如 embedding、lm_head、norm，或其它 root-level trainable 参数。

## Statistical Validation

本报告是 smoke/debug report，不是多 seed 统计实验。当前证据来自：

- 1 个成功 0.6B 2GPU smoke run。
- batch size 4，其中 math/code 各 2 个 sample。
- 1 个 training step。
- log scalar、audit JSONL、top-param parity JSONL。

因此当前可以支持的结论是工程调试级别：

- sample/token/domain 在 audit target space 内已经闭合。
- 真实 training total 与 audit total 仍存在差异。
- mismatch 的候选区域被缩小到 root-level FSDP flat parameter。

当前不能支持的结论：

- 不能说 4B 正式配置已经通过。
- 不能说所有 step 都稳定闭合。
- 不能说真实训练 gradient 已经与 audit gradient 完全一致。

## Figure-by-Figure Interpretation

本轮没有生成图表。TensorBoard event 文件已下载，但本地环境缺少 `tensorboard` Python 包，未在本报告中做 event-level 解析。当前数值来自训练日志和 audit JSONL。

后续如果要补图，建议至少画三类：

1. `audit_total_vs_training_total` 的 norm/cosine/rel_l2/projection_share。
2. sample/token closure 的 per-domain share sum 与 cosine。
3. top-param parity 的 per-param diff norm，突出 root flat param。

## Failure Cases / Negative Results / Limitations

### 4B run 失败

4B student smoke 因 Ray CPU memory OOM 失败，失败点是 worker memory 超过 Ray threshold。这说明当前完整 4B debug 配置的 CPU memory 压力过高，不能直接作为验证依据。

### 0.6B run 仍未证明 training parity 完成

虽然 0.6B run 完整跑完，但 audit total 与真实 training total 仍未闭合：

```text
rel_l2 = 0.2483
cosine = 0.9705
```

这意味着当前版本还不能声明“gradient audit 与真实训练梯度完全一致”。

### Smoke 覆盖范围有限

当前只有 1 step、batch size 4、2GPU、0.6B student。它足够用于定位工程问题，但不能替代正式 4B/6GPU audit 验证。

### TensorBoard 本地解析缺失

event 文件已经下载，但本地 Python 环境没有 `tensorboard` 包。因此本报告没有直接从 event 文件二次解析标量。训练日志与 audit JSONL 已足够支撑当前结论。

## What Changed Our Belief

本轮实验改变了几个判断：

1. 之前怀疑 sample/token gradient 的比例问题来自 replica scaling；现在在 audit target space 内，sample/token 都已经闭合，所以这个不是当前主线。
2. 之前怀疑 FSDP finalize 或 auxiliary debug backward 会污染 `.grad`；新增指标显示 finalize 前后 `.grad` 完全一致，因此该假设被排除。
3. 之前怀疑真实 training total 是 `0.5 * math + code` 这类 first-domain scaling 问题；新 hypothesis 指标显示该形态更差，因此也被排除。
4. 当前更可信的主线是 root-level FSDP flat parameter 在训练路径与 audit recompute 路径中参与方式不一致。

## Next Actions

P0. 定位 root flat param 的真实含义。

- 打印 `_fsdp_wrapped_module._flat_param` 对应的原始参数范围或 FSDP handle metadata。
- 判断它是否包含 embedding、lm_head、final norm 或其它 root-level trainable 参数。
- 记录该参数在训练 forward/backward 中是否 `requires_grad=True`，是否进入 optimizer param group。

P1. 对比 root flat param 在训练路径与 audit recompute 路径中的梯度来源。

- 在真实训练 backward 后记录 root flat param grad norm。
- 在 audit domain recompute 后记录 root flat param grad norm。
- 对 root flat param 做 loss component attribution，确认是 top-k distill loss、KL loss、entropy 或其它项产生。

P2. 验证是否是训练路径清零或跳过 root-level 参数。

- 检查 optimizer step 前 `.grad` 是否被 clip、zero、offload/finalize 逻辑改写。
- 检查 FSDP root module 是否与子层 FSDP module 的 grad reduce/finalize 行为不同。

P3. 在 0.6B 上补一个 targeted smoke。

建议下一次只增加 root flat param metadata 和 per-param parity 记录，不扩大 batch/model，避免引入新的资源变量。

P4. 4B 正式 smoke 需要降低 CPU memory 压力后再跑。

可选手段：

- 降低 debug top param 数量。
- 关闭大 vocab vector 输出。
- 降低 max response length。
- 降低 batch size。
- 先保持 0.6B 定位问题，再迁移到 4B。

## Artifact and Reproducibility Index

成功 run id：

```text
codex_training_parity_nosync_0p6b_b4_step1_20260627_013540
```

本地产物：

```text
temp/remote_codex_training_parity_nosync_0p6b_b4_step1_20260627_013540/
```

核心文件：

```text
gradient_smoke_summary.md
codex_training_parity_nosync_0p6b_b4_step1_20260627_013540.log
_codex_training_parity_nosync_0p6b_b4_step1_20260627_013540.yaml
audit/domain_step_metrics.jsonl
audit/sample_grad_metrics.jsonl
audit/token_grad_metrics.jsonl
audit/gradient_recompute_debug.jsonl
audit/training_gradient_parity_top_params.jsonl
```

当前代码分支：

```text
bowen
```

当前代码状态尚未整理为干净 commit；本报告记录的是当前 working tree 上远端同步后跑出的 smoke 结果。
