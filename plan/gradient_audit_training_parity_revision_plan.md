# Gradient Audit / Training Parity Revision Plan

## 目标

当前首要目标不是让 audit 指标通过额外缩放看起来闭合，而是验证并修复真实训练 `.grad` 与 audit recompute gradient 的 parity：

```text
training_total_grad ~= audit_math_grad + audit_code_grad
```

只有这个关系成立后，domain / sample / token 的 projection share 才能解释为真实 training gradient contribution ratio。

## 当前判断

根据已有 smoke 现象和修改建议，最可疑的问题链路是：

```text
sequential backward / FSDP sync-finalize / selected top-k fast path
```

典型异常形式是：

```text
training_total ~= 0.5 * first_domain + rest_domain
```

这说明问题更可能发生在真实训练 backward accumulation，而不是 audit projection 公式本身。

## P0：保留并完善 hypothesis parity metrics

目的：明确当前真实训练 `.grad` 更像哪一个假设。

需要记录：

```text
audit_total_vs_training_total
first_domain_scaled_{domain}_vs_training_total
rest_domains_scaled_after_{domain}_vs_training_total
```

判断标准：

```text
如果 first_domain_scaled_math_vs_training_total 明显优于 audit_total_vs_training_total，
说明真实训练梯度很可能是 0.5 * math + code。
```

实现策略：

1. 保留已有 generic domain-target hypothesis 逻辑。
2. 如果日志命名不够直观，补充 first-domain 命名指标，方便和报告中的建议对齐。
3. 不通过手动乘除 `replica_count` 修正 audit target。

## P1：确认 parity/debug 不会污染真实训练 `.grad`

目的：避免 `training_backward_parity_metrics()` 本身改变 optimizer step 使用的 `.grad`。

修改方向：

```text
snapshot raw training .grad
manual finalize / auxiliary finalize
snapshot finalized training .grad
emit before_vs_after metrics
```

新增指标：

```text
global/full_grad_training_parity/training_grad_before_vs_after_manual_finalize/rel_l2
global/full_grad_training_parity/training_grad_before_vs_after_manual_finalize/cosine
global/full_grad_training_parity/training_grad_before_vs_after_manual_finalize/norm_ratio
global/full_grad_training_parity/training_grad_before_vs_after_manual_finalize/projection_share
```

判断标准：

```text
rel_l2 ~= 0, cosine ~= 1
```

如果这里不闭合，先暂停信任 parity debug，并考虑在 parity 统计前后保存/恢复 `.grad`，或避免在真实训练路径里调用会 mutate `.grad` 的 finalize helper。

## P2：默认禁用 selected top-k training fast path

目的：排除 `FSDP.summon_full_params()` 的 selected top-k head fast path 对真实 backward accumulation 的影响。

修改方向：

```text
_can_use_selected_topk_head() 默认在 training 中返回 false
只有 selected_topk_head_train_enabled: true 时才开启
```

建议配置默认值：

```yaml
policy_loss:
  selected_topk_head_train_enabled: false
```

同时记录一个 debug metric，确认 smoke run 实际是否进入 fast path：

```text
global/audit/selected_topk_head_train_fast_path_used
```

## P3：将真实训练 sequential backward 改为标准 FSDP no_sync accumulation

目的：让多 micro-batch accumulation 遵循 FSDP 推荐语义。

修改方向：

```text
前 N-1 个 tracked micro-batch:
  with actor_module.no_sync():
      forward + loss + backward

最后 1 个 tracked micro-batch:
  forward + loss + backward with sync
```

实现上使用 `ExitStack` 同时管理：

```text
fsdp no_sync context
selected_topk_param_context
forward
loss.backward()
```

这样可以保证 context 覆盖 forward + backward，并且异常时能正常退出。

## P4：把 domain-boundary checkpoint 调整为 accumulation checkpoint

原建议里的 domain-boundary checkpoint 假设真实训练 batch 按 domain block 排序。当前我们倾向于 pre-audit + 真实训练不按 domain 重排，所以该 checkpoint 需要改成更通用的形式：

```text
after_micro_batch_{i}_vs_audit_total
after_micro_batch_{i}_vs_first_domain
after_final_sync_vs_audit_total
```

如果临时打开旧的 domain-block diagnostic，再额外记录：

```text
after_domain_math
after_domain_code
```

这一步主要用于定位缩放发生在：

```text
第一个 backward 当下
后续 backward/finalize 处理中
最终 FSDP sync/reduce 阶段
```

## P5：single backward smoke 作为定位实验

目的：判断问题是否来自连续多次 backward accumulation。

增加 smoke-only 配置：

```yaml
mopd_full_gradient:
  single_backward_smoke: true
```

实验逻辑：

```text
每个 micro-batch 只 forward 并保存 loss
mini-batch 结束后 sum(losses).backward()
再跑 training_backward_parity_metrics()
```

使用限制：

1. 只用于 smoke/debug，不作为默认训练逻辑。
2. 先禁用 selected top-k fast path，再做 single backward。
3. 如果 single backward 闭合而普通 sequential backward 不闭合，问题基本锁定在 FSDP accumulation/sync。

## P6：可选记录 true-training snapshot domain target

如果目标是解释“当前 optimizer step 实际更新了什么”，可以从真实训练 backward 中 snapshot：

```text
training_total_grad
training_first_block_grad
training_second_block_grad = training_total_grad - training_first_block_grad
```

但这只能忠实记录真实训练 `.grad`，不能证明真实训练 `.grad` 是正确的。

因此优先级低于 P0-P5。

## P7：长期统一 loss builder

当前训练路径和 audit 路径仍有两套 loss 构造：

```text
dp_actor.py inline loss
actor_loss.py / audit recompute loss
```

长期目标是抽出唯一 loss builder：

```python
build_actor_micro_batch_loss(...)
```

训练和 audit 都调用同一份逻辑，统一：

```text
top-k support selection
teacher selection
teacher prefix mask
distill_response_mask
loss_token_mask
top-k loss
KL loss
loss_scale_factor
selected top-k context
```

这一步建议在 parity 问题定位清楚后再做，避免同时改变过多变量。

## Smoke 验证矩阵

最小验证顺序：

```text
Run A: 当前 baseline + P0/P1 metrics
目的：确认 0.5 * first_domain + rest 是否仍成立。

Run B: 禁用 selected top-k fast path
目的：看 fast path 是否是主因。

Run C: 禁用 selected top-k fast path + FSDP no_sync accumulation
目的：验证最终推荐训练路径是否闭合。

Run D: 禁用 selected top-k fast path + single backward smoke
目的：如果 Run C 仍不闭合，用来判断是不是 sequential backward 问题。
```

如果显存压力过大，优先降级：

```text
student model -> 0.6B
micro batch / train batch -> 更小
storage dtype -> bfloat16
```

## 关键验收指标

主要看：

```text
global/full_grad_training_parity/audit_total_vs_training_total/rel_l2
global/full_grad_training_parity/audit_total_vs_training_total/cosine
global/full_grad_training_parity/audit_total_vs_training_total/norm_ratio
global/full_grad_training_parity/audit_total_vs_training_total/projection_share
```

目标：

```text
rel_l2 <= 1e-3
cosine ~= 1
norm_ratio ~= 1
projection_share ~= 1
```

辅助判断：

```text
training_grad_before_vs_after_manual_finalize/rel_l2 ~= 0
first_domain_scaled_* 不再明显优于 audit_total_vs_training_total
selected_topk_head_train_fast_path_used = 0
training_backward_no_sync_used = 1
```

## 不采用的修法

不把问题修成：

```text
audit_math *= 0.5
projection_share 手动除以 replica_count
```

原因是这会让 audit 去适配一个可能已经错误的真实训练 `.grad`，而不是修复真实训练 gradient parity。

## 建议实施顺序

1. 对齐当前代码和本计划，确认已有 P0/no_sync 改动是否完整。
2. 先实现 P1 finalize 前后 `.grad` mutation guard。
3. 实现 P2：training 默认禁用 selected top-k fast path，并加日志指标。
4. 整理 P3：把当前手写 no_sync 改成 `ExitStack` 版本。
5. 补 P4 accumulation checkpoint，避免依赖 domain-block 排序。
6. 加 P5 single backward smoke 配置，但默认关闭。
7. 本地 compile / diff check。
8. 远端按 Run A-D 做 smoke，下载日志并只分析 gradient 相关指标。
9. parity 闭合后，再决定是否进入 P7 common loss builder 重构。
