# Gradient Audit Debug Plan

## Goal

把 `domain target`、`sample gradient`、`token gradient` 放到同一个可验证的 gradient space 中，并证明 `token_gradient_top_p=1.0` 时 full-token gradient 能闭合到可信 domain target。

## Current Interpretation

当前建议的方向基本合理，但要分两层看：

1. `token top_p=1` 的 math 指标出现 `projection ~= 0.5`、`norm/cos ~= 0.7071`，强烈支持 replica scaling convention 有问题，尤其可能是 token stats 对 optimizer-space gradient 又除了一次 `replica_count=2`。
2. 这个解释不能覆盖全部问题。code 即使撤销一次 replica scaling，projection 也大约只有 `0.764`，所以仍需要排查 `domain target construction`、`_actor_micro_batch_loss` 复刻训练 loss 的一致性，以及 `response_mask_override` 的 full-mask identity。

因此执行顺序应该是：先加无侵入诊断指标，再根据证据修 scaling，最后才做 shared loss builder。

## Phases

- [x] Phase 0: 暂停信任 raw projection share
- [x] Phase 1: 加 token no-replica-scale 指标
- [x] Phase 2: 记录 chosen target source 与 chosen target closure
- [ ] Phase 3: 打开并增强 gradient_recompute_debug
- [x] Phase 4: 加 full-mask identity test
- [ ] Phase 5: 根据诊断结果修复 gradient space / replica scaling
- [ ] Phase 6: 若确认 loss drift，再重构 shared loss builder
- [ ] Phase 7: 远端 1-step smoke 验证 closure

## Phase Details

### Phase 0: 暂停信任 raw projection share

当前先把 raw `projection_share` 视为 debug signal，不作为论文或实验结论。

暂时可参考：

- `sample_projection_share_normalized`
- token selection 的 `loss_abs_mass_frac`
- token selection 是否覆盖 full candidate tokens

但不要解释成真实 contribution ratio。

### Phase 1: Token no-replica-scale metrics

目的：验证 math 的 `0.5 / 0.7071` 是否来自 token 统计多除一次 `replica_count`。

改动位置：

- `/Users/linghuazhang/Desktop/Project/OPD/code/mopd_verl/full_gradient/tracker.py`
- `_recompute_token_selection_gradient_stats`

新增指标：

```text
token_grad_norm_no_replica_scale
{domain}_projection_share_no_replica_scale
{domain}_cos_no_replica_scale
{domain}_norm_ratio_no_replica_scale
token_grad_replica_scale_applied
token_grad_replica_count
```

判断标准：

```text
如果 math top_p=1 no-replica-scale 后 projection/cos/norm_ratio 接近 1：
  token 当前 scaling 基本确认有问题。

如果 code no-replica-scale 后仍明显小于 1：
  code 还存在 target 或 loss/mask path 问题。
```

### Phase 2: Chosen target source and closure

目的：明确 sample/token 实际投影到哪个 domain target，以及这个最终 target 是否闭合。

改动位置：

- `finish_mini_batch`
- `_finish_direct_domain_gradient_metrics`
- `_finish_domain_gradient_metrics`
- `_finish_single_domain_gradient_metrics`

新增指标：

```text
global/audit/full_gradient_domain_target_source
global/full_grad_closure/chosen_target/rel_l2
global/full_grad_closure/chosen_target/cosine
global/full_grad_closure/chosen_target/norm_ratio
global/full_grad_closure/chosen_target/projection_share
```

target source 编码：

```text
0 = none
1 = direct_recompute
2 = sequential_snapshot_diff
3 = single_domain_snapshot
4 = hook_accumulation
```

判断标准：

```text
chosen_target/rel_l2 接近 0，cos/norm/projection 接近 1：
  target construction 可暂时认为可信。

chosen target 不闭合：
  sample/token projection 没有可信解释，优先修 target。
```

### Phase 3: Gradient recompute debug

目的：把差异拆成四类：

```text
loss 不一致
backward path 不一致
parameter.grad/finalize 不一致
domain target construction 不一致
```

配置：

```yaml
mopd_audit.gradient_recompute_debug_enabled: true
mopd_audit.gradient_recompute_debug_max_micro_batches: 8
mopd_audit.gradient_recompute_debug_top_param_count: 16
mopd_audit.gradient_recompute_debug_storage_dtype: float32
mopd_audit.storage_dtype: float32
mopd_audit.token_gradient_top_p: 1.0
trainer.total_training_steps: 1
trainer.total_epochs: 1
```

重点指标：

```text
loss_rel_diff
actual_vs_recompute_hook_selected/rel_l2
actual_hook_vs_recompute_grad_selected/rel_l2
domain_sum_vs_training/rel_l2
chosen_target/rel_l2
```

判断标准：

```text
loss_rel_diff 大：
  _actor_micro_batch_loss 与 update_policy inline loss 不一致。

loss_rel_diff 小，但 hook rel_l2 大：
  RNG/dropout/scaler/top-k/FSDP context 不一致。

hook rel_l2 小，但 parameter.grad rel_l2 大：
  FSDP finalize / parameter.grad snapshot / grad restore 有问题。

micro-batch 闭合，但 domain target 不闭合：
  domain partition / replica reduce / target construction 有问题。
```

### Phase 4: Full-mask identity test

目的：验证 token top_p=1 的基础前提。

同一个 micro-batch 上比较：

```text
loss_default = _actor_micro_batch_loss(response_mask_override=None)
loss_full_mask = _actor_micro_batch_loss(response_mask_override=response_mask)
```

新增指标：

```text
full_mask_loss_abs_diff
full_mask_loss_rel_diff
grad_default_vs_full_mask_rel_l2
grad_default_vs_full_mask_cos
grad_default_vs_full_mask_projection_share
```

判断标准：

```text
如果 full-mask identity 不成立：
  token top_p=1 不闭合优先查 response_mask_override、token_mask_contribution_scale、teacher_prefix_masks、loss_agg_mode。

如果 full-mask identity 成立：
  token 不闭合更可能是 target/scaling 问题。
```

### Phase 5: Fix gradient target space / replica scaling

只有在 Phase 1/2 证明 scaling convention 有问题后再改。

建议默认语义：

```yaml
mopd_audit.gradient_target_space: distributed_chunks
```

含义：

```text
candidate 和 target 都以 rank-local chunks 表示；
norm/dot 统一 all_reduce sum；
不要额外除 replica_count，除非 target_map 本身也是 optimizer-average space。
```

修复后保留对照指标一段时间：

```text
current projection
no_replica_scale projection
target_space
replica_scale_applied
```

### Phase 6: Shared loss builder

只有在 Phase 3 证明 `_actor_micro_batch_loss` 与 `update_policy` inline loss 存在 drift，或者为了长期维护性，再做这个重构。

原则：

```text
不能再写第三份 loss rebuilder；
要把训练和 audit 都改成调用同一个 shared loss builder。
```

目标接口：

```python
compute_actor_micro_batch_loss(
    actor,
    micro_batch,
    *,
    loss_scale_factor,
    on_policy,
    response_mask_override=None,
    return_debug=False,
)
```

注意事项：

- top-k selected param context 必须覆盖 forward 到 backward。
- builder 只负责 forward/loss/component metrics。
- backward、GradScaler、FSDP finalize/restore 留在 caller。
- 先让 audit 调 shared builder，通过 debug 后再切 training path。

### Phase 7: Remote smoke validation

最小远端验证：

```text
configs/mopd_formal_audit_loss_only_6gpu_smoke.yaml
trainer.total_training_steps=1
trainer.total_epochs=1
mopd_audit.token_gradient_top_p=1.0
mopd_audit.gradient_recompute_debug_enabled=true
```

验收标准：

```text
chosen_target/rel_l2 <= 1e-4 或至少显著下降
chosen_target/cosine ~= 1
chosen_target/norm_ratio ~= 1
chosen_target/projection_share ~= 1

token top_p=1 projection_share ~= 1
token top_p=1 cos_to_domain ~= 1
token top_p=1 norm_ratio ~= 1

full_mask_loss_rel_diff ~= 0
grad_default_vs_full_mask_rel_l2 ~= 0
grad_default_vs_full_mask_cos ~= 1
```

## Execution Order

第一批提交只做诊断，不改变训练结果：

1. Phase 1: token no-replica-scale metrics
2. Phase 2: chosen target source + chosen target closure
3. Phase 4: full-mask identity test
4. Phase 3: 打开 debug config 并跑 1-step

第二批提交根据结果修复：

1. 如果 no-replica-scale 闭合：修 token replica scaling / target space。
2. 如果 chosen target 不闭合：修 domain target construction，必要时引入 hook accumulation target。
3. 如果 loss diff 不为 0：做 shared loss builder。

## Status

当前已完成 Phase 1、Phase 2、Phase 4 的诊断代码。下一步等待远端 1-step smoke 验证，优先查看 `no_replica_scale`、`chosen_target` 与 `full_mask_identity` 三组指标。
