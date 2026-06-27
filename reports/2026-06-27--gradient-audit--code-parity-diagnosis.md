# Gradient Audit Code Parity Diagnosis

Date: 2026-06-27

Source diagnosis: `/Users/linghuazhang/Downloads/gradient_audit_parity_diagnosis.md`

Related smoke run: `codex_training_parity_nosync_0p6b_b4_step1_20260627_013540`

## Core Conclusion

当前代码中最可疑的问题不是 sample/token/domain projection 公式，也不是 parity debug 的 FSDP finalize 污染了 `.grad`，而是：

```text
audit recompute path 与真实 training path 没有使用完全相同的 forward/backward parameter space。
```

更具体地说：

```text
training path 已经禁用 selected top-k head fast path；
但 audit recompute path 仍然会无条件尝试进入 selected-topk recompute context。
```

这会让 audit backward 可能在 `FSDP.summon_full_params(...)` / selected-head path 中运行，而真实训练 backward 在普通 `actor_module(...)` / full-logits path 中运行。

当前 smoke 的 top-param parity 证明 mismatch 几乎全部集中在 root-level FSDP `_flat_param`：

```text
reference = audit_total
candidate = training_total
diff = training_total - audit_total

_fsdp_wrapped_module._flat_param:
  audit/reference norm = 0
  training/candidate norm = 1.376 on rank 0
  training/candidate norm = 2.584 on rank 1
```

因此更准确的判断是：

```text
audit_total 漏掉 root flat param gradient；
不是 training_total 漏掉 root flat param gradient。
```

## Evidence From Code

### 1. Parity reference/candidate 方向容易误读

位置：`mopd_verl/full_gradient/tracker.py`

`training_backward_parity_metrics()` 中：

```python
reference_chunks = self._last_audit_total_chunks
raw_training_chunks = _snapshot_current_grad_chunks(...)

stats, top_rows = _gradient_chunk_pair_stats(
    self.actor,
    reference_chunks,
    raw_training_chunks,
    ...
)
```

`_gradient_chunk_pair_stats()` 中：

```python
diff = candidate_float - reference_float
```

所以：

```text
reference = audit_total
candidate = training_total
diff = training_total - audit_total
```

当前 JSONL 只写：

```text
reference_norm
candidate_norm
diff_norm
```

没有写：

```text
reference_label = audit_total
candidate_label = training_total
diff_label = training_total_minus_audit_total
```

这导致上一份实验报告中把 root flat param 的方向读反。

### 2. Training path 已经显式禁用 selected top-k head

位置：`third_party/verl/verl/workers/actor/dp_actor.py`

训练侧读取：

```python
selected_topk_head_train_enabled = bool(
    mopd_full_gradient_cfg.get(
        "selected_topk_head_train_enabled",
        self.config.policy_loss.get("selected_topk_head_train_enabled", False),
    )
)
```

本次 smoke 配置中该值为 `false`，日志也显示：

```text
selected_topk_head_train_fast_path_requested = 1
selected_topk_head_train_fast_path_used = 0
```

训练 forward 调用：

```python
forward_output = self._forward_micro_batch(
    ...,
    allow_selected_topk_head=selected_topk_head_train_enabled,
)
```

因此真实训练虽然满足 selected-head 的 request 条件，但实际没有进入 selected-head fast path。

### 3. Audit recompute path 没有对应的禁用开关

位置：`mopd_verl/full_gradient/actor_loss.py`

`_actor_micro_batch_loss()` 当前签名是：

```python
def _actor_micro_batch_loss(
    actor,
    micro_batch,
    *,
    loss_scale_factor,
    on_policy,
    safe_logprob_backward=False,
    response_mask_override=None,
):
```

它没有 `allow_selected_topk_head` 参数。

内部 forward：

```python
forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
```

由于没有传 `allow_selected_topk_head=False`，`actor._forward_micro_batch()` 使用默认值：

```python
allow_selected_topk_head: bool = True
```

所以 audit loss recompute 默认允许 selected-head path。

### 4. Audit recompute 多处无条件进入 selected-topk context

位置：`mopd_verl/full_gradient/tracker.py`

domain direct recompute：

```python
with _actor_selected_topk_recompute_context(self.actor, micro_batch):
    loss = _actor_micro_batch_loss(...)
    loss.backward()
```

gradient recompute debug、token gradient、sample gradient 中也存在同样模式。

这意味着 audit recompute 当前会无条件尝试：

```python
actor._selected_topk_param_context()
```

位置：`mopd_verl/full_gradient/actor_loss.py`

```python
return actor._selected_topk_param_context()
```

而 actor 侧这个 context 是：

```python
FSDP.summon_full_params(
    self.actor_module,
    recurse=True,
    writeback=True,
    rank0_only=False,
    offload_to_cpu=False,
)
```

这个 context 可能改变 backward 时 gradient materialization 的位置，尤其是 root-level flat param。

### 5. `chosen_target` 闭合不等于 training parity 闭合

位置：`mopd_verl/full_gradient/tracker.py`

pre-audit 下：

```python
chosen_reference_chunks = self._summed_domain_target_reference_chunks(domain_targets)
```

随后：

```python
_domain_target_closure_metrics(
    domain_targets,
    reference_chunks=chosen_reference_chunks,
)
```

因此：

```text
chosen_target closure = sum(domain_targets) vs sum(domain_targets)
```

它只能证明 audit target space 内部加和自洽，不能证明：

```text
audit_total == training_total
```

sample/token top-p=1 的闭合也同理：它们闭合到的是当前 audit domain target。如果 audit domain target 已经漏掉 root flat param，那么 sample/token 仍然可以完美闭合到这个不完整 target。

## Current Interpretation

当前现象最一致的解释是：

```text
training backward:
  ordinary actor_module forward/backward
  root-level FSDP flat_param.grad 被正常 materialize

audit recompute backward:
  selected-topk recompute context / summon_full_params path
  root-level FSDP flat_param.grad 没有落到 optimizer flat parameter 上

result:
  audit_total 在 root flat param 上为 0
  training_total 在 root flat param 上非零
  总 diff_norm 几乎完全由 root flat param 解释
```

这也解释了为什么：

```text
普通 transformer layer flat params 基本对齐；
root flat param 不对齐；
projection_share ≈ 1，但 rel_l2 仍然明显非零。
```

root flat param 很可能包含未被 transformer layer auto-wrap 包进去的参数，例如：

```text
embed_tokens.weight
lm_head.weight
final norm
其它 root-level trainable parameters
```

selected-topk head path 正好最容易涉及 `lm_head` / embedding / final norm 这类参数。

## Confirmed Issues

### P0: Audit recompute 与 training forward path 不一致

这是 correctness 问题。当前 training 可以关 selected-head，但 audit 不能关。即使配置中：

```yaml
selected_topk_head_train_enabled: false
```

audit recompute 仍会尝试 selected-head/summon path。

### P0: `_actor_micro_batch_loss()` 缺少 `allow_selected_topk_head`

该函数是 audit loss builder 的核心入口，但不能把 selected-head 开关传给 `actor._forward_micro_batch()`。

### P1: parity JSONL 缺少 reference/candidate label

这已经导致实际分析中方向被读反。应该显式记录：

```text
reference_label = audit_total
candidate_label = training_total
diff_label = training_total_minus_audit_total
```

并额外写：

```text
audit_norm
training_norm
training_minus_audit_norm
```

### P1: 缺少 root flat param metadata

当前 top-param row 只告诉我们名字是：

```text
_fsdp_wrapped_module._flat_param
```

但没有告诉这个 flat param 包含哪些原始参数。需要记录 FlatParameter metadata，例如：

```text
_fqns
_numels
_shapes
_param_infos
requires_grad
parameter_type
```

### P1: 缺少 per-phase root flat grad debug

需要直接记录：

```text
audit_domain_recompute_math/root_flat_grad_norm
audit_domain_recompute_code/root_flat_grad_norm
training_before_finalize/root_flat_grad_norm
training_after_finalize/root_flat_grad_norm
```

这样可以明确 root flat param 是在哪个阶段漏掉的。

## Recommended Fix Direction

### Step 1: 增加 audit selected-topk 开关，默认关闭

新增配置：

```yaml
audit_selected_topk_head_enabled: false
```

tracker 初始化：

```python
self.audit_selected_topk_head_enabled = bool(
    cfg.get("audit_selected_topk_head_enabled", False)
)
```

### Step 2: 让 `_actor_micro_batch_loss()` 支持 forward path 控制

增加参数：

```python
allow_selected_topk_head: bool = False
```

forward 调用改成：

```python
forward_output = actor._forward_micro_batch(
    model_inputs,
    **forward_kwargs,
    allow_selected_topk_head=allow_selected_topk_head,
)
```

默认建议设为 `False`，因为 audit correctness 优先于 selected-head fast path。

### Step 3: audit context 和 forward flag 必须一起控制

不能只关 context，也不能只关 forward flag。建议封装 helper：

```python
def _audit_selected_topk_context(self, micro_batch):
    if not self.audit_selected_topk_head_enabled:
        return nullcontext()
    return _actor_selected_topk_recompute_context(self.actor, micro_batch)
```

所有 audit recompute 调用统一写成：

```python
with self._audit_selected_topk_context(micro_batch):
    loss = _actor_micro_batch_loss(
        ...,
        allow_selected_topk_head=self.audit_selected_topk_head_enabled,
    )
```

### Step 4: 给 parity/top-param debug 加明确标签与 flat metadata

至少写入：

```python
reference_label = "audit_total"
candidate_label = "training_total"
diff_label = "training_total_minus_audit_total"
```

再补 root flat param metadata 和 per-phase grad norm。

## Next Validation

建议下一次 targeted smoke：

```yaml
audit_selected_topk_head_enabled: false
selected_topk_head_train_enabled: false
training_backward_no_sync_enabled: true
single_backward_smoke: false
token_gradient_top_p: 1.0
```

预期：

```text
audit_total root flat param norm 不再为 0
audit_total_vs_training_total/diff_norm 从 2.93 明显下降
audit_total_vs_training_total/rel_l2 从 0.248 明显下降
root flat param 不再支配 top-param diff
```

如果仍不闭合，再做隔离实验：

```yaml
single_backward_smoke: true
```

以及：

```yaml
training_backward_no_sync_enabled: false
```

用于判断是否还有 sequential/no_sync accumulation 对 root param 的特殊影响。

## Bottom Line

诊断报告的大方向是对的，但 root flat param 的方向必须修正：

```text
audit reference 中 root flat param 为 0；
training candidate 中 root flat param 非零。
```

当前代码最大问题是 audit recompute 仍可能走 selected-topk/summon_full_params path，而 training 已经禁用该 path。这会让二者不在同一个 optimizer flat-param gradient space 中，正好解释当前 root-level FSDP `_flat_param` 漏梯度现象。
