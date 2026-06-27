---
type: gradient-audit-log-analysis
date: 2026-06-26
experiment_line: mopd-gradient-audit
run_id: grad_diagnostic_smoke_20260626_203323
config: configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml
status: active
source_artifacts:
  - temp/remote_grad_diagnostic_smoke_20260626_203323/grad_diagnostic_smoke_20260626_203323.log
  - temp/remote_grad_diagnostic_smoke_20260626_203323/audit/domain_step_metrics.jsonl
  - temp/remote_grad_diagnostic_smoke_20260626_203323/audit/gradient_recompute_debug.jsonl
  - temp/remote_grad_diagnostic_smoke_20260626_203323/audit/sample_grad_metrics.jsonl
  - temp/remote_grad_diagnostic_smoke_20260626_203323/audit/token_grad_metrics.jsonl
  - temp/grad_diagnostic_smoke_20260626_203323_report.md
---

# Gradient Audit Log Analysis Report

本文档用于交接当前 `MOPD / OPD` gradient audit 的日志分析方法。目标不是只复述一次 smoke run 的结果，而是给后续分析者一套稳定流程：拿到日志后应该看哪些文件、读哪些指标、如何判断 domain/sample/token gradient 是否在同一个 gradient space 内闭合，以及看到异常时如何定位。

## 1. Executive Summary

当前最新可复现实验是：

```text
run_id = grad_diagnostic_smoke_20260626_203323
config = configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml
hardware = 2 x A800 80GB
steps = 1
train_batch_size = 8
result = completed, no traceback, no OOM
```

这次 run 的核心结论：

1. 训练 inline loss 与 audit recompute loss 已经闭合。`loss_rel_diff=0`。
2. 训练 hook gradient 与 recompute hook gradient 已经闭合。`actual_vs_recompute_hook_selected_cosine=1`，`rel_l2=0`。
3. `response_mask_override=response_mask` 与默认 full loss 已经闭合。`full_mask_loss_rel_diff=0`，`default_grad_vs_full_mask_grad_selected_rel_l2=0`。
4. 在 2 GPU 下，`parameter.grad` 相比 hook gradient 存在明确的 `1 / replica_count = 0.5` 缩放。这个解释了 math token top-p=1 的 `0.5 share / 0.707 norm_ratio` 问题。
5. math domain 的 token top-p=1 在 no-replica-scale 指标下几乎完全闭合：`share=1.0001`，`cos=1.00005`，`norm_ratio=1.00005`。
6. code domain 的 token top-p=1 仍未完全闭合：no-replica-scale 后 `share=0.7643`，`cos=0.8178`，`norm_ratio=0.9345`。这说明 code 还有 residual mismatch，不能只归因于 replica scaling。
7. sample gradient 的 normalized share 可用于 domain 内相对排序，但 raw projection share 目前不能解释为严格真实 contribution ratio。math raw share sum 为 `1.3484`，code raw share sum 为 `1.0177`。

当前最重要的判断边界：

```text
可以信：
- loss recompute path 对 debug samples 是一致的；
- hook-level training gradient 与 recompute gradient 是一致的；
- math token top-p=1 在 no-replica target space 下闭合；
- sample normalized share 可作为相对排序指标。

暂时不能完全信：
- sample raw projection share 作为真实 contribution ratio；
- code token top-p=1 的 closure；
- domain_sum_vs_training 直接作为 domain target 是否正确的唯一标准。
```

## 2. Relevant Code Paths

主要实现位于：

```text
mopd_verl/full_gradient/tracker.py
```

关键入口：

| 逻辑 | 代码位置 |
| --- | --- |
| mini-batch audit 总入口 | `finish_mini_batch()` around line 1231 |
| direct domain target recompute | `_recompute_direct_domain_targets()` around line 1571 |
| domain target closure | `_domain_target_closure_metrics()` around line 1664 |
| recompute debug | `_gradient_recompute_debug_metrics()` around line 1732 |
| sample gradient metrics | `_sample_cos_metrics()` around line 2181 |
| token gradient metrics | `_token_gradient_metrics()` around line 2438 |
| token selection gradient recompute | `_recompute_token_selection_gradient_stats()` around line 2936 |
| sample-to-domain gradient recompute | `_recompute_sample_to_domain_stats()` around line 3570 |

配置入口：

| 配置 | 位置 |
| --- | --- |
| config smoke profile | `configs/mopd_formal_audit_grad_consistency_2gpu_smoke.yaml` |
| audit config dataclass | `mopd_verl/settings.py` |
| Hydra/OmegaConf audit parsing | `mopd_verl/verl_audit.py` |

## 3. Smoke Config Requirements

用于 gradient consistency debug 的 config 至少应满足：

```yaml
data:
  train_batch_size: 8

actor:
  ppo_mini_batch_size: 8
  ppo_micro_batch_size_per_gpu: 1
  use_dynamic_bsz: false

trainer:
  n_gpus_per_node: 2
  total_training_steps: 1
  save_freq: -1

audit:
  full_gradient_enabled: true
  full_gradient_direct_recompute_enabled: true
  sample_gradient_enabled: true
  sample_gradient_backward_recompute_enabled: true
  token_gradient_enabled: true
  token_gradient_top_p: 1.0
  token_gradient_backward_recompute_enabled: true
  token_gradient_backward_sync_enabled: true
  token_gradient_replica_average_enabled: false
  gradient_recompute_debug_enabled: true
  gradient_recompute_debug_max_micro_batches: 2
  gradient_recompute_debug_top_param_count: 4
```

解释：

- `full_gradient_direct_recompute_enabled=true`：domain target 直接按 domain 重算，不依赖 `final_grad - first_snapshot`。
- `token_gradient_top_p=1.0`：让 token selection 覆盖全部 valid response tokens，用来做 identity/closure test。
- `token_gradient_replica_average_enabled=false`：保留 no-replica-scale 对照，便于判断 mismatch 是否只是 replica scaling。
- `gradient_recompute_debug_top_param_count=4`：只对 top 4 个大参数记录详细 diff，降低 debug 成本。它不会改变训练或 gradient 计算，只影响 debug 记录量。

## 4. Audit Files and What They Mean

每次 run 下载后，优先看这些文件：

```text
temp/remote_<run_id>/<run_id>.log
temp/remote_<run_id>/audit/domain_step_metrics.jsonl
temp/remote_<run_id>/audit/gradient_recompute_debug.jsonl
temp/remote_<run_id>/audit/sample_grad_metrics.jsonl
temp/remote_<run_id>/audit/token_grad_metrics.jsonl
temp/remote_<run_id>/audit/training_cost.jsonl
```

文件用途：

| 文件 | 用途 |
| --- | --- |
| main log | 判断是否完成、是否 OOM/traceback、确认 Hydra config 与运行时覆盖 |
| `domain_step_metrics.jsonl` | domain loss、token count、response length、domain-level statistics |
| `gradient_recompute_debug.jsonl` | 对齐训练 inline loss、audit recompute loss、hook gradient、`.grad`、full-mask identity |
| `sample_grad_metrics.jsonl` | 每个 sample gradient 投影到 domain target 的 share/cos/norm |
| `token_grad_metrics.jsonl` | top-k/top-p selected token gradient 投影到 domain target 的 share/cos/norm |
| `training_cost.jsonl` | audit 开销、update_actor 时间等 |

## 5. Metric Definitions

设 domain target 为 `G_d`，sample 或 token selection gradient 为 `g`。

核心统计：

```text
cos(g, G_d) = <g, G_d> / (||g|| * ||G_d||)

projection_share(g -> G_d) = <g, G_d> / ||G_d||^2

norm_ratio(g -> G_d) = ||g|| / ||G_d||
```

直觉：

- `cos` 看方向是否一致。
- `projection_share` 看 `g` 沿着 domain target 方向贡献了多少 signed projection。
- `norm_ratio` 看 `g` 的长度和 target 的长度比例。
- `normalized_share` 是同一 domain 内把 sample share 归一化到和为 1，只能用于相对排序，不代表 raw contribution 一定闭合。

在 top-p=1 token identity test 中，如果 selection 覆盖所有 valid response tokens，且 target space 完全一致，理论上应满足：

```text
token_all_grad ~= G_domain
cos ~= 1
projection_share ~= 1
norm_ratio ~= 1
```

## 6. Current Run: Gradient Ratios

### 6.1 Domain Gradient

本次 domain target 指标：

| metric | value |
| --- | ---: |
| `math/full_grad/grad_norm` | 2.0994 |
| `code/full_grad/grad_norm` | 2.6813 |
| `code/math norm ratio` | 1.277 |
| `math_vs_code cosine` | -0.4016 |
| `math_to_total projection_share` | 0.3034 |
| `code_to_total projection_share` | 0.6966 |

解释：

- code domain gradient norm 比 math 大约 `1.28x`。
- total gradient 的 signed projection 中，code 约占 `69.7%`，math 约占 `30.3%`。
- `math_vs_code cosine=-0.4016`，说明两个 domain 的 gradient 方向明显冲突。

### 6.2 Domain Closure

Chosen target closure：

| metric | value |
| --- | ---: |
| `chosen_target/rel_l2` | 0.001722 |
| `chosen_target/cosine` | 0.999992 |
| `chosen_target/norm_ratio` | 1.000003 |
| `chosen_target/projection_share` | 0.999995 |

旧的 `domain_sum_vs_training`：

| metric | value |
| --- | ---: |
| `domain_sum_vs_training/rel_l2` | 0.591962 |
| `domain_sum_vs_training/cosine` | 0.913583 |
| `domain_sum_vs_training/norm_ratio` | 1.343714 |
| `domain_sum_vs_training/projection_share` | 1.227594 |

解释：

- `chosen_target` closure 说明当前选择的 domain target restore/snapshot 过程内部一致。
- `domain_sum_vs_training` 不闭合，说明 direct recompute domain target 的和与训练最终 `.grad` 仍不在完全相同的统计空间中。这个指标现在更像 diagnostic，不应单独用来否定 chosen direct target。

### 6.3 Recompute Debug

`gradient_recompute_debug.jsonl` 本次记录了 4 行，全部是 `domain=math`。关键指标：

| metric | value |
| --- | ---: |
| `loss_rel_diff` | 0 |
| `actual_vs_recompute_hook_selected_cosine` | 1.0 |
| `actual_vs_recompute_hook_selected_rel_l2` | 0 |
| `actual_vs_recompute_hook_selected_norm_ratio` | 1.0 |
| `actual_hook_vs_recompute_grad_selected_cosine` | 1.0 |
| `actual_hook_vs_recompute_grad_selected_norm_ratio` | 0.5 |
| `actual_hook_vs_recompute_grad_selected_projection_share` | 0.5 |
| `full_mask_loss_rel_diff` | 0 |
| `default_grad_vs_full_mask_grad_selected_rel_l2` | 0 |

解释：

- 训练 path 和 recompute path 在 hook gradient 层面完全一致。
- `_actor_micro_batch_loss(...)` 对 debug samples 复刻训练 loss 成功。
- full-mask identity 成功，说明 `response_mask_override=response_mask` 不改变 full loss。
- hook gradient 与 `.grad` 的比例为 `0.5`，符合 2 GPU 下 `1 / replica_count` 的 replica average scale。

限制：

- 本次 debug rows 全部是 math，没有覆盖 code。由于 code token top-p=1 仍不闭合，下一轮必须让 `gradient_recompute_debug` stratified 覆盖 code samples。

### 6.4 Sample Gradient

sample gradient 每个 sample 单独重算 all-token loss gradient，然后投影到对应 `G_domain`。

汇总：

| domain | sample norm mean | projection_share_sum | raw_projection_share_sum | normalized_sum | valid_frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| math | 2.2125 | 1.3484 | 2.6968 | 1.0 | 1.0 |
| code | 2.7573 | 1.0177 | 2.0354 | 1.0 | 1.0 |

解释：

- sample recompute 可用率是 100%。
- code sample share 基本闭合，scaled share sum 为 `1.0177`。
- math sample share 明显偏高，scaled share sum 为 `1.3484`。
- `raw_projection_share_sum` 约等于 scaled share 的 2 倍，符合 `replica_count=2`。
- `normalized_sum=1.0` 是归一化构造出来的，不能作为 gradient closure 成功的证据。

当前使用建议：

```text
sample_projection_share_normalized:
  可以用于 domain 内样本相对排序。

sample_projection_share / raw_projection_share:
  暂时不要解释为真实 contribution ratio，尤其是 math。
```

### 6.5 Token Gradient

token gradient 有两组主要 selection：

```text
top50_loss_abs
topp100_loss_abs_mass   # token_gradient_top_p=1.0，覆盖全部 valid response tokens
```

Top-50：

| domain | share | no-replica share | cos | no-replica cos | norm_ratio | no-replica norm_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math top50 | 0.0959 | 0.1918 | 0.4927 | 0.6967 | 0.1947 | 0.2753 |
| code top50 | 0.1405 | 0.2811 | 0.5027 | 0.7110 | 0.2795 | 0.3953 |

Top-p=1 identity test：

| domain | selected tokens | selected all tokens | share | no-replica share | cos | no-replica cos | norm_ratio | no-replica norm_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 6526 | 1.0 | 0.5001 | 1.0001 | 0.7071 | 1.0001 | 0.7071 | 1.0001 |
| code | 3743 | 1.0 | 0.3821 | 0.7643 | 0.5783 | 0.8178 | 0.6608 | 0.9345 |

解释：

- math top-p=1 在 no-replica-scale 下闭合，说明 token gradient 计算逻辑在 math 上是可用的。
- code top-p=1 覆盖了全部 tokens 和 samples，但 no-replica-scale 后仍只有 `0.7643` share、`0.8178` cosine、`0.9345` norm ratio。这个 residual mismatch 需要继续查。
- top50 是 token attribution 指标，不应该期望 close to 1；top-p=1 才是 closure test。

## 7. How To Analyze A New Run

### Step 1: Confirm Run Health

先看主日志：

```bash
rg -n "Traceback|OutOfMemory|OOM|error|Step|update_actor|total_training_steps" temp/remote_<run_id>/<run_id>.log
```

判断：

- 有 traceback/OOM：先处理运行稳定性，后续 gradient 统计不可信。
- 1 step 没跑完：不要分析 closure。
- `update_actor` 极慢或 GPU memory 接近上限：后续 debug 可能受 memory pressure 影响。

### Step 2: Confirm Config

检查 config 是否真的启用了 diagnostic flags：

```bash
rg -n "full_gradient_direct_recompute_enabled|token_gradient_top_p|token_gradient_replica_average_enabled|gradient_recompute_debug" temp/remote_<run_id>/<run_id>.log
```

必须确认：

```text
full_gradient_direct_recompute_enabled=true
token_gradient_top_p=1.0
token_gradient_replica_average_enabled=false
gradient_recompute_debug_enabled=true
use_dynamic_bsz=false
```

否则不能拿该 run 做 gradient consistency 结论。

### Step 3: Inspect Domain Target

重点看：

```text
global/full_grad_closure/chosen_target/*
global/full_grad_closure/domain_sum_vs_training/*
global/audit/full_gradient_replica_count
math/full_grad/grad_norm
code/full_grad/grad_norm
global/full_grad_conflict/math_vs_code/*
global/full_grad_contribution/*_to_total/*
```

判断标准：

```text
chosen_target/cosine ~= 1
chosen_target/norm_ratio ~= 1
chosen_target/projection_share ~= 1
chosen_target/rel_l2 <= 1e-3 ~ 1e-2
```

如果 `chosen_target` 不闭合：

- 先查 target restore/snapshot 逻辑。
- 后续 sample/token projection 都不能信。

如果只有 `domain_sum_vs_training` 不闭合：

- 目前不能直接判定 sample/token 逻辑错。
- 这可能反映 direct recompute target 与训练最终 `.grad` 的 target space 不完全一致。

### Step 4: Inspect Recompute Debug

看 `gradient_recompute_debug.jsonl`：

```text
loss_rel_diff
actual_vs_recompute_hook_selected_cosine
actual_vs_recompute_hook_selected_rel_l2
actual_vs_recompute_hook_selected_norm_ratio
actual_hook_vs_recompute_grad_selected_norm_ratio
actual_hook_vs_recompute_grad_selected_projection_share
full_mask_loss_rel_diff
default_grad_vs_full_mask_grad_selected_rel_l2
domain
```

判断树：

```text
loss_rel_diff != 0
  -> loss builder / mask / scale / top-k context 可能不一致。

actual_vs_recompute_hook_selected_rel_l2 != 0
  -> backward path 或 hook 捕获逻辑不一致。

actual_hook_vs_recompute_grad_selected_norm_ratio ~= 1 / replica_count
  -> 是 replica scaling，不是 loss 复刻错误。

full_mask_loss_rel_diff != 0
  -> response_mask_override 或 full-mask identity 有问题。

debug rows 只覆盖某一个 domain
  -> 不能用于解释另一个 domain 的 residual mismatch。
```

### Step 5: Inspect Token Top-p=1

看 `token_grad_metrics.jsonl` 中：

```text
selection = topp100_loss_abs_mass
closure_selected_all_tokens
closure_selected_all_samples
*_projection_share
*_projection_share_no_replica_scale
*_cos
*_cos_no_replica_scale
*_norm_ratio
*_norm_ratio_no_replica_scale
token_grad_available
token_grad_autograd_error
token_grad_restore_*_rel_l2
```

判断标准：

```text
closure_selected_all_tokens = 1
closure_selected_all_samples = 1
token_grad_available = 1
token_grad_autograd_error = null
restore rel_l2 = 0
```

如果上述条件满足，则 top-p=1 是有效 closure test。

理想闭合：

```text
*_projection_share_no_replica_scale ~= 1
*_cos_no_replica_scale ~= 1
*_norm_ratio_no_replica_scale ~= 1
```

如果 raw 指标约为：

```text
projection_share ~= 0.5
norm_ratio ~= 0.7071
```

而 no-replica-scale 指标约为 1，则问题基本是 `replica_count=2` 的 scaling convention。

如果 no-replica-scale 仍明显偏离 1，例如本次 code：

```text
projection_share_no_replica_scale = 0.7643
cos_no_replica_scale = 0.8178
norm_ratio_no_replica_scale = 0.9345
```

则说明还有额外 mismatch，需要继续查 target construction、selection aggregation 或 code-specific recompute coverage。

### Step 6: Inspect Sample Gradient

看 `sample_grad_metrics.jsonl`：

```text
sample_recompute_available
sample_recompute_autograd_error
sample_recompute_replica_count
sample_projection_share
sample_projection_share_raw
sample_projection_share_normalized
sample_to_domain_cos
sample_recompute_restore_*_rel_l2
```

判断：

- `sample_projection_share_normalized` 只适合相对排序。
- `sum(sample_projection_share)` 如果接近 1，说明 scaled sample shares 与 domain target 大致闭合。
- `sum(sample_projection_share_raw)` 在 2 GPU 下可能约为 scaled share 的 2 倍。
- 如果 normalized sum 为 1，不代表 raw share 正确，因为 normalized 是后处理。

本次：

```text
math scaled share sum = 1.3484   # 偏高
code scaled share sum = 1.0177   # 接近闭合
```

所以 sample raw contribution 仍需谨慎解释。

## 8. Current Failure Hypotheses

### Hypothesis A: Replica Scaling

证据强：

```text
actual_hook_vs_recompute_grad_selected_norm_ratio = 0.5
actual_hook_vs_recompute_grad_selected_projection_share = 0.5
replica_count = 2
math top-p=1 no-replica share ~= 1
```

结论：

replica scaling 是已经确认的问题来源之一，尤其解释 math 的 raw `0.5 / 0.7071`。

### Hypothesis B: Loss Builder Mismatch

当前证据不支持它作为主因：

```text
loss_rel_diff = 0
actual_vs_recompute_hook_selected_rel_l2 = 0
full_mask_loss_rel_diff = 0
```

限制：

debug rows 当前只覆盖 math，没有覆盖 code。因此不能完全排除 code-specific loss/recompute mismatch。

### Hypothesis C: Code Domain Target Or Aggregation Mismatch

当前最需要查：

```text
code top-p=1 selected_all_tokens = 1
code top-p=1 selected_all_samples = 1
code no-replica share = 0.7643
code no-replica cos = 0.8178
code no-replica norm_ratio = 0.9345
```

如果 token top-p=1 已覆盖全部 code tokens，却仍不等于 `G_code`，可能原因包括：

1. code domain direct target construction 和 token selected recompute 不在同一个 target space。
2. code samples 没有被 `gradient_recompute_debug` 覆盖，仍有 code-specific loss/mask/scale 差异未被证明。
3. token aggregation 的 per-rank / all-reduce / candidate scale 对 code path 有额外偏差。
4. domain target 的 norm/dot 汇总与 token gradient 的 norm/dot 汇总使用了不同 replica convention。

### Hypothesis D: Sample Gradient Raw Share Not Fully Closed

证据：

```text
math sample scaled share sum = 1.3484
code sample scaled share sum = 1.0177
```

解释：

sample gradient 和 token gradient 的问题可能部分同源，都依赖 target space / replica convention 对齐。code sample 基本闭合，但 math sample 偏高；这和 math token top-p=1 已闭合并不完全一致，因此 sample 还有额外 aggregation 或 per-sample summation 差异需要独立确认。

## 9. Acceptance Criteria

后续修复后，至少要满足以下标准。

### 9.1 Recompute Debug Closure

对每个 domain 至少覆盖 2 个 samples：

```text
loss_rel_diff <= 1e-8
actual_vs_recompute_hook_selected_rel_l2 <= 1e-6 ~ 1e-4
actual_vs_recompute_hook_selected_cosine ~= 1
full_mask_loss_rel_diff <= 1e-8
default_grad_vs_full_mask_grad_selected_rel_l2 <= 1e-6 ~ 1e-4
```

如果使用 bf16 storage，极小误差可以放宽，但不能出现系统性偏差。

### 9.2 Token Top-p=1 Closure

对 math 和 code 都需要：

```text
closure_selected_all_tokens = 1
closure_selected_all_samples = 1
token_grad_available = 1
*_projection_share_no_replica_scale ~= 1
*_cos_no_replica_scale ~= 1
*_norm_ratio_no_replica_scale ~= 1
```

建议阈值：

```text
abs(projection_share_no_replica_scale - 1) <= 1e-2
abs(norm_ratio_no_replica_scale - 1) <= 1e-2
cos_no_replica_scale >= 0.99
```

### 9.3 Sample Share Closure

如果目标是把 sample raw share 解释为真实 contribution ratio，则每个 domain 需要：

```text
sum(sample_projection_share) ~= 1
```

建议阈值：

```text
abs(sum(sample_projection_share) - 1) <= 0.02
```

否则只能使用：

```text
sample_projection_share_normalized
```

作为相对排序。

### 9.4 Domain Target Closure

至少需要：

```text
chosen_target/cosine ~= 1
chosen_target/norm_ratio ~= 1
chosen_target/projection_share ~= 1
chosen_target/rel_l2 small
```

`domain_sum_vs_training` 是否必须闭合，取决于后续是否决定把训练最终 `.grad` 作为 canonical target。如果 canonical target 改成 direct recompute domain target，则 `domain_sum_vs_training` 是额外 diagnostic，不是 blocking criterion。

## 10. Practical Parser Script

下面这个脚本可用于快速汇总关键 gradient 指标。把 `AUDIT_DIR` 改成对应 run 的 audit 目录即可。

```bash
AUDIT_DIR=temp/remote_grad_diagnostic_smoke_20260626_203323/audit
export AUDIT_DIR
python - <<'PY'
import json
import os
from pathlib import Path

audit_dir = Path(os.environ.get("AUDIT_DIR", "temp/remote_grad_diagnostic_smoke_20260626_203323/audit"))

def rows(name):
    path = audit_dir / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

print("== Recompute Debug ==")
debug = rows("gradient_recompute_debug.jsonl")
print("rows:", len(debug))
for key in [
    "loss_rel_diff",
    "actual_vs_recompute_hook_selected_cosine",
    "actual_vs_recompute_hook_selected_rel_l2",
    "actual_hook_vs_recompute_grad_selected_norm_ratio",
    "actual_hook_vs_recompute_grad_selected_projection_share",
    "full_mask_loss_rel_diff",
    "default_grad_vs_full_mask_grad_selected_rel_l2",
]:
    vals = [r.get(key) for r in debug if r.get(key) is not None]
    if vals:
        print(key, "min=", min(vals), "max=", max(vals))
print("debug domains:", sorted({r.get("domain") for r in debug}))

print("\n== Token Top-p=1 ==")
for r in rows("token_grad_metrics.jsonl"):
    if r.get("selection") != "topp100_loss_abs_mass":
        continue
    d = r.get("domain")
    print(
        d,
        "tokens=", r.get("selected_token_count"),
        "all_tokens=", r.get("closure_selected_all_tokens"),
        "share=", r.get(f"{d}_projection_share"),
        "share_no_replica=", r.get(f"{d}_projection_share_no_replica_scale"),
        "cos_no_replica=", r.get(f"{d}_cos_no_replica_scale"),
        "norm_no_replica=", r.get(f"{d}_norm_ratio_no_replica_scale"),
    )

print("\n== Sample Share Sums ==")
by_domain = {}
for r in rows("sample_grad_metrics.jsonl"):
    d = r.get("domain")
    by_domain.setdefault(d, {"share": 0.0, "raw": 0.0, "norm": 0.0, "n": 0})
    by_domain[d]["share"] += float(r.get("sample_projection_share") or 0.0)
    by_domain[d]["raw"] += float(r.get("sample_projection_share_raw") or 0.0)
    by_domain[d]["norm"] += float(r.get("sample_projection_share_normalized") or 0.0)
    by_domain[d]["n"] += 1
for d, v in sorted(by_domain.items()):
    print(d, v)
PY
```

## 11. Recommended Next Debug Actions

优先级从高到低：

1. 让 `gradient_recompute_debug` stratified 覆盖每个 domain，尤其 code。当前 debug rows 全部是 math，无法解释 code residual mismatch。
2. 对 code top-p=1 记录 selected token gradient 与 direct `G_code` 的 top-param diff，类似 `actual_vs_recompute_hook_selected_param_diffs`。
3. 明确 canonical target space：后续是以 direct recompute domain target 为准，还是以训练最终 `.grad` 为准。两者不要混在同一个 projection share 解释里。
4. 对 sample gradient 增加 domain-level sum closure 日志，直接记录 `sum(sample_grad) vs G_domain` 的 cos/rel_l2/norm/share，而不仅是逐 sample projection share。
5. 如果 code top-p=1 仍不闭合，逐项打印 code micro-batch 的 loss scalar、loss scale factor、selected token count、rank token split、grad norm、per-parameter dot/norm。

## 12. Final Interpretation For Handoff

目前不是“gradient audit 已完全正确”的状态，而是“关键问题已经被缩小”的状态。

已经排除或基本排除：

```text
- math debug samples 上的 loss builder mismatch；
- full-mask override mismatch；
- hook-level recompute backward mismatch；
- FSDP 状态破坏导致的第二步崩溃；
- math top-p=1 的主要 replica scaling 偏差。
```

仍然开放：

```text
- code top-p=1 为什么 no-replica-scale 后仍不闭合；
- sample raw projection share 为什么 math 偏高；
- direct domain target 与 training final .grad 是否需要统一为一个 canonical target space；
- gradient_recompute_debug 是否需要按 domain 分层采样。
```

给后续分析者的核心提醒：

```text
不要直接把 raw projection_share 当真实 contribution ratio。
先看 top-p=1 closure，再看 no-replica-scale 指标。
先确认 loss/hook/full-mask identity，再讨论 sample/token attribution。
math 当前基本闭合，code 是下一步 debug 的主线。
```
