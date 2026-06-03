# MOPD Metrics 最新说明

本文档只描述当前代码实际计算和输出的 metrics。低成本 `grad` / `grad_anchor` / `grad_conflict` proxy 已删除；gradient 相关诊断只保留 full-parameter gradient。

重要边界：当前实现里的“full training gradient”指**当前训练 step 的完整 training batch**，并按 train domain 分组 backward。正式配置中 `full_gradient_train_max_samples_per_domain: null` 表示对当前 batch 内该 domain 的样本不截断。它不是“每个 step 扫完整 train parquet / 整个训练集”，后者成本接近每 step 额外跑一个 epoch，不适合作为默认训练期 audit。

核心实现位置：

- `code/mopd_verl/verl_audit.py`：每个 training step 的 loss / teacher / calibration / coverage audit。
- `code/mopd_verl/full_gradient_worker.py`：full-parameter train gradient、validation gradient anchor、domain conflict。
- `code/mopd_verl/audit_validation.py`：full-gradient validation anchor 的调度标记。
- `code/mopd_verl/audit_scalar_logging.py`：validation gain 与 cost metrics。
- `code/mopd_verl/tensorboard_filter.py`：TensorBoard core 过滤规则。

## TensorBoard 层级

一级层级直接是 domain 名或 `global`：

```text
<train_domain>/<category>/<metric>
<train_domain>/<category>/<validation_domain>/<metric>
global/<category>/<metric>
global/<category>/<domain_i>_vs_<domain_k>/<metric>
```

示例：

```text
math/loss/token_opd_loss_mean
math/full_grad/grad_norm
math/full_grad_anchor/AIME2024/full_grad_cosine_i_j
math/full_grad_anchor/AIME2024/predicted_val_opd_loss_delta_i_j
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/cost/step_seconds
```

## Domain Data

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/data/domain_sample_count` | 当前 training batch 中属于该 train domain 的样本数。 | 按 `opd_teacher/domain/source_domain/ability` 解析 domain label 后计数。 |
| `<domain>/data/domain_token_count` | 当前 training batch 中该 domain 的 response token 数。 | 对该 domain 样本的 `response_mask` 求和。 |
| `<domain>/data/domain_token_frac` | 该 domain token 占整个 batch token 的比例。 | `domain_token_count / total_tokens`。 |
| `global/data/total_samples` | 当前 training batch 总样本数。 | `batch_size`。 |
| `global/data/total_tokens` | 当前 training batch 总 response token 数。 | `response_mask.sum()`。 |
| `global/data/domain_mix_entropy` | batch 内 domain token 分布熵。 | `-sum_i p_i log(p_i)`，其中 `p_i=domain_token_frac_i`。 |

## Loss Metrics

每个样本先计算 token-level OPD signal：

```text
if lambda_vals == 1:
    reverse_kl = old_log_prob - teacher_log_prob
else:
    reverse_kl = old_log_prob - base_log_prob
                 - lambda_vals * (teacher_log_prob - base_log_prob)
```

然后分别在 token-level 和 sample-level 做统计，且两类统计都会输出 global 与 per-domain 指标：

- token-level：把当前 scope 内所有有效 response tokens 的 `reverse_kl` 展平后计算 mean / std / variance。
- sample-level：每个样本先累加 `sum(reverse_kl * response_mask)` 得到一个 sample OPD loss，再在当前 scope 的样本维度计算 mean / std / variance。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/loss/token_opd_loss_mean` | 当前 domain 内 token-level OPD loss 均值。 | 对该 domain 的全部有效 response tokens 计算 `mean(reverse_kl)`。 |
| `<domain>/loss/token_opd_loss_std` | 当前 domain 内 token-level OPD loss 标准差。 | 对该 domain 的全部有效 response tokens 计算 `std(reverse_kl)`。 |
| `<domain>/loss/token_opd_loss_variance` | 当前 domain 内 token-level OPD loss 方差。 | 对该 domain 的全部有效 response tokens 计算 `mean(reverse_kl^2)-mean(reverse_kl)^2`。 |
| `<domain>/loss/sample_opd_loss_mean` | 当前 domain 内 sample-level OPD loss 均值。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对该 domain 样本求 mean。 |
| `<domain>/loss/sample_opd_loss_std` | 当前 domain 内 sample-level OPD loss 标准差。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对该 domain 样本求 std。 |
| `<domain>/loss/sample_opd_loss_variance` | 当前 domain 内 sample-level OPD loss 方差。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对该 domain 样本求 variance。 |
| `global/loss/token_opd_loss_mean` | 当前 batch 全局 token-level OPD loss 均值。 | 对全 batch 所有有效 response tokens 计算 `mean(reverse_kl)`。 |
| `global/loss/token_opd_loss_std` | 当前 batch 全局 token-level OPD loss 标准差。 | 对全 batch 所有有效 response tokens 计算 `std(reverse_kl)`。 |
| `global/loss/token_opd_loss_variance` | 当前 batch 全局 token-level OPD loss 方差。 | 对全 batch 所有有效 response tokens 计算 variance。 |
| `global/loss/sample_opd_loss_mean` | 当前 batch 全局 sample-level OPD loss 均值。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对全 batch 样本求 mean。 |
| `global/loss/sample_opd_loss_std` | 当前 batch 全局 sample-level OPD loss 标准差。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对全 batch 样本求 std。 |
| `global/loss/sample_opd_loss_variance` | 当前 batch 全局 sample-level OPD loss 方差。 | 每个样本先算 `sum(reverse_kl * response_mask)`，再对全 batch 样本求 variance。 |
| `<domain>/loss/high_variance_sample_rate` | 该 domain 中 token-level loss 波动较大的样本比例。 | 统计 `sample_loss_cv > high_variance_cv_threshold` 的比例。 |
| `<domain>/loss/advantage_mean` | 当前 actor update 使用的 advantage 均值 sanity check。 | 若 batch 有 `advantages`，对其做 masked mean；否则使用 `-reverse_kl`。 |

## Training Reward Metrics

这组 metric 来自当前 training batch 的 `token_level_scores`。如果 batch 中没有 `token_level_scores`，这些 metric 不会写入 TensorBoard。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/reward/training_reward_mean` | 当前 train domain 的 training reward 均值。 | 每个样本先算 `sum(token_level_scores * response_mask)`，再对 domain 内样本求均值。 |
| `<domain>/reward/training_accuracy` | 当前 train domain 中 reward 为正的样本比例。 | `mean(training_reward > 0)`。 |

## Full-Parameter Gradient Metrics

这组是真实 actor 参数梯度。worker 会对当前 `DataProto` 按 domain 分组，分别 backward，收集完整 actor 参数 `.grad`，再计算 norm / cosine / 一阶 validation OPD surrogate objective delta。

full-gradient audit 使用与 actor update 更接近的 PPO policy-loss path：先按 MOPD teacher 构造 `advantages=-reverse_kl`，再调用 verl 的 `get_policy_loss_fn()`，并纳入 entropy / KL loss 系数非零时的项。因此它不再是手写的 `reverse_kl * ratio` proxy。不过它仍然是**一阶梯度诊断**，不是临时执行 Adam/optimizer step 后的真实 validation delta。

正式配置：

```yaml
full_gradient_enabled: true
full_gradient_freq_steps: 1
full_gradient_train_max_samples_per_domain: null
full_gradient_validation_max_samples_per_domain: null
full_gradient_micro_batch_size_per_gpu: 1
```

其中 `null` 表示不截断：

- train mode：使用当前 training step batch 内该 domain 的全部样本。
- validation-anchor mode：在同一次 validation step 内，对 validation batches 持续累计该 validation domain 的全部样本，并按 response token count 维护 token-weighted running mean gradient。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<train_domain>/full_grad/grad_norm` | 当前 step 该 train domain 的真实参数梯度范数。 | 对该 domain 的完整当前 batch subset backward，收集 actor 参数梯度，计算 `||g_train_i||_2`。 |
| `<train_domain>/full_grad/sample_count` | 该 train domain full-gradient backward 使用的样本数。 | 当前 DataProto 中该 domain 样本数，经 distributed sum。 |
| `<validation_domain>/full_grad_anchor/validation_anchor_sample_count` | validation gradient anchor 累计样本数。 | 同一 validation step 内，该 validation domain 已用于 anchor backward 的样本数。 |
| `<validation_domain>/full_grad_anchor/validation_anchor_token_count` | validation gradient anchor 累计 response token 数。 | 同一 validation step 内，该 validation domain 已用于 anchor backward 的 response token 数，经 distributed sum。 |
| `<validation_domain>/full_grad_anchor/validation_anchor_grad_norm` | validation gradient anchor 范数。 | `||g_val_j||_2`，其中 `g_val_j` 是按 response token count 维护的 running mean gradient，不再是各 validation batch mean gradient 的直接求和。 |
| `<train_domain>/full_grad_anchor/<validation_domain>/full_grad_cosine_i_j` | train gradient 与 validation gradient 的方向一致性。 | `(g_train_i · g_val_j) / (||g_train_i||_2 ||g_val_j||_2)`。 |
| `<train_domain>/full_grad_anchor/<validation_domain>/predicted_val_opd_loss_delta_i_j` | train domain 一步 SGD 对 validation OPD surrogate objective 的一阶预测变化。 | `-learning_rate * (g_train_i · g_val_j)`；负值表示预测 validation OPD surrogate objective 下降。该值不包含 Adam preconditioning、gradient clipping、实际 optimizer state 等二阶/优化器效应。 |
| `global/full_grad_conflict/<domain_i>_vs_<domain_k>/full_grad_cosine_train_i_k` | 两个 train domains 的真实参数梯度方向相似度。 | `cosine(g_train_i, g_train_k)`。 |
| `global/full_grad_conflict/<domain_i>_vs_<domain_k>/conflict_magnitude_i_k` | 真实参数梯度冲突强度。 | `max(0, -full_grad_cosine_train_i_k)`。 |
| `global/full_grad_cost/backward_seconds` | full-gradient audit 的额外 backward 墙钟耗时。 | worker 内 `time.perf_counter()` 差值。 |
| `global/full_grad_cost/max_memory_allocated_gb` | full-gradient audit 后的 CUDA peak allocated memory。 | `torch.cuda.max_memory_allocated() / 1024^3`。 |

已删除：所有 `grad/*`、`grad_anchor/*`、`grad_conflict/*` proxy metrics。

## Teacher / Calibration / Coverage

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/teacher/teacher_student_gap_mean` | teacher 与 student chosen-token log-prob 的平均差距。 | 对每个样本算 `masked_mean(teacher_log_prob - old_log_prob)`，再对 domain 内样本求均值。 |
| `<domain>/teacher/teacher_confidence_mean` | teacher chosen-token confidence proxy。 | 对每个样本算 `exp(masked_mean(teacher_log_prob))` 并 clip 到 `[0,1]`，再求均值。 |
| `<domain>/calibration/calibration_error` | ECE-style calibration proxy。 | confidence 使用 teacher confidence；correctness 来自 `token_level_scores` 是否为正；按 `calibration_bins` 分桶后计算 weighted absolute gap。若没有 `token_level_scores`，该指标为空。 |
| `<domain>/coverage/duplicate_rate` | 当前 domain batch 中重复 sample_id 比例。 | 维护每个 domain 已见 `sample_id` 集合，统计当前 batch 中已经出现过的比例。 |

## Validation Gain Metrics

原始 validation 分数仍由 verl 原生 metrics 写入 TensorBoard，例如 `val-core/*`。MOPD audit 只记录相邻两次 validation 的 gain。

TensorBoard tag 的一级层级取决于 validation metric key 能否解析出配置里的 domain 名：

- 如果 key 形如 `val/math/score` 或包含配置 domain `math` / `code`，tag 会写成 `<domain>/validation_gain/...`。
- 如果 key 形如 `val-core/AIME2024/reward/mean@1`，其中 `AIME2024` 不是配置里的 train domain，则 tag 会写成 `global/validation_gain/val-core_AIME2024_reward_mean_1`。也就是说，benchmark dataset 名会被折叠进 metric 名，而不是成为一级层级。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain_or_global>/validation_gain/<metric>` | 当前 validation 分数相对上一次 validation 的变化。 | `current_metric - previous_metric`。第一次 validation 没有 previous，因此没有 gain。 |
| `<domain_or_global>/validation_gain_stats/<metric>/mean` | 最近窗口内 validation gain 均值。 | 对最近 `tier2_window_size` 个 gain 求均值。 |
| `<domain_or_global>/validation_gain_stats/<metric>/variance` | 最近窗口内 validation gain 方差。 | 对最近 `tier2_window_size` 个 gain 求 `np.var`。 |

## Cost Metrics

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `global/cost/step_seconds` | 一个训练 step 的耗时。 | 优先读取 `timing_s/step`，否则读取 `perf/time_per_step`。 |
| `global/cost/gpu_seconds_step` | 当前 step 消耗的 GPU 秒。 | `step_seconds * n_gpus`。 |
| `global/cost/tokens_per_second` | training throughput。 | `perf/total_num_tokens / step_seconds`。 |
| `global/cost/memory_peak_step` | 当前 step 的 peak allocated memory。 | 优先读取 `perf/max_memory_allocated_gb`，否则读取 `perf/max_memory_reserved_gb`。 |

## JSONL 输出

当前保留的 audit JSONL 文件：

| 文件 | 内容 |
| --- | --- |
| `domain_step_metrics.jsonl` | 每 step、每 train domain 的核心 data/loss/teacher/calibration/coverage rows。 |
| `loss_variance_domain_step.jsonl` | 每 step、每 train domain 的 loss variance 摘要。 |
| `loss_variance_sample.jsonl` | 每 step、每 train domain 最多 `max_samples_per_domain` 条样本级 `opd_loss`、`sample_token_opd_loss_mean` 和 `sample_token_opd_loss_variance`。 |
| `validation_probe.jsonl` | 原始 validation value、previous value、gain，用于复盘 validation gain。 |
| `validation_gain_variance.jsonl` | validation gain history、mean、variance。 |
| `training_cost.jsonl` | step seconds、GPU seconds、tokens/sec、peak memory。 |
| `audit_errors.jsonl` | audit 过程中发生异常时的防御性错误记录。 |

已删除的 JSONL：`validation_anchor.jsonl`、`gradient_anchor_alignment.jsonl`、`domain_conflict.jsonl`、`trend_stability.jsonl`、`gradient_noise.jsonl`、`rank_stability.jsonl`、`teacher_logits_reliability.jsonl`、`calibration.jsonl`、`sample_influence.jsonl`、`coverage_diversity.jsonl`、`shadow_probe.jsonl`。

## 当前限制

当前训练期 full-gradient audit 的“全量”是当前 step batch 级别，不是完整训练集级别。如果要计算完整训练集 gradient，需要单独实现离线全数据 dataloader backward，并明确评估频率；不建议每个训练 step 都做。

validation anchor 是 full-parameter gradient anchor，不再是 token-level proxy anchor。它会在一次 validation pass 内累计 validation batches，并按 response token count 形成 running mean；如果 validation dataloader 没有覆盖完整 validation set，则 anchor 也只覆盖该 pass 实际看到的数据。
