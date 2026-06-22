# MOPD Metrics 最新说明

本文档只描述当前代码实际计算和输出的 metrics。低成本 gradient proxy 和 validation-gradient anchor 已删除；gradient 相关诊断只保留 train-side full-parameter gradient。

重要边界：当前实现里的“full training gradient”指**真实 actor update mini-batch 内按 train domain 分解出来的完整 actor 参数梯度**。train-side tracker 复用 `dp_actor.update_policy()` 的真实 `loss.backward()` 累积梯度，不为 train domains 额外 recompute forward/backward；正式配置中 `full_gradient_train_max_samples_per_domain: null` 表示对当前 mini-batch 内该 domain 的样本不截断。它不是“每个 step 扫完整 train parquet / 整个训练集”，后者成本接近每 step 额外跑一个 epoch，不适合作为默认训练期 audit。

核心实现位置：

- `code/mopd_verl/verl_audit.py`：每个 training step 的 loss / teacher / calibration / coverage audit。
- `code/mopd_verl/full_gradient_worker.py`：full-parameter train gradient、sample gradient、domain conflict。
- `code/mopd_verl/audit_scalar_logging.py`：validation gain 与 cost metrics。
- `code/mopd_verl/tensorboard_filter.py`：TensorBoard core 过滤规则。

## TensorBoard 层级

一级层级直接是 domain 名或 `global`：

```text
<train_domain>/<category>/<metric>
global/<category>/<metric>
global/<category>/<domain_i>_vs_<domain_k>/<metric>
global/<category>/<domain_i>_vs_total/<metric>
global/<category>/<domain_i>_to_total/<metric>
```

示例：

```text
math/loss/token_opd_loss_mean
math/advantage/positive_frac
math/length/response_mean
math/full_grad/grad_norm
math/sample_grad/norm_mean
math/sample_grad_cos/domain_cos_mean
math/sample_grad_contribution/projection_share_mean
math/token_conflict/proxy_mass
global/full_grad_conflict/math_vs_code/full_grad_cosine_train_i_k
global/full_grad_alignment/math_vs_total/full_grad_cosine_domain_total
global/full_grad_contribution/math_to_total/signed_projection_share
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
| `<domain>/loss/advantage_mean` | audit batch 中 sample-level advantage 均值 sanity check。 | 若 batch 有 `advantages`，每个样本先算 `masked_mean(advantages, response_mask)`，再对 domain 内样本求 mean；否则回退为 `masked_mean(-reverse_kl, response_mask)`。 |

## Gap / Entropy Distribution Metrics

这组 metric 来自 `verl_audit.py`，按当前 training batch 的 train domain 统计有效 response token 分布，并把 raw vector 写入 JSONL，方便后续离线画图。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/token_gap/gap_signed_*` | teacher chosen-token log-prob 与 student/old log-prob 的 signed gap 分布。 | 对该 domain 的有效 response token occurrence 计算 `gap_signed = teacher_logp - student_logp`，输出 mean/std/p05/p50/p95/max/sum；raw occurrence vector 写入 `token_gap_vectors.jsonl` 的 `gap_vector_domain`。 |
| `<domain>/token_gap/gap_abs_*` | signed gap 的绝对值分布。 | `gap_abs = abs(teacher_logp - student_logp)`，输出同一组分布统计；raw occurrence vector 写入 `gap_abs_vector_domain`。 |

如果开启 `token_gap_vocab_vector_enabled=true`，还会写 `token_gap_vocab_vectors.jsonl`。该文件是全词表 dense vector 口径：第 `v` 维对应 token id `v`，长度来自 tokenizer vocab size、`token_gap_vocab_size` 配置，或无 tokenizer 时的当前 batch 最大 token id + 1。每行包含：

- `token_count_vector_vocab`: 当前 step/domain 中每个 token id 的 occurrence count。
- `gap_signed_sum_vector_vocab`: 每个 token id 的 `teacher_logp - student_logp` 总和。
- `gap_abs_sum_vector_vocab`: 每个 token id 的 `abs(teacher_logp - student_logp)` 总和。
- `gap_signed_mean_vector_vocab` / `gap_abs_mean_vector_vocab`: 对 count 非零 token 取 mean，count 为 0 的维度为 0。
- `nonzero_token_ids`: 当前 step/domain 中实际出现过的 token id，方便离线稀疏读取。

同时会输出 domain-pair scalar：

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `global/token_gap_vocab_cosine/<domain_i>_vs_<domain_j>/gap_signed_sum_cosine` | 两个 domain 的全词表 signed gap sum vector cosine。 | 对两个 domain 的 `gap_signed_sum_vector_vocab` 计算 cosine；任一 vector 为零向量时不写该指标。 |
| `global/token_gap_vocab_cosine/<domain_i>_vs_<domain_j>/gap_abs_sum_cosine` | 两个 domain 的全词表 abs gap sum vector cosine。 | 对两个 domain 的 `gap_abs_sum_vector_vocab` 计算 cosine；任一 vector 为零向量时不写该指标。 |
| `<domain>/entropy/sum_teacher_entropy` | 当前 domain teacher entropy 总和。 | `sum_t H(p_teacher(. | state_t))`，只统计 `response_mask > 0` 的 token。 |
| `<domain>/entropy/sum_student_entropy` | 当前 domain student entropy 总和。 | `sum_t H(p_student(. | state_t))`，来自 actor log-prob/entropy pass。 |
| `<domain>/entropy/sum_teacher_student_cross_entropy` | 当前 domain teacher-student cross entropy 总和。 | `sum_t H(p_teacher, p_student)`。开启 top-k distill 时，这个值使用 teacher top-k local support 的重归一化分布，不是 full-vocab CE。 |
| `<domain>/entropy/teacher_entropy_*` | teacher entropy 分布。 | 对 token-level teacher entropy vector 输出 mean/std/p05/p50/p95/max/sum；raw vector 写入 `entropy_distribution_vectors.jsonl`。 |
| `<domain>/entropy/student_entropy_*` | student entropy 分布。 | 对 token-level student entropy vector 输出同一组统计。 |
| `<domain>/entropy/teacher_student_cross_entropy_*` | teacher-student cross entropy 分布。 | 对 token-level CE vector 输出同一组统计；top-k distill 下为 local support CE。 |

如果开启 `entropy_vocab_vector_enabled=true`，还会写 `entropy_vocab_vectors.jsonl`，用同一套 token-id 坐标统计 student entropy 和 teacher-student cross entropy 的全词表 dense vector：

- `student_entropy_sum_vector_vocab`: 每个 token id 的 `H(p_student)` 总和。
- `student_entropy_mean_vector_vocab`: 每个 token id 的 `H(p_student)` 均值，count 为 0 的维度为 0。
- `teacher_student_cross_entropy_sum_vector_vocab`: 每个 token id 的 `H(p_teacher, p_student)` 总和。
- `teacher_student_cross_entropy_mean_vector_vocab`: 每个 token id 的 `H(p_teacher, p_student)` 均值，count 为 0 的维度为 0。
- `token_count_vector_vocab` / `nonzero_token_ids`: 与 `token_gap_vocab_vectors.jsonl` 相同，记录 token occurrence count 和非零 token id。

同时会输出 `global/entropy_vocab_cosine/<domain_i>_vs_<domain_j>/student_entropy_sum_cosine` 和 `global/entropy_vocab_cosine/<domain_i>_vs_<domain_j>/teacher_student_cross_entropy_sum_cosine`，用于比较两个 domain 的 entropy / CE mass 是否集中在同一批 token id 上。

## Advantage Metrics

这组 metric 描述当前 training batch 里 advantage 的符号分布。它不是 loss，也不是 reward 本身；它反映的是 batch 中有多少样本在当前 advantage 定义下是正向训练信号。

当前 `verl_audit.py` 的口径是：如果 batch 带有 `advantages`，直接使用 batch 中的 `advantages`；否则使用 `-reverse_kl` 作为回退。随后每个样本先做 `masked_mean(advantages, response_mask)`，再在 domain 内统计。因此它是 sample-level 指标，不是 token-level 正 token 比例。

注意：actor 内部若开启 `actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True`，`dp_actor.py` 会在实际 policy loss 计算时把 actor 使用的 advantage 临时替换为 `-reverse_kl`。`<domain>/advantage/positive_frac` 记录的是 audit 看到的 batch advantage 符号分布，不等价于 actor 内部每个 token 最终参与 loss 时的临时 advantage 分布。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/advantage/positive_frac` | 当前 domain 中 sample-level advantage 为正的样本比例。 | 对每个样本计算 `masked_mean(advantages, response_mask)`，再统计 `mean(sample_advantage_mean > 0)`。 |

## Response Length Metrics

这组 metric 描述不同 train domain 的 response 长度分布，用来判断某个 domain 的训练信号、耗时、梯度范数或 loss 是否主要被长输出样本驱动。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/length/response_mean` | 当前 domain 的平均有效 response token 数。 | 每个样本计算 `response_mask.sum()`，再对 domain 内样本求 mean。 |
| `<domain>/length/response_p95` | 当前 domain 的 response token 数 95 分位。 | 对 domain 内样本的 `response_mask.sum()` 取 95 percentile。 |
| `<domain>/length/response_clip_ratio` | 当前 domain 中 response 达到最大 response 长度的样本比例。 | 统计 `response_token_count >= response_mask.shape[-1]` 的比例。 |

## Training Reward Metrics

这组 metric 来自当前 training batch 的 `token_level_scores`。如果 batch 中没有 `token_level_scores`，这些 metric 不会写入 TensorBoard。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/reward/training_reward_mean` | 当前 train domain 的 training reward 均值。 | 每个样本先算 `sum(token_level_scores * response_mask)`，再对 domain 内样本求均值。 |
| `<domain>/reward/training_accuracy` | 当前 train domain 中 reward 为正的样本比例。 | `mean(training_reward > 0)`。 |

## Full-Parameter Gradient Metrics

这组是真实 actor 参数梯度。指标来自真实 `dp_actor.update_policy()` 的同一次 optimizer step：tracker 会把当前 actor mini-batch 内的 micro-batches 按配置里的 train domain 顺序执行 backward，第一组 domain backward 完成后 snapshot 一次 `g_math`，全部 domain backward 完成后读取 `g_total`，再用 `g_code = g_total - g_math` 还原第二个 domain 的梯度。

full-gradient audit 的 train tracker 不再走同图 `torch.autograd.grad()`，因为 FSDP/sharded 参数下该路径可能拿不到有效 `.grad`。新路径直接读取真实 backward 后的参数 `.grad` snapshot。因此它不再是手写的 `reverse_kl * ratio` proxy，也不再是 train-side 额外 forward/backward。不过它仍然是**一阶梯度诊断**，不是临时执行 Adam/optimizer step 后的真实 validation delta。

当前 verl FSDP worker 启用 gradient checkpointing 时显式传入 `use_reentrant: false`。这保留了标准 backward 的兼容性，也避免重新引入 reentrant checkpoint 对 `autograd.grad()` 的限制；train-side full-gradient tracker 本身只复用 `loss.backward()` 产生的 `.grad`。

当前 train-side sequential tracker 只支持两个 train domains，且要求每个 micro-batch 只包含一个 domain；正式配置中 `full_gradient_micro_batch_size_per_gpu: 1` 满足这个条件。如果出现 mixed-domain micro-batch，tracker 会写入 `global/audit/full_gradient_domain_sequential_unsupported=1`，并跳过本 mini-batch 的 domain gradient metrics。

multi-GPU 下还要求各 rank 的 domain micro-batch 数量一致，以保证每次 FSDP backward collective 都处于同一个 domain block。若任一 rank 不满足单-domain micro-batch，或各 rank 的 domain 边界不对齐，所有 rank 都会跳过该 mini-batch 的 domain gradient metrics，避免把不同 domain 的梯度错误聚合。

gradient scalar statistics 按 FSDP topology 归并：FULL_SHARD 对各 rank 持有的参数 shard 做 sum；HYBRID_SHARD 先对所有 rank 做 sum，再除以相同 shard 的 DDP replica 数。实现只 collective norm/dot 等 FP64 scalars，不 all-gather 完整参数梯度，因此不会额外复制 multi-billion-parameter gradient vector。

正式配置：

```yaml
full_gradient_enabled: true
full_gradient_freq_steps: 1
full_gradient_train_max_samples_per_domain: null
full_gradient_micro_batch_size_per_gpu: 1
full_gradient_storage_dtype: bfloat16
sample_gradient_enabled: true
sample_gradient_norm_enabled: true
sample_gradient_cos_enabled: false
sample_gradient_cos_freq_steps: 1
sample_gradient_log_sample_level: true
token_gradient_enabled: false
token_gradient_freq_steps: 10
token_gradient_gap_selection_enabled: true
token_gradient_gap_abs_selection_enabled: true
token_gradient_loss_abs_selection_enabled: true
token_gradient_top_k: 100
token_gradient_top_p: 0.10
token_gradient_strict_grad_restore: false
```

其中 `null` 表示 train probe 使用当前 actor update mini-batch 内该 domain 的全部样本；多 mini-batch 时由 verl 原有 metrics reducer 做均值聚合。token-gradient audit 的 candidate pool 也是当前 step 内该 domain 的全部 valid response tokens，不再做 per-sample top-k 截断；`token_gradient_top_k` 和 `token_gradient_top_p` 都在 domain 全局候选分布上生效。`token_gradient_enabled` 是总开关，`token_gradient_gap_selection_enabled`、`token_gradient_gap_abs_selection_enabled` 和 `token_gradient_loss_abs_selection_enabled` 分别控制 signed log-prob gap、absolute log-prob gap 与 loss score 三套 top-k/top-p selection。

当前正式配置使用 `bfloat16` 保存两个 domain gradient target，不再额外保存 total-gradient CPU snapshot。norm、dot product 和 cosine 都会先转换为 FP32 再累加。BF16 与 FP16 同为 2 bytes，但 exponent range 更大，更不容易把小梯度分量下溢成 0。

sample-gradient 配置的含义：

- `sample_gradient_norm_enabled: true`：每个 step 统计 domain 内每个样本对真实 backward 的 sample grad norm 分布，不额外 recompute。
- `sample_gradient_cos_enabled: false`：默认禁用 sample-to-domain gradient cosine。当前实现依赖 `torch.autograd.grad()` 重算 sample gradient，实测容易出现全参数断图；如需诊断可临时开启。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<train_domain>/full_grad/grad_norm` | 当前 actor update mini-batch 中该 train domain 的真实参数梯度范数。 | 从真实 backward 后的参数 `.grad` snapshot 计算 `||g_train_i||_2`。 |
| `<train_domain>/full_grad/sample_count` | 该 train domain probe 使用的样本数。 | 当前 actor update mini-batch 中该 domain 样本数，经 distributed sum。 |
| `global/full_grad_conflict/<domain_i>_vs_<domain_k>/full_grad_cosine_train_i_k` | 两个 train domains 的真实参数梯度方向相似度。 | `cosine(g_train_i, g_train_k)`。 |
| `global/full_grad_conflict/<domain_i>_vs_<domain_k>/conflict_magnitude_i_k` | 真实参数梯度冲突强度。 | `max(0, -full_grad_cosine_train_i_k)`。 |
| `global/full_grad_alignment/<domain_i>_vs_total/full_grad_cosine_domain_total` | train domain 梯度与总梯度的方向一致性。 | `(g_i · g_total) / (||g_i||_2 ||g_total||_2)`。 |
| `global/full_grad_contribution/<domain_i>_to_total/signed_projection_share` | train domain 梯度对总梯度方向的有符号投影贡献。 | `(g_i · g_total) / ||g_total||_2^2`；两个 domain 的 share 理论上相加为 1，负值表示该 domain 在抵消总更新方向。 |
| `global/full_grad_cost/backward_seconds` | train tracker 从 mini-batch backward 开始到 full-gradient summary 前的墙钟耗时。 | 主要覆盖真实 actor backward 和 domain gradient snapshot，token-gradient diagnostic 不包含在这个字段里。 |
| `global/full_grad_cost/domain_summary_seconds` | full-gradient domain summary 的额外耗时。 | 统计 domain gradient norm、domain-domain cosine、domain-total alignment 和 projection share 的 summary 计算时间。 |
| `global/full_grad_cost/finish_mini_batch_seconds` | tracker finish 阶段总耗时。 | 从进入 `finish_mini_batch()` 到返回 metrics 的墙钟时间，包含 full/sample/token gradient summary。 |
| `global/full_grad_cost/max_memory_allocated_gb` | full-gradient audit 后的 CUDA peak allocated memory。 | `torch.cuda.max_memory_allocated() / 1024^3`。 |

## Sample-Gradient Metrics

这组 metric 来自 `full_gradient_worker.py` 的 train-side tracker，用来观察 domain 内部样本梯度强弱和样本梯度相对 domain 梯度的方向关系。

sample grad norm 不需要额外 backward：tracker 在真实 actor backward 过程中按 micro-batch/single-sample 记录 sample-level grad norm。sample-to-domain cosine 和 projection share 会对当前 actor update mini-batch 内的全部样本逐个执行 GPU recompute backward，并在 GPU 上按参数分块计算 gradient norm 和 dot product。

注意：在当前 verl/FSDP actor 路径下，部分 recompute forward 的 `autograd.grad()` 可能返回全 None。token-gradient diagnostic 已提供 backward fallback；sample-gradient cosine 也可以复用同类思路，但成本约等于“每个样本一次额外 forward/backward”，比 hook-based sample norm 贵很多，应只在小 batch 或低频 diagnostic 中开启。开启 token-gradient 时，domain target chunks 会临时使用 FP32 存储，用于降低 fallback restore 的量化误差；普通 full-gradient audit 仍遵循 `full_gradient_storage_dtype`。如果需要证明 fallback 不扰动训练梯度，可以打开 `token_gradient_strict_grad_restore`：fallback 前把原始 `.grad` clone 到 CPU，fallback 后直接恢复原始快照，并记录 restored-vs-original 的误差指标；该模式会额外占用一份完整梯度 CPU 内存，因此建议只用于小 batch debug。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/sample_grad/norm_mean` | 当前 domain 内样本级梯度范数均值。 | 对该 domain 所有样本的 `sample_grad_norm` 求 mean。 |
| `<domain>/sample_grad/norm_p50` | 当前 domain 内样本级梯度范数中位数。 | 对该 domain 所有样本的 `sample_grad_norm` 取 50 percentile。 |
| `<domain>/sample_grad/norm_p95` | 当前 domain 内样本级梯度范数 95 分位。 | 对该 domain 所有样本的 `sample_grad_norm` 取 95 percentile。 |
| `<domain>/sample_grad/norm_max` | 当前 domain 内最大样本级梯度范数。 | `max(sample_grad_norm)`。 |
| `<domain>/sample_grad/norm_cv` | 当前 domain 内样本级梯度范数变异系数。 | `std(sample_grad_norm) / (abs(mean(sample_grad_norm)) + 1e-12)`。 |
| `<domain>/sample_grad/sample_count` | 当前 domain 参与 sample grad norm 统计的样本数。 | 当前 actor update mini-batch 中该 domain 的样本数。 |
| `<domain>/sample_grad_cos/domain_cos_mean` | 全部样本的 sample gradient 与该 domain gradient 的平均 cosine。 | 对全部样本计算 `cosine(g_sample, g_domain)` 后求 mean。 |
| `<domain>/sample_grad_cos/domain_cos_p05` | 全部样本 sample-to-domain cosine 的 5 分位。 | 对全部样本的 `cosine(g_sample, g_domain)` 取 5 percentile。 |
| `<domain>/sample_grad_cos/domain_cos_negative_frac` | 全部样本中与 domain gradient 方向相反的比例。 | 统计 `cosine(g_sample, g_domain) < 0` 的比例。 |
| `<domain>/sample_grad_cos/sample_count` | 当前 domain 完成 sample-to-domain cosine 的样本数。 | `len(samples_with_valid_cosine)`。 |
| `<domain>/sample_grad_contribution/projection_share_mean` | 全部样本对 domain gradient 方向的平均投影贡献。 | 对全部样本计算 `(g_sample · g_domain) / ||g_domain||_2^2` 后求 mean。 |
| `<domain>/sample_grad_contribution/projection_share_min` | 全部样本 projection share 最小值。 | `min(projection_share)`。 |
| `<domain>/sample_grad_contribution/projection_share_max` | 全部样本 projection share 最大值。 | `max(projection_share)`。 |
| `<domain>/sample_grad_contribution/projection_share_negative_frac` | 全部样本中抵消 domain gradient 方向的比例。 | 统计 `projection_share < 0` 的比例。 |
| `<domain>/sample_grad_contribution/top1_abs_share` | 全部样本里绝对投影贡献最大的单样本强度。 | `max(abs(projection_share))`。 |
| `<domain>/sample_grad_contribution/projection_share_sum` | sample projection share 的求和 sanity check。 | 若 sample 重算 loss 与 domain gradient 口径一致，且统计覆盖该 domain 全部样本，此值应接近 1；明显偏离表示 sample/domain 梯度口径不一致或样本未完整聚合。 |

已删除：所有 `grad/*`、`grad_anchor/*`、`grad_conflict/*` proxy metrics。

## Token Conflict Attribution Metrics

这组 metric 来自 `verl_audit.py`，目标是回答“哪些 response tokens 更可能贡献 domain conflict”。它不是精确 token-level gradient decomposition；精确 token gradient 见下一节。当前实现先用低成本 token-level diff 找候选 token：

```text
teacher_teacher_diff =
    abs(selected_teacher_logp - alternate_teacher_logp)

student_teacher_diff =
    abs(student_or_rollout_logp - selected_teacher_logp)

combined_diff =
    teacher_teacher_diff * student_teacher_diff
```

其中 selected teacher 由 `opd_teacher` 决定；`code` 使用 `code_teacher_log_prob`，其它 domain 使用 primary/math teacher。若 batch 没有 alternate teacher log-prob，`teacher_teacher_diff` 回退为 `abs(selected_teacher_logp - old_log_probs)`。所有统计只在有效 `response_mask` token 上计算。

旧字段 `proxy_*` 仍保留作兼容；当 `lambda_vals=1.0` 时它等价于 `combined_diff_*`。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/token_conflict/teacher_teacher_diff_mean` | 当前 domain 的 teacher-vs-teacher 平均 token-level disagreement。 | `mean(abs(selected_teacher_logp - alternate_teacher_logp))`。 |
| `<domain>/token_conflict/teacher_teacher_diff_p95` | teacher-vs-teacher disagreement 的 95 分位。 | 对有效 tokens 取 p95，用于观察少量强分歧 token。 |
| `<domain>/token_conflict/teacher_teacher_diff_max` | teacher-vs-teacher disagreement 最大值。 | 对有效 tokens 取 max。 |
| `<domain>/token_conflict/student_teacher_diff_mean` | student/rollout 与 selected teacher 的平均 mismatch。 | `mean(abs(old_log_probs - selected_teacher_logp))`。 |
| `<domain>/token_conflict/student_teacher_diff_p95` | student-teacher mismatch 的 95 分位。 | 对有效 tokens 取 p95。 |
| `<domain>/token_conflict/combined_diff_mass` | 当前 domain 的候选冲突总量。 | `sum(teacher_teacher_diff * student_teacher_diff)`。 |
| `<domain>/token_conflict/combined_diff_mean` | 平均每 token 的候选冲突强度。 | `combined_diff_mass / domain_token_count`。 |
| `<domain>/token_conflict/combined_diff_p95` | combined diff 的 95 分位。 | 对有效 tokens 取 p95。 |
| `<domain>/token_conflict/proxy_mass` | 兼容旧面板的 combined diff 总量。 | 当前等价于 `combined_diff_mass`。 |
| `<domain>/token_conflict/token_abs_opd_loss_mean` | 当前 domain 的平均 token-level OPD signal 强度。 | 对有效 tokens 计算 `mean(abs(reverse_kl))`。 |
| `<domain>/token_conflict/top1_token_share` | top-1 token id 对该 domain combined diff mass 的贡献占比。 | 对 token id 聚合 combined diff 后取最大值并除以 `combined_diff_mass`。 |
| `<domain>/token_conflict/top1_teacher_diff_share` | top-1 token id 对 teacher diff mass 的贡献占比。 | 对 token id 聚合 `teacher_teacher_diff` 后取最大值并除以 teacher diff mass。 |
| `<domain>/token_conflict/unique_token_count` | 当前 step 该 domain 中有非零 teacher/combined diff 的 unique token id 数。 | 对 top-token 聚合前的有效 response token ids 去重计数。 |

如果 batch 中存在 `responses`、`response_ids` 或可切片的 `input_ids`，还会写 `token_conflict_attribution.jsonl`。每行是某个 step/domain 的一个 top token id，按 `teacher_teacher_diff_sum` 排名，包含 `token_id`、`rank`、`token_count`、`teacher_teacher_diff_sum`、`student_teacher_diff_mean`、`combined_diff_sum`、`combined_diff_frac`、`token_abs_opd_loss_mean` 和 `response_position_mean`。要看 token 文本，可用当前 tokenizer 对 `token_id` 离线 decode。

## Exact Token-Gradient Diagnostic

这组 metric 来自 `full_gradient_worker.py`，默认关闭。开启 `token_gradient_enabled: true` 后，tracker 会在命中 `token_gradient_freq_steps` 的 step 中，先按 domain 收集当前 step 全局所有 valid response token。每个 token 会记录两类 selection score：

- `gap = teacher_logp - student_logp`：signed chosen-token teacher/student log-prob 差距；top-p mass 使用正向 gap 质量，避免正负抵消。
- `gap_abs = abs(teacher_logp - student_logp)`：absolute chosen-token teacher/student log-prob 差距。
- `loss_abs = abs(token_loss_score)`：训练 loss 层面的 token score。top-k distillation 开启时来自 per-token top-k distill loss；非 top-k 路径下使用 chosen-token reverse-KL proxy。

随后 tracker 会根据 selection 开关，分别在 `gap`、`gap_abs` 和/或 `loss_abs` 的 domain 全局分布上选择 `top{token_gradient_top_k}_*` 与覆盖 `token_gradient_top_p` 比例 score mass 的最小 token 集合，例如 `top100_gap`、`topp10_gap_mass`、`top100_gap_abs`、`topp10_gap_abs_mass`、`top100_loss_abs`、`topp10_loss_abs_mass`。每个 selection 都会对选中 token 做额外 gradient recompute。若 `token_gradient_loss_abs_selection_enabled=false`，不会额外 forward 计算 loss score。

```text
g_token = grad(token_loss_with_original_aggregation_scale, actor_params)
```

注意这里的 `token_loss` 会按原始 `loss_agg_mode` 重新缩放。例如 `token-mean` 下，单 token mask 的 loss 会乘以 `1 / effective_tokens`，避免把单 token 的梯度贡献放大。

| Metric | 含义 | 计算方式 |
| --- | --- | --- |
| `<domain>/token_grad/global_candidate_sample_count` | 当前 domain 全局候选分布覆盖的样本数。 | 对所有 actor rank 的 valid response token metadata 做 all-gather 后按 `sample_id` 去重。 |
| `<domain>/token_grad/global_candidate_token_count` | 当前 domain 全局候选分布覆盖的 token occurrence 数。 | 所有 actor rank、所有 sample、所有 `response_mask > 0` 的 valid response token。 |
| `<domain>/token_grad/global_candidate_gap_mass` | 当前 domain 全局候选分布的正向 `gap` 总质量。 | `sum(max(0, teacher_logp - student_logp))`，覆盖全局候选 token。 |
| `<domain>/token_grad/global_candidate_gap_abs_mass` | 当前 domain 全局候选分布的 `gap_abs` 总质量。 | `sum(abs(teacher_logp - student_logp))`，覆盖全局候选 token。 |
| `<domain>/token_grad/global_candidate_loss_abs_mass` | 当前 domain 全局候选分布的 `loss_abs` 总质量。 | `sum(abs(token_loss_score))`，覆盖全局候选 token。 |
| `<domain>/token_grad/selected_sample_count` | 当前 domain 被所有 top-k/top-p selection 覆盖的样本数合计。 | 对 selection rows 的 `selected_sample_count` 求和；集合重叠时会按 audit workload 计数。 |
| `<domain>/token_grad/selected_token_count` | 当前 domain 做 exact token gradient 的 token occurrence 数合计。 | 所有 selection rows 的 token 数之和；重叠 token 会按 recompute workload 计数。 |
| `<domain>/token_grad/top100_gap_selected_token_count` | 当前 domain 全局 signed `gap` 最大的最多 100 个 token 数。 | 在 domain 全局候选分布上按 signed `gap` 排序后取前 100。 |
| `<domain>/token_grad/topp10_gap_mass_selected_token_count` | 当前 domain 覆盖 top-p positive gap mass 的 token 数。 | selection 名称里的 `10` 随 `token_gradient_top_p` 变化。 |
| `<domain>/token_grad/top100_gap_abs_selected_token_count` | 当前 domain 全局 `gap_abs` 最大的最多 100 个 token 数。 | 在 domain 全局候选分布上按 `gap_abs` 排序后取前 100。 |
| `<domain>/token_grad/topp10_gap_abs_mass_selected_token_count` | 当前 domain 覆盖 top-p gap_abs mass 的 token 数。 | selection 名称里的 `10` 随 `token_gradient_top_p` 变化。 |
| `<domain>/token_grad/top100_loss_abs_selected_token_count` | 当前 domain 全局 `loss_abs` 最大的最多 100 个 token 数。 | 在 domain 全局候选分布上按 `loss_abs` 排序后取前 100。 |
| `<domain>/token_grad/topp10_loss_abs_mass_selected_token_count` | 当前 domain 覆盖 top-p loss_abs mass 的 token 数。 | selection 名称里的 `10` 随 `token_gradient_top_p` 变化。 |
| `<domain>/token_grad/top100_gap_abs_gap_abs_mass_frac` | top100 gap tokens 占当前 domain 全局 gap_abs mass 的比例。 | `top100_gap_abs_mass / global_candidate_gap_abs_mass`。 |
| `<domain>/token_grad/top100_loss_abs_loss_abs_mass_frac` | top100 loss tokens 占当前 domain 全局 loss_abs mass 的比例。 | `top100_loss_abs_loss_abs_mass / global_candidate_loss_abs_mass`。 |
| `<domain>/token_grad/<selection>_score_mass_frac` | 当前 selection 覆盖其排序 score 总质量的比例。 | `selected_score_mass / global_candidate_score_mass`；对 signed gap selection 是 positive gap mass，对 loss selection 是 loss mass。 |
| `<domain>/token_grad/norm_mean` | 选中 token 的真实 gradient norm 均值。 | 对 `||g_token||_2` 求 mean。 |
| `<domain>/token_grad/norm_p95` | 选中 token 的真实 gradient norm 95 分位。 | 对 `||g_token||_2` 取 p95。 |
| `<domain>/token_grad_conflict/other_cos_mean` | 选中 token gradient 与另一个 domain full gradient 的平均 cosine。 | `mean(cos(g_token, g_other_domain))`。 |
| `<domain>/token_grad_conflict/other_cos_negative_frac` | 选中 token 中与另一个 domain full gradient 方向相反的比例。 | 统计 `cos(g_token, g_other_domain) < 0`。 |
| `<domain>/token_grad_conflict/conflict_to_other_mean` | 选中 token 对另一个 domain 的平均冲突强度。 | `mean(max(0, -cos(g_token, g_other_domain)))`。 |
| `<domain>/token_grad_contribution/own_projection_share_sum` | 选中 token 对当前 domain full gradient 的投影贡献总量。 | `sum((g_token · g_domain) / ||g_domain||_2^2)`。只覆盖 top-k token，因此不是完整 domain contribution。 |
| `<domain>/token_grad_contribution/negative_other_projection_share_sum` | 选中 token 对另一个 domain full gradient 的负向投影总量。 | `sum(max(0, -(g_token · g_other_domain) / ||g_other_domain||_2^2))`。 |
| `<domain>/token_grad_cost/seconds_sum` | 当前 domain 选中 token 的 exact gradient 总耗时。 | 对该 domain token rows 的 `token_grad_seconds` 求和。 |
| `<domain>/token_grad_cost/seconds_per_selected_token` | 当前 domain 平均每个选中 token 的 exact gradient 耗时。 | `seconds_sum / selected_token_count`。 |
| `<domain>/token_grad_cost/backward_fallback_count` | 当前 domain 使用 backward fallback 的 token 数。 | 当 `autograd.grad()` 断图且 fallback 成功时计 1。 |
| `<domain>/token_grad_cost/backward_fallback_seconds_sum` | 当前 domain backward fallback 总耗时。 | 对 `token_grad_backward_fallback_seconds` 求和。 |
| `<domain>/token_grad_cost/valid_frac` | 当前 domain exact token gradient 可用比例。 | `available_token_count / selected_token_count`。 |
| `<domain>/token_grad_cost/restore_original_rel_l2_max` | strict restore 后 `.grad` 与 fallback 前原始 `.grad` 的最大相对 L2 误差。 | 仅 `token_gradient_strict_grad_restore=true` 时写出；理想值为 0。 |
| `<domain>/token_grad_cost/restore_original_max_abs_max` | strict restore 后 `.grad` 与 fallback 前原始 `.grad` 的最大绝对误差。 | 仅 strict restore 时写出；用于确认训练梯度没有被 token backward 覆盖。 |
| `global/token_grad_cost/global_candidate_token_count` | 全 domain 全局候选 token 数。 | 按 domain 去重后求和，避免 top100/top-p selection rows 重复计数。 |
| `global/token_grad_cost/global_candidate_sample_count` | 全 domain 全局候选样本数。 | 按 domain 去重后求和。 |
| `global/token_grad_cost/global_candidate_gap_mass` | 全 domain 全局候选正向 `gap` 总质量。 | 按 domain 去重后求和。 |
| `global/token_grad_cost/global_candidate_gap_abs_mass` | 全 domain 全局候选 `gap_abs` 总质量。 | 按 domain 去重后求和。 |
| `global/token_grad_cost/global_candidate_loss_abs_mass` | 全 domain 全局候选 `loss_abs` 总质量。 | 按 domain 去重后求和。 |
| `global/token_grad_cost/seconds` | 本 step exact token-gradient diagnostic 总耗时。 | 从进入 `_token_gradient_metrics()` 到写出 rows/聚合 metrics 的墙钟时间。 |
| `global/token_grad_cost/seconds_per_selected_token` | 全局平均每个选中 token 的 exact gradient 耗时。 | `seconds / selected_token_count`。 |
| `global/token_grad_cost/backward_fallback_count` | 全局使用 backward fallback 的 token 数。 | 用于判断是否仍存在 `autograd.grad()` 断图。 |

同时会写 `token_grad_metrics.jsonl`。每行是一个 selection summary，包含 `selection_scope=global`、`selection_score`、`global_candidate_scope=all_valid_response_tokens`、`global_candidate_token_count`、`global_candidate_sample_count`、`global_candidate_gap_mass`、`global_candidate_gap_abs_mass`、`global_candidate_loss_abs_mass`、`selected_token_count`、`selected_gap_mass_frac`、`selected_gap_abs_mass_frac`、`selected_loss_abs_mass_frac`、`selected_score_mass_frac`、`own_domain_cos`、`other_domain_cos`、`conflict_to_other`、`own_projection_share` 和 `other_projection_share`。这份文件适合离线比较 signed-gap、gap-abs 与 loss-based top-k/top-p token 集合和 domain gradient 的方向关系。

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
| `token_gap_vectors.jsonl` | 每 step、每 train domain 的 `teacher_logp - student_logp` 与 `abs(teacher_logp - student_logp)` raw response-token occurrence vectors。 |
| `token_gap_vocab_vectors.jsonl` | 开启 `token_gap_vocab_vector_enabled` 后，每 step、每 train domain 的全词表 dense token-id vectors；第 `v` 维对应 token id `v`。 |
| `entropy_distribution_vectors.jsonl` | 每 step、每 train domain 的 teacher entropy、student entropy、teacher-student cross entropy raw token vectors。 |
| `entropy_vocab_vectors.jsonl` | 开启 `entropy_vocab_vector_enabled` 后，每 step、每 train domain 的 student entropy 与 teacher-student cross entropy 全词表 dense token-id vectors。 |
| `token_conflict_attribution.jsonl` | 每 step、每 train domain 的 top token diff rows，用于定位哪些 token id 的 teacher disagreement / combined diff 最大。 |
| `token_grad_metrics.jsonl` | 开启 `token_gradient_enabled` 后，每 step、每 train domain 的 top-k token occurrence exact gradient rows，用于定位真实 token-level gradient norm、冲突和投影贡献。 |
| `sample_grad_metrics.jsonl` | 每 step、每样本的 sample grad norm、sample-to-domain cosine、projection share、recompute grad norm 和 `computed_for_cos` 标记。 |
| `validation_probe.jsonl` | 原始 validation value、previous value、gain，用于复盘 validation gain。 |
| `validation_gain_variance.jsonl` | validation gain history、mean、variance。 |
| `training_cost.jsonl` | step seconds、GPU seconds、tokens/sec、peak memory。 |
| `audit_errors.jsonl` | audit 过程中发生异常时的防御性错误记录。 |

已删除的 JSONL：`domain_conflict.jsonl`、`trend_stability.jsonl`、`gradient_noise.jsonl`、`rank_stability.jsonl`、`teacher_logits_reliability.jsonl`、`calibration.jsonl`、`sample_influence.jsonl`、`coverage_diversity.jsonl`、`shadow_probe.jsonl`。

## 当前限制

当前训练期 full-gradient audit 的“全量”是当前 step batch 级别，不是完整训练集级别。如果要计算完整训练集 gradient，需要单独实现离线全数据 dataloader backward，并明确评估频率；不建议每个训练 step 都做。

普通 validation score、validation gain 和 validation gain variance 仍然保留，但 validation pass 不再执行额外 gradient backward。
