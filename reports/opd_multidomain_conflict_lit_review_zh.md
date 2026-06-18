# OPD 多 Domain 冲突诊断与优化调研

日期：2026-06-13

## 背景

当前 MOPD 训练里，math/code 两个 domain 共享同一个 student policy，但 teacher signal 来自不同 domain-specialized teachers。冲突可能来自三个层次：

1. domain full gradient 方向冲突：`cos(g_math, g_code) < 0`。
2. sample gradient 内部异质性：少数样本的 `sample_grad_norm` 或 negative projection 主导 domain update。
3. token-level teacher disagreement：某些 response token 上 selected teacher 与 alternate teacher 的 log-prob 差异很大，同时 student-teacher mismatch 仍然较大。

本轮代码已经把第 3 层拆成低成本 diff 统计和 top-k exact token gradient diagnostic。

## 相关文献脉络

| 方向 | 代表工作 | 关键思想 | 对 OPD 的启发 |
| --- | --- | --- | --- |
| Gradient surgery | PCGrad, Yu et al., NeurIPS 2020 | 当两个 task gradients 冲突时，把一个梯度投影到另一个梯度的法平面，去掉负向分量。 | 可以在 domain full gradient 层面做 OPD-PCGrad：`g_math` 与 `g_code` 冲突时只去掉 destructive component。 |
| Conflict-aware multi-objective update | CAGrad, Liu et al., NeurIPS 2021 | 在平均 loss 优化和每个 task 的 worst local improvement 之间做 regularized balance。 | 比 PCGrad 更稳，但需要每步 domain gradient，适合先做小规模 ablation。 |
| Bargaining / Pareto update | Nash-MTL, Navon et al., ICML 2022 | 把 gradient combination 看成 bargaining game，求 Nash bargaining update direction。 | 可作为强 baseline，但计算复杂度和实现复杂度都高于 PCGrad/CAGrad。 |
| Relatedness-aware alignment | Gradient Vaccine, Wang et al., ICLR 2021 | 不是所有 task 都应强制对齐；相近任务才鼓励 gradient 几何对齐。 | 对 math/code 可引入 domain relatedness：只对同类子域强对齐，对跨域冲突用 projection 或 sampling 控制。 |
| Multi-domain LLM data curriculum | EVIC, ICML 2025 | 用 gradient-based sample influence 发现 sample conflict 随训练动态变化，并动态选择有益样本。 | 和我们的 sample/token diagnostics 很贴：可把 `negative_other_projection` 高的样本降采样或延后。 |
| Adaptive data mixing | PiKE, 2025 | 根据 task gradient interaction 动态调整 data mixture，强调大语言模型里常有 positive interactions。 | 不要默认所有跨域 signal 都是坏的；应先区分 positive vs negative token/sample，再调 mixing weight。 |
| Multi-objective LLM alignment | GAPO, ACL 2025 | 用 multiple-gradient descent 平衡多种偏好目标。 | 如果把 math/code teacher signals 看成两个 objectives，可做 OPD-GAPO 式 domain gradient rescaling。 |
| OPD 机制 | GKD/OPD, Agarwal et al., ICLR 2024；OPD survey 2026；OPSD 2026；Rethinking OPD 2026 | OPD 的价值在 student-generated states 上提供 dense token supervision；近期工作强调 top-k / token overlap / high-probability token alignment。 | 我们的 token diff 与 exact token gradient 可以进一步回答：哪些 teacher-disagreement tokens 是有效 dense supervision，哪些只是跨域冲突噪声。 |

## 对当前 OPD 最值得尝试的方案

### 1. 先做诊断闭环：token diff → token gradient → sample/domain decision

当前新增指标建议这样使用：

1. 用 `teacher_teacher_diff` 找 selected teacher 与 alternate teacher 分歧大的 token。
2. 用 `combined_diff = teacher_teacher_diff * student_teacher_diff` 找“分歧大且 student 还没学好”的 token。
3. 对 top-k token 打开 `token_gradient_enabled`，记录：
   - `token_grad_norm`
   - `other_domain_cos`
   - `conflict_to_other`
   - `own_projection_share`
   - `other_projection_share`
4. 将 token rows 聚合回 sample/domain：
   - 如果 high-diff token 的 `own_projection_share` 为正、`other_projection_share` 不强负，说明它是有益 domain-specific signal。
   - 如果 high-diff token 的 `other_projection_share` 强负，且 token text 主要是格式/风格/模板 token，则更像冲突噪声。

### 2. OPD-PCGrad：domain full gradient 层面的最小改动

当 `cos(g_math, g_code) < 0` 时，用 PCGrad 风格更新：

```text
g_math' = g_math - min(0, dot(g_math, g_code) / ||g_code||^2) * g_code
g_code' = g_code - min(0, dot(g_code, g_math) / ||g_math||^2) * g_math
g_update = w_math * g_math' + w_code * g_code'
```

优点：

- 和现有 `full_grad_conflict` 指标直接对齐。
- 不改变 OPD loss 本身，只改 gradient aggregation。
- 最适合作为第一版 optimization ablation。

风险：

- 需要真实训练 update 使用 surgery 后的 gradient，而不仅是 audit。
- 如果 math/code 冲突其实是必要 trade-off，过度 surgery 会削弱 domain specialization。

### 3. Token-aware OPD clipping / masking

基于 exact token gradient rows，给 token 级 loss 一个 gate：

```text
gate_t = 1
if teacher_teacher_diff_t high and conflict_to_other_t high:
    gate_t = downweight
```

更稳的版本不是直接 mask，而是只对冲突 token 降低 OPD coefficient：

```text
lambda_t = lambda_base * clip_or_decay(conflict_to_other_t, teacher_teacher_diff_t)
```

优点：

- 比 domain-level PCGrad 更细，可以只压制冲突 token，不牺牲整个 domain batch。
- 对“格式 token / boilerplate token 主导冲突”的情况尤其合适。

风险：

- exact token gradient 成本高，训练期不能全量跑。建议先用 offline/periodic diagnostic 学出规则，再用低成本 proxy 近似执行。

### 4. Dynamic domain sampling / curriculum

借鉴 EVIC 和 PiKE，把 domain/sample sampling weight 做成动态项：

```text
score_sample =
    own_projection_share
    - alpha * negative_other_projection_share
    + beta * validation_gain_proxy
```

可以先做最简单版本：

- 每 N step 统计 sample/token conflict。
- 对 conflict 高且 validation gain 差的 domain 降低 sampling weight。
- 对 positive interaction 高的 domain pair 保持或提高 sampling weight。

优点：

- 不改 optimizer，更容易和当前 pipeline 兼容。
- 适合先做 batch-level/domain-level ablation。

风险：

- 如果统计窗口太短，会被噪声带偏。
- 需要和 domain mix entropy、validation gain 一起看。

## 推荐实验顺序

1. **诊断实验**：打开 `token_gradient_enabled=true`，小步数、低频率，只验证 high `teacher_teacher_diff` token 是否真的高 `conflict_to_other`。
2. **低成本 proxy 验证**：检查 `combined_diff` 与 exact `token_grad_norm/conflict_to_other` 的相关性。如果相关性好，用 `combined_diff` 做训练期 gate。
3. **OPD-PCGrad ablation**：只在 domain full gradient cosine 为负时做 projection，比较 validation math/code trade-off。
4. **Token-aware clipping ablation**：对高 conflict token 降低 OPD coefficient，观察是否能降低 cross-domain negative projection，同时保留 own-domain gain。
5. **Dynamic sampling ablation**：基于 sample/domain projection share 调整 math/code sampling weight。

## 参考来源

- Yu et al. 2020, Gradient Surgery for Multi-Task Learning: https://arxiv.org/abs/2001.06782
- Liu et al. 2021, Conflict-Averse Gradient Descent for Multi-task Learning: https://arxiv.org/abs/2110.14048
- Navon et al. 2022, Multi-Task Learning as a Bargaining Game: https://arxiv.org/abs/2202.01017
- Wang et al. 2020/2021, Gradient Vaccine: https://arxiv.org/abs/2010.05874
- Agarwal et al. 2024, On-Policy Distillation of Language Models: https://openreview.net/forum?id=3zKtaqxLhW
- Song & Zheng 2026, A Survey of On-Policy Distillation for LLMs: https://arxiv.org/abs/2604.00626
- Zhao et al. 2026, Self-Distilled Reasoner / OPSD: https://arxiv.org/abs/2601.18734
- Li et al. 2026, Rethinking OPD: https://arxiv.org/html/2604.13016v1
- Liang et al. 2025, EVIC multi-domain fine-tuning: https://openreview.net/forum?id=Si0HHbjBfU
- Li et al. 2025, GAPO: https://aclanthology.org/2025.acl-long.549/
- PiKE adaptive data mixing: https://arxiv.org/html/2502.06244v1
