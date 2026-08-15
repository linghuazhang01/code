# Control-Anchored Region-DPO

该实现是在标准 OPD actor update 中加入一个可选的、behavior-referenced
Region-DPO auxiliary loss。它默认关闭，不会改变既有 OPD/EOPD/Top-k KL
训练。

## 训练流程

对每条 base student rollout：

1. 在 response 中找到 frozen control-token taxonomy 对应的 occurrences；
2. 最多选择 `points_per_rollout` 个位置；
3. 将 prefix 截断在 control token 之前；
4. 从同一个 frozen rollout policy 为每个 prefix 采样
   `branches_per_point` 条自然 sibling rerollouts；
5. 把原 response prefix 与 rerollout suffix 拼回完整 response，使用现有
   rule-based reward/verifier 评分；
6. 对每个 point 取 reward 最高与最低的两个 siblings；reward 无严格差异或
   两条序列没有实际 divergence 时，不产生训练 pair；
7. 从 first-divergence token 开始计算 Region-DPO。

Region prefix 不包含原 control token。这让 siblings 可以共同探索：是否此时
发生 transition、使用哪种 transition，以及 transition 后如何展开。

## Loss

对 chosen/rejected regions `r+`、`r-`，令 `d` 为 first divergence，
rerollout behavior policy 为 `pi_old`：

```text
S(r) = sum_{t=d}^{T_r} [log pi_theta(r_t | h, r_<t)
                        - log pi_old(r_t | h, r_<t)]

L_region_dpo = -log sigmoid(beta * (S(r+) - S(r-)))
```

实现中先对同一条 base rollout 上已确认的 points 求平均，再对 base
rollouts 求平均；没有有效 pair 的 rollout 贡献 0。这样 gradient
accumulation 的 micro-batch 划分不会改变 Region-DPO 的相对权重，也不会让
control-token 更密集的样本支配 auxiliary loss。

`pi_old` log-prob 在 rerollout 时由 vLLM 记录并冻结。按上述两级 reduction
得到的 loss 再乘 `loss_weight`，与当前 OPD loss 在同一个 actor
backward/update 中相加。Inactive packed slots 不进入 actor forward。

## Config

在任意 MOPD YAML 顶层加入：

```yaml
region_dpo:
  enabled: true

  # 每条 base rollout 最多选几个 control occurrences。
  points_per_rollout: 2

  # 每个 occurrence 自然采样几条 sibling rerollouts；必须 >= 2。
  branches_per_point: 4

  # 每条 rerollout 最多生成多少 token。
  max_new_tokens: 256

  beta: 0.1
  loss_weight: 0.1

  # best - worst 必须严格大于该 margin。
  min_reward_margin: 0.0

  # first | random | uniform；均不读取 outcome。
  selection_strategy: random
  seed: 42

  # 可显式配置；省略时复用 audit 中冻结的 control-token taxonomy。
  domain_control_token_ids:
    math: [TOKEN_ID_1, TOKEN_ID_2]
    code: [TOKEN_ID_3]
```

如果现有 config 已包含：

```yaml
audit:
  domain_control_token_ids:
    math: [...]
    code: [...]
```

则 `region_dpo.domain_control_token_ids` 可以省略。即使 audit logging 关闭，
只要该 taxonomy 仍在 YAML 中，也可以作为 Region-DPO anchor source。

## Runtime Constraints

- `rollout.name: vllm`
- `rollout.mode: sync`
- `rollout.calculate_log_probs: true`
- single-turn rollout
- rule-based `reward_fn`/exact verifier
- 当前不与 teacher-prefix roll-in 或 dynamic domain budgeting 同时启用

## Metrics

- `region_dpo/eligible_point_count`
- `region_dpo/selected_point_count`
- `region_dpo/candidate_count`
- `region_dpo/generated_token_count`
- `region_dpo/confirmed_pair_count`
- `region_dpo/confirmed_pair_fraction`
- `region_dpo/reward_margin_mean`
- `actor/region_dpo_loss`
- `actor/region_dpo_weighted_loss`
- `actor/region_dpo_preference_accuracy`
- `actor/region_dpo_logit_mean`

`candidate_count` 的上界是：

```text
base rollout count * points_per_rollout * branches_per_point
```

因此 smoke run 建议从 `1 point × 2 branches × 64 tokens` 开始，确认
pair yield、显存和 generation cost 后再扩大。
