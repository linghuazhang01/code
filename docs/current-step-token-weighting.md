# Current-step token selection 与低权重 tail

当前实现支持在同一个 optimizer step 中评分、选取 token、再加权反向传播。
默认仍为 `next_step`；显式配置 `current_step` 才启用新增路径。
本文以下的 tail/TC 例子保留为历史机制说明。原有 V4/V5 与 legacy head-only
Math+Code 配置已迁至上一 step 结果选择：这 37 个 active public profiles 均为
`next_step`、`token_id`、`top_loss`、无 tail；旧 `current_step` 文件名只是兼容 alias，
展开后仍为 `next_step`。2026-10-01 新增的 V6/V7/V8 候选池矩阵为真正的
`current_step`；全部池内 ID 在各 domain 共享 TopLoss/TopP，Code 无位置门控。
候选池及数量统一以 [Token.md §0.4](../Token.md) 为准。实验对照应以实际展开配置
而不是文件名判断时序。

## 最小配置

以下字段可叠加到现有 regular Math+Code TopLoss profile：

```yaml
audit:
  control_token_online_selection_timing: current_step
  control_token_online_selection_unit: token_id
  control_token_online_selection_mode: top_loss
  control_token_tail_selection_mode: bottom_loss
  control_token_tail_top_p: 0.0
  control_token_tail_top_p_by_domain:
    math: 0.05
    code: 0.01
  control_token_tail_weight: 0.5  # 0.0 drops the direct tail loss gradient.
```

tail 比例是示例预算，尚未通过训练验证。head 沿用原配置的 Math 5% / Code 1%。
tail 的候选范围沿用同一份 configured domain C+S pool，不包含全词表 Other；
具体 taxonomy 以 `../Token.md` 当前定义为准。增权、降权范围均按实际生效的 pool 计算。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `control_token_online_selection_timing` | `next_step` | 新路径使用 `current_step` |
| `control_token_online_selection_unit` | `token_id` | 类型均值排名；`occurrence` 按具体位置排名 |
| `control_token_tail_top_p` | `0.0` | 降权组 occurrence 预算占该 domain 全部 valid tokens 的比例 |
| `control_token_tail_top_p_by_domain` | `{}` | 可只覆盖部分已知 domain；0 表示该 domain 不选 tail |
| `control_token_tail_weight` | `1.0` | raw tail multiplier，允许 `[0,1]` |
| `control_token_tail_selection_mode` | `bottom_loss` | 降权组评分方向，见下表 |

tail 的默认预算为 0、权重为 1，不改变旧训练。非默认 tail 配置必须搭配显式
`current_step`，否则报错，避免配置被静默忽略。设预算但保持 tail weight=1，
与同一 current-step head-only 配置的有效权重相同。

## 时序、排名与预算

1. 在 optimizer 更新前，对本步所有 micro-batches 做一次 `no_grad` student 评分。
   使用未乘选择权重的 raw teacher-support renormalized reverse KL。
2. 按 domain 跨 actor ranks 汇总。`token_id` 使用 occurrence-weighted mean
   absolute loss；`occurrence` 使用各位置 absolute loss。频次门槛仍按 ID 判断，
   regular profile 严格要求本步全局 count > 20。
3. head 按高分排名，先分配预算；tail 按低分排名，排除已经进入 head 的单位。
   预算始终相对于该 domain 的全部 valid response tokens。
4. 评分后还原 RNG、module mode、buffer 和已有 gradients，在参数未更新时执行
   正式 forward/backward，再更新 optimizer。首步即可生效，每步重新选择。

`token_id` 选整个 ID，因此会覆盖该 ID 在本步该 domain 的所有 eligible 出现位置，预算可能
overshoot；`occurrence` 精确选位置。ties 分别按 token ID、或 rank/batch/row/column
打破。候选不足时记录 shortfall，不自动扩展候选池。未通过频次门槛或未观察到的 ID
不会作为 loss=0 的 tail。token-ID 通信只发送每个类型的统计，避免广播全部出现位置。

legacy 路径要求 Fixed4、configured C+S pool；V4/V5 revision 3 使用 Control-only pool，
支持 C/S 同为 Fixed4 或 Fixed8。两者均要求 top-p 预算、window=interval=1、一个完整
actor minibatch、一个 PPO epoch、SP=1，及纯 teacher-support renormalized reverse KL。
不支持混合 KL、TIP/FiRE baseline、辅助 entropy/KL loss、region DPO、动态 domain
weighting 或其他 token weighting controller。原有校验会拒绝不兼容组合。

## Token V4/V5 revision 3：Control 与 Structure 分离

Token V4/V5 profile 使用更严格的组合语义：

```text
Control candidate --position gate--previous-step TopLoss--> selected Control, raw w
Structure IDs --------position gate---------------------> accepted Structure, raw w
两者与普通 token(raw w=1) 合并 -----------> 每个 microbatch/domain 归一化一次
```

主设置 `w=4`，显式变体 `w=8` 同时提高 Control 与 Structure 的 raw 权重。
集合以 `Token.md §0.3` 为准：V4 C=119/153，V5 C=61/72，S=9/45（Math/Code）。
C 类答案词 `final`/`answer`/`conclusion` 全部划入 Structure，V5 只保留 A+B。
V5 包含新增 4/8 个 sibling，不是 V4 子集；两者 Control intersection 为 57/64。

Structure 不进入 TopLoss candidate pool，因此不占 Math/Code 的 5%/5%、5%/2% 或
5%/1% budget。trainer 在 batch balance 前逐个 decode 原始 token id、保留原始生成边界，并生成
`mopd_structure_position_mask` 与 `mopd_control_position_mask`；Code 没有闭合代码块时不加权
code-body/fence token，但正文答案标签及 EOS 仍可生效。mask 随 `DataProto.reorder` 进入 actor，
不会在 actor rank 上重新加载 tokenizer。

Code Control 的 mask 排除所有 backtick/tilde fenced blocks，包括从未闭合的 fence
到回复末尾；scoring 的频次、均值与最终 ID 加权共用同一 mask。top-p 分母仍为该 domain
全部 valid response tokens。Math Control 的有效位置不另收窄。prepass 的 raw selector
validity 仍必须等于完整 response mask，位置筛选独立发生在该完整 mask 内；这条
prepass 说明仅适用于历史 Current-Step 路径。当前 Next-Step 路径直接使用上一步
正式 forward 的未加权原始 loss。

Code Structure 的非答案词只取最终闭合 block 的围栏/语言行及 I/O、函数签名、入口逻辑语句，
排除注释和 string literal token 位置，支持多行语句和 `input = sys.stdin.readline`。
普通 `int(value)`、`value.split()` 等计算行不命中；输出 `.join(...)` 作为格式化语句。
Python lexical parsing 失败时该 block body fail closed。两 domain 的答案词在大小写不敏感的
Markdown 标题或闭合加粗 label 中生效，Code 必须位于所有 fenced blocks 外，不能作为
I/O/signature 行中的变量获得权重。Math 还允许与字面 `\boxed` 命令同一行的答案词；
下一行不继承，`boxed`/EOS 本身任意位置生效。`Answer`（16141）未通过支持条件。

新增配置字段如下：

| 字段 | V4/V5 值 |
|---|---|
| `token_taxonomy_version` | `token_v4` 或 `token_v5` |
| `token_taxonomy_artifact_sha256` | 冻结 artifact SHA256 |
| `domain_control_token_candidate_ids` | 对应版本的 Control-only IDs |
| `structure_token_loss_weighting_enabled` | `true` |
| `structure_token_loss_weight` | `4.0`，或显式变体 `8.0`；必须等于 Control raw weight |
| `structure_token_position_profile` | `token_v4_v5_fourbaseline_r3_answer_format` |
| `domain_structure_token_ids` | 对应版本的 Structure IDs |

validator 要求版本、SHA、domain、ID 集合、C/S 互斥与一致的 Fixed4/Fixed8 匹配冻结定义，
防止 YAML 静默漂移。source metrics 也按显式 version/domain membership 分类，不再借用
旧 809-token global taxonomy。

新 JSON SHA256 为 `2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8`。
revision-1 JSON/CSV、配置快照保存在 `mopd_verl/domain_gradient/token_taxonomy_history/revision_1/`。
revision-2 JSON/CSV/provenance 与 44 份配置快照保存在同级 `revision_2/`。
所有新 run、audit、eval、checkpoint namespace 含 `-r3-`，继续 `wandb_resume: never`
与 `trainer.resume_mode=disable`，拒绝 revision-1/2 SHA/profile。

## Teacher confidence

confidence 为 teacher 对**实际生成 token**的完整词表 log-probability：
`log p_T(y_i | context_i)`，不是 teacher entropy、最大类别概率或 Top-K 内重归一化概率。
读取已有 teacher 输出，不新增 teacher inference。token-ID 模式取 mean log-probability，
对应概率的 geometric mean；不宣称它等于 mean probability。

| head 选项 | tail 选项 | 排名信号 |
|---|---|---|
| `top_loss` | `bottom_loss` | raw loss：高分增权、低分降权 |
| `top_teacher_confidence` | `bottom_teacher_confidence` | chosen-token logp：高 confidence 增权、低 confidence 降权 |
| `top_loss_teacher_confidence` | `bottom_loss_teacher_confidence` | `L_hat + C_hat + L_hat*C_hat` |

head 和 tail 可独立搭配，例如 head 用 Loss+TC、tail 只用 bottom loss。
head 还沿用现有 per-domain selection mode 配置。composite 在同一 domain 的全部
eligible units 上分别做 loss/logp 的 Q2–Q98 clipping 和 min–max normalization；
tail 排除 head 后不会重新拟合 normalization。

低 chosen-token confidence 表示 teacher 不认可该生成 token，不能据此断言 teacher
不可靠。`bottom_loss_teacher_confidence` 是 composite 的低分端，并非同时满足
low-loss 与 low-confidence 两个阈值；它与纯 bottom-loss 消融回答不同的问题。

## 梯度与归一化

raw weights 为 `head=4 / middle=1 / tail=alpha`。显式 `current_step` 的两个粒度
都沿用 regular token-ID 路径的 **rank-local micro-batch/domain mean-one**：

```text
D = 1 + 3*r_head - (1-alpha)*r_tail
effective weights = 4/D, 1/D, alpha/D
```

其中比例取自本次局部归一化范围，不能用 source 全局预算直接替代。
alpha 降低也会放大 head 和 middle 的有效系数。mean-one 不保证梯度范数不变。
若该 domain 的整个局部有效块都属于 tail 且 alpha=0，则该块保持全零，并记录
zero-weight event，不产生除零，不回退到普通监督。padding 权重为 0。

与原实现一致，gradient gate 保持 forward loss 数值，用 detached mask 改变梯度。
alpha=0 只移除 tail 位置自身损失的直接梯度，共享参数仍可被其他位置更新；Adam
动量与 weight decay 也不会因此自动停用。tail 位置下次仍重新评分，可以退出 tail。

历史 `selection_unit: occurrence` 且未显式设置 current_step 的 profile 保持其已有
same-step / global-domain normalization；它与这里的显式 current_step normalization
不同。比较实验时必须同时记录 timing、unit 与 normalization scope。

## 开销、日志与恢复

以下开销与日志也适用于 V6/V7/V8 Current-Step 路径；原有 Next-Step 配置不执行
额外 student 评分 forward，而是利用上一 step 正式 forward 的原始 loss 更新下一步选择。

每步额外一遍 student scoring forward，以及全局评分统计通信。teacher 已缓存结果
继续复用。不能把置零比例当作 FLOPs 节约，也不能未经测量声称总时间翻倍。

日志包含 `domain/current_step/{head,tail}/budget_count`、`selected_count`、`coverage`、
`shortfall`、`overshoot`，以及 `normalizer_min`、`effective_weight_max`、
`zero_weight_microbatch_count`。后面三个指标跨 rank 分别取 min、max、sum。
另有 scoring prepass 时间、forward 数及 gathered candidates 数。
开启 full-gradient audit 时，loss amplification 使用实际位置 mask；不输出依赖旧
static-ID membership 的 Control 计数或没有独立 oracle 的 gradient-mask error。

本步 mask 绑定 micro-batch 对象，保留至 backward 和 post-step source metrics 完成。
新 prepass 清空旧 mask；current-step 不读取或持久化 lagged active-ID state，resume
后重新评分。eval metadata 关闭当前步选择和 tail 权重。

## 已提供的 Math+Code 配置

均位于 `configs/token_selection/math_code/taxonomy/`。regular Math+Code head-only 配置为
`mopd_math_code_next_step_toploss_m05_c01_fixed4_4gpu.yaml`：batch 528，Math/Code
top-p 分别为 5%/1%，首步无选中 ID，之后使用上一 step 生产 loss 的统计。旧同名
`current_step` 文件为指向此配置的 alias。

原四个非默认 tail/TC 例子（TopLoss tailhalf/tailzero、Teacher Confidence tailhalf、
Loss+Teacher Confidence tailhalf）仍只实现 same-step 语义，未作为 active YAML 保留，
不能把它们的 timing 改为 `next_step` 后直接运行。原始 YAML 保存在
`tests/fixtures/next_step_migration/current-step-configs.json.gz` 及 Git
commit `806795e`；后续若需比较这些机制，应先实现并验证对应的 lagged selector。

Token V4/V5 提供 18 个主设置 public profile，命名为：

```text
mopd_math_code_next_step_token_v{4,5}_toploss_
m05_c{05,02,01}_fixed4_{3,4,8}gpu_colocated.yaml
```

另有同一矩阵的 18 个 `fixed8` profile，分别继承对应 `fixed4`，仅覆盖 C/S raw weight
为 8 及独立 namespace（含 `-r3-`、`-next-` 和 `-f8-`）。Math/Code top-p、batch 和
topology 保持一致。首步只有位置门控 Structure 加权；第 t 步产生的未加权原始 loss
用于第 t+1 步选择 Control IDs，不做额外的评分 forward。全部 36 个旧 `current_step`
文件名保留 alias，但解析后的机制与相应 `next_step` profile 完全相同。
本轮不包含 S-only sweep。后续实验需同报 Math4/Code4 以及 code extraction failure、
无 `\\boxed` 比例、截断率和平均长度；当前实现测试不等同于这些训练/评测结果。

所有 profile 都使用 Math/Code 1:1 sampling、global batch 528、PPO minibatch 528。
3/4/8 actor ranks 分别处理 176/132/66 samples；每 domain/rank 为 88/66/33。
8-GPU 版本让 student actor 与 8-way FSDP teacher shard 共存于全部 GPU，关闭
dedicated-teacher performance wrapper。现有 6-student+2-teacher 配置保留为 fallback；
colocated 版本是否更快必须用 1–2 step smoke/profile 实测，不能仅由 topology 推断。
