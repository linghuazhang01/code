# Control、Structure、Other 与 VR Token 定义

更新日期：2026-08-31

本文档是当前项目中 Control、Control-44、Connective、Structure、Other 和 VR token
集合的规范定义。除非另有说明，token ID 均绑定 Qwen3 tokenizer，不能直接复用于
其他 tokenizer。

## 0. 项目全局 Four-Baseline-Supported Taxonomy（当前主定义）

后续 token composition、Rising/Stable Top-200 和 recall 分析统一使用本节。第 1–7 节
保留 Semantic Control、Control-44、Control Structure V1/V2 与 VR selector 的历史
provenance，但不能覆盖本节的 active taxonomy。

令 `BaseControl` 为现有 V2 `connective_structure ∪ frozen Control-44`，令
`BaseStructure` 为全局 `format_structure ∪ code_lexical_structure` 并排除
`BaseControl`。二者首先在完整 tokenizer vocabulary 上互斥定义，且不随 domain 改变。

对四条固定 baseline `B={1.7B-OPD, 1.7B-EOPD, 4B-OPD, 4B-EOPD}` 定义：

```text
M_b(t) = max_{d ∈ {math,code,science}, step} occurrence(b,d,step,t)

GlobalControl   = {t ∈ BaseControl   | ∀b ∈ B, M_b(t) > 20}
GlobalStructure = {t ∈ BaseStructure | ∀b ∈ B, M_b(t) > 20}
GlobalOther     = Vocabulary \ (GlobalControl ∪ GlobalStructure)
```

这里严格使用 `>20`，不是 `>=20`。冻结结果为：

| Global type | Token count |
|---|---:|
| Control | 175 |
| Structure | 634 |
| Control ∪ Structure | 809 |

Domain subset 只能从这 809 个 global token 二次筛选。令
`M_{b,d}(t)=max_step occurrence(b,d,step,t)`，则：

```text
DomainControl_d   = {t ∈ GlobalControl   | ∀b ∈ B, M_{b,d}(t) > 20}
DomainStructure_d = {t ∈ GlobalStructure | ∀b ∈ B, M_{b,d}(t) > 20}
```

| Domain | Control | Structure | Total |
|---|---:|---:|---:|
| Math | 124 | 266 | 390 |
| Code | 157 | 551 | 708 |
| Science | 128 | 244 | 372 |

同一 token 可以进入多个 domain subset，但 global type 不得改变。candidate pool 不是
taxonomy 来源；任意 pool `P_d` 的有效 selector 集合必须为
`P_d ∩ (DomainControl_d ∪ DomainStructure_d)`。

机器可读完整清单、字符、provenance、四 baseline maximum occurrence、domain subset 与
三套 candidate-pool audit 位于：
`analysis-output/four-baseline-global-token-taxonomy/`。其中
`tables/global-taxonomy.csv` 是 809-token global inventory，`tables/domain-subsets.csv`
是 domain membership，`tables/candidate-pool-membership.csv` 保留所有有效与被排除项；
重建入口为 `build_taxonomy.py`。

### 0.1 Rising/Stable Top-200（当前 active 定义）

每条 baseline 的 phase endpoints 固定如下，不能根据 Top-200 结果重新调 boundary：

| Baseline | Rising | Stable |
|---|---:|---:|
| 1.7B-OPD | 1→35 | 36→52 |
| 1.7B-EOPD | 1→37 | 37→65 |
| 4B-OPD | 6→20 | 21→45 |
| 4B-EOPD | 1→20 | 21→55 |

对每个 `baseline × domain × phase × token` 定义：

```text
speed_j = (gap_start(j) - gap_end(j)) / (end_step - start_step)
```

其中 gap 是该 token 的 mean absolute teacher–student log-probability gap。只有 start 和
end occurrence 都严格 `>20` 且 gap 有限的 token eligible。在完整 eligible vocabulary
中按 speed 降序、token ID 升序打破 tie，Rising/Stable 各自独立取 Top-200；然后才用
本节的 domain Control/Structure subsets 分类，余下 token 归 Other。因此这些统计严格
表示 **Top-200 内的 taxonomy composition**，不表示完整 taxonomy 中 Control 或
Structure token 总量发生了变化。

### 0.2 Candidate-pool Recall（当前 active 定义）

三套 raw pool 为 ExpandedPruned-V2、Robust190 和 Control-44；每个 domain 的实际
候选集合先与 `DomainControl ∪ DomainStructure` 相交。对 `window=1..20`、`K=1..30`
重放 A/C/E/F，并分别测试 next-step、next-window 与 split/unified：

- selection boundary 从每个 window 最早的 supported step 开始，随后每隔恰好
  `window` 个 optimizer steps 触发；score 只使用 boundary 及历史数据；
- token 在 source metric 所需每个 snapshot 的 occurrence 必须严格 `>20`；
- next-step 是 `t→t+1` endpoint speed；next-window 是 `t→t+window` endpoint speed，
  且要求 `t..t+window` 每个 snapshot occurrence 严格 `>20`；两者都从完整 eligible
  vocabulary 中选 global Top-200；
- A/C 是 source window 内 occurrence-weighted mean gap/Student entropy；E 是 aggregate
  vector 层面可恢复的 normalized `A+C+A*C` proxy，并非 raw occurrence-level E；
  F 是历史 endpoint speed `(gap[t-window]-gap[t])/window`；
- split 为 Control、Structure 各选 K，unified 为二者并集共选 K。模式比较另用等总
  budget：split `K/类型` 对 unified `2K/并集`。

对当次通过 source occurrence gate 的动态候选集合 `DynamicPool` 定义：

```text
Actual       = |DynamicPool ∩ FutureTop200|
Hits         = |Selected ∩ FutureTop200|
Precision    = Hits / requested K
PoolRecall   = Hits / Actual
PoolF1       = 2 * Precision * PoolRecall / (Precision + PoolRecall)
TypeActual   = |DomainTaxonomyType ∩ FutureTop200|
PoolCapacity = Actual / TypeActual
TypeRecall   = Hits / TypeActual
```

`Actual=0` 时 PoolRecall/PoolF1 unavailable，不能填 0。主 grid optimum 必须包含所有
expected `baseline × domain × token type` cells 且 full-K rate 为 100%；同时报告
PoolCapacity、TypeRecall、TypeF1 和每 cell event count。Control-44 的 Structure 集合
为空，因此 split complete evaluation unavailable。完整机器可读结果位于
`analysis-output/four-baseline-global-token-taxonomy/phase-top200-recall/`。

## 1. 名称与状态

| 名称 | 状态 | 含义 |
|---|---|---|
| Semantic Control | 分析口径 | 由 48 个归一化词面定义的 broad semantic family |
| Control Structure V0 / Control-44 | 历史版本 | 4B-OPD audit 得到的 44 个 global token IDs |
| Control Structure V1 | 历史版本 | PDTB Connective + Format/boundary + Code-only CodeLex |
| Control Structure V2 | 当前版本 | 将 Control-44 并入 Connective，Structure 保持独立 |
| All Other Token | 当前补集定义 | Qwen3 vocabulary 中不属于 V2 Connective/Structure family 的全部 token |
| VR 全集（VR Full Set） | 统计集合，当前不启用 | 每个 domain 的完整 4B-OPD Rising Top-500，即 `R_d^500` |
| VR Control and Structure Token | 独立 frozen selector 定义 | V2 Connective/Structure 与 `R_d^200` 的交集，Math/Code/Science 分别为 48/58/40 个 token；不等于 `domaincand_v1/v2` config whitelist |

特别注意：**Semantic Control 不等于 V0/Control-44**。前者由词面规则得到，可以
映射到大量 token IDs；后者是固定的 44 个 token IDs，不能从 48 个词面重新推导。

### 1.1 `domaincand_v1/v2` 与 Control Structure V1/V2 是两套版本号

config、run name 与 output path 中的 `domaincand_v1/v2` 表示 Semantic Control
candidate whitelist 的 source-audit 版本，不表示本文第 5–6 节的 Control Structure
taxonomy 版本：

| Config candidate version | Source whitelist | Math | Code | Science | Union |
|---|---|---:|---:|---:|---:|
| `domaincand_v1` | 4B source-audit Semantic Control | 185 | 206 | 188 | 209 |
| `domaincand_v2` | 1.7B source-audit Semantic Control | 188 | 207 | 188 | 210 |

例如
`mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_topk32_control_online_topklentropy_i3_w3_f20_k30_w4_b528.yaml`
直接把 `domaincand_v2` 的三张 ID 表写入
`domain_control_token_candidate_ids`。runtime 在每个 domain 的这一张联合 whitelist
内做 occurrence filter、score ranking 与 Top-K；它不会按本文的 Connective、
Structure 或 Other 再分 quota。

因此必须使用完整名称区分：

```text
domaincand V1/V2       = config Semantic Control whitelist version
Control Structure V1/V2 = full-vocabulary taxonomy version
VR ControlStructure V1/V2 = taxonomy 与 4B-OPD Rising Top-200 的交集
```

三者 membership 和用途均不同，不能仅凭 `V1`/`V2` 后缀互换。

另需区分 ranking interval 与使用区间：本文的 `R_d^K` 只根据 4B-OPD Rising
phase（step 6→20）的两个 endpoint 排名，**不是全训练过程统计**。所谓 frozen
training selector，是指从该 Rising interval 得到 membership 后，在后续训练中保持
不变；不能解释成训练全程持续重算 Top-K。

## 2. Control Token 的词面定义

对 decoded token 使用以下归一化：

```text
normalized_surface(token) = strip(lower(decoded_token(token)))
```

当归一化结果精确属于下列 48 个 surface 时，该 token 属于 broad
`Semantic Control` family：

| 子类 | 归一化 surface |
|---|---|
| Conditional/causal | `if`, `because`, `therefore`, `thus`, `hence`, `however`, `so`, `assume`, `assuming`, `suppose`, `imply` |
| Sequence/transition | `then`, `also`, `instead`, `similarly`, `alternative`, `conclusion`, `conclude` |
| Logical operator | `and`, `or`, `but`, `not`, `either`, `necessarily` |
| Output/reasoning structure | `answer`, `note`, `notes`, `example`, `problem`, `step`, `steps`, `case`, `cases`, `option`, `output`, `reason`, `explanation`, `instruction` |
| Procedural control | `wait`, `try`, `check`, `correct`, `use`, `initialize`, `generate`, `find`, `recall`, `proceed` |

这是 exact-match 规则，不使用 substring、stemming 或 embedding similarity。因为大小写、
前导空格和 BPE 形式不同，同一个 surface 可以对应多个 global token IDs。在现有
inventory 中，1.7B/4B 分别观察到 210/209 个 Semantic Control token IDs。

## 3. Control Structure Token V0：历史 Control-44

为统一 V0/V1/V2 的版本命名，本文将历史 fixed `Control-44` selector 记为
`Control Structure V0`，简称 `V0`。旧 config、W&B run 和分析 artifact 通常只写
`control-fixed`、`Control-44` 或 `C44`，未必显式包含 `v0`；这些名称在本项目的
版本对齐语境中指向同一个 44-ID frozen selector。V0 不区分 Math/Code/Science，
三个 domain 使用同一集合。

记冻结集合为 `C44`。它包含下列 44 个 Qwen3 global token IDs；`␠` 表示一个
前导空格：

| Token ID | Decoded token | Token ID | Decoded token |
|---:|---|---:|---|
| 300 | `as` | 641 | `In` |
| 758 | `␠In` | 983 | `to` |
| 1083 | `␠also` | 1156 | `␠first` |
| 1249 | `To` | 1416 | `␠If` |
| 1431 | `␠now` | 1986 | `This` |
| 2014 | `␠To` | 2055 | `␠So` |
| 2121 | `As` | 2461 | `For` |
| 2679 | `If` | 2938 | `␠That` |
| 4226 | `␠answer` | 4354 | `␠However` |
| 4416 | `So` | 4695 | `␠Now` |
| 5005 | `␠Then` | 5338 | `First` |
| 7039 | `Now` | 7281 | `␠Also` |
| 8704 | `␠Since` | 9112 | `Note` |
| 9211 | `␠Because` | 9658 | `␠assume` |
| 11209 | `However` | 11284 | `since` |
| 12209 | `Then` | 12549 | `Since` |
| 13023 | `␠Final` | 13394 | `Also` |
| 15277 | `␠Therefore` | 16085 | `␠hence` |
| 16141 | `Answer` | 17949 | `Because` |
| 19357 | `Final` | 21806 | `␠Answer` |
| 22477 | `␠suppose` | 44500 | `Thus` |
| 54815 | `Therefore` | 73877 | `␠Conclusion` |

形式化表示：

```text
C44 = {
  300, 641, 758, 983, 1083, 1156, 1249, 1416, 1431, 1986, 2014,
  2055, 2121, 2461, 2679, 2938, 4226, 4354, 4416, 4695, 5005,
  5338, 7039, 7281, 8704, 9112, 9211, 9658, 11209, 11284, 12209,
  12549, 13023, 13394, 15277, 16085, 16141, 17949, 19357, 21806,
  22477, 44500, 54815, 73877
}
```

Control-44 是 outcome-conditioned audit set。它适合复现历史分析和构造 V2
selector，但不应被解释为与模型、tokenizer 或训练轨迹无关的通用语言学集合。

## 4. 公共基础集合

后续两个版本使用相同的基础集合：

- `V`：当前 Qwen3 tokenizer 的完整 global token vocabulary。
- `T_{b,d,p}^K`：baseline `b`、domain `d`、phase `p` 中按 endpoint absolute
  chosen-token log-probability gap reduction 选出的动态 Top-K token types；token
  必须在该 phase 的两个 endpoint 都至少出现 20 次。
- `R_d^K`：domain `d` 在 4B-OPD Rising（step 6→20）中的动态 Top-K token
  types，即 `R_d^K = T_{4B-OPD,d,Rising}^K`。`R_d^500` 命名为 `VR 全集`，仅作
  统计且当前不用于训练；VR frozen-selector 口径使用 V2
  Connective/Structure 与 `R_d^200` 的交集，并命名为
  `VR Control and Structure Token`。它不等于 `domaincand_v1/v2` whitelist；Top-100
  只作为 sensitivity analysis。
- `P`：official PDTB-3 explicit connective inventory 的 exact single-token match。
- `F`：deterministic Format/boundary token。满足以下任一条件：sequence special
  token、decoded surface 仅含 whitespace/newline、或所有 non-space 字符均为
  Unicode punctuation。
- `L_d`：CodeLex。仅当 `d=code` 时启用，使用 case-sensitive exact match，覆盖
  Python keyword/builtin、programming language、module、method/API 和 programming
  lexicon；Math/Science 中 `L_d=∅`。

## 5. Control Structure Token V1（历史版本）

V1 是原 `Structural+CodeLex Rising Top-200` selector。为与 V2 做严格的
`token type × domain × baseline` 对比，V1 的三类 token 也必须先在完整
Qwen3 vocabulary `V` 上定义，Top-K、phase 和 support 均不参与类别定义：

```text
ConnectiveToken_v1 = P

StructureToken_v1(d) = (F ∪ L_d) \ P

AllOtherToken_v1(d)
  = V \ (ConnectiveToken_v1 ∪ StructureToken_v1(d))

ConnectiveToken_v1 ⊎ StructureToken_v1(d) ⊎ AllOtherToken_v1(d) = V
```

其中 `AllOtherToken_v1` 是 V1 在完整词表上的补集，不能复用 V2 的
`AllOtherToken_v2`。只有在进行 Top-K 展示时，才将这三个全词表类别分别与
`R_d^K` 或逐 step 动态 Top-K 取交集。原 frozen selector 的互斥 presentation
因此写为：

```text
Connective_v1(d, K) = P ∩ R_d^K

Structure_v1(d, K) = ((F ∪ L_d) \ P) ∩ R_d^K

ControlStructure_v1(d, K)
  = Connective_v1(d, K) ∪ Structure_v1(d, K)
```

也就是说，V1 的 Connective 仅包含 PDTB connective；Structure 包含
Format/boundary，以及仅在 Code domain 生效的 CodeLex。Connective 优先级高于
Structure，因此两类互斥。

### V1 数量

| Domain | Top-K | Connective V1 | Structure V1 | Union V1 |
|---|---:|---:|---:|---:|
| Math | 100 | 18 | 6 | 24 |
| Code | 100 | 12 | 15 | 27 |
| Science | 100 | 15 | 2 | 17 |
| Math | 200 | 26 | 13 | 39 |
| Code | 200 | 23 | 27 | 50 |
| Science | 200 | 24 | 9 | 33 |

V1 已退役，不应再作为当前 Qwen3-4B follow-up config 的解释口径。

### V1 token 清单（4B-OPD Rising Top-200 frozen selector）

以下是 V1 的完整 `Connective ∪ Structure` membership。清单按 Rising rank 排序；
`R<n>` 表示 token 在对应 domain 的 Rising Top-200 中排名为 `<n>`，`␠` 表示空格，
`↵` 表示换行。V1 的总数为 Math/Code/Science `39/50/33`。

#### Math V1（39）

```text
Connective (26):
54815=Therefore [R6], 67691=Similarly [R8], 15277=␠Therefore [R12],
3830=From [R17], 2870=where [R20], 2679=If [R23], 16085=␠hence [R33],
641=In [R42], 2121=As [R44], 1752=␠For [R48], 1416=␠If [R49],
17949=Because [R52], 12209=Then [R59], 4416=So [R80], 2461=For [R85],
333=if [R86], 269=or [R89], 5005=␠Then [R90], 2058=␠still [R108],
1075=␠like [R113], 13394=Also [R123], 476=␠or [R142],
4518=␠instead [R162], 11209=However [R168], 2055=␠So [R183],
438=␠as [R198]

Structure (13):
151645=<|im_end|> [R3], 2303=␠␠↵ [R40], 44364=---↵↵ [R41],
820=#### [R45], 549=␠: [R69], 608=␠/ [R78], 715=␠↵ [R105],
2137=}, [R119], 96065={( [R174], 56177=**↵↵ [R175],
19788=␠(\ [R185], 1959=␠— [R196], 10293=␠(- [R197]
```

#### Code V1（50）

```text
Connective (23):
17949=Because [R4], 4197=with [R11], 10694=after [R12], 1083=␠also [R21],
4416=So [R34], 5005=␠Then [R41], 300=as [R45], 269=or [R52],
1393=␠while [R69], 2055=␠So [R73], 758=␠In [R79], 1416=␠If [R97],
12209=Then [R103], 44500=Thus [R105], 4518=␠instead [R108], 3243=only [R111],
2679=If [R114], 1221=␠then [R156], 1576=␠because [R175],
1790=␠next [R186], 323=␠and [R187], 641=In [R188], 773=␠so [R190]

Structure (27):
1159=␠import [R1], 13027=␠Python [R2], 746=set [R13], 7252=)] [R25],
943=␠type [R26], 2822=()↵↵ [R27], 3767=any [R42], 474=import [R44],
5097=Output [R55], 151645=<|im_end|> [R58], 750=def [R61],
3270=␠write [R78], 44364=---↵↵ [R84], 1494=␠pass [R94], 1648=): [R98],
7213=↵␠␠␠␠↵ [R107], 2550=␠output [R122], 7609=[- [R142],
1887=␠main [R150], 1173=␠print [R151], 4318=␠[[ [R152],
10907=]))↵ [R157], 565=## [R167], 3647=␠abs [R181], 5563=)]↵ [R185],
981=␠␠␠␠␠␠ [R194], 516=', [R195]
```

#### Science V1（33）

```text
Connective (24):
11209=However [R6], 15277=␠Therefore [R10], 641=In [R12],
54815=Therefore [R14], 333=if [R25], 2679=If [R48], 8704=␠Since [R53],
1752=␠For [R56], 300=as [R59], 269=or [R60], 1416=␠If [R63],
1573=␠before [R88], 4416=So [R91], 758=␠In [R95], 8450=␠thus [R99],
2474=␠since [R112], 44500=Thus [R129], 2058=␠still [R138],
11284=since [R139], 2055=␠So [R145], 1499=from [R147],
1083=␠also [R157], 773=␠so [R170], 1221=␠then [R171]

Structure (9):
508=␠[ [R4], 44364=---↵↵ [R52], 1019=**↵ [R116], 568=). [R128],
14374=### [R166], 2376=)( [R172], 624=.↵ [R174], 35702={\ [R178],
330=␠" [R179]
```

## 6. Control Structure Token V2 与 All Other Token（当前版本）

V2 的三类 token **先在完整 Qwen3 vocabulary `V` 上定义**。Top-K、phase、support
和 speed 都不参与 token 类别定义：

```text
ConnectiveToken_v2 = P ∪ C44

StructureToken_v2(d) = (F ∪ L_d) \ ConnectiveToken_v2

AllOtherToken_v2(d)
  = V \ (ConnectiveToken_v2 ∪ StructureToken_v2(d))
```

因此，`AllOtherToken_v2`（亦称 `Other Token` 或 `Others Token`）是完整词表中除
`ConnectiveToken_v2` 与 `StructureToken_v2` 以外的**全部剩余 token**，不是
Top-200 的补集，也不局限于任何 Top-K。三类在完整词表上两两互斥且完备：

```text
ConnectiveToken_v2 ⊎ StructureToken_v2(d) ⊎ AllOtherToken_v2(d) = V
```

Top-K 只用于统计展示。对任意 baseline `b`、domain `d`、phase `p`，先独立得到
`T_{b,d,p}^K`，再把上述全词表类别与它取交集：

```text
TopKConnective_v2(b, d, p, K)
  = ConnectiveToken_v2 ∩ T_{b,d,p}^K

TopKStructure_v2(b, d, p, K)
  = StructureToken_v2(d) ∩ T_{b,d,p}^K

TopKOther_v2(b, d, p, K)
  = AllOtherToken_v2(d) ∩ T_{b,d,p}^K

T_{b,d,p}^K
  = TopKConnective_v2 ⊎ TopKStructure_v2 ⊎ TopKOther_v2
```

其中 `⊎` 表示 disjoint union。因此每个 Top-200 cell 的三类 token count 必须严格
合计为 200，三类 token-type share 必须严格合计为 100%。图表中的灰色
`All Other` 段仅表示全局 `AllOtherToken_v2` 中落入该 cell Top-200 的成员；这不会
把 `AllOtherToken_v2` 的定义域缩小到 Top-200。类别定义不得附加 paired-support、
fixed-Rising、speed availability 或其他二次过滤。若 speed 分析因跨 phase support
另行过滤 token，该 support-filtered control 只能标为分析子集，不能改称
`All Other Token`。

V2 不是把全部 44 个 ID 无条件写入每个 domain。一个 Control-44 ID 只有进入相应
baseline/domain/phase 的 Top-K 时才计入 dynamic composition；当前 frozen config
固定的是 V2 Connective/Structure 与 `R_d^200` 的交集，而不是完整的
`R_d^200`。对该 frozen selector，可继续使用简写：

```text
ConnectiveSelector_v2(d, K) = ConnectiveToken_v2 ∩ R_d^K

StructureSelector_v2(d, K) = StructureToken_v2(d) ∩ R_d^K

OtherSelector_v2(d, K) = AllOtherToken_v2(d) ∩ R_d^K

ControlStructure_v2(d, K)
  = ConnectiveSelector_v2(d, K) ∪ StructureSelector_v2(d, K)
```

### VR 命名与集合关系

`VR` 在本文中是集合命名前缀，不改变 V2 taxonomy。两个 VR 集合定义为：

```text
VRFullSet(d) = R_d^500

VRControlStructureToken(d)
  = (ConnectiveToken_v2 ∪ StructureToken_v2(d)) ∩ R_d^200
```

因此，`VR 全集` 是每个 domain 完整的 500 个 Rising-ranked token，包含 Connective、
Structure 和 Other；`VR Control and Structure Token` 则只保留 Rising Top-200 中
属于 V2 Connective 或 Structure 的 token。二者既不是同一大小，也不是同一用途。

`VR 全集` 当前只保留统计，不写入训练 config、不参与 reweight，也不建立 checkpoint
或 experiment namespace。其 V2 composition 如下：

| Domain | VR 全集 | Connective V2 | Structure V2 | All Other V2 | C+S Union |
|---|---:|---:|---:|---:|---:|
| Math | 500 | 62 | 46 | 392 | 108 |
| Code | 500 | 52 | 73 | 375 | 125 |
| Science | 500 | 58 | 49 | 393 | 107 |

### VR Control and Structure Token 的 V2 数量（Rising Top-200）

下表保留 Top-100 sensitivity analysis 作为参考；只有 Top-200 的 C+S Union
`48/58/40` 是当前 `VR Control and Structure Token`。

| Domain | Top-K | Connective V2 | Structure V2 | All Other V2 | C+S Union |
|---|---:|---:|---:|---:|---:|
| Math | 100 | 23 | 6 | 71 | 29 |
| Code | 100 | 18 | 15 | 67 | 33 |
| Science | 100 | 20 | 2 | 78 | 22 |
| Math | 200 | 35 | 13 | 152 | 48 |
| Code | 200 | 31 | 27 | 142 | 58 |
| Science | 200 | 31 | 9 | 160 | 40 |

### VR Control and Structure Token 清单（V2 Rising Top-200）

以下是当前 `VR Control and Structure Token` 的完整 V2
`Connective ∪ Structure` membership，按 Rising rank 排序。其总数为
Math/Code/Science `48/58/40`。`AllOtherToken_v2` 是完整 vocabulary 补集，不属于
训练时冻结的 `VR Control and Structure Token`，因此不在此逐项展开。

#### Math V2（48）

```text
Connective (35):
73877=␠Conclusion [R1], 13023=␠Final [R5], 54815=Therefore [R6],
67691=Similarly [R8], 15277=␠Therefore [R12], 3830=From [R17],
2870=where [R20], 1986=This [R22], 2679=If [R23], 16085=␠hence [R33],
641=In [R42], 2121=As [R44], 1752=␠For [R48], 1416=␠If [R49],
17949=Because [R52], 12209=Then [R59], 4416=So [R80], 4226=␠answer [R81],
2461=For [R85], 333=if [R86], 269=or [R89], 5005=␠Then [R90],
21806=␠Answer [R91], 2058=␠still [R108], 1075=␠like [R113],
13394=Also [R123], 9112=Note [R126], 7039=Now [R135], 476=␠or [R142],
1156=␠first [R144], 4518=␠instead [R162], 11209=However [R168],
9658=␠assume [R170], 2055=␠So [R183], 438=␠as [R198]

Structure (13):
151645=<|im_end|> [R3], 2303=␠␠↵ [R40], 44364=---↵↵ [R41],
820=#### [R45], 549=␠: [R69], 608=␠/ [R78], 715=␠↵ [R105],
2137=}, [R119], 96065={( [R174], 56177=**↵↵ [R175],
19788=␠(\ [R185], 1959=␠— [R196], 10293=␠(- [R197]
```

#### Code V2（58）

```text
Connective (31):
17949=Because [R4], 1249=To [R7], 4197=with [R11], 10694=after [R12],
983=to [R17], 1083=␠also [R21], 7039=Now [R30], 4416=So [R34],
5005=␠Then [R41], 300=as [R45], 269=or [R52], 2014=␠To [R65],
1393=␠while [R69], 2055=␠So [R73], 758=␠In [R79], 13023=␠Final [R91],
1986=This [R92], 1416=␠If [R97], 1431=␠now [R102], 12209=Then [R103],
44500=Thus [R105], 4518=␠instead [R108], 3243=only [R111],
2679=If [R114], 2938=␠That [R126], 1221=␠then [R156],
1576=␠because [R175], 1790=␠next [R186], 323=␠and [R187],
641=In [R188], 773=␠so [R190]

Structure (27):
1159=␠import [R1], 13027=␠Python [R2], 746=set [R13], 7252=)] [R25],
943=␠type [R26], 2822=()↵↵ [R27], 3767=any [R42], 474=import [R44],
5097=Output [R55], 151645=<|im_end|> [R58], 750=def [R61],
3270=␠write [R78], 44364=---↵↵ [R84], 1494=␠pass [R94], 1648=): [R98],
7213=↵␠␠␠␠↵ [R107], 2550=␠output [R122], 7609=[- [R142],
1887=␠main [R150], 1173=␠print [R151], 4318=␠[[ [R152],
10907=]))↵ [R157], 565=## [R167], 3647=␠abs [R181], 5563=)]↵ [R185],
981=␠␠␠␠␠␠ [R194], 516=', [R195]
```

#### Science V2（40）

```text
Connective (31):
16141=Answer [R3], 11209=However [R6], 15277=␠Therefore [R10],
641=In [R12], 54815=Therefore [R14], 9658=␠assume [R22], 333=if [R25],
2679=If [R48], 8704=␠Since [R53], 1752=␠For [R56], 300=as [R59],
269=or [R60], 1416=␠If [R63], 21806=␠Answer [R64], 4226=␠answer [R67],
1573=␠before [R88], 4416=So [R91], 758=␠In [R95], 1986=This [R96],
8450=␠thus [R99], 2474=␠since [R112], 44500=Thus [R129],
2058=␠still [R138], 11284=since [R139], 2055=␠So [R145],
1499=from [R147], 1083=␠also [R157], 773=␠so [R170],
1221=␠then [R171], 1431=␠now [R191], 2938=␠That [R198]

Structure (9):
508=␠[ [R4], 44364=---↵↵ [R52], 1019=**↵ [R116], 568=). [R128],
14374=### [R166], 2376=)( [R172], 624=.↵ [R174], 35702={\ [R178],
330=␠" [R179]
```

### V1 → V2 新增的 Control-44-only token

下表列出 4B-OPD Rising Top-200 中不属于 V1、但因加入 Control-44 而进入 V2
Connective 的 token。`R≤100` 表示同时位于 Rising Top-100。

| Domain | Token ID | Decoded token | Rising rank | R≤100 |
|---|---:|---|---:|:---:|
| Math | 73877 | `␠Conclusion` | 1 | ✓ |
| Math | 13023 | `␠Final` | 5 | ✓ |
| Math | 1986 | `This` | 22 | ✓ |
| Math | 4226 | `␠answer` | 81 | ✓ |
| Math | 21806 | `␠Answer` | 91 | ✓ |
| Math | 9112 | `Note` | 126 |  |
| Math | 7039 | `Now` | 135 |  |
| Math | 1156 | `␠first` | 144 |  |
| Math | 9658 | `␠assume` | 170 |  |
| Code | 1249 | `To` | 7 | ✓ |
| Code | 983 | `to` | 17 | ✓ |
| Code | 7039 | `Now` | 30 | ✓ |
| Code | 2014 | `␠To` | 65 | ✓ |
| Code | 13023 | `␠Final` | 91 | ✓ |
| Code | 1986 | `This` | 92 | ✓ |
| Code | 1431 | `␠now` | 102 |  |
| Code | 2938 | `␠That` | 126 |  |
| Science | 16141 | `Answer` | 3 | ✓ |
| Science | 9658 | `␠assume` | 22 | ✓ |
| Science | 21806 | `␠Answer` | 64 | ✓ |
| Science | 4226 | `␠answer` | 67 | ✓ |
| Science | 1986 | `This` | 96 | ✓ |
| Science | 1431 | `␠now` | 191 |  |
| Science | 2938 | `␠That` | 198 |  |

V2 相对 V1 的新增量为 Math/Code/Science `+9/+8/+7`。当前数据中这些新增项
都不是 Format/boundary 或 CodeLex，因此 Structure 的数量保持 `13/27/9`。

### 6.1 VR Seed-Sibling Expanded ControlStructure Candidate Pool

为避免把 `domaincand_v1/v2` Semantic Control whitelist 错当成 Control/Structure
大池子，当前新增一套独立 candidate family：`VR Seed-Expanded ControlStructure V1/V2`。
它以第 5–6 节的原始 VR Rising Top-200 Control/Structure selector 为 Small seed，
对每个 seed 做确定性的 tokenizer sibling closure：

```text
Small_v(d) = VRControlStructure_v(d)

Expanded_v(d) = SeedLabelPreservingSiblingClosure(Small_v(d))

ExpandedPruned_v(d; 20)
  = Small_v(d)
    ∪ {t ∈ Expanded_v(d): max_archived_step_occurrence(t,d) ≥ 20}
```

扩充规则如下：

- Control：`casefold(strip(NFKC(decoded_token)))` 相同；
- Format Structure：special exact、whitespace signature exact 或 punctuation core exact；
- CodeLex Structure：仅 Code domain 启用规范化 lexical sibling；
- Control 与 Structure 冲突时 Control 优先；
- 每个新增 candidate 必须保存 source seed ID 和 match key。

这里的 `candidate_type` 是 **seed-label provenance**，不是对新增 sibling 重新执行
第 6 节完整 V2 taxonomy 后得到的类别。两者不得混用。例如，一个普通 content token
可能因为与某个 Control seed 具有相同规范化词面而被标记为 pool `Control`，但在完整
V2 taxonomy 中仍属于 `All Other`。因此：

```text
PoolCandidateType(t) ≠ FullVocabularyTaxonomyType_v2(t)  （允许）
```

- selector 的 pool composition、quota 和 provenance 可以使用 `candidate_type`；
- Top-200 三分类、Type Recall denominator 和“是否遗漏 Control/Structure”的审计，
  必须重新使用第 6 节的完整词表 taxonomy；
- 不得把 `Expanded_v(d)` 中所有 pool-Control sibling 都宣称为 V2 Control token。

`Small_v(d) ⊆ ExpandedPruned_v(d) ⊆ Expanded_v(d)` 是硬约束。occurrence pruning
只能删除新增 sibling，不能删除 Small seed；扩充和裁剪均不使用 future optimization
target。统一 threshold=20 后的最终数量为：

| Version | Domain | Small C/S | Raw Expanded C/S | Final@20 C/S | Final Total |
|---|---|---:|---:|---:|---:|
| V1 | Math | 26/13 | 107/73 | 64/24 | 88 |
| V1 | Code | 23/27 | 102/93 | 68/57 | 125 |
| V1 | Science | 24/9 | 94/71 | 63/24 | 87 |
| V2 | Math | 35/13 | 153/73 | 92/24 | 116 |
| V2 | Code | 31/27 | 137/93 | 89/57 | 146 |
| V2 | Science | 31/9 | 123/71 | 81/24 | 105 |

机器可读 membership、provenance、config-ready whitelist 与完整 replay 位于：
`analysis-output/V2_token_precision_recall/vr_seed_expanded_pool_search/`。
逐 baseline、逐 step 的完整 taxonomy Top-200 coverage audit 位于其子目录
`top200_coverage_audit/`。

### 6.2 Omitted-token Tier-A 特别高频定义

对 Future-5 boundary Top-200 中被 `ExpandedPruned-V2` 遗漏的完整-taxonomy
Control/Structure token，分别在每条 baseline 计算：

```text
OccurrenceRate_b(t) = # recorded steps with occurrence_b,s(t) >= 20
                      / # recorded steps in baseline b

MeanOccurrence_b(t) = mean_s occurrence_b,s(t)

FutureTop200Rate_b(t) = # Future-5 boundary cells where t enters its domain/type Top-200
                        / # valid Future-5 boundary cells in baseline b
```

这里的“最弱 baseline”对每个指标分别取四条 baseline 的 minimum；未进入某条
baseline 的 Future Top-200 必须补零，不能把该 baseline 从分母中删除。**Tier A**
同时满足：

1. `min_b OccurrenceRate_b(t) >= 0.80`；
2. `min_b MeanOccurrence_b(t) >= 20`；
3. `min_b FutureTop200Rate_b(t) >= 0.20`。

第三条把“最弱 baseline 里 Top-200 出现次数很高”操作化为至少 20% 的有效
Future-5 boundaries。四条 baseline 分别有 `9/14/7/10` 个 boundary，因此整数命中
下限分别为 `2/3/2/2`。只满足前两条的 token 记作 `occurrence-only high`，不能再称为
Tier A。注意 `occurrence_b,s(t)` 是一个 optimizer step 内的 token occurrence 数，
而 FutureTop200Rate 是跨 selection-boundary cells 的出现率，两者不是同一分母。

该 family 与以下两类都不同：

- `VR Control and Structure Token`：Small frozen Rising Top-200 intersection；
- `domaincand_v1/v2`：Semantic Control config whitelist，不是 Control/Structure 扩充池。

## 7. 当前 config families 与机器可读来源

当前仓库同时存在两类不可混用的 config family：

1. `structural_codelex_rising_top200` profiles 生效的是 V2
   `VR Control and Structure Token`；`VR 全集` 仅统计、不启用。
2. `control_online_*domaincand_v1/v2*` profiles 生效的是相应 Semantic Control
   domain whitelist，并在 whitelist 内动态选 Top-K。指定的 1.7B KL+Entropy config
   属于这一类，候选数为 `188/207/188`。

Qwen3-4B 的 5/6/8-GPU VR fixed 和 speed profiles 使用：

```text
configs/mopd_qwen4b_30b_a3b_instruct_2507_*gpu_math_code_science_topk32_
structural_codelex_rising_top200_control_{fixed_w4_b528,speed_pwl_u2}.yaml
```

文件名和 runtime 的 `control_token_*`/`domain_control_token_ids` 字段属于兼容性
legacy naming；这些 profile 内的实际语义是 V2 `Connective+Structure`，也就是
本文的 `VR Control and Structure Token`。训练、W&B、experiment 和 checkpoint
namespace 均含 `connective-structure` 标记，
audit/eval path 均含 `connective_structure` 标记，并统一使用 `v2` 后缀；不得
resume V1 namespace，也不得把未启用的 `VR 全集` 写成 active selector。

机器可读事实来源：

- Control-44 inventory：
  `../research_analysis/baseline-control-taxonomy-20260819/control44/tables/candidate-token-inventory.csv`
- `VR 全集` Top-500 ranking 原始 endpoint vectors：
  `../experiments_records/baseline/qwen4b_30b_a3b_instruct_2507_math_code_science/token_gap_vocab_vectors.jsonl`
- 四 baseline V2 membership：
  `../research_analysis/token-category-rising-stable-20260819/tables/four-baseline-top200-set-membership.csv`
- 四 baseline composition：
  `../research_analysis/token-category-rising-stable-20260819/tables/four-baseline-top200-set-composition-by-cell.csv`
- 四 baseline 三分类 macro composition：
  `../research_analysis/token-category-rising-stable-20260819/tables/four-baseline-top200-three-class-macro.csv`
- 四 baseline Rising/Stable 三分类柱状图：
  `../research_analysis/token-category-rising-stable-20260819/figures/10-four-baseline-top200-three-class-faceted.pdf`
- 完整分析说明：
  `../research_analysis/token-category-rising-stable-20260819/README.md`

如果文档、图表与机器可读 membership 发生冲突，应先重新运行分析构建脚本并核对
source manifest，而不是手工修改 token 数量。

## 8. A–F online selector 指标与 future-accuracy 口径

这一节冻结多指标 selector replay 的定义。设当前 valid response occurrence 为
位置 (i)，teacher/student 对实际生成 token 的 log-probability 分别为
\(\ell_i^T\) 与 \(\ell_i^S\)，Student entropy 为 \(H_i^S\)。

| 指标 | 定义 | 方向 |
|---|---|---|
| A | `logp diff` = \(|\ell_i^T-\ell_i^S|\) | 越大越优先 |
| B | TIP/FiRe weight 与 rollout-IS 生效前的 detached raw Top-K KL loss | 越大越优先 |
| C | `student entropy` = \(H_i^S\) | 越大越优先 |
| D | \(N(B_i)+N(C_i)+N(B_i)N(C_i)\) | 越大越优先 |
| E | \(N(A_i)+N(C_i)+N(A_i)N(C_i)\) | 越大越优先 |
| F | 历史 token optimization speed：\(F_w(t)=\frac{1}{w}\sum_{s=t-w}^{t-1}(G_s-G_{s+1})=\frac{G_{t-w}-G_t}{w}\)，其中 \(G_t\) 是 step \(t\) 的 token-ID mean absolute logp gap | 越大越优先 |

这里的 \(N(x_i)=x_i/\max_{j\in r}x_j\) 是在同一条 valid response `r`
内部的 max-normalization。它不是跨 token ID、跨 domain 或跨 step 的全局
normalization。D/E 随后与当前代码一致：先对 occurrence 算 score，再按
`(domain, token_id)` 在 rolling window 内用 occurrence count 聚合 mean；不得先对
token-ID mean 做相乘后声称是 exact online score。

标准 replay 使用 3-step historical window、每 5 optimizer steps 触发一次、每个
`(domain, token type)` 独立选 Top-20；mean occurrences per step 至少为 20，score
相同时按 token ID 升序打破平局。这里的 Control/Structure 使用第 6 节 V2
全词表定义，不限于 Top-200；Top-200 只用于 future outcome。

F 只使用 selection boundary 及其之前的 `t-w,...,t`，不使用任何 future
snapshot。为避免用稀疏 endpoint 估计速度，F 要求全部 `w+1` 个历史 endpoint 的
token count 均至少为 20。单独 replay F 时应报告自己的 candidate-pool reference；
与 A/C/E 做 controlled comparison 时，四者应使用 A/C count support 与 F endpoint
support 的交集；比较多个 window 时还必须取各 window eligible sets 的交集，使
window 之间的 candidate pool 完全相同。虽然 F 在代数上可 telescoping 为
\((G_{t-w}-G_t)/w\)，实现保留
逐段计算以明确时间方向和 endpoint 检查。

future accuracy 使用 chosen-token absolute logp gap 的下降速度，并冻结为以下两个
互补 outcome：

1. **主 target：`Future-5 start–end net optimization`**。对 token `j` 定义
   \(T_{\mathrm{net}}(j)=(G_t(j)-G_{t+5}(j))/5\)。只检查 start 与 end 两个
   gap，但 occurrence eligibility 同时要求：`count_t(j) > 20`、
   `count_{t+5}(j) > 20`，以及未来五个 snapshots
   `mean(count_{t+1:t+5}(j)) > 20`。这里的 mean 明确包含
   `t+1,t+2,t+3,t+4,t+5`；中间 gap 不参与速度公式，但中间 occurrence 通过该均值
   进入 eligibility。对满足条件的全词表 token 按 \(T_{\mathrm{net}}\) 降序排名，并取
   Top-200。令
   `Hits = |Selected Top-20 ∩ Future Top-200|`。原先泛称的 selection accuracy 必须
   严格记为 `Precision@20 = Hits/20`。对于当前 token type \(\tau\)，主召回率定义为
   `Type Recall@20 = Hits/|Future Top-200 ∩ C_{d,τ}|`，其中 \(C_{d,\tau}\) 是第 6 节
   在完整词表上定义的 domain-aware Control 或 Structure 集合。也就是说，Control
   selector 只召回 Future Top-200 中的 Control tokens，Structure 同理。另保留
   `Global Recall@20 = Hits/min(200, N_eligible)` 作为 audit metric；加入上述未来五步
   occurrence 均值限制后，当前四 baseline 所有 boundary 的 `N_eligible≥736`，所以
   Global Recall 分母仍恒为 200，但它不再是
   precision–recall 综合分数的主 recall。另报
   `Pool Recall = Hits/|Candidate Pool ∩ Future Top-200|`，用于衡量 candidate pool 中
   可达正例的召回；若可达正例数为 0，则该值 unavailable，不能填 0。start 或 end
   不满足门槛的 selected token 计为未命中，因而 Precision 分母始终为 20。标准
   classification accuracy 会被大量 true negatives 主导，不作为 selector 主指标。
   跨 baseline/domain 比较 rank 时使用相对于该 cell start–end eligible vocabulary
   的 rank percentile，并单独报告 endpoint coverage。
2. **辅助诊断：`Future-5 valid-step mean rank`**。分别对五个 one-step interval
   `s→s+1` 计算全词表 signed speed rank。令
   \(I_{s,j}=1\) 当且仅当 token `j` 在该 interval 两个 endpoint 的 count 均至少为
   20 且 gap 有限，并令 \(m_j=\sum_{s=t}^{t+4}I_{s,j}\)。逐-step 平均必须使用
   token-specific 动态分母：
   \[
   T_{\mathrm{step}}(j)=
   \frac{1}{m_j}\sum_{s=t}^{t+4}I_{s,j}\frac{r_{s,j}}{N_s},
   \qquad m_j>0,
   \]
   其中 `r_{s,j}` 是 interval 内 rank，`N_s` 是该 interval 的 eligible vocabulary
   数。某个中间 interval 因 occurrence 过少被忽略时，分子不含该 interval，分母
   `m_j` 也同步减一；`m_j=0` 时该辅助 target 记为 unavailable，而不是 1.0。
   必须同时报告 token coverage `m_j/5` 与 event coverage
   `Σ_j m_j/(20×5)`，防止只看 surviving intervals 产生 survivor bias。把缺失 interval
   固定惩罚为 percentile 1.0 的旧指标仅保留为 `legacy penalized diagnostic`，不再作为
   selector 比较的主 target。

### 8.1 Selection-window ablation

比较多个 selection window 时，A/C/E 分别聚合最近 (w) 个 signal snapshots；
F 使用过去 (w) 个完整优化区间。所有 window 必须限制在共同可用的 selection
boundaries，不能让短 window 多出的早期 step 进入主比较。`window=3/5/7/9`
sweep 还必须对四个 history-qualified eligible sets 取四者交集，并让全部 metric
都从这个固定 candidate pool 选择。window 之间的主比较统一使用上述
`Future-5 start–end net optimization` rank percentile 与 Top-200；逐-step 结果只作
辅助诊断，并严格采用随有效 interval 数变化的动态分母，同时展示 coverage。旧的
all-six-endpoint comprehensive target 与 fixed-penalty rank 不再进入主结论。

历史四 baseline 只共同保存了 A 与 C 的 token-ID aggregate vectors，没有保存
raw Top-K KL token vector；`teacher_student_cross_entropy` 是 CE，不能替代 B。
因此历史 replay 中 A/C 是 exact aggregate，F 是由 exact aggregate gap snapshots
得到的 exact derived metric，E 只能使用 domain-step/token-ID max normalization
的明确 proxy，B/D 必须标为 unavailable。只有 future run 落盘
response-level raw Top-K KL 与 normalization denominator 后，才可以把 B/D/E
标为 exact online replay。

### 8.2 Selection-size harmonic objective

当 metric A 的 selection cutoff 从固定 Top-20 扩展到 Top-K 时，必须在完全相同的
selection boundaries、四个 window 共同 candidate pool 和 Future-5 start–end target
上重放 A 排序。令 \(h_K\) 为前 K 个 A-selected token 中进入 Future Top-200 的数量，
并令 \(M_\tau=|Future Top\text{-}200\cap C_{d,\tau}|\)，则：

\[
P_K=\frac{h_K}{K},\qquad
R_K^{type}=\frac{h_K}{M_\tau}.
\]

用户指定的 precision–recall 综合分数冻结为：

\[
S_K=\left(\frac{1}{P_K}+\frac{1}{R_K}\right)^{-1}
=\frac{P_KR_K}{P_K+R_K}.
\]

标准 F1 为 \(F1_K=2S_K\)，因此是否乘 2 不会改变 K 的排序或最优 K。对 Type
Recall，分数还可化简为 \(S_K=h_K/(K+M_\tau)\)；若 \(h_K=0\)，约定
\(S_K=F1_K=0\)。Global audit 版本将 \(M_\tau\) 替换为
\(\min(200,N_{eligible})\)；Pool 版本则替换为

```text
M_pool = |Candidate Pool ∩ Future Top-200|
```

且 `M_pool=0` 时 Pool Recall、Pool score 和 Pool F1 均记为 unavailable，不能填 0。
Type score 是主 target；Global score 只作跨完整 Top-200 的 audit，Pool score 是
candidate-pool retrieval efficiency 的辅助 target，三者不得混称为同一种 recall。

聚合时，先在每个 `baseline×domain` cell 内累计 hits、selected count 与 target count，
由 ratio-of-sums 计算 cell-level Precision/Recall/F1；再对 12 个 cell 等权 macro
average。不能先平均 event-level F1。若比较不同 K，主表只保留全部 event 都支持的
范围；当前四 baseline、window=3/5/7/9 replay 的公平范围为 Control `K=1…34`、
Structure `K=1…47`，跨 token type 合并时为 `K=1…34`。并列最高时选择最小 K。
若最优 K 落在公平范围上界，结论必须写成 right-censored optimum 或“至少为该 K”，
不能声称已证明更大 K 不会继续改善。
