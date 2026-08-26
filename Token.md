# Control 与 Structure Token 定义

更新日期：2026-08-27

本文档是当前项目中 Control、Control-44、Connective 和 Structure token 的
规范定义。除非另有说明，token ID 均绑定 Qwen3 tokenizer，不能直接复用于其他
tokenizer。

## 1. 名称与状态

| 名称 | 状态 | 含义 |
|---|---|---|
| Semantic Control | 分析口径 | 由 48 个归一化词面定义的 broad semantic family |
| Control-44 | 历史冻结集合 | 4B-OPD audit 得到的 44 个 global token IDs |
| Control Structure V1 | 历史版本 | PDTB Connective + Format/boundary + Code-only CodeLex |
| Control Structure V2 | 当前版本 | 将 Control-44 并入 Connective，Structure 保持独立 |

特别注意：**Semantic Control 不等于 Control-44**。前者由词面规则得到，可以映射
到大量 token IDs；后者是固定的 44 个 token IDs，不能从 48 个词面重新推导。

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

## 3. 历史 Control-44 的精确定义

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

- `R_d^K`：domain `d` 在 4B-OPD Rising（step 6→20）中的动态 Top-K token
  types。排名指标是 endpoint absolute chosen-token log-probability gap reduction；
  token 必须在两个 endpoint 都至少出现 20 次。当前 frozen training selector 使用
  `K=200`，Top-100 只作为 sensitivity analysis。
- `P`：official PDTB-3 explicit connective inventory 的 exact single-token match。
- `F`：deterministic Format/boundary token。满足以下任一条件：sequence special
  token、decoded surface 仅含 whitespace/newline、或所有 non-space 字符均为
  Unicode punctuation。
- `L_d`：CodeLex。仅当 `d=code` 时启用，使用 case-sensitive exact match，覆盖
  Python keyword/builtin、programming language、module、method/API 和 programming
  lexicon；Math/Science 中 `L_d=∅`。

## 5. Control Structure Token V1（历史版本）

V1 是原 `Structural+CodeLex Rising Top-200` selector。其互斥 presentation
定义为：

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

## 6. Control Structure Token V2（当前版本）

V2 将完整 `C44` 加入 Connective 的定义，但在 Top-K 统计和 frozen selector 中只
保留相应 domain 的 `R_d^K` 内实际出现的 token：

```text
Connective_v2(d, K) = (P ∪ C44) ∩ R_d^K

Structure_v2(d, K) = ((F ∪ L_d) \ (P ∪ C44)) ∩ R_d^K

ControlStructure_v2(d, K)
  = Connective_v2(d, K) ∪ Structure_v2(d, K)
```

因此：

```text
Connective_v2 ∩ Structure_v2 = ∅
```

V2 不是把全部 44 个 ID 无条件写入每个 domain。一个 Control-44 ID 只有进入该
domain/phase 的 Top-K 时才计入 composition；当前 frozen config 只冻结
4B-OPD Rising Top-200 中的 membership。

### V2 数量

| Domain | Top-K | Connective V2 | Structure V2 | Union V2 |
|---|---:|---:|---:|---:|
| Math | 100 | 23 | 6 | 29 |
| Code | 100 | 18 | 15 | 33 |
| Science | 100 | 20 | 2 | 22 |
| Math | 200 | 35 | 13 | 48 |
| Code | 200 | 31 | 27 | 58 |
| Science | 200 | 31 | 9 | 40 |

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

## 7. 当前 config 与机器可读来源

当前生效的是 V2。Qwen3-4B 的 5/6/8-GPU fixed 和 speed profiles 均使用：

```text
configs/mopd_qwen4b_30b_a3b_instruct_2507_*gpu_math_code_science_topk32_
structural_codelex_rising_top200_control_{fixed_w4_b528,speed_pwl_u2}.yaml
```

文件名和 runtime 的 `control_token_*`/`domain_control_token_ids` 字段属于兼容性
legacy naming；这些 profile 内的实际语义是 V2 `Connective+Structure`。训练、
W&B、experiment 和 checkpoint namespace 均含 `connective-structure` 标记，
audit/eval path 均含 `connective_structure` 标记，并统一使用 `v2` 后缀；不得
resume V1 namespace。

机器可读事实来源：

- Control-44 inventory：
  `../research_analysis/baseline-control-taxonomy-20260819/control44/tables/candidate-token-inventory.csv`
- 四 baseline V2 membership：
  `../research_analysis/token-category-rising-stable-20260819/tables/four-baseline-top200-set-membership.csv`
- 四 baseline composition：
  `../research_analysis/token-category-rising-stable-20260819/tables/four-baseline-top200-set-composition-by-cell.csv`
- 完整分析说明：
  `../research_analysis/token-category-rising-stable-20260819/README.md`

如果文档、图表与机器可读 membership 发生冲突，应先重新运行分析构建脚本并核对
source manifest，而不是手工修改 token 数量。
