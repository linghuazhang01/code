# Token V4/V5 四 baseline revision 2 实现报告

状态：本地实现、文档、集中验证及主代理审核完成。主代理独立核对了全部 CSV/NPZ 计数，并检查解析样例、mask 评分/应用与 actor dispatch、配置权重及 revision 标识。未启动 GPU 训练、远端同步、commit 或 push。

## 定义与来源

采用用户 2026-09-28 第二版定义，权威输入目录：
`../plan/code-gap-20260927/structure-budget-context-20260927/`（相对于 code 仓库）。

| Version | Math C/S | Code C/S |
|---|---:|---:|
| V4 | 121/6 | 154/37 |
| V5 | 64/6 | 74/37 |

- JSON SHA256：`c45b5f7b6bd824140922706b2cf75630dce7130d580c346c638a3358833fd97b`。
- Membership SHA256：`f8efb0b8346e9145af0adabe065205b810fa04e7d9567beaa1632d9d6562ad70`。
- JSON/CSV 已逐字节与来源比较；每个 keep 集与 JSON 精确一致。
- 647 membership rows × 4 baseline = 2,588 个最大出现次数，逐项与 NPZ 比较为 0 mismatch。
- 所有 keep token 在每条 baseline 的相应 domain 均严格 `>20`；完整支持集 Math 3,686、Code 6,573。
- V5 保留 `control_sibling_added`，Math +5、Code +9；V4/V5 Control intersection 为 59/65。V5 不再是 V4 子集。
- Math Structure 用 1590 替换不满足支持条件的 16141；Code Structure 移除 5138 并采用源定义中的 37 IDs。

新 `token_v4_v5_provenance.json` 记录以上核验及源 JSON、CSV、四份 NPZ、三份构建脚本的 hash。
finalizer 已改为验证/导入入口，默认只验证，显式 `--install` 才导入；不再用旧 tokenizer-only
逻辑生成过时集合。它不重跑原始 trajectory 统计或完整 vocabulary closure，而是核验用户提供的冻结产物。

## 运行时语义

- Code Control：排除所有 fenced blocks，包括未闭合块直到回复末尾。prepass 频次和 mean loss
  聚合、selected ID 的最终加权，共用同一 `mopd_control_position_mask`。top-p 分母仍为完整 domain valid tokens。
- Code Structure：只使用最终闭合块的围栏/语言行，以及 I/O、函数签名、入口逻辑语句；排除 comments
  和 string literal token 位置。支持 multiline I/O/signature、stdin alias、输出 join；普通 int/split 计算不命中。
- Python lexical parsing 失败（包括未闭合字符串）时该 body fail closed，fence/EOS 独立处理。
- Math：大小写不敏感的 heading、闭合 `**...**`/`__...__` label；label 后续正文不命中；boxed/EOS 任意位置。
- 两种 mask 均在 trainer batch reorder 前生成，actor `update_policy` 的 tensor whitelist 同时保留它们。
- C/S 合并后每个 microbatch/domain 只进行一次 mean-one normalization。

## 配置与旧版本保护

原 18 public profiles 保持 Math top-p .05 × Code .05/.02/.01 × GPU 3/4/8，global batch/PPO
mini-batch 均为 528；sampling、既有 colocated topology、TP、teacher FSDP 与 65-step 限制保持不变。

主设置 w4；另提供 18 个 `fixed8` overlays，各自继承对应 `fixed4`，将 Control 与 Structure raw weight
同时设为 8。版本化 validator 接受一致的 4/4 或 8/8，拒绝其他值或 C/S 不一致；legacy Current-Step 仍要求 Fixed4。

新 run、audit、eval、checkpoint namespace 含 `-r2-`，w8 含独立 `-f8-`；继续禁止隐式 resume。
旧 revision 的 JSON/CSV 和 26 份配置源文本保留在
`mopd_verl/domain_gradient/token_taxonomy_history/revision_1/`，旧实现报告保持历史记录。
没有引入 .025/.035 top-p 或 S-only sweep。

## 修改文件

- Artifacts：`mopd_verl/domain_gradient/token_v4_v5.json`、`token_v4_v5_membership.csv`、
  `token_v4_v5_provenance.json`、`token_taxonomy_history/revision_1/`。
- Runtime：`token_taxonomy_registry.py`、`code_positions.py`、`structure_positions.py`、
  `current_step_weights.py`、`occurrence.py`、`occurrence_config.py`（均在 domain_gradient 下）；
  `third_party/verl/verl/workers/actor/dp_actor.py` 仅增补一个 mask key。
- Configs：`configs/token_selection/math_code/taxonomy/` 的两个 V4/V5 base、18 个 public w4 profiles、18 个 w8 overlays。
- Tools：`scripts/finalize_token_v4_v5.py`；本计划目录 `update_configs.py` 记录可重复的配置机械升级。
- Tests：`tests/test_token_v4_v5.py`、`tests/test_structure_positions.py`、`tests/test_token_v4_v5_runtime.py`。
- `.gitignore`：为新增的 runtime regression 文件添加精确例外，保留原 secrets ignore 规则。
- Docs：`Token.md`、`docs/current-step-token-weighting.md`、本计划文件和 `MANIFEST.md`。

既有 dirty changes 未还原或覆盖。revision-1 config snapshot 是配置源文本备份，不是独立可运行的历史仓库 checkout。

## 实际验证

使用 `/opt/anaconda3/bin/python`（Python 3.12 / torch 2.5.1）与 `PYTHONPATH=.`：

```sh
python -m pytest tests/test_token_v4_v5.py tests/test_structure_positions.py \
  tests/test_token_v4_v5_runtime.py tests/test_current_step_config.py \
  tests/test_current_step_selection.py tests/test_current_step_runtime.py \
  tests/test_amplification_sources.py tests/test_audit_occurrence_overrides.py \
  tests/test_config_profiles.py tests/test_domain_gradient_rebuild.py \
  -k 'not real_two_rank' -q
# 203 passed, 3 deselected

python -m pytest tests/test_current_step_runtime.py -k real_two_rank -q
# 3 passed, 15 deselected; automatically approved local CPU execution outside sandbox
```

共 206 个不同 test cases 通过。覆盖 36 配置的 launch/worker metadata chain、strict >20 边界、
V5 sibling 关系、旧 SHA/profile 拒绝、位置语义、代码块内高 loss 不影响外部 Control 排名、
仅外部 occurrence 通过频次门槛、w4/w8 正式加权，以及既有真实双 rank 聚合/collective error 路径。

Ruff format/check、JSON/CSV `cmp` 和 `git diff --check` 全部通过。

初次完整运行的 3 个 Gloo cases 在 sandbox 内因 OpenMP `Can't open SHM`/SIGABRT 未进入业务断言；
仅这三项在自动批准的 sandbox 外本地 CPU 重跑后通过。系统 python 缺少 NumPy，`uv --offline`
遇到 macOS system-configuration panic；使用现有 Anaconda 环境解决，没有安装依赖或更改宿主环境。

## 限制

未进行 GPU smoke、速度/显存测试或模型评测，因此不声称 colocated topology 性能收益。格式指标
（提取失败、无 boxed、截断、平均长度）与 Math4/Code4 作为后续实验记录要求，未新增评测 pipeline。
