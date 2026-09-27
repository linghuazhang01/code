# Token V4/V5 revision 3 迁移报告

状态：本地迁移、focused verification 与主代理最终 review 完成。
主代理独立核验了全部 CSV/NPZ 计数、JSON 字节一致性、集合互斥，以及位置规则和配置保留情况。
采用用户 2026-09-28 第三版定义；未启动 GPU、远端同步、commit 或 push。

## 定义与核验

权威来源为 `../plan/code-gap-20260927/structure-budget-context-20260927/`。

| Version | Math Control/Structure | Code Control/Structure |
|---|---:|---:|
| V4 | 119/9 | 153/45 |
| V5 | 61/9 | 72/45 |

- C 类 `final`/`answer`/`conclusion` 全部归入 Structure；V4 只保留 A/B/D，V5 只保留 A/B。
- V5 sibling additions 为 Math 4 / Code 8，V4/V5 Control intersection 为 57/64。
- JSON SHA256：`2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8`。
- CSV SHA256：`7c27427726f985036f2c6b415358508af61b51444782a2fd9cedb8ca15e4d16b`。
- finalizer 核对 666 rows × 4 baseline = 2,664 个 CSV/NPZ 计数，0 mismatch；所有 keep 项严格 `>20`。
- 完整词表支持集保持 Math 3,686 / Code 6,573；C/S 互斥且 V4/V5 Structure 相同。
- finalizer 额外验证 AnswerWord 不留在 Control、V5 不包含 C/D 类、所有有支持的答案词进入 Structure。

JSON/CSV 由验证入口逐字节导入；`token_v4_v5_provenance.json` 保存报告与 source hashes。
该入口只验证冻结产物，不重新运行 baseline trajectories 或 full-vocabulary closure。

## 最小实现范围

1. `token_taxonomy_registry.py`、finalizer 与两个 taxonomy base 配置同步 r3 SHA/数量。
2. `structure_positions.py` 的答案词分支扩展到两 domain，增加 conclusion。
   Markdown heading 或闭合 `**...**` / `__...__` 标签中生效；Math 额外允许与字面
   `\boxed` 命令同一行，支持 CRLF，排除 `\boxedness` 等较长命令，前后行不继承。
   `boxed` token 本身和 EOS 的原位置规则不变。
3. Code 的 8 个答案标记只在全部 fenced blocks 外的正文标签生效，无闭合代码块时也可生效。
   它们明确排除于既有 I/O/signature/entry 分支，代码变量、comments、strings、docstrings 不加权。
4. 原 Code Control 块外 mask、最终闭合 block 的其他 Structure 规则、distributed wiring 和
   C/S 合并后一次 mean-one normalization 保持不变。
5. 36 个公共 profile 在原路径改用独立 `-r3-` namespaces；既有 w4/w8、Math .05、
   Code .05/.02/.01、batch 528、3/4/8-GPU colocated topology 均保持。没有增加实验矩阵。

新测试发现原 `__Answer__` 无冒号标签因正则 word boundary 与 underscore 冲突而漏判；
已通过要求闭合 marker 的最小修改修复，并覆盖两种加粗写法。

## 历史保护与文件

`mopd_verl/domain_gradient/token_taxonomy_history/revision_2/` 保存 r2 JSON、CSV、provenance
原始字节及 44 份配置源文本（8 个 base、36 个 public）。r1 history 未修改；快照用于
provenance 恢复，不是独立可运行的旧代码 checkout。旧报告保留历史状态。

本轮修改集中在：

- `mopd_verl/domain_gradient/token_v4_v5{.json,_membership.csv,_provenance.json}`；
  `token_taxonomy_registry.py`、`structure_positions.py`、新增 `token_taxonomy_history/revision_2/`。
- `scripts/finalize_token_v4_v5.py`。
- `configs/token_selection/math_code/taxonomy/` 中两个 taxonomy base 和 36 个公共配置；
  6 个 GPU topology base 仅备份，未改动。
- `tests/test_token_v4_v5.py`、`test_structure_positions.py`、`test_token_v4_v5_runtime.py`；
  新增 `tests/test_answer_structure_positions.py`。
- `Token.md §0.3`、`docs/current-step-token-weighting.md`、本计划与报告、`MANIFEST.md`。

无关 dirty changes 保留。project memory 与 Obsidian daily 由主代理维护，不重复写入。

## 实际验证

使用 `/opt/anaconda3/bin/python` 与 `PYTHONPATH=.`：

```sh
python -m pytest tests/test_token_v4_v5.py tests/test_structure_positions.py \
  tests/test_answer_structure_positions.py tests/test_token_v4_v5_runtime.py \
  tests/test_current_step_config.py tests/test_current_step_selection.py \
  tests/test_current_step_runtime.py tests/test_amplification_sources.py \
  tests/test_audit_occurrence_overrides.py tests/test_config_profiles.py \
  tests/test_domain_gradient_rebuild.py -k 'not real_two_rank' -q
# 253 passed, 3 deselected (before adding the four integration cases below)

python -m pytest tests/test_token_v4_v5_runtime.py -k answer_structure -q
# 4 passed, 4 deselected
```

合计 **257 个不同 cases 通过**。包括所有 36 配置、strict `>20`、r1/r2 SHA/profile 拒绝、
历史快照、全部答案词的正反位置例、Code 变量/注释/docstring、Math boxed 换行边界。
新增 integration cases 覆盖 Math/Code × w4/w8：高 loss 答案词不占 Control budget，
正文标签获得 Structure 权重，普通同词保持 raw 1，并只归一化一次。

本轮 Python 文件 Ruff format/check 通过；`git diff --check` 通过。
未改变 distributed wiring，因此按计划未重跑现有 3 个 real-two-rank Gloo cases。

## 限制

未进行 GPU smoke、速度/显存测试或模型评测；未声称格式指标或 Math4/Code4 已改善。
未改训练预算、top-p 或引入新的 evaluation pipeline。
