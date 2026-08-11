# MOPD 远端调试规则

- 本地仓库是源码唯一 source of truth；所有修改必须先在本地完成。
- 固定流程：本地修改与检查 → `rsync --dry-run` → 上传到 `/home/shuang_qiu/mopd_code` → 远端运行 → 下载日志 → 回本地修复。
- 禁止直接修改远端源码；远端只允许运行、测试、训练和生成运行产物。
- `../ssh.sh` 第 1 行是 SSH 连接命令，第 2 行是密码。不得修改、打印、提交或整体执行该文件；只执行第 1 行，并在密码提示时输入第 2 行。
- 默认禁止 `rsync --delete`，不得覆盖远端独有的 dataset、model、logs 或 checkpoints。

## Control Token 选择规则

为 Qwen3 OPD/EOPD 实验生成 `domain_control_token_ids` 时，必须使用固定的
common-support Logic/Control 规则，不得根据目标 run 的后验收益临时挑选 token：

1. 对每个 domain 分别建立 common-support vocabulary。token 必须在以下三个
   已对齐实验的 Rising 与 Stable 全部端点中各出现至少 20 次：
   - 4B OPD：Rising `6 -> 20`，Stable `21 -> 45`；
   - 1.7B OPD：Rising `1 -> 35`，Stable `36 -> 52`；
   - 1.7B EOPD：Rising `1 -> 37`，Stable `37 -> 65`。
2. 使用冻结的 token surface-form taxonomy 做一次语义分类，只保留
   `semantic_category == logic_control` 的 token。
3. 权威 artifact 是
   `analysis-output/category-optimization-loss-vs-gap-20260809/tables/common-support-vocabulary.csv`。
   在当前 Qwen3 tokenizer 下，完整性检查必须得到 Math 44、Code 30、Science 27。
4. 这里的 occurrence threshold 是跨模型、算法、阶段的 support gate，不是
   frequency Top-K。不得额外使用 Rising Top-100、late-window improvement、
   reward 或目标 run 的其他 outcome 对 token ID 做并集、删减或重排。
5. 只有 tokenizer、实验集合或 endpoint 定义发生变化时才重建集合；重建时必须
   生成带版本的新 artifact，并同步更新本规则、配置和完整性测试。
6. 向用户汇报选择结果时，必须同时给出 domain、token ID 和可读的 decoded token，
   不能只展示整数 ID。
