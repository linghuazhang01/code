# MOPD 远端调试规则

- 本地仓库是源码唯一 source of truth；所有修改必须先在本地完成。
- 固定流程：本地修改与检查 → `rsync --dry-run` → 上传到 `/home/shuang_qiu/mopd_code` → 远端运行 → 下载日志 → 回本地修复。
- 禁止直接修改远端源码；远端只允许运行、测试、训练和生成运行产物。
- `../ssh.sh` 第 1 行是 SSH 连接命令，第 2 行是密码。不得修改、打印、提交或整体执行该文件；只执行第 1 行，并在密码提示时输入第 2 行。
- 默认禁止 `rsync --delete`，不得覆盖远端独有的 dataset、model、logs 或 checkpoints。
- 后续所有远端 Slurm 任务（训练、评测和 smoke）的内存请求 hard cap 为 `400G`，`--mem` 不得超过 `400G`。若 launcher 默认值高于 `400G`，提交前必须显式设置 `MOPD_SLURM_MEMORY=400G`（或更低）；无法满足时不得提交。
