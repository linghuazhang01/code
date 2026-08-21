# MOPD 远端调试规则

- 本地仓库是源码唯一 source of truth；所有修改必须先在本地完成。
- 固定流程：本地修改与检查 → `rsync --dry-run` → 上传到 `/home/shuang_qiu/mopd_code` → 远端运行 → 下载日志 → 回本地修复。
- 禁止直接修改远端源码；远端只允许运行、测试、训练和生成运行产物。
- `../ssh.sh` 第 1 行是 SSH 连接命令，第 2 行是密码。不得修改、打印、提交或整体执行该文件；只执行第 1 行，并在密码提示时输入第 2 行。
- 默认禁止 `rsync --delete`，不得覆盖远端独有的 dataset、model、logs 或 checkpoints。
