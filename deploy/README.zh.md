# deploy：OPD 训练 + 评测环境的一键安装

训练和评测统一用同一个环境：**torch 2.6.0、vLLM 0.8.5.post1、transformers 4.51.3**（`environment.yml` 声明包，`env.lock.txt` 把全部 pip 包钉到 CityU H200 训练环境的版本）。安装和校验都会检查这三个版本，不一致就失败；Blackwell GPU（算力 ≥ 10）不支持 torch 2.6，会直接拒绝。

## 怎么启动：一条命令

在自己电脑上运行 `deploy/opd_deploy.sh`，SSH 信息作为参数传入。脚本按步骤依次执行（connect → sync → network → base → env / data / models / ckpts → evalrt → verify），过程中每 20 秒报告一次各步骤状态，结束时打印汇总表：每一步成功、失败还是没跑，用了多久；有失败时指出第一个失败的步骤，并显示它日志的最后几行。

```bash
cd code
bash deploy/opd_deploy.sh --ssh "ssh -p 30022 root@1.2.3.4" --root /data/opd --gpus 0,1,2,3
#   SSH 也可以分开写：--host 1.2.3.4 --user root --port 30022 --key ~/.ssh/id_ed25519 [--jump 跳板机]
#   要下的模型：--students "Qwen/Qwen3-1.7B Qwen/Qwen3-4B" --teacher Qwen/Qwen3-30B-A3B-Instruct-2507（默认就是这些，--teacher "" 不下老师）
#   评测 checkpoint：--checkpoints "q1p7b_math_termlook q1p7b_code_termlook"（名字在 eval_models.tsv）
#   网络：--proxy auto|direct|http|ssh-reverse|mirror，--http-proxy URL，--hf-mirror URL，--github-proxy URL，--pip-mirror URL，--conda-mirror URL
#   其他：--min-free-gib 500（CityU H200）、--sequential（不并行）、--steps a,b（只跑部分步骤）、--force（重做）、--plan deploy/plans/x.tsv（装完接着跑计划）
bash deploy/opd_deploy.sh --ssh "ssh -p 30022 root@1.2.3.4" --root /data/opd --status   # 只看进度和失败位置
```

- 中途断网或按 Ctrl-C：机器上的安装继续跑，用 `--status` 查看；修好原因后用同一条命令重跑，已完成的步骤会跳过。
- 机器要密码登录时，开始时在终端里输一次即可（之后复用同一个连接）。
- 机器上需要有 `bash`、`curl`、`git`；有 `rsync` 用 rsync 同步代码，没有就用 tar。

下面两种是同一套安装的分步用法。

### 方式一：分步命令（`bootstrap.sh`，配置写在文件里）

```bash
cd code
bash deploy/bootstrap.sh init           # 生成 deploy/target.env、deploy/network.env（不进 git）
#   编辑 deploy/target.env：SSH_HOST / SSH_USER / SSH_PORT / SSH_KEY，REMOTE_ROOT（新目录），GPU_IDS，要下载的模型
bash deploy/bootstrap.sh check          # 测 SSH、探测 HF / GitHub / PyPI / conda-forge 线路
bash deploy/bootstrap.sh setup env      # 环境脚本：装环境 + GPU 校验
bash deploy/bootstrap.sh setup data     # 数据脚本：训练数据、评测数据、模型、评测 checkpoint
#   （两个可以同时开；也可以一次 setup all。两者都完成后自动装评测打分运行环境）
bash deploy/bootstrap.sh status         # 看进度；logs 看日志
```

装好后在目标机器上：

```bash
source <REMOTE_ROOT>/activate.sh        # 进入统一环境（PATH、PYTHONPATH、HF 缓存、代理设置），并 cd 到代码目录
bash start.sh -c configs/<config>.yaml --local     # 训练
bash deploy/remote/eval_queue.sh --gpu 0 --jobs math4:<name>:42,code4:<name>:42   # 评测（名字在 eval_models.tsv）
```

### 方式二：直接在目标机器上（代码已经放在机器上，比如 `<REMOTE_ROOT>/code`）

```bash
cd <REMOTE_ROOT>/code
cp deploy/target.env.example deploy/target.env      # 可选：改 REMOTE_ROOT、GPU_IDS、模型；不建就全部装在代码目录旁边
cp deploy/network.env.example deploy/network.env    # 可选：需要代理或镜像时
bash deploy/install_env.sh               # 环境脚本
bash deploy/install_data.sh              # 数据脚本（可以和上面同时开）
source <REMOTE_ROOT>/activate.sh
```

| 脚本 | 做什么 |
|---|---|
| `deploy/opd_deploy.sh` | **主入口**：SSH 参数传入，一步步安装并跟踪进度，最后汇总成功 / 失败位置 |
| `deploy/bootstrap.sh` | 分步入口：读 `target.env`，同步代码、启动安装或计划、拉回结果、中转上传、复制密钥 |
| `deploy/install_env.sh` | 环境：Miniforge → `mopd-verl` 环境（锁定版本）→ 版本 + CUDA + 多卡 NCCL 校验 |
| `deploy/install_data.sh` | 数据：`icemoon28/MOPD-Training-Data`、G-OPD 评测数据（commit `37371a4c`）、LiveCodeBench v5/v6、`STUDENT_MODELS`、`TEACHER_MODEL`、`EVAL_CHECKPOINTS` |
| `deploy/remote/setup.sh` | 上面两个的实现，分阶段，可 `--phases` 单独跑某一阶段 |
| `deploy/remote/eval_queue.sh` | 单卡评测队列，协议同 Burgundy A100 表格 |
| `deploy/remote/run_pipeline.sh` | 按计划文件（`plans/*.tsv`）跑训练 → 评测 → 上传 |

所有东西都在 `REMOTE_ROOT` 下：`code/`、`miniforge3/`、`models/`、`eval_models/`、`eval_outputs/`、`logs/`、`state/`。`REMOTE_ROOT` 必须是新目录或空目录（代码同步用 `rsync --delete`，脚本会拒绝写进已有内容的目录）。每个阶段完成后写 `state/<阶段>.done`，中断后重跑会跳过已完成的阶段（`FORCE=1` 重做）。

## 容器：只装环境 / 只下数据和模型（`deploy/container/`）

适用于不能 SSH 进去、镜像固定、要在启动脚本或 `docker build` 里自己装环境的平台。两个脚本互相独立，都在代码目录里运行，重复运行会跳过已完成的部分，不读 `target.env` / `network.env`。

| 脚本 | 做什么 |
|---|---|
| `deploy/container/install_env.sh` | 只装环境：Miniforge（核对 SHA-256）→ `mopd-verl`（`environment.yml` + `env.lock.txt`，torch 2.6.0 / vLLM 0.8.5.post1 / transformers 4.51.3）→ 版本检查 → 写 `<prefix>/opd_env.sh`。不需要 GPU，可以放进 `docker build`；`--verify <GPU>` 在有 GPU 时做 CUDA + 多卡 NCCL 检查 |
| `deploy/container/download_data_models.sh` | 只下数据和模型，全部放进 `--dir`：训练 parquet（DeepMath level-6 + Eurus code）、学生和老师模型（逐文件核对大小和 SHA-256）。再在 `<代码上一级>/mopd/` 下建链接，让配置里的 `../mopd/models/...`、`../mopd/code/data/...` 能找到；代码目录本身不改 |

```bash
# 1. 环境。启动时安装就放持久盘，每次挂到同一路径（conda 环境不能换路径），之后启动会跳过安装；docker build 里加 --clean
bash deploy/container/install_env.sh --prefix /mnt/disk/miniforge3 --verify 0,1,2,3   # 没有 GPU 时去掉 --verify
source /mnt/disk/miniforge3/opd_env.sh            # PATH、PYTHON、PYTHONPATH
# 2. 数据和模型，下到指定目录
bash deploy/container/download_data_models.sh --dir /mnt/disk/opd_data
#   --students "Qwen/Qwen3-1.7B Qwen/Qwen3-4B"（默认）  --teacher Qwen/Qwen3-30B-A3B-Instruct-2507（默认，"" = 不下）
#   --no-data / --no-models：只做其中一样
# 3. 训练
GPU_IDS=0,1,2,3 bash start.sh -c configs/<config>.yaml --local
```

- 空间（Hub 上的大小）：训练数据 1.7 GB，Qwen3-1.7B 4.1 GB，Qwen3-4B 8.1 GB，Qwen3-30B-A3B-Instruct-2507 61.1 GB；环境另算。
- 网络：两个脚本都认 `HTTPS_PROXY` / `HTTP_PROXY`。没有代理时，环境脚本用 `GITHUB_PROXY`（前缀式，如 `https://ghfast.top`）、`PIP_MIRROR`、`CONDA_MIRROR`；下载脚本用 `HF_ENDPOINT=https://hf-mirror.com`。镜像只能下载，不能上传。
- 镜像里要有 `bash`、`curl`、`git`；只支持 Linux x86_64、非 Blackwell GPU、支持 CUDA 12.4 的驱动。
- 不包括：评测数据和官方打分环境（G-OPD、LiveCodeBench、EvalPlus）、密钥、上传。

## 安装阶段

| 阶段 | 内容 | 属于 |
|---|---|---|
| base | 磁盘、GPU（拒绝 Blackwell）、网络探测、Miniforge（核对 SHA-256）、下载用小环境 `opd-tools` | 两个脚本都会先跑 |
| env | `mopd-verl` 环境，装完检查 torch / vLLM / transformers 版本 | 环境脚本 |
| verify | 版本、CUDA 计算、多卡时 `scripts/gpu_connectivity_check.sh` | 环境脚本 |
| data | 训练 parquet、G-OPD 评测数据、LiveCodeBench | 数据脚本 |
| models | `STUDENT_MODELS`、`TEACHER_MODEL`，逐文件核对大小和 SHA-256 | 数据脚本 |
| ckpts | `EVAL_CHECKPOINTS`（`eval_models.tsv` → `icemoon28/opd-checkpoints`） | 数据脚本 |
| evalrt | G-OPD `code_eval/coding`、LiveCodeBench 源文件（核对行数和哈希）、固定版本的 EvalPlus 运行环境、`anthropic`、Qwen3 tokenizer | env 和 data 都完成后自动跑 |

env 和 data/models/ckpts 并行：下载只用 `opd-tools` 小环境，不等大环境。

## 代理和镜像（`network.env`）

| PROXY_MODE | 做法 | 适用 |
|---|---|---|
| auto（默认） | 能直连就直连；否则用配置的代理；再不行按服务换镜像 | 大多数情况 |
| http | 全部走 `HTTP_PROXY_URL`（目标机器能访问的代理） | 平台提供代理，或容器里自己起了代理 |
| ssh-reverse | `ssh -R` 把本机代理（Clash 7890）映射到目标机器 `127.0.0.1:17890` | 目标机器不通外网但 SSH 允许端口转发；本机要一直在线 |
| mirror | `HF_MIRROR`（hf-mirror.com）、`GITHUB_PROXY`（前缀式，如 ghfast.top）、`PIP_MIRROR`、`CONDA_MIRROR` | 只需要下载的国内机器 |

- 这些设置只作用于脚本本身：git 用 `GIT_CONFIG_COUNT` 环境变量改写 GitHub 地址，conda 用 `state/condarc`，pip 用 `PIP_INDEX_URL`，不改用户全局配置；`environment.yml` 只在复制件里给 GitHub 地址加前缀。
- **上传只能连 huggingface.co**（镜像只读）。`net_check.sh` 判断目标机器能否直连或经代理访问 huggingface.co：能（direct）则训练按配置里的 `huggingface_checkpoint` 自己上传 step 60，计划里的 `upload` 上传评测结果；不能（relay）则结果留在目标机器，本机 `bootstrap.sh fetch` 后 `bootstrap.sh relay-upload <目录>` 用本机代理上传，训练配置里的上传要关掉。
- ssh-reverse 依赖本机在线，适合装环境；长时间训练中的上传建议用平台代理或 relay。

## 密钥

HF_TOKEN、WANDB_API_KEY 只放在 `~/.config/opd-private/opd.env`。`bootstrap.sh secrets` 确认后复制到目标机器同名位置（权限 600），`activate.sh` 和计划脚本从那里读。`icemoon28/opd-checkpoints` 是公开仓库，下载不需要 token。

## 计划文件（`plans/*.tsv`）

```
train      configs/<config>.yaml    0,1,2,3     [hydra overrides]
eval       math4:<name>:42,code4:<name>:42,...  0,1
wait-eval
upload     <suite 名通配>
```

- train：多卡时先跑 GPU 连通检查，再 `start.sh -c <config> --local --foreground`。
- eval：每张卡一个 `eval_queue.sh`，共用任务列表，靠 `eval_outputs/.locks/<tag>` 分配；K=8、T=1.0、16,384 tokens、CUDA graph。suite 名里的硬件标签（`a100`、`h200`…）取自 GPU 型号。被打断的 suite 要手动删锁才会重跑（已生成的 shard 会续用）。
- 官方 LiveCodeBench 打分进程数默认取 CPU 数减 3（`MOPD_LCB_PROCS`；Burgundy A100 表格用的是 5）。

## 已知限制

- 只支持 Linux x86_64 + 非 Blackwell 的 NVIDIA GPU，驱动需支持 CUDA 12.4。
- CityU 服务器：按 AGENTS.md 的账号规则登录，`MIN_FREE_GIB` 设 500，`REMOTE_ROOT` 用新目录，不要指向已有的训练代码目录。
