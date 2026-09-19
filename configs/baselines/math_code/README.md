# Math + Code baselines：四卡 / 八卡

共 12 个训练入口；`_common.yaml` 仅作为继承基配置。单 seed=42，Qwen3-1.7B student，Qwen3-30B-A3B-Instruct-2507 teacher。六种方法的 objective 对齐 `../opd_baselines.yaml` 的对应 profile。

## 固定训练预算

- global train batch = PPO mini batch = **516**。516 是最接近 512、同时能被 4 和 6 整除的 batch，避免不同布局改变训练预算。
- Math : Code = 1 : 1，每步精确分配 258 / 258 条；rollout `n=1`，每步一次 optimizer update。
- 单 seed=42；训练 60 steps，总计 30,960 个 prompt draws，两个领域各 15,480（有放回抽样，不等于 unique examples）。
- learning rate=5e-6，prompt 上限 2,048，response 上限 16,384，TP=1，micro batch/GPU=1，actor FSDP size=1，开启参数与优化器 offload。
- vLLM `gpu_memory_utilization=0.75`，`max_num_seqs=64`，`max_num_batched_tokens=32768`。
- 每 5 steps 保存，保留最近 2 个 actor checkpoint；HF 仅上传 Step60，沿用项目公开 checkpoint repo。没有在本次创建或上传 checkpoint。
- 只启用轻量 domain audit；不启用动态领域预算、额外 token weighting 或高开销梯度审计。

## 配置入口

| 方法 | 四卡共置 | 八卡 6 student + 2 teacher |
|---|---|---|
| OPD Top-K32 Uniform | `topk32_uniform_4gpu_colocate_b516.yaml` | `topk32_uniform_8gpu_6s2t_b516.yaml` |
| OPD Native | `opd_native_4gpu_colocate_b516.yaml` | `opd_native_8gpu_6s2t_b516.yaml` |
| EOPD Native | `eopd_native_4gpu_colocate_b516.yaml` | `eopd_native_8gpu_6s2t_b516.yaml` |
| ExOPD, λ=1.25 | `exopd_native_lambda1p25_4gpu_colocate_b516.yaml` | `exopd_native_lambda1p25_8gpu_6s2t_b516.yaml` |
| TIP Top-K32, ρ=0.5 | `tip_topk32_rho50_4gpu_colocate_b516.yaml` | `tip_topk32_rho50_8gpu_6s2t_b516.yaml` |
| FiRE-OPD Native | `fire_opd_native_4gpu_colocate_b516.yaml` | `fire_opd_native_8gpu_6s2t_b516.yaml` |

OPD Top-K32 使用 teacher support 上的 renormalized reverse KL，无 tail bucket。OPD Native 使用 sampled chosen-token policy-gradient objective。

EOPD Native 沿用 canonical profile：entropy threshold=0.8、forward KL weight=1、辅助 Top-K=16，并关闭 rollout importance sampling；这不是另一个 Teacher50 variant。ExOPD 使用冻结的初始 student reference，λ=1.25。TIP 保留 50% token，采用 `seq-mean-token-mean`。FiRE 的 confidence/confusion 指数均为 1、trajectory drop ratio=0.2、trajectory filtering 开启，同样采用 `seq-mean-token-mean`。FiRE 的 516 是 filtering 之前的采样预算，不保证相同有效 token 数。

## 布局评估

| 项目 | 四卡共置 | 八卡分离 |
|---|---|---|
| 总卡数 / student worker 数 | 4 / 4 | 8 / 6 |
| 每个 student worker 的 prompt 数 | 129 | 86 |
| Teacher | 与 student 共用四张卡，FSDP size=4 | 独立两张卡，FSDP size=2 |
| Teacher 含义 | 同一个 teacher 的分片 | 同一个 teacher 的两卡分片，并非 Math/Code 各占一个 teacher |
| Teacher 自适应 batching | 关闭，保留共置 collective 顺序 | 开启，使用现有 multi-rank 同步边界实现 |
| 主要优势 | 卡数少；若有八张卡可安排两组独立四卡作业 | teacher 与 vLLM 不共享显存；student/rollout 有更多 worker |
| 主要代价 | 模型切换/offload 和共享显存压力；长 response 需确认峰值 | 两卡 teacher 可能成为瓶颈；总卡时不一定划算 |

**排程建议：**若四卡在目标机器上可以稳定容纳 `.75` 的 vLLM 显存预算，完成六个 baseline 优先考虑两组四卡并行；若四卡共置显存不足，或实测单作业耗时更重要，则使用八卡 6+2。所有正式比较尽量采用统一布局；切换布局须记录，单 seed 也不保证跨拓扑 bitwise 一致。

目前没有 GPU 实测，不能给出可靠小时数或保证八卡更快。设同一方法四卡单次耗时为 T4、八卡为 T8；忽略排队/共享主机争用时，两组四卡完成六方法约需 3×T4，单组八卡依次运行约需 6×T8。八卡需要 **T8 < T4/2** 才同时赢得这组排程的总时间与 GPU-hours。

`gpu_memory_utilization=0.75` 是显存预算比例，不是 GPU 算力利用率，也不保证留下的 25% 足以容纳共置 teacher/activation。ExOPD 还要计入冻结 student reference；EOPD、Top-K 方法需关注 logits 峰值。硬件型号、显存、互联、CPU/RAM 和回答长度都会改变结论。建议正式首跑记录 warm-up 后 rollout/ref_logprob/update 分段时间与每卡显存峰值；这是性能测量，不新增训练 seed。

## 启动与环境

在远端 `code/` 目录使用项目标准入口。四卡示例（仅在对应物理 GPU 已获分配时执行）：

```bash
GPU_IDS=0,1,2,3 bash start.sh --local --config configs/baselines/math_code/topk32_uniform_4gpu_colocate_b516.yaml
```

八卡示例（需要已分配的完整八卡，不能占用他人 GPU）：

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash start.sh --local --config configs/baselines/math_code/topk32_uniform_8gpu_6s2t_b516.yaml
```

替换文件名即可启动其他方法；两种布局是备选，不要求每个方法都训练两次。重复运行同一配置前应更改 `trainer.default_local_dir`、`audit.output_dir` 等输出目录，避免覆盖同名实验。

数据与模型路径沿用现有远端配置的 `../mopd/code/data/...`、`../mopd/models/...` 部署约定，均相对于启动工作目录；目标机器必须存在这些路径，或先在 `_common.yaml` 中改为实际路径。ExOPD 两个入口的 `model.gopd_reference_path` 也需同步更新。当前本地没有完整模型和 Eurus 数据，不能在本地完成训练 preflight 或 GPU smoke test。

## 评测与验证边界

训练内 greedy validation 和旧 `paper_eval` 自动评测关闭。`paper_eval.datasets` 保留的 `lcb` 是旧辅助入口字段，不能代表官方 v5/v6 双 split 的 K=8 评测。正式结果另走项目标准评测流程：Step60、Math 四项 + Code 四项、每项 K=8，Code 使用官方评分。本次两领域结果应单独标记，不能冒充仓库原有十数据集标准总表。

本地验证覆盖全部 12 个 YAML 的继承展开、typed config validation、六个 canonical objective、batch 整除、两领域集合、placement、0.75 显存参数和 launcher dry-run。验证不加载权重，不检查 CUDA OOM、真实吞吐或远端文件存在性。本轮没有启动训练。
