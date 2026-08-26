# OPD / MOPD 配置说明

本文档说明当前保留的正式配置矩阵。训练入口仍是 `scripts/run_mopd.sh` 或 `scripts/run_local_mopd_training.sh`；YAML 会由 `mopd_verl/launch.py` 转换成 `verl.trainer.main_ppo` 的 Hydra overrides。

## 配置文件总览

| 配置 | 适用场景 | 主要特点 |
| --- | --- | --- |
| `configs/mopd_formal_audit_all_2gpu.yaml` | 2 卡正式诊断训练 | 4B student，math/code 4B teachers，teacher top-k distillation，domain gradient 与 observation metrics |
| `configs/mopd_formal_audit_all_4gpu.yaml` | 4 卡正式诊断训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_all_6gpu.yaml` | 6 卡正式诊断训练 | TP=2，6 卡 batch，沿用 audit-off 实测显存安全 profile |
| `configs/mopd_formal_audit_all_8gpu.yaml` | 8 卡 OPD 正式诊断训练 | 6 student + 2 teacher 分离部署，policy-gradient objective，audit-only CE/logp vectors |
| `configs/mopd_formal_audit_loss_only_2gpu.yaml` | 2 卡兼容配置 | 保留旧 loss-only 命名；nested token backward 已关闭 |
| `configs/mopd_formal_audit_loss_only_4gpu.yaml` | 4 卡兼容配置 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_loss_only_6gpu.yaml` | 6 卡兼容配置 | TP=2，6 卡 batch，`fsdp_size=2` domain-gradient audit |
| `configs/mopd_formal_audit_loss_only_8gpu.yaml` | 8 卡兼容配置 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_off_2gpu.yaml` | 2 卡无 audit 训练 | 同样的模型、数据和 objective，关闭所有 MOPD audit 输出 |
| `configs/mopd_formal_audit_off_4gpu.yaml` | 4 卡无 audit 训练 | 同 objective，batch 按卡数放大 |
| `configs/mopd_formal_audit_off_6gpu.yaml` | 6 卡无 audit 训练 | TP=2，vLLM memory 0.6，max_num_seqs 24 |
| `configs/mopd_formal_audit_off_8gpu.yaml` | 8 卡无 audit 训练 | TP=4，8 卡 batch |
| `configs/mopd_formal_audit_all_smoke.yaml` | 指标 smoke 测试 | 2 卡 one-step，domain gradient 与 full-vocab observation vectors |
| `configs/mopd_formal_audit_loss_only_smoke.yaml` | 兼容 smoke 测试 | 2 卡 one-step，保留旧 loss-only 输出命名 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math.yaml` | 原始 6 卡单域训练 | 4 actor + 2 teacher，math-only，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_code.yaml` | 原始 6 卡单域训练 | 4 actor + 2 teacher，code-only，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_if.yaml` | 原始 6 卡单域训练 | 4 actor + 2 teacher，IF-only，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_science.yaml` | 原始 6 卡单域训练 | 4 actor + 2 teacher，science-only，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml` | 原始 6 卡双域训练 | 4 actor + 2 teacher，math/code 等权采样，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science.yaml` | 原始 6 卡三域训练 | 4 actor + 2 teacher，math/code/science 等权采样，保持原配置 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math.yaml` | 8 卡单域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，math-only |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_code.yaml` | 8 卡单域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，code-only |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_if.yaml` | 8 卡单域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，IF-only，IFBench validation |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_science.yaml` | 8 卡单域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，science-only，GPQA validation |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code.yaml` | 8 卡双域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，math/code 等权采样 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science.yaml` | 8 卡三域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，math/code/science 等权采样 |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_topk32.yaml` | 8 卡 Top-32 三域训练 | 6 actor + 2 teacher，actor `fsdp_size=1`，batch 504，Top-32 reverse-KL distillation |
| `configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_if_topk32_dynamic_budget.yaml` | 8 卡动态四域训练 | math/code/science/IF；capability gap 决定 `q`，sequence OPD-loss variance 决定 sampling `p`，实际 batch 比例决定 `lambda=q/p` |

GPU integration configs 已收敛到 `test_grad_configs/`：一个 FSDP reliability
matrix，以及一个四领域 dynamic data ratio / loss-scale smoke profile。

卡数 scaling：

- `data.max_prompt_length=2048`
- `data.max_response_length=16384`，其中 6 卡 compatibility profile 为了降低 audit 峰值改为 `10240`，并显式设置 `rollout.max_model_len=12288`

| GPU 数 | 配置后缀 | `trainer.n_gpus_per_node` | `rollout.tensor_model_parallel_size` | `data.train_batch_size` | `actor.ppo_mini_batch_size` | `ray_kwargs.ray_init.num_cpus` |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | `_2gpu` | 2 | 2 | 256 | 256 | 8 |
| 4 | `_4gpu` | 4 | 4 | 512 | 512 | 16 |
| 6 | `_6gpu` | 6（标准 profile）/ 4（Qwen30B split） | 2 | 768（标准）/ 512（Qwen30B 单/双域）/ 504（Qwen30B 三域） | 768（标准）/ 512（Qwen30B 单/双域）/ 504（Qwen30B 三域） | 24 |
| 8 | `_8gpu` | 8（标准 profile）/ 6（OPD split profile） | 4（标准 profile）/ 2（OPD split profile） | 1024（标准 profile）/ 768（formal OPD split）/ 504（Qwen30B split） | 1024（标准 profile）/ 768（formal OPD split）/ 504（Qwen30B split） | 32（标准）/ 24（split） |

指标 smoke profile 使用独立设置：`trainer.n_gpus_per_node=2`、`rollout.tensor_model_parallel_size=2`、`data.train_batch_size=32`、`actor.ppo_mini_batch_size=32`、`trainer.total_training_steps=1`；response 长度保持正式配置的 `data.max_response_length=16384`。

## 模型与数据

```yaml
model:
  student_path: ../models/Qwen3-4B
  math_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Math-Step500
  code_teacher_path: ../models/Qwen3-4B-Non-Thinking-RL-Code-Step300
```

## 按 step 上传模型到 Hugging Face

在任意训练 YAML 顶层加入以下配置，并让 actor checkpoint 的
`save_contents` 包含 `hf_model`，即可在指定 global step 生成并上传可由
Transformers 加载的模型：

```yaml
huggingface_checkpoint:
  enabled: true
  steps: [20, 50, 100]
  repo_id: your-hf-account/opd-checkpoints
  private: true
  path_prefix: checkpoints
  token_env_var: HF_TOKEN
```

```yaml
extra_overrides:
  - actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]
```

`steps` 是正整数数组。命中其中任一步时，即使该 step 不满足
`trainer.save_freq`（包括 `save_freq: -1`），训练器也会先强制保存
`global_step_<STEP>`，再把本地 `actor/huggingface/` 中的模型权重、config 和
tokenizer 同步上传到仓库中的 `checkpoints/global_step_<STEP>/`。optimizer、
scheduler/RNG state、dataloader、critic 和 MOPD controller state 只保留在本地
restartable checkpoint 中，不会上传到 Hugging Face。

推荐直接在启动 `start.sh` 的同一个 shell 中 export token。若本机已通过
`hf auth login` 登录，可以从 Hugging Face cache 安全读取，命令本身不会包含
raw token：

```bash
HF_TOKEN_FILE="${HF_HOME:-${HOME}/.cache/huggingface}/token"
if [ -s "${HF_TOKEN_FILE}" ]; then
  export HF_TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
else
  echo "Hugging Face token cache not found" >&2
fi
unset HF_TOKEN_FILE
test -n "${HF_TOKEN}" && echo "HF_TOKEN is set"

bash start.sh --config configs/your-training-config.yaml

# 训练提交完成后可从当前 shell 清除。
unset HF_TOKEN
```

`start.sh`、launcher 与训练进程会继承已 export 的 `HF_TOKEN`；无需把 token
写入 `.env.local`。上传逻辑始终优先读取当前环境变量，只有环境变量不存在时才
尝试配置中的 env-file fallback。token 本身不会进入 YAML、Hydra 配置、日志或
Hugging Face commit。
若目标 repo 尚不存在，训练器会按 `private` 设置自动创建。每个目标 step 保存模型后，
trainer 会先在同一文件系统建立仅包含 HF model 的 hard-link snapshot，再由单独线程
顺序上传；网络传输不会阻塞后续训练 step。训练 loop 退出时才等待剩余上传并汇总错误。
如果 checkpoint filesystem 不支持 hard link，训练器会 fail-fast，绝不会退化为同步复制
大模型。上传失败时 snapshot 会保留在 `.huggingface_uploads/`，下次启动同一配置时会在
加载 checkpoint 前自动补传；成功后会自动清理，并在仍存在的本地 step 目录写入 upload
receipt。正常结束、`val_only` 和训练异常退出都会 drain 已排队任务；训练异常与上传异常
同时发生时保留训练异常，并额外记录上传失败信息。每次上传同时清除对应 Hub step 目录中的
旧文件，避免旧版整 checkpoint 上传留下 optimizer 或 dataloader。当前只支持单节点 Ray
cluster 与 `trainer.nnodes: 1`；检测到多节点时会 fail-fast。

## Dynamic Domain Budgeting

动态 profile 的训练入口为：

```bash
bash scripts/run_mopd.sh \
  configs/mopd_qwen4b_30b_a3b_instruct_2507_8gpu_math_code_science_if_topk32_dynamic_budget.yaml \
  --dry-run

# 确认 teacher score、模型路径和数据路径后，移除 --dry-run 启动训练。
```

每个 validation window 先计算 domain-level capability gap，并做 causal
time-series normalization：

```text
gap[d] = max(teacher_score[d] - student_score[d], 0)
g[d]   = clip(EMA(gap[d]) / max(initial_gap[d], gap_floor, eps), 0, gap_max)
q[d]   ∝ prior[d] * (g[d] + eps)^alpha
```

每个 actor step 从未加权的 configured token loss 计算每条 sequence 的
token mean，再估计 domain 方差。controller 使用
`p[d] ∝ q[d] * sqrt(EMA_variance[d])` 更新下一个 batch 的 sampler。当前
batch 的整数计数可能和连续 `p` 略有差异，所以真正施加的是
`lambda[d] = q[d] / p_active[d]`。这里 `p_active` 是 optimizer 的
`seq-mean-token-mean` 实际纳入聚合的非空 sequence 比例；通常它等于当前
batch 的 `p_observed`，出现 fully masked response 时则以 `p_active` 为准。

必须满足以下 runtime contract：

- `actor.loss_agg_mode: seq-mean-token-mean`；
- 必须使用正权重的 top-k OPD objective，且首版不支持 teacher-prefix training；
- `actor.ppo_epochs: 1`，且一个 actor batch 只对应一个 optimizer mini-batch；
- `actor.entropy_coeff: 0` 且 `actor.kl_loss_coef: 0`；当前 scale 施加在 actor gradient 上，首版只允许纯 OPD objective；
- `data.domain_sampling_replacement: true`；
- `data.dataloader_num_workers: 0`；
- `min_samples_per_domain >= variance_min_samples`，确保每个 step 都能更新每个 domain 的方差；
- `trainer.val_before_train: true` 且 `trainer.test_freq > 0`；
- 不能同时开启 `audit.dynamic_domain_loss_weighting_enabled`；
- `teacher_scores` 必须是在同一固定 probe set、同一 score scale 上预先测得的 teacher 结果。
- 正式启动前必须显式设置 `teacher_scores_calibrated: true`；占位分数只允许 dry-run。

四域示例中的 `teacher_scores: 1.0` 只是用于 config validation 和 dry-run
的占位值（相当于把 task ceiling 当作临时 reference），不能当作正式的
teacher probe score。正式训练前必须用 Qwen3-30B 在上述每组
`validation_metric_keys` 上、按同一 reducer 得到的实测值替换，并将
`teacher_scores_calibrated` 改为 `true`。`gap_normalization_floor` 防止初始
gap 为 0 时，微小回退被 `epsilon` 放大并直接撞上 normalization cap。

`loss variance` 在这里是 stochastic-gradient noise 的低成本 proxy，不是
gradient variance 的等价量。controller 不修改 optimizer；它只改变未来
batch 的 domain 配额，并给当前 batch 的每条 sequence 施加 domain scale。
`mopd_audit.loss_variance_signal: opd_loss_token` 是兼容模式；若填写具体名称
（例如 `topk_renormalized_reverse_kl`），audit 会校验 production forward
返回的 configured token loss 名称完全一致，不一致时立即失败，避免指标名
与实际训练 loss 静默错配。
状态同时写入 `domain_budgeting.output_dir/state.json` 和训练 checkpoint 的
`domain_budgeting_state.json`。恢复训练时以 trainer checkpoint 内的状态为
唯一恢复源，避免 model、dataloader 和 controller step 不一致。sampler
checkpoint 同时保存 RNG、整数配额和 allocation version，恢复后不会重播
已经消费过的 batch；controller checkpoint 会校验全部训练语义参数。

```yaml
data:
  domain_train_files:
    math:
      - data/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
    code:
      - data/G-OPD-Training-Data/Eurus/code_train.parquet
    science:
      - data/G-OPD-Training-Data/Science/train.parquet
    if:
      - data/G-OPD-Training-Data/IF/train.parquet
  domain_sampling_weights:
    math: 1
    code: 1
    science: 1
    if: 1
```

IF 训练样本的 `data_source` 为 `m2rl_ifbench`。dataset loader 会按
`domain_train_files.if` 强制写入 `domain/opd_teacher/source_domain=if`，从而
选择 `if_teacher_path`；`mopd_verl/mixed_reward.py` 会把该批样本路由到
IFBench instruction-following reward。运行前需保证
`verifiable_instructions` 可导入，或按 `scripts/prepare_ifbench_runtime.sh`
准备 official IFBench evaluator。该脚本会同时检查 Python dependencies、
下载所需 NLTK resources 并验证 `evaluation_lib` 能实际 import；应在已激活的
training environment 中运行。

## 蒸馏目标

除 8 卡 OPD split profile 外，formal 配置默认使用 teacher top-k local-support distillation：

```yaml
actor:
  distill_mode: chosen_token_reverse_kl
  topk_distill_enabled: true
  topk_distill_support_source: teacher
  topk_distill_kl_direction: reverse
  topk_distill_k: 32
  topk_distill_tail_bucket: false
```

`topk_distill_enabled` 只控制训练 objective 是否使用 teacher top-k distillation。Policy-gradient 配置可以保持该开关关闭，同时通过 audit 的 `topk_teacher_student_cross_entropy_vocab_enabled` 独立收集 teacher/student cross-entropy vocab vector，不改变训练 loss。

8 卡 OPD split profile 使用：

```yaml
actor:
  distill_loss_builder: policy_gradient
  distill_mode: chosen_token_policy_gradient
  topk_distill_enabled: false
  topk_distill_loss_weight: 0.0
```

## Audit All

`configs/mopd_formal_audit_all_*gpu.yaml` 保留 domain-gradient audit 与
无需额外 backward 的 observation metrics。为避免重复同步、污染 `.grad`
以及极高的重放成本，nested sample/token backward 已从当前实现退役：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_<gpu>
  log_sample_level: true
  log_validation_metrics: true
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_enabled: true
  token_gap_vocab_vector_enabled: true
  vocab_per_occurrence_mean_vector_enabled: true
  logp_vocab_per_occurrence_mean_vector_enabled: null
  logp_abs_vocab_per_occurrence_mean_vector_enabled: null
  entropy_vocab_per_occurrence_mean_vector_enabled: null
  entropy_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

`logp_vector_enabled` 显式输出 signed gap
`teacher_logp - old_student_logp`，`logp_abs_vector_enabled` 输出其绝对值。
`vocab_per_occurrence_mean_vector_enabled` 是 legacy global 开关，并继续直接控制
token-gap family。`logp_vocab_per_occurrence_mean_vector_enabled`、
`logp_abs_vocab_per_occurrence_mean_vector_enabled` 和
`entropy_vocab_per_occurrence_mean_vector_enabled` 可以分别覆盖 logp、logp_abs 与
entropy family；值为 `null` 或省略时回退 global，显式 `true/false` 时独立生效。
entropy override 同时控制 student entropy 与 teacher-student cross-entropy 的 mean
vector。所有 `*_mean_vector_vocab` 都对每个 token id 使用
`sum / occurrence_count`，未出现 token 的维度保持 0。该统计是当前 step、当前
domain 内的 conditional mean，不是 `count / total_count` 的 token-frequency
probability。

它还设置：

```yaml
actor:
  use_dynamic_bsz: false
rollout:
  gpu_memory_utilization: 0.6
```

这让 full/domain-gradient audit 的统计路径保持固定 micro-batch，避免
dynamic batching 影响 domain-gradient 对比。`sample_gradient_norm_enabled`
等旧字段可能仍保留在 YAML 中，但在 `sample_gradient_enabled: false` 时不会
触发 sample backward；token selector 字段同理。

## Audit Loss Only

`configs/mopd_formal_audit_loss_only_*gpu.yaml` 现在是兼容旧实验名和输出目录
的 aliases。当前实现关闭 nested sample/token backward，因此 selector 字段
不会产生 token gradient；6 卡 profile 继续用 `fsdp_size: 2` 统计 domain
gradient：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_loss_only_<gpu>
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: true
  sample_gradient_cos_enabled: true
  token_gap_enabled: true
  token_gap_vocab_vector_enabled: true
  vocab_per_occurrence_mean_vector_enabled: true
  entropy_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: false
  token_gradient_gap_abs_selection_enabled: false
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  # 6gpu fsdp=2 profile uses 0.15; other loss-only formal profiles use 0.10.
  token_gradient_top_p: 0.10
```

因此当前配置不会生成 `token_grad_metrics.jsonl`，也不会做 sample/token
级别的额外 backward。`sequence_masked_target_*` 只服务于 domain-gradient
target：

```yaml
audit:
  sequence_masked_target_enabled: true
  sequence_masked_target_use_as_primary: true
```

这一路径不要求每个 worker 拥有完整 local params。不要再用
`token_gradient_top_p` 作为 closure sanity check；当前 closure 应查看
domain-sum、audit-total 和 training-total 的 cosine/relative-L2 指标。

## 指标 Smoke

`configs/mopd_formal_audit_all_smoke.yaml` 用于快速验证 TensorBoard scalar、
domain-gradient JSONL、full-vocab token gap vector 和 entropy vector 的记录逻辑：

```yaml
audit:
  enabled: true
  output_dir: audit/formal_audit_all_smoke
  full_gradient_enabled: true
  sample_gradient_enabled: false
  sample_gradient_cos_enabled: true
  token_gap_vocab_vector_enabled: true
  token_gap_vocab_size: null
  vocab_per_occurrence_mean_vector_enabled: true
  entropy_vocab_vector_enabled: true
  topk_teacher_student_cross_entropy_vocab_enabled: true
  logp_vector_enabled: true
  logp_abs_vector_enabled: true
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

其中 `token_gap_vocab_size: null` 表示使用 tokenizer 的完整词表维度，不是压缩到小词表的假 smoke。

`configs/mopd_formal_audit_loss_only_smoke.yaml` 使用同样的 one-step smoke
设置并保留旧 selector 元数据，但不会执行 token backward：

```yaml
audit:
  output_dir: audit/formal_audit_loss_only_smoke
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: false
  token_gradient_gap_abs_selection_enabled: false
  token_gradient_loss_abs_selection_enabled: true
```

## Audit Off

`configs/mopd_formal_audit_off_*gpu.yaml` 保持同样的训练 objective，但关闭所有 audit：

```yaml
audit:
  enabled: false
  output_dir: audit/formal_audit_off_<gpu>
  log_sample_level: false
  log_validation_metrics: false
  full_gradient_enabled: false
  sample_gradient_enabled: false
  sample_gradient_norm_enabled: false
  sample_gradient_cos_enabled: false
  token_gap_enabled: false
  token_gap_vocab_vector_enabled: false
  entropy_enabled: false
  entropy_vocab_vector_enabled: false
  topk_teacher_student_cross_entropy_vocab_enabled: false
  logp_abs_vector_enabled: false
  token_gradient_enabled: false
  token_gradient_gap_selection_enabled: true
  token_gradient_gap_abs_selection_enabled: true
  token_gradient_loss_abs_selection_enabled: true
  token_gradient_top_k: 100
  token_gradient_top_p: 0.10
```

## Control-Anchored Region-DPO

Region-DPO 默认关闭。启用后，可以分别控制每条 base rollout 选择多少个
rerollout points，以及每个 point 采样多少条 sibling branches：

```yaml
region_dpo:
  enabled: true
  points_per_rollout: 2
  branches_per_point: 4
  max_new_tokens: 256
  beta: 0.1
  loss_weight: 0.1
  min_reward_margin: 0.0
  selection_strategy: random
  seed: 42
```

默认复用 `audit.domain_control_token_ids` 中冻结的 domain-specific control
taxonomy，也可以在 `region_dpo.domain_control_token_ids` 中显式覆盖。完整的
candidate construction、loss 公式、runtime constraints 与 metrics 见
[docs/region-dpo.md](docs/region-dpo.md)。

## Online Control-token selection

Online selector 在固定的 domain candidate pools 上提供四种 ranking mode，
并把 token-ID selection 与入选后的 weighting 独立配置：

```yaml
audit:
  control_token_loss_weighting_enabled: true
  control_token_online_selection_enabled: true
  # top_loss | top_speed | top_kl_student_entropy |
  # top_teacher_confidence_student_entropy
  control_token_online_selection_mode: top_kl_student_entropy
  control_token_online_weight_mode: paired  # fixed | paired
  control_token_online_audit_interval_steps: 3
  control_token_online_window_steps: 3
  control_token_online_min_mean_occurrences_per_step: 10.0
  control_token_online_top_k: 30
  control_token_loss_weight: 4.0
```

`top_loss` 按 rolling window 内的 occurrence-mean absolute configured loss
降序选择。`top_speed` 对每个 `(domain, token_id)` 的 per-step
occurrence-mean absolute configured loss 做 occurrence-count-weighted linear
regression，并使用负 slope 作为 optimization speed；正值表示 loss 正在下降。
Top-speed 按 signed speed 降序选择，不取绝对值，也不丢弃负 speed。

两个 paired-signal mode 都先在每条 valid response 内把信号归一化到
`[0, 1]`，再按

\[
s=A+B+AB=(1+A)(1+B)-1
\]

聚合 `(domain, token_id)` 的 rolling occurrence mean，并选择 Top-K：

- `top_kl_student_entropy`：`A` 是在 TIP/FiRe token weight 与 rollout-IS
  生效前捕获的 detached raw Top-K KL loss，`B` 是 Student entropy；
- `top_teacher_confidence_student_entropy`：`A` 是
  `1 - normalized Teacher entropy`，`B` 是 Student entropy。

`control_token_online_weight_mode=fixed` 沿用
`control_token_loss_weight`。`paired` 则让入选 token ID 在下一 selection
interval 使用历史窗口估计的 `1 + mean(s)`，即 mean
`(1+A)(1+B)`；未入选 token 的 raw weight 仍为 1。若
`control_token_normalize_per_domain=true`，最后仅做一次 domain mean-one
normalization。`paired` 只允许搭配上述两个 paired-signal selector，因此可形成
`2 selectors × 2 weight modes` 的四种组合。

每次 selection audit boundary 都会记录实际 ranking score 的 token-ID-level
分布。`online_control_selection.jsonl` 包含 eligible/selected 两组完整 summary；
训练 metrics 使用
`{domain}/token_weight/{eligible|selected}_selection_score_{count|mean|std|min|p10|p50|p90|max}`。
这里的 score 是 rolling window 内每个 token ID 的 occurrence-mean ranking score，
不是单个 token occurrence 的原始 score。

四种 mode 共用 occurrence eligibility、audit cadence 和 next-step lag。Top-speed
要求 window 至少包含两个 step，且 token 在至少两个不同 step 有 observation。
`control_token_speed_weighting_enabled` 是另一套 domain-level speed-to-weight
controller，不应与 online selector 同时开启。Qwen4B Top-speed profile 位于
`configs/mopd_qwen4b_30b_a3b_instruct_2507_4gpu_math_code_science_topk32_control_online_topspeed_i3_w3_f10_k30_w4_b525.yaml`。

## 常用启动

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_2gpu.yaml \
  --run-id mopd_audit_all_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_off_2gpu.yaml \
  --run-id mopd_audit_off_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_2gpu.yaml \
  --run-id mopd_audit_loss_only_2gpu_$(date +%Y%m%d_%H%M%S)
```

```bash
GPU_IDS=0,1,2,3 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_4gpu.yaml \
  --run-id mopd_audit_all_4gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_6gpu.yaml \
  --run-id mopd_audit_all_6gpu_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_8gpu.yaml \
  --run-id mopd_audit_all_8gpu_$(date +%Y%m%d_%H%M%S)
```

指标 smoke 测试直接使用维护中的 smoke YAML：

```bash
GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_all_smoke.yaml \
  --run-id mopd_metrics_smoke_$(date +%Y%m%d_%H%M%S)

GPU_IDS=0,1 bash scripts/run_local_mopd_training.sh \
  configs/mopd_formal_audit_loss_only_smoke.yaml \
  --run-id mopd_metrics_loss_only_smoke_$(date +%Y%m%d_%H%M%S)
```

详细 metric 口径见 [metrics_zh.md](metrics_zh.md)。

## EOPD baseline 切换

EOPD 作为独立的可选 loss builder 接入，不会替换现有 OPD 或 Top-k KL。
配对配置保证两种 baseline 共用相同的数据、模型、optimizer、rollout 和
evaluation 设置：

```bash
# OPD
bash scripts/run_mopd.sh --dry-run \
  'configs/matrices/eopd_baseline_matrix.yaml::opd'

# EOPD: tau=0.8, alpha=1.0, teacher top-k=16
bash scripts/run_mopd.sh --dry-run \
  'configs/matrices/eopd_baseline_matrix.yaml::eopd'
```

EOPD 的公式语义、论文参数与本项目参数差异见
[docs/eopd-baseline.md](docs/eopd-baseline.md)。
