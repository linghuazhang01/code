# FSDP 4B/8B Commit Regression Prompt

> 使用方法：把本文作为后续修改 training、domain-gradient audit、FSDP、checkpoint
> 或配置逻辑后的验证任务交给 Codex。必须验证当前 candidate commit，不得为通过
> 实验而修改源码、放宽阈值或替换配置。

## 固定验证对象

以下两份配置共同构成最小 real-model topology regression matrix：

1. `test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml`
   - actor world size：2
   - `fsdp_size=1`
   - effective FSDP1 strategy：同步的 `NO_SHARD` replication
   - expected replica count：2
2. `test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml`
   - actor world size：2
   - `fsdp_size=2`
   - effective FSDP1 strategy：`FULL_SHARD`
   - expected replica count：1

当前 golden config SHA256：

```text
0c3552fce4ed9ce15ce4e3a205f714217e60c816f61286bf6726dc6d9f864924  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml
10c9ff9da9764f91356d1216b36139ce4ce6de9f3728d521bd43e1579fb4e32d  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml
```

若任一 hash 不匹配，停止 GPU 实验并展示 config diff；不得静默接受新配置。

---

## 给后续执行者的 Prompt

你负责验证 OPD 当前 candidate commit 是否破坏正常训练、domain-gradient audit
或 FSDP size 1/2。严格执行下列阶段，最终给出有完整证据的 PASS/FAIL 报告。

### 0. 不可违反的约束

1. 这是验证任务，不是修复任务。失败后只收集证据并报告，不得自动修改代码。
2. 不得执行 `git reset`、`git checkout`、`git stash` 或删除用户工作区内容。
3. 不得修改 seed、batch size、rollout sampling、`fsdp_size`、audit frequency、
   gradient storage dtype、threshold 或 loss 配置来使测试通过。
4. 两个 real-model smoke 必须顺序运行，不能并行争抢 GPU、CPU RAM 或 Ray。
5. 必须使用 3 张 dedicated idle GPU：2 张 student actor/rollout，1 张 teacher/ref。
6. 所有命令默认设置 `STOP_STALE_RAY=0`，禁止 launcher 自动执行
   `ray stop --force`。启动前运行 `ray status`；如果存在任何 Ray job，必须确认
   它属于本次验证且获得明确授权后才能停止，否则不得启动实验。
7. 不得把 SSH 密码、API key、`.env`、模型凭据或内部地址写入日志、Prompt、commit。
8. 不提交 `logs/`、`audit/`、`checkpoints/`、TensorBoard、W&B 或临时实验产物。

### 1. Provenance 与 config gate

在真正的 Git repo 根目录执行并记录：

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
shasum -a 256 \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  environment.blackwell.yml
```

要求：

- 记录 candidate commit 的完整 SHA。
- 默认要求 clean worktree；若存在未提交修改，停止并报告，因为实验无法归属于
  唯一 commit。
- 两份 config hash 必须与本文 golden hash 一致。
- 两份 YAML 只能在以下字段不同：
  `actor.fsdp_size`、`audit.output_dir`、`trainer.experiment_name`、
  `trainer.default_local_dir`。

执行 YAML parse 和 pair-diff 检查：

```bash
python - <<'PY'
from pathlib import Path
import yaml

paths = [
    Path("test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml"),
    Path("test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml"),
]
configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]

def flatten(value, prefix=()):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            output.update(flatten(child, prefix + (str(key),)))
        return output
    return {prefix: value}

left, right = map(flatten, configs)
different = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
allowed = {
    ("actor", "fsdp_size"),
    ("audit", "output_dir"),
    ("trainer", "experiment_name"),
    ("trainer", "default_local_dir"),
}
if different != allowed:
    raise SystemExit(f"Unexpected config differences: {sorted(different)}")

for config, expected_size in zip(configs, (1, 2), strict=True):
    assert config["actor"]["fsdp_size"] == expected_size
    assert config["trainer"]["n_gpus_per_node"] == 2
    assert config["worker_placement"]["actor_rollout"]["n_gpus_per_node"] == 2
    assert config["worker_placement"]["ref_policy"]["n_gpus_per_node"] == 1
    assert config["trainer"]["total_training_steps"] == 4
    assert config["data"]["train_batch_size"] == 16
    assert config["rollout"]["do_sample"] is False
    assert config["rollout"]["temperature"] == 1.0
    assert config["audit"]["full_gradient_enabled"] is True
    assert config["audit"]["full_gradient_freq_steps"] == 2
    assert config["audit"]["full_gradient_storage_dtype"] == "bfloat16"
    assert config["audit"]["full_grad_training_parity_rel_l2_threshold"] == 2.0e-2
    assert "trainer.resume_mode=disable" in config["extra_overrides"]
print("CONFIG_PAIR_GATE=PASS")
PY
```

### 2. 环境和静态验证

目标环境必须满足：

- Python 3.10
- PyTorch `2.8.0+cu128`
- CUDA 12.8
- vLLM 0.11.0
- 3 张可见且空闲的 Blackwell `sm_120` GPU
- Qwen3-4B student、Qwen3-8B teacher 和两份训练 parquet 均存在

先显式激活 Blackwell 环境；不能只给 launcher 传 `ENV_NAME`：

```bash
test -f logs/activate_training_env.sh
source logs/activate_training_env.sh
python -c 'import sys; print(sys.executable)'
```

记录环境，不得只相信 environment 文件：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python - <<'PY'
import platform
import sys
import torch
import vllm

print("python", platform.python_version())
print("python_executable", sys.executable)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("gpu_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.startswith("2.8.0+cu128")
assert torch.version.cuda == "12.8"
assert vllm.__version__ == "0.11.0"
assert torch.cuda.device_count() >= 3
assert all(torch.cuda.get_device_capability(index) == (12, 0) for index in range(3))
PY
```

先做两个 dry-run。dry-run 不允许停止共享 Ray：

```bash
DRY_SHORT=$(git rev-parse --short=12 HEAD)
DRY_STAMP=$(date +%Y%m%d_%H%M%S)
DRY_FSDP1="dryrun_${DRY_SHORT}_fsdpsize1_${DRY_STAMP}"
DRY_FSDP2="dryrun_${DRY_SHORT}_fsdpsize2_${DRY_STAMP}"
```

```bash
STOP_STALE_RAY=0 ENV_NAME=mopd-verl-blackwell GPU_IDS=0,1,2 \
  bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml \
  --run-id "$DRY_FSDP1" --foreground --dry-run -- \
  actor_rollout_ref.model.use_remove_padding=false \
  trainer.save_freq=-1
```

```bash
STOP_STALE_RAY=0 ENV_NAME=mopd-verl-blackwell GPU_IDS=0,1,2 \
  bash scripts/run_local_mopd_training.sh \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  --run-id "$DRY_FSDP2" --foreground --dry-run -- \
  actor_rollout_ref.model.use_remove_padding=false \
  trainer.save_freq=-1
```

launcher config/path/GPU-count 错误在此停止；Hydra/runtime 错误由后续 gate 捕获。

### 3. Focused tests 与两卡 CUDA oracle

先运行与 domain-gradient/FSDP 直接相关的测试：

```bash
python -m pytest -q \
  tests/test_fsdp1_replication_contract.py \
  tests/test_grad_reliability_profiles.py \
  tests/test_domain_gradient_optimization_contracts.py \
  tests/test_domain_gradient_rebuild.py \
  tests/test_fsdp_checkpoint_topology.py \
  tests/test_audit_vocab_cosine.py \
  tests/test_mopd_profiles.py
```

再运行 world size 2 的 `fsdp_size=1/2` CUDA oracle：

```bash
ORACLE_GPU_IDS=0,1 PYTHON_BIN=python \
  bash tests/run_minimal_fsdp_domain_gradient.sh
```

oracle 必须确认：

- size 1：`NO_SHARD`、replica count 2、两个 rank gradient/parameter 一致，
  不执行非法 reshard。
- size 2：`FULL_SHARD`、replica count 1、reshard 路径正常。
- 两种 topology 的 analytic gradient、micro-batch accumulation、optimizer update、
  FP32/BF16 closure 和 training parity 均通过。

任一测试或 oracle 失败都不得进入 real-model smoke。

### 4. 顺序执行两份 real-model config

只在 dedicated node 上运行。启动每个 run 前先执行 `ray status` 并检查 Ray
进程；如果发现的 cluster 不属于本次验证，立即停止。如果它属于已经结束的本次
验证 run，确认没有活动 job 后才能手动清理，不能让 launcher 自动清理。

在同一个 shell/session 中执行下面的完整 block。它会生成唯一 run ID、顺序运行
size 1/2，并分别保存 exit code：

```bash
set -e
SHORT_SHA=$(git rev-parse --short=12 HEAD)
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_FSDP1="commit_${SHORT_SHA}_fsdpsize1_${STAMP}"
RUN_FSDP2="commit_${SHORT_SHA}_fsdpsize2_${STAMP}"
printf '%s\n' "$RUN_FSDP1" "$RUN_FSDP2"

run_smoke() {
  local config="$1"
  local run_id="$2"
  local status
  set +e
  STOP_STALE_RAY=0 ENV_NAME=mopd-verl-blackwell GPU_IDS=0,1,2 \
    bash scripts/run_local_mopd_training.sh \
    "$config" --run-id "$run_id" --foreground -- \
    actor_rollout_ref.model.use_remove_padding=false \
    trainer.save_freq=-1
  status=$?
  set -e
  printf '%s\n' "$status" > "logs/${run_id}.exit"
  RUN_SMOKE_STATUS=$status
}

run_smoke \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml \
  "$RUN_FSDP1"
STATUS_FSDP1=$RUN_SMOKE_STATUS

# Before continuing, confirm the first process exited and its Ray/GPU resources
# are no longer active. Do not stop a cluster that is not owned by this run.
run_smoke \
  test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize2_audit_freq2_b16_4step_smoke.yaml \
  "$RUN_FSDP2"
STATUS_FSDP2=$RUN_SMOKE_STATUS

test "$STATUS_FSDP1" -eq 0
test "$STATUS_FSDP2" -eq 0
```

config 已固定 `trainer.resume_mode=disable`。仅允许两个命令行 override：
`actor_rollout_ref.model.use_remove_padding=false`、`trainer.save_freq=-1`，后者避免
4B checkpoint 占满磁盘。主 regression 不验证 checkpoint correctness；
验证 real-model save/resume 时，必须改用包含 run ID 的全新 checkpoint 目录。

### 5. Hard acceptance gates

每个 run 都必须满足：

1. 进程 exit code 为 0，完成且只完成 4 个 optimizer steps。
2. 日志中没有 Traceback、OOM、NCCL timeout、hang、nonfinite gradient 或
   `Expects sharded strategy`。
3. `audit_errors.jsonl` 不存在或为空；所有 JSONL/TensorBoard scalar 均 finite。
4. `actor/pg_loss` 和 `actor/grad_norm` 在 step 1–4 全部存在且 finite。
   `training/rollout_probs_diff_valid` 在四步都必须等于 1。
5. `global/audit/domain_gradient_source_step` 只能出现在：
   `(step=2, value=2)`、`(step=4, value=4)`；step 1/3 不得出现 stale value。
6. `global/audit/full_gradient_replica_count`：
   - `(world=2, size=1)` 在 step 2/4 必须等于 2；
   - `(world=2, size=2)` 在 step 2/4 必须等于 1；
   - conditional `(world=4, size=2)` 在 step 2/4 必须等于 2。
7. 以下两组指标只能在 step 2/4 出现，并且每次都必须：
   - `passed == 1`；
   - `rel_l2 <= 0.02`；
   - cosine、norm、diff 等所有组成量 finite。

```text
global/full_grad_closure/domain_sum_vs_audit_total/*
global/full_grad_training_parity/audit_total_vs_training_total/*
```

8. `global/full_grad_closure/domain_sum_vs_audit_total/`
   `storage_roundoff_may_exceed_threshold` 在 step 2/4 必须为 0。
   `global/audit/domain_gradient_backward_replay_count` 在 step 2/4 必须等于 3，
   `global/audit/domain_gradient_coverage_fraction` 必须等于 1；math/code 的
   `full_grad/sample_count` 必须分别等于 8。
9. 每个 run 的以下文件必须有 4 steps × 2 domains = 8 rows，且没有 NaN/Inf：
   - `domain_step_metrics.jsonl`
   - `loss_variance_domain_step.jsonl`
   - `token_gap_vectors.jsonl`
   - `token_gap_vocab_vectors.jsonl`
   - `entropy_distribution_vectors.jsonl`
   - `entropy_vocab_vectors.jsonl`
   - `topk_teacher_student_cross_entropy_vocab_vectors.jsonl`
   `training_cost.jsonl` 必须有 4 rows。
10. vocab rows 必须满足：vector 长度与 `vocab_size` 一致、
    `sum(token_count_vector_vocab) == observed_token_count`、
    `dropped_token_count == 0`。
11. 两个 topology 的 step 1 使用相同初始 model/data；比较其 `pg_loss` 与
    `grad_norm`，若差异超过 `1e-6` 则标记 WARN 并调查，但不要仅因此判 FAIL。
    只有固定 response batch oracle 才能作为严格 topology-parity gate。
    step 2–4 允许因为微小数值差异和 greedy rollout 分支而产生不同 response，
    不要求后续 loss/vector 逐项完全相等。

如果 training parity 的 `rel_l2` 大于 `1e-5` 但仍小于等于配置阈值 `0.02`，
hard gate 可以 PASS，但必须额外标记 WARN 并检查 gradient restore 路径。

历史 Blackwell baseline 仅作为异常检测参考，不作为必须精确复现的 hardcode：

| Topology | Closure rel-L2, step 2/4 | Parity rel-L2, step 2/4 | Peak GPU |
|---|---|---|---|
| `fsdp_size=1` | 0.005404 / 0.004829 | 0 / 4.19e-8 | 79.94 GiB |
| `fsdp_size=2` | 0.004433 / 0.004472 | 3.60e-8 / 0 | 63.21 GiB |

若 peak memory 相对 baseline 增长超过 10%，或 audit-step duration 增长超过
20%，标记 WARN 并调查，但在没有 OOM/hang/correctness failure 时不要单独判 FAIL。

### 6. 产物与最终报告

记录并归档每个 run 的：

- candidate commit SHA、branch、初始 `git status --short`
- config 和 environment SHA256
- 完整 launch command 与生成的 `.launch.sh`
- console log、exit code、GPU CSV
- audit JSONL 目录
- TensorBoard event 文件
- Python、PyTorch、CUDA、vLLM、GPU 型号
- `pip freeze` 与本次验证涉及的 source SHA256
- 每个 hard gate 的实际值和 PASS/FAIL

从训练 log 中解析实际 TensorBoard 输出路径，不要根据 experiment name 猜目录。
最后将上述文本产物和报告打包为 tar.gz，并另存 SHA256；不得把 secrets、模型、
数据集、checkpoint 或 `.env` 放入归档。

最终报告必须明确区分：

```text
STATIC/UNIT GATE: PASS | FAIL
CUDA ORACLE SIZE 1: PASS | FAIL
CUDA ORACLE SIZE 2: PASS | FAIL
REAL MODEL FSDP SIZE 1: PASS | FAIL
REAL MODEL FSDP SIZE 2: PASS | FAIL
OVERALL: PASS | FAIL
```

只要一个 hard gate 失败，`OVERALL=FAIL`。不得用“曲线看起来正常”替代 closure、
training parity、replica count 和 audit frequency 的逐项验证。

## Conditional experiments

以下实验不要求每个 commit 都运行：

1. 修改 audit replay、RNG、gradient restore 或 optimizer interaction 时，比较
   `test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_freq2_b16_4step_smoke.yaml`
   与 `test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw2_fsdpsize1_audit_off_b16_4step_smoke.yaml`。
   两份 config 已固定 `trainer.resume_mode=disable`；比较所有非 audit、非 timing
   training metrics。比较 optimizer state 时，为两次 run 设置不同的全新目录。
2. 修改 checkpoint manager 或 topology metadata 时：运行 size 1/2 checkpoint
   save-load-resume oracle；磁盘充足时再做一次 real 4B resume。
3. 准备正式长跑或修改 offload/storage 时：运行至少 20-step soak，确认 CPU/GPU
   memory 不随 audit 次数单调泄漏。
4. 若生产使用 `HYBRID_SHARD`，增加
   `test_grad_configs/mopd_grad_reliability_qwen4b_8b_aw4_fsdpsize2_audit_freq2_b16_4step_smoke.yaml`；
   它使用 `world_size=4, fsdp_size=2, replica_count=2`，需要 4 张 actor GPU
   加 1 张 teacher GPU；step 2/4 的 replica count=2、closure/parity 均通过，checkpoint metadata=`HYBRID_SHARD`。
5. 若需要声称两种 topology 产生相同参数更新：使用固定的 pre-generated
   response batch 做一次 4B update，并比较 update 前后的参数 checksum；不能用
   已经分叉的后续 rollout 证明 topology parity。
