#!/usr/bin/env bash
# Multi-GPU connectivity check, run before every multi-GPU launch (author rule 2026-10-17, OPD/CLAUDE.md).
# 1) nvidia-smi answers within a timeout, no GPU reports "requires reset" and every GPU's recovery action is None;
# 2) a short NCCL broadcast + all-reduce across exactly the GPUs the job will use finishes and returns correct values.
# Exit 0 = pass, non-zero = do not launch.
#
# Usage (qiushuang, from /home/qiushuang/mopd_code):
#   bash scripts/gpu_connectivity_check.sh 0,1,2
#   PYTHON=/home/qiushuang/miniconda3/envs/mopd-verl/bin/python bash scripts/gpu_connectivity_check.sh 0,1,2,3
set -uo pipefail
GPUS="${1:?usage: gpu_connectivity_check.sh <comma-separated GPU ids>}"
PYTHON="${PYTHON:-/home/qiushuang/miniconda3/envs/mopd-verl/bin/python}"
TORCHRUN="$(dirname "${PYTHON}")/torchrun"
N=$(awk -F, '{print NF}' <<< "${GPUS}")
WORK="$(mktemp -d /tmp/gpu_conn_check.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

echo "[1/2] nvidia-smi health on GPUs ${GPUS}"
if ! out=$(timeout 30 nvidia-smi -i "${GPUS}" -q 2>&1); then
  echo "FAIL: nvidia-smi did not answer within 30 s (driver may be wedged)"; exit 2
fi
if grep -q "requires reset" <<< "${out}"; then
  echo "FAIL: a GPU reports 'GPU requires reset'"; exit 3
fi
if grep -i "Recovery Action" <<< "${out}" | grep -v -q ": None"; then
  echo "FAIL: a GPU has a pending recovery action:"; grep -i "Recovery Action" <<< "${out}"; exit 3
fi
echo "ok: $(grep -c 'Recovery Action' <<< "${out}") GPUs, recovery action None"

echo "[2/2] NCCL broadcast + all-reduce across ${N} GPUs (${GPUS})"
cat > "${WORK}/nccl_check.py" <<'PY'
import datetime, time, torch, torch.distributed as dist
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
n = 132_382_976  # numel of the first broadcast that hung on 2026-10-17
x = torch.full((n,), float(r + 1), dtype=torch.bfloat16, device="cuda")
t = time.time(); dist.broadcast(x, src=0); torch.cuda.synchronize(); tb = time.time() - t
y = torch.ones(n, dtype=torch.bfloat16, device="cuda")
t = time.time(); dist.all_reduce(y); torch.cuda.synchronize(); ta = time.time() - t
ok = bool((x == 1).all()) and float(y[0]) == float(w)
print(f"rank{r}: broadcast {tb:.2f}s all_reduce {ta:.2f}s ok={ok}", flush=True)
dist.barrier(); dist.destroy_process_group()
raise SystemExit(0 if ok else 1)
PY
if CUDA_VISIBLE_DEVICES="${GPUS}" NCCL_DEBUG=WARN timeout 180 "${TORCHRUN}" --standalone --nproc_per_node="${N}" \
     "${WORK}/nccl_check.py" > "${WORK}/nccl.log" 2>&1; then
  grep -E "^rank" "${WORK}/nccl.log"; echo "PASS: GPUs ${GPUS} connected"; exit 0
fi
echo "FAIL: NCCL test failed or timed out (180 s); last lines:"; tail -n 15 "${WORK}/nccl.log"; exit 4
