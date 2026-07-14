#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${CODE_DIR}/third_party/verl}"
ORACLE_GPU_IDS="${ORACLE_GPU_IDS:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found: ${VERL_RUNTIME_DIR}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${ORACLE_GPU_IDS}"
if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
  echo "ORACLE_GPU_IDS must expose exactly two GPUs, got: ${ORACLE_GPU_IDS}" >&2
  exit 2
fi
if [[ -z "${gpu_ids[0]}" || -z "${gpu_ids[1]}" || "${gpu_ids[0]}" == "${gpu_ids[1]}" ]]; then
  echo "ORACLE_GPU_IDS must contain two distinct, non-empty GPU IDs: ${ORACLE_GPU_IDS}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${ORACLE_GPU_IDS}"
export PYTHONPATH="${CODE_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"

cd "${CODE_DIR}"

for fsdp_size in 1 2; do
  echo "Running world_size=2, fsdp_size=${fsdp_size}"
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/fsdp_domain_gradient_oracle.py \
    --fsdp-size "${fsdp_size}" \
    --tolerance 1e-5 \
    --bf16-tolerance 2e-2
done

"${PYTHON_BIN}" -m unittest discover \
  -s tests -p 'test_fsdp1_replication_contract.py'
"${PYTHON_BIN}" -m unittest discover \
  -s tests -p 'test_grad_reliability_profiles.py'
