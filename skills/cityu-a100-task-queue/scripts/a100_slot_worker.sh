#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'QUEUE_SLOT_ERROR %s\n' "$*" >&2
  exit 2
}

expected_gpu_count="${OPD_SLOT_GPU_COUNT:-}"
case "${expected_gpu_count}" in
  3|4) ;;
  *) fail "OPD_SLOT_GPU_COUNT must be 3 or 4" ;;
esac

job_id="${SLURM_JOB_ID:-}"
[[ -n "${job_id}" ]] || fail "SLURM_JOB_ID is required"

queue_root="${OPD_SLOT_QUEUE_ROOT:-${HOME}/scratch/opd/queue}"
command_file="${OPD_SLOT_COMMAND_FILE:-${queue_root}/commands/${job_id}.sh}"
ready_file="${OPD_SLOT_READY_FILE:-${queue_root}/ready/${job_id}.ready}"
activation_script="${OPD_SLOT_ACTIVATION_SCRIPT:-${HOME}/scratch/opd/logs/activate_training_env.sh}"
poll_seconds="${OPD_SLOT_POLL_SECONDS:-10}"

mkdir -p "$(dirname "${command_file}")" "$(dirname "${ready_file}")"

printf 'QUEUE_SLOT_START job_id=%s node=%s gpu_count=%s\n' \
  "${job_id}" "$(hostname)" "${expected_gpu_count}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

recovery_state="$(nvidia-smi -q | grep -A 2 'GPU Recovery Action' || true)"
printf '%s\n' "${recovery_state}"
if grep -Eq ':[[:space:]]*Reset([[:space:]]|$)' <<<"${recovery_state}"; then
  fail "GPU Recovery Action requires Reset"
fi

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
)
[[ "${#gpu_rows[@]}" -eq "${expected_gpu_count}" ]] ||
  fail "expected ${expected_gpu_count} visible GPUs, got ${#gpu_rows[@]}"

for row in "${gpu_rows[@]}"; do
  gpu_name="${row%%,*}"
  gpu_memory="${row##*,}"
  gpu_memory="${gpu_memory//[[:space:]]/}"
  [[ "${gpu_name}" == *A100* ]] || fail "non-A100 device: ${gpu_name}"
  [[ "${gpu_memory}" =~ ^[0-9]+$ ]] || fail "invalid GPU memory: ${gpu_memory}"
  ((gpu_memory >= 80000)) || fail "A100 memory below 80000 MiB: ${gpu_memory}"
done

[[ -f "${activation_script}" ]] || fail "missing activation script: ${activation_script}"
# shellcheck disable=SC1090
source "${activation_script}"

OPD_EXPECTED_GPUS="${expected_gpu_count}" python - <<'PY'
import json
import os

import torch

expected = int(os.environ["OPD_EXPECTED_GPUS"])
actual = torch.cuda.device_count()
assert actual == expected, (actual, expected)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
witnesses = []
for index in range(actual):
    device = torch.device(f"cuda:{index}")
    properties = torch.cuda.get_device_properties(device)
    assert "A100" in properties.name
    assert properties.total_memory >= 80_000 * 1024 * 1024
    left = torch.randn(64, 64, device=device)
    right = torch.randn(64, 64, device=device)
    result = left @ right
    assert result.shape == (64, 64)
    assert torch.isfinite(result).all()
    witnesses.append(
        {
            "device": index,
            "name": properties.name,
            "shape": list(result.shape),
        }
    )
print("CUDA_WITNESS", json.dumps(witnesses, sort_keys=True))
PY

umask 077
printf 'job_id=%s\nnode=%s\ngpu_count=%s\nready_at=%s\n' \
  "${job_id}" "$(hostname)" "${expected_gpu_count}" "$(date -Is)" >"${ready_file}"
printf 'QUEUE_SLOT_READY job_id=%s node=%s gpu_count=%s command_file=%s\n' \
  "${job_id}" "$(hostname)" "${expected_gpu_count}" "${command_file}"

while [[ ! -f "${command_file}" ]]; do
  sleep "${poll_seconds}"
done
[[ ! -L "${command_file}" ]] || fail "command file must not be a symlink"

claimed_file="${command_file}.claimed"
mv "${command_file}" "${claimed_file}"
chmod 700 "${claimed_file}"
printf 'QUEUE_TASK_START job_id=%s command_file=%s\n' "${job_id}" "${claimed_file}"
set +e
bash "${claimed_file}"
task_status=$?
set -e
printf 'QUEUE_TASK_END job_id=%s exit_code=%s\n' "${job_id}" "${task_status}"
exit "${task_status}"
