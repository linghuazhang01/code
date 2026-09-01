# Burgundy access and Slurm allocation

## Two different requests

### 1. Administrative HPC account/access request

The official [CityUHK HPC Account Request page](https://www.cityu.edu.hk/its/services-facilities/list-of-services-facilities/h/hpc-account-request)
states that teaching staff, research staff, and research degree students are
eligible. It asks interested users to describe their computing needs and
desired applications to `csc.hpc@cityu.edu.hk`, and says account applications
are submitted through an online CSC Work Request at the
[CityUHK Service Portal](https://service.cityu.edu.hk/sp).

Prepare identity/EID, department, project purpose, supervisor/PI context,
software list, expected GPU count/memory/runtime, CPU/RAM, and storage needs.
The public page does not state that a separate PDF or a supervisor-signed form
is universally required; department-level approval may still apply.

### 2. Slurm GPU resource request

After the account has partition permission, every computation is requested as
a Slurm job. Successful `sbatch`/`srun` submission proves scheduler access; it
does not prove the assigned physical GPU is healthy.

## Connect and inspect

```bash
ssh lzhan37@burgundy.hpc.cityu.edu.hk
sinfo -o "%20P %8a %12l %8D %10t %G"
bash skills/cityu-a100/scripts/a100_inventory.sh
```

Do not run training on the login node. File management, job submission,
downloads, installation, and compilation are the intended lightweight login
operations described by the official page.

## Default OPD debug allocation

Interactive request:

```bash
srun \
  --partition=gpu_a100 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --pty bash
```

Short batch verification request:

```bash
sbatch --parsable \
  --job-name=opd-a100-verify \
  --partition=gpu_a100 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=00:20:00 \
  --output="$HOME/scratch/opd/logs/opd-a100-verify-%j.log" \
  --wrap="cd '$HOME/scratch/opd/mopd_code' && PHASES=verify bash scripts/setup_burgundy.sh"
```

Use `--exclude=<nodes>` only from refreshed evidence. As of the inventory
snapshot, `gpu-a100-07` and `gpu-a100-11` had reset-required free cards. Do not
turn these exclusions into permanent policy after CSC repairs the nodes.

## Inspect the exact allocation

```bash
squeue -j <JOB_ID> -o "%i|%T|%M|%L|%N|%R"
scontrol show job -dd <JOB_ID>
sacct -j <JOB_ID> \
  --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList,AllocTRES%80 \
  -n -P
```

`scontrol show job -dd` can expose the physical GRES detail such as `IDX:2`.
Within a Slurm cgroup, `CUDA_VISIBLE_DEVICES=0` may be a remapped view rather
than physical GPU index zero.

## Search recent allocation history

Every A100 status search includes both current `squeue` state and bounded
`sacct` history. Use a 7-day partition-visible window and a 30-day owned window
unless the user requests another range:

```bash
sacct -X -S now-7days -r gpu_a100 -n -P \
  --format=JobIDRaw,User,JobName,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,NodeList,ReqTRES%80,AllocTRES%80

sacct -X -S now-30days -u "$USER" -n -P \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,NodeList,ReqTRES%80,AllocTRES%80
```

Always identify any `opd-queue-slot-3g` and `opd-queue-slot-4g` rows. Preserve
Job ID, state, exit code, node, requested TRES, and allocated TRES so a terminal
failure is not confused with a scheduler-capacity shortage. State when Slurm
accounting privacy or retention restricts the visible rows. Historical records
do not prove that a node is currently free or that its GPU is CUDA-healthy.

## Mandatory in-allocation health gate

First inspect the assigned GPU:

```bash
printf 'host=%s\n' "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,compute_mode \
  --format=csv,noheader
nvidia-smi -q | grep -A 2 "GPU Recovery Action"
```

Then activate the OPD environment and dispatch a seeded kernel:

```bash
source "$HOME/scratch/opd/logs/activate_training_env.sh"
python - <<'PY'
import json

import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
left = torch.randn(64, 64, device="cuda")
right = torch.randn(64, 64, device="cuda")
result = left @ right
assert result.shape == (64, 64)
assert torch.isfinite(result).all()
print(
    "CUDA_WITNESS",
    json.dumps(
        {
            "device": torch.cuda.get_device_name(0),
            "shape": list(result.shape),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        sort_keys=True,
    ),
)
PY
```

Pass requires an A100 with at least 80,000 MiB, no reset requirement, and a
successful witness. `nvidia-smi`, imports, and `torch.cuda.is_available()` are
only preliminary checks.

## Fault handling and stopping condition

If `GPU Recovery Action: Reset` appears or the seeded allocation returns
`CUDA-capable device(s) is/are busy or unavailable` with no compute process:

1. Save Job ID, node, partition, GPU UUID/GRES detail, `nvidia-smi -q`, and the
   failing witness log.
2. Stop retrying that node. Do not issue `nvidia-smi --gpu-reset` as a user.
3. Cancel only the exact owned job if it has no remaining useful work:
   `scancel <JOB_ID>`.
4. Contact `csc.hpc@cityu.edu.hk` and request that CSC reset or drain the
   affected physical GPU.
5. A single-GPU job cannot normally select a physical index. Testing another
   free card on the same node may require a bounded multi-GPU diagnostic and
   explicit user authorization.

Do not submit repeated replacement jobs once all scheduler-free cards show the
same admin-reset condition. Wait for CSC action or a healthy occupied card to
be released.
