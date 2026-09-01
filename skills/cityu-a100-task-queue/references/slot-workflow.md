# CityU A100 queued slot workflow

## State model

The local queue is durable, LIFO, and independent of Slurm:

```text
pending -> claimed -> running -> completed | failed
   |          |
   |          `-> pending (release only before dispatch)
   `-> canceled
```

Use `scripts/queue_manager.py`. Its default state file is
`<repo>/.codex/state/cityu-a100-task-queue.json`; the path is gitignored and
written atomically under a file lock.

Common operations from the OPD repository root:

```bash
python skills/cityu-a100-task-queue/scripts/queue_manager.py init

python skills/cityu-a100-task-queue/scripts/queue_manager.py add \
  --title "descriptive task name" \
  --gpu-counts 3,4 \
  --remote-cwd '$HOME/scratch/opd/mopd_code' \
  --command 'a command that branches on OPD_SLOT_GPU_COUNT'

python skills/cityu-a100-task-queue/scripts/queue_manager.py list
python skills/cityu-a100-task-queue/scripts/queue_manager.py peek --gpu-count 3
```

Do not put credentials in a queued command. Both text and JSON `list` output
omit command bodies; use `show --task-id ...` only when the user needs the exact
stored task.

## Adaptive 3-/4-GPU commands

Queue OPD training tasks for both slot sizes unless the user explicitly limits
one. The command must inspect `OPD_SLOT_GPU_COUNT` and apply a validated resource
profile for the winning slot; metadata alone does not make a command portable.

For the common separate-ref layout, reserve one GPU for `ref_policy` and use
the remaining cards for `actor_rollout`:

| Slot | Actor GPUs | Ref GPUs | Batch near 255 |
|---|---:|---:|---:|
| 3 A100 | 2 | 1 | 256 |
| 4 A100 | 3 | 1 | 255 |

Select the nearest user-acceptable batch divisible by the actor world size.
Dry-run every branch and verify the final expanded values for batch size,
mini-batch size, worker placement, method, domains, and training steps. Keep
branch-specific experiment and checkpoint names when their effective resource
profiles differ. Fail closed for an unexpected slot count.

## One queue cycle

Run this workflow only when the queue has a pending task. The user's explicit
request to add a runnable task to this managed queue authorizes one bounded
paired-slot cycle for pending work; a status-only request does not.

1. Use `Cityu-A100 Skill` to refresh inventory, reservations, current faults,
   account/QoS limits, and the user's exact owned jobs.
2. Reconcile existing `opd-queue-slot-3g` and `opd-queue-slot-4g` Job IDs before
   submitting. Never create a duplicate of either slot for one cycle.
3. Submit one 3-A100 and one 4-A100 batch slot by piping
   `scripts/a100_slot_worker.sh` to remote `sbatch`. Do not copy `ssh2.sh` or its
   password to Burgundy.
4. Inspect both jobs with `squeue`, `scontrol show job -dd`, and `sacct`. A
   pending reason is not a CUDA-health result.
5. When a job is `RUNNING`, inspect its output or ready marker. Only
   `QUEUE_SLOT_READY` permits dispatch.
6. Choose the newest pending task compatible with the ready slot. If both slots
   become ready together, prefer the 4-GPU slot when the newest task supports
   both; otherwise use the slot compatible with that task.
7. Atomically `claim` the task using the winning Job ID. Render it only after
   the claim succeeds, upload the rendered script atomically to that worker's
   job-specific command path, then mark the task `running`.
8. Cancel the exact unused sibling Job ID from the authorized pair after a
   healthy winner is chosen. If both become ready together, choose first, then
   cancel only the other exact owned Job ID before dispatching. Never cancel a
   job that was not recorded as this cycle's sibling.
9. Reconcile the winning job's terminal state with `sacct`, then mark the local
   task `completed` or `failed`. Preserve the worker log and claimed command
   file for audit until the user asks to clean them.
10. If pending tasks remain, a later authorized cycle may submit a new pair.

If delivery fails after a local claim but before the command file becomes
visible to the worker, use `release` with the same task and Job ID. Never
release a task after its worker may have begun executing it.

## Resource envelope

Derive limits live on every submission. With the observed
`partition-gpu-a100` limit of 7200 GPU-minutes per job:

| Slot | CPUs | RAM | Maximum wall time from 7200 GPU-minutes |
|---|---:|---:|---:|
| 3 A100 | 24 | 192 GiB | 2400 minutes = `1-16:00:00` |
| 4 A100 | 32 | 256 GiB | 1800 minutes = `1-06:00:00` |

The requested time is the minimum of the partition MaxTime, QoS MaxWall, and
`floor(MaxTRESMinsPerJob / GPU count)`. Do not hardcode this table after live
limits change. Keep memory at or below the Cityu-A100 project ceiling.

Suggested job names and output paths:

```text
opd-queue-slot-3g -> $HOME/scratch/opd/logs/opd-queue-slot-3g-%j.log
opd-queue-slot-4g -> $HOME/scratch/opd/logs/opd-queue-slot-4g-%j.log
```

Pass `OPD_SLOT_GPU_COUNT`, plus a unique command and ready path containing the
Job ID. Exclude nodes only from refreshed fault evidence. The worker validates
all visible cards, writes `QUEUE_SLOT_READY`, waits for exactly one command
file, executes it once, and exits with the task's status.

## Atomic dispatch commands

After selecting the winning Job ID and GPU count:

```bash
python skills/cityu-a100-task-queue/scripts/queue_manager.py claim \
  --gpu-count <3-or-4> \
  --slurm-job-id <JOB_ID> \
  --json

python skills/cityu-a100-task-queue/scripts/queue_manager.py render \
  --task-id <TASK_ID> \
  --slurm-job-id <JOB_ID>
```

Pipe the render output to a restrictive remote temporary file, then atomically
rename it to the exact `OPD_SLOT_COMMAND_FILE` for the winning worker. Do not
print the rendered command into normal status output. Once the remote rename is
confirmed:

```bash
python skills/cityu-a100-task-queue/scripts/queue_manager.py start \
  --task-id <TASK_ID> \
  --slurm-job-id <JOB_ID> \
  --node <NODE>
```

## Status-only requests

Status inspection is read-only even when an active cycle exists:

- Run local `list --json` and `peek` for both GPU counts.
- Use `Cityu-A100 Skill` for live inventory and SSH handling.
- Query exact owned slot jobs with `squeue`; use `sacct` for recently terminal
  jobs and worker logs for `QUEUE_SLOT_READY` or `QUEUE_TASK_END`.
- On every search, query a bounded 7-day `gpu_a100` partition history and
  30-day owned history. Always surface `opd-queue-slot-3g` and
  `opd-queue-slot-4g` rows with state, exit code, node, requested TRES, and
  allocated TRES; state when accounting visibility or retention is limited.
- Join records by `slurm_job_id`. Report mismatches instead of silently changing
  local state.
- Separate `PENDING`, `RUNNING`, worker-ready, task-running, and task-terminal.

Do not submit, cancel, claim, release, or finish anything merely because the
user asked what is happening.
