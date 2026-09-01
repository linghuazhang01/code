---
name: cityu-a100-task-queue
description: Manage a persistent LIFO queue of OPD tasks and paired 3-/4-A100 Slurm slot jobs on CityUHK Burgundy; use when adding or listing queued work, dispatching the newest compatible task, or jointly reporting local queue and remote job status.
metadata:
  short-description: CityU A100 LIFO task queue
---

# CityU A100 Task Queue

Keep user work in a local, durable LIFO queue and bind the newest compatible
pending task only after a Burgundy 3- or 4-A100 slot is actually running and
healthy. Do not freeze a task into a pending Slurm job: selection happens at
dispatch time so a task added later can run first.

## Route the request

- For `add`, `list`, `peek`, local status changes, or rendering a claimed task,
  use `scripts/queue_manager.py`.
- For submitting the paired Slurm slots, dispatching a task, reconciling a
  completed job, or explaining the state machine, read
  `references/slot-workflow.md`.
- For any live Burgundy capacity, Slurm, reservation, allocation, or GPU-health
  decision, also use the project `Cityu-A100 Skill` at
  `../cityu-a100/SKILL.md` and its routed reference. Live Slurm evidence wins
  over queue metadata.

## Queue semantics

- The queue is LIFO: select the highest sequence number among pending tasks
  compatible with the active slot's GPU count. "Top" means newest compatible
  task, not the first item originally added.
- Treat OPD training tasks as compatible with both 3- and 4-GPU slots by
  default. Store `gpu_counts` as `3,4` and make the queued command branch on
  `OPD_SLOT_GPU_COUNT`; do not merely relabel a fixed-placement command.
- Validate both command branches before enqueueing. Adapt worker placement and,
  when required, choose the nearest user-acceptable global batch divisible by
  the actor world size. Preserve the requested method, data domains, training
  budget, and other experimental semantics across branches.
- An explicit request to add a runnable command to this managed queue authorizes
  that task's later execution and one bounded paired-slot cycle while pending
  work exists. A status-only request never grants submission or cancellation
  authority, and an empty queue never justifies holding GPUs.
- Claim a task atomically before delivering its command to a running slot.
  Never dispatch one task to both the 3-GPU and 4-GPU jobs.
- Keep completed, failed, and canceled records for auditability. Do not delete
  history merely to make the active queue shorter.
- Store no passwords, API keys, tokens, or private environment values in task
  titles, commands, notes, or queue state. The queue file is local runtime
  state and is gitignored.

## Remote slot invariants

- Maintain at most one owned 3-GPU slot and one owned 4-GPU slot for a queue
  cycle. Check exact Job IDs and names before submitting or canceling anything.
- A `PENDING` Slurm job is queued, not allocated. A `RUNNING` job is not ready
  until its worker logs `QUEUE_SLOT_READY` after the NVIDIA recovery-state and
  seeded CUDA witnesses pass on every assigned card.
- When one healthy slot wins, atomically claim the newest compatible task,
  deliver its rendered command only to that Job ID, mark it running, and cancel
  the exact unused sibling from that authorized pair. Never leave a second
  allocation idle or cancel a job outside the pair.
- A reset-required card is unusable. Preserve evidence, do not issue a GPU
  reset, and do not automatically launch replacement allocations after a
  health-gate failure.

## Status answers

For a status-only question, perform no mutations. Report together:

1. Local queue counts and newest compatible pending task.
2. Exact owned slot Job IDs, GPU counts, Slurm states, nodes or pending reasons,
   time limits, and remaining time.
3. Worker readiness, claimed/running task mapping, and terminal results.
4. Bounded `sacct` history: recent partition-visible `gpu_a100` jobs and the
   user's owned OPD/slot jobs, including state, exit code, node, requested TRES,
   and allocated TRES. Report visibility or retention limits.

Label local queue state separately from scheduler state and CUDA health. If SSH
is blocked, report the local queue as current and the remote portion as unknown;
never reuse an old Slurm snapshot as live status.
