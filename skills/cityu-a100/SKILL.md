---
name: cityu-a100
description: This skill should be used when the user asks about "Cityu-A100 Skill", CityUHK Burgundy A100 availability or Slurm allocation, or deploying the OPD repository, environment, data, and models to Burgundy; do not use it for unrelated clusters.
metadata:
  short-description: CityUHK Burgundy A100 与 OPD 部署
---

# Cityu-A100 Skill

Manage CityUHK Burgundy A100 work for the OPD repository without treating a
stale scheduler snapshot or a successful import as proof that a GPU is usable.

## Route the request

- For current A100 capacity, free-card counts, node states, reservations, or
  known faults, read `references/resource-inventory.md`.
  Run `bash skills/cityu-a100/scripts/a100_inventory.sh` before answering a
  current-status question.
- For CSC account access, Slurm resource requests, job inspection, GPU health
  gates, cancellation, or fault escalation, read
  `references/slurm-allocation.md`.
- For putting the OPD repository on Burgundy or installing its Conda
  environment, data, and Qwen models, read `references/opd-bootstrap.md`.

Read only the references required by the request. When a task spans allocation
and deployment, read both allocation and bootstrap references.

## Required workflow

1. Refresh the live Slurm inventory. Distinguish node state, CPU availability,
   configured GRES, allocated GRES, reservations, and CUDA health. Every status
   search must also include bounded `sacct` history for the `gpu_a100`
   partition and the user's owned jobs; history is context, never a substitute
   for current `squeue`/`scontrol` evidence.
2. Confirm whether the user means an administrative CSC/HPC account request or
   a Slurm job request. Do not conflate them.
3. Before submitting, report partition, GPU count/type, CPUs, RAM, wall time,
   exclusions, and expected command. Default debug shape for this project is
   one A100/80GB, 8 CPUs, 64 GiB RAM, and a short bounded wall time.
4. Inside the allocation, require both the NVIDIA recovery-state check and a
   seeded CUDA kernel witness. `nvidia-smi` visibility and
   `torch.cuda.is_available()` alone are insufficient.
5. Bootstrap through the repository-level Burgundy wrapper; use phase
   resumption instead of reinstalling completed assets.
6. Record Job ID, node, partition, paths, versions, witness, and faults. Do not
   declare the environment ready until the kernel witness succeeds.

## Safety and project invariants

- Treat repo-local `ssh2.sh`, passwords, tokens, and `.env` files as secrets.
  Never print, copy, upload, commit, or quote their contents. Never source the
  whole credential file as shell code.
- Use the local OPD tree as source of truth. Preview remote sync with
  `rsync --dry-run`, use a narrow allowlist, and never use `rsync --delete`.
- Keep Conda, models, datasets, logs, and caches under `~/scratch/opd`; Burgundy
  home quota is much smaller. Keep requested memory at or below 400 GiB unless
  the user explicitly changes the resource envelope.
- Do not interpret Slurm `MIXED` as a free GPU. A node may have idle CPUs while
  every GPU GRES is allocated.
- `GPU Recovery Action: Reset` means the card requires administrator action.
  Preserve evidence and stop using that physical card; users must not attempt
  an NVIDIA GPU reset on shared nodes.
- Submit, cancel, requeue, or expand a live allocation only within the user's
  authorization. Resolve the exact Job ID before `scancel`.
- Current fault exclusions are evidence, not permanent policy. Refresh them
  after CSC maintenance or a reset.

## Completion criteria

A deployment is complete only when the environment/data/model phases finish,
the expected asset paths validate, and a seeded CUDA kernel prints its witness
on an A100/80GB. If the assets are installed but every available card requires
a reset, report the deployment as staged and GPU verification as blocked by
cluster health.
