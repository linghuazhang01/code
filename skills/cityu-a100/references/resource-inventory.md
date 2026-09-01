# Burgundy A100 resource inventory

## Source hierarchy

1. Live Slurm (`scontrol`, `sinfo`, `squeue`) is authoritative for current
   scheduling state.
2. Bounded `sacct` history must accompany every status search to expose recent
   node assignments, failures, cancellations, and prior owned slot jobs. Its
   visibility and retention may be account-limited, and it does not establish
   current availability.
3. An in-allocation `nvidia-smi -q` plus seeded CUDA kernel is authoritative
   for whether the allocated card can execute work.
4. The official [CityUHK HPC Account Request page](https://www.cityu.edu.hk/its/services-facilities/list-of-services-facilities/h/hpc-account-request)
   describes the designed capacity: twelve A100/80GB nodes with four
   NVLink-based A100 accelerators per node. It does not supersede live outages,
   reservations, or temporary GRES removal.

## Live snapshot: 2026-08-28T20:39:55+08:00

| Node | Live A100 configuration | Partition/state | Interpretation |
|---|---:|---|---|
| `gpu-a100-01` | registered 4, configured 0 | no visible partition; `DOWN+DRAIN+NOT_RESPONDING` | Drained by `TRIX-DRAINER`; not schedulable. |
| `gpu-a100-02` | 0 | no visible partition; `IDLE` | CPU node record is online, but A100 GRES is removed. |
| `gpu-a100-03` | 4, allocated 0 | `stingy`/`special_cs`/`testing`; `RESERVED` | Reserved through 2026-09-01 15:45 +08:00 for another user/account. |
| `gpu-a100-04` | 4, allocated 4 | `stingy`; `ALLOCATED` | No free GPU GRES. |
| `gpu-a100-05` | 4, allocated 2 | `stingy`; `MIXED` | Two scheduler-free cards; one selected card was confirmed reset-required, the other is unverified. |
| `gpu-a100-06` | 4, allocated 4 | `stingy`; `ALLOCATED` | No free GPU GRES. |
| `gpu-a100-07` | 4, allocated 3 | `gpu_a100`; `MIXED` | The scheduler-free card was confirmed reset-required. |
| `gpu-a100-08` | 4, allocated 4 | `gpu_a100`; `ALLOCATED` | No free GPU GRES. |
| `gpu-a100-09` | 4, allocated 4 | `gpu_a100`; `ALLOCATED` | No free GPU GRES. |
| `gpu-a100-10` | 4, allocated 4 | `gpu_a100`; `MIXED` | GPUs are full; `MIXED` reflects idle CPU/RAM only. |
| `gpu-a100-11` | 4, allocated 3 | `gpu_a100`; `MIXED` | The scheduler-free card was confirmed reset-required. |
| `gpu-a100-12` | 4, allocated 3 | `gpu_a100`; `MIXED` | One scheduler-free card appeared after the earlier snapshot; CUDA health is unverified. |

At this timestamp, nodes `03` through `12` contribute 40 A100 GRES to active
Slurm configuration. Node `01` is named and registered with four GPUs but is
drained; node `02` exposes zero GPU GRES. Therefore the `01..12` naming scheme
does not imply 48 currently schedulable cards.

## Confirmed reset-required evidence

| Job | Partition/node | Result |
|---|---|---|
| `514838` | `gpu_a100 / gpu-a100-07` | Assets installed; seeded CUDA allocation failed. Physical GRES detail included `IDX:2`; `GPU Recovery Action: Reset`. |
| `514901`, `514902` | `gpu_a100 / gpu-a100-11` | Independent A100/80GB showed `GPU Recovery Action: Reset`; `torch.ones(..., device="cuda")` failed. |
| `514904` | `stingy / gpu-a100-05` | A100-SXM4-80GB, 81920 MiB, showed `GPU Recovery Action: Reset`; kernel failed. |

`torch.cuda.is_available() == True`, 16 MiB reported usage, and no compute
process were present in these cases. Those facts do not override the reset
requirement.

## Refresh commands

From a Burgundy login node at the OPD repository root:

```bash
bash skills/cityu-a100/scripts/a100_inventory.sh
```

For raw scheduler evidence:

```bash
sinfo -a -N -o "%N|%P|%T|%C|%m|%G"
scontrol show node gpu-a100-01
scontrol show node gpu-a100-12
scontrol show reservation -o
sacct -X -S now-7days -r gpu_a100 -n -P \
  --format=JobIDRaw,User,JobName,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,NodeList,ReqTRES%80,AllocTRES%80
sacct -X -S now-30days -u "$USER" -n -P \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,NodeList,ReqTRES%80,AllocTRES%80
```

Calculate GPU availability from `CfgTRES` and `AllocTRES`, not from CPU state.
Do not call a free slot usable until an allocated health gate passes.
Filter the owned 30-day history for `opd-queue-slot-3g` and
`opd-queue-slot-4g`, while retaining other OPD A100 jobs as fault context.
Explicitly report when accounting privacy or retention limits the visible
history.

## Updating this snapshot

Append a new timestamped subsection after a meaningful scheduler change or CSC
maintenance. Preserve old fault evidence for auditability, but mark exclusions
as cleared only after a fresh kernel witness succeeds on the repaired card.
