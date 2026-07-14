# FSDP Synchronized Gradient Accumulation Bug

## Summary

The sequence-replay gradient audit previously accumulated multiple synchronized
FSDP backward passes directly into the same parameter `.grad` buffers. Under the
`fsdp_size: 1` replicated actor layout, a later synchronized backward could
rescale gradient values already present in `.grad`. Consequently, early
micro-batches and early domains received exponentially smaller weights, and the
reported domain gradients depended on `mopd_audit.domains` ordering.

This was not a domain-mask, `full_gradient_freq_steps`, or manual FSDP finalize
bug. The gradient was already biased before finalize was called.

## Observed Failure

Two 3-GPU, 2-step runs used identical data, seed, model, and batch settings. The
only difference was domain order:

| Domain order | Math gradient norm | Code gradient norm |
| --- | ---: | ---: |
| `[math, code]` | 0.0125373855 | 1.9829976951 |
| `[code, math]` | 0.8023926733 | 0.0309843390 |

Moving either domain from first to second changed its norm by exactly 64 times:

```text
0.8023926733 / 0.0125373855 = 64
1.9829976951 / 0.0309843390 = 64
```

Six later micro-batches produced the observed `2^6` factor. The same issue also
biased earlier micro-batches within a domain, so taking the result from whichever
domain happened to be last was not a valid workaround.

## Root Cause

The unsafe pattern was equivalent to:

```python
optimizer.zero_grad()
for micro_batch in replay_schedule:
    loss = replay_loss(micro_batch)
    loss.backward()  # synchronized backward into an existing .grad
```

The intended gradient is a linear sum:

```text
G = g1 + g2 + ... + gN
```

The observed behavior instead gave earlier contributions geometric attenuation:

```text
G_biased ~= g1 / 2^(N-1) + g2 / 2^(N-2) + ... + gN
```

Zero-masked non-target slots still execute backward and synchronization, so they
can rescale earlier nonzero contributions even though their own contribution is
zero.

## Why Existing Checks Missed It

### Manual finalize parity

The historical metric reported:

```text
training_grad_before_vs_after_manual_finalize/rel_l2 = 0
training_grad_before_vs_after_manual_finalize/cosine = 1
training_grad_before_vs_after_manual_finalize/projection_share = 1
```

This only proves that manual finalize did not mutate the gradient it received.
It does not prove that preceding synchronized backward accumulation was correct.

### Availability and token masks

`sequence_target_domain_<domain>_available=1` and a positive `token_mask_sum`
only prove that the target existed and produced a nonzero gradient. They do not
detect incorrect weighting across replay slots.

### Closure

The old total replay and domain replay used the same unsafe accumulation
mechanism. A self-consistent but biased total can therefore pass a closure gate.
Closure must be combined with a domain-order invariance regression.

## Considered Fix: One Long `no_sync` Region

Wrapping the first `N-1` long-sequence replays in `no_sync()` removes repeated
gradient synchronization, but it also lets actor ranks progress independently
across the entire replay. Independent per-micro-batch accumulation was selected
because it preserves the existing collective cadence and directly represents a
linear sum of synchronized micro-gradients.

The first `no_sync` validation attempt reached the NCCL watchdog. However, an
identical run after restoring the original training backward path timed out at
the same first collective, and a standalone two-GPU `all_gather` also hung.
Therefore the timeout must not be attributed to `no_sync`. The machine's default
NCCL P2P path was faulty at validation time; the standalone collective and both
training runs completed after setting `NCCL_P2P_DISABLE=1`. This environment
workaround does not change gradient mathematics.

## Implemented Fix

Each replay micro-batch now owns an independent synchronized gradient:

```text
for each micro-batch:
    clear parameter .grad
    run synchronized backward
    finalize FSDP backward state
    add this micro-gradient to a CPU float32 accumulator
```

Only after all replay slots finish is the accumulator converted to the configured
gradient storage dtype. A later synchronization can no longer modify an earlier
micro-gradient because the earlier value no longer resides in `.grad`.

Relevant implementation:

- `mopd_verl/full_gradient/tracker.py`
- `mopd_verl/tensorboard_filter.py`
- `tests/test_mopd_verl.py`

## Validation

After the fix, swapping domain order produced identical results:

| Domain order | Math gradient norm | Code gradient norm |
| --- | ---: | ---: |
| `[math, code]` | 1.2280323691 | 11.8056730417 |
| `[code, math]` | 1.2280323691 | 11.8056730417 |

The swapped run also reported:

```text
sequence_target_domain_<domain>_sync_backward_count = 12
sequence_target_domain_<domain>_no_sync_count = 0
sequence_target_domain_<domain>_independent_accumulation = 1
full_grad_sequence/domain_sum_vs_total/rel_l2 = 7.115399077e-09
full_grad_sequence/domain_sum_vs_total/cosine ~= 1
```

Validation runs:

- `[math, code]`: `https://wandb.ai/lz101-rice-university/MOPD/runs/4tksrxpn`
- `[code, math]`: `https://wandb.ai/lz101-rice-university/MOPD/runs/glj4r0rz`

## Required Regression Checks

Any future change to gradient replay, FSDP finalize handling, micro-batch
scheduling, domain masking, or gradient storage must satisfy all checks below:

1. Run identical data and seed with at least two domain orders. Each domain's
   gradient norm and vector metrics must be invariant to ordering.
2. Require one independent synchronized backward per executed replay slot.
   `sync_backward_count` must equal `executed_micro_batch_count`.
3. Require `independent_accumulation=1` whenever sequence replay is used.
4. Require every configured domain to be present and available with a positive
   token-mask sum.
5. Require domain-sum versus total-replay closure below the configured threshold;
   a healthy deterministic smoke should be much smaller than `1e-6`.
6. Do not interpret finalize before/after equality as end-to-end gradient
   correctness. It is only a mutation check for the finalize operation.
7. Do not accumulate repeated synchronized backward calls directly into the same
   `.grad` buffers unless the distributed wrapper's behavior is independently
   proven and covered by an order-invariance test.

The historical four-domain 3-GPU profile used for this report has been
retired. The maintained topology regression suite now lives in
`test_grad_configs/`; use its five profiles for `NO_SHARD`, `FULL_SHARD`, and
`HYBRID_SHARD` coverage. For the domain-order check above, run
`configs/mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code.yaml` twice with the same
seed/data and reversed `math`/`code` mapping order, then compare the per-domain
gradient vectors and closure metrics.
