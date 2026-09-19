# Current-batch occurrence selector

Implementation scope: local source and CPU tests only. No training, remote edits,
sync, or commit. The existing `token_id` mean-ranking selector remains the default.

## Configuration and semantics

`audit.control_token_online_selection_unit: token_id | occurrence` propagates
through settings, launcher overrides, `MOPDAuditLogger.full_gradient_meta`, and
`DomainGradientConfig`. Evaluation metadata retains `token_id` with weighting off.

The new runnable overlay is
`configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_4gpu_m05_c01_s01_cs_occurrence_rkl32_fixed4_b528_colocated.yaml`.
It inherits the latest pure teacher-support RKL32 C+S 4GPU configuration, batch
528, Math/Code/Science budgets 0.05/0.01/0.01, Fixed4, strict ID count >20, and
domain normalization. All run, audit, eval, and checkpoint identities are new.
It has not been launched.

The A-preserving Loss+TC extension is documented in `A-occurrence-20260919.md`.
It additionally supports `top_loss_teacher_confidence`: globally eligible
occurrences (not ID means) provide Q2–Q98 normalization statistics, and each
position is scored as `L_hat + C_hat + L_hat*C_hat`. C is the teacher's
chosen-token log-probability, not teacher entropy or top-k mass. The original
token-ID path and its normalization remain unchanged.

For each domain, let N be **all valid response positions** across the complete
actor minibatch and all actor DP ranks, including Other and ineligible IDs.
The target count is `ceil(p * N)` (decimal rounding, shared with existing budget
code). Eligible positions have a configured C+S ID. If strict occurrence gating
is enabled, that ID must appear strictly more than the configured threshold
(20 in the overlay) in this domain's current global minibatch. Non-strict gating
uses >=; threshold zero disables the count restriction. This gate never averages
or ranks by ID loss. The current batch is the only source window.

Rank eligible positions by detached raw `selector_token_loss` descending,
captured before TIP/FiRe or rollout-IS corrections. Pure RKL32 is required by the
overlay; mixed KL routing and unsupported weighting/baseline combinations fail
closed. Ties use actor rank, local microbatch index, row, response position;
ties are deterministic for a fixed partition. Duplicate token IDs can receive
different weights. Select at most the target count; insufficient candidates
produce an explicitly smaller selected_count. This is count budgeting, not loss
mass. Masked positions are never eligible and have weight zero. Nonfinite raw
loss at **any valid position**, including non-candidates, fails on all ranks;
it is never silently removed from N. Selector validity must equal response validity.

For actual selected count S, raw weights are 4 for selected positions and 1 for
all remaining valid positions. Divide by `1 + 3*S/N` separately for each domain.
The global valid-position mean is therefore one, even with rank imbalance or
candidate scarcity. There is no subsequent ID-based weight application.

The current-minibatch no-grad prepass precedes all training microbatches. Audit
RNG/buffers/modes/gradients are restored using existing `AuditState`. Cached masks
are keyed by exact microbatch identity with strong references, remain available
through audit replay and post-update source metrics, and are retired after
`observe_completed_step`; a new prepass clears the cache first. No positional
mask is stored in an actor checkpoint. Unknown batch objects fail closed.
Both YAML validation and the actor entrypoint require one PPO epoch and one
full actor minibatch; sequence parallelism >1 is unsupported.

## Efficiency overhead

Every optimizer minibatch adds **one complete actor forward without autograd**,
executed using the same microbatch split, plus CPU candidate extraction, one
actor-world `all_gather_object`, global candidate sorting, and mask construction.
Teacher-support tensors already in the batch are reused; no extra rollout or
teacher inference is requested. Training still performs its normal forward and
backward. If those cost F and B, added actor compute is F, or F/(F+B) relative to
ordinary actor update compute (roughly 33% if B is 2F). This is an analytical
estimate, not a measured GPU or end-to-end slowdown. Existing enabled audit
replays remain additional work.

No backward graph is retained for the scoring pass. Persistent positional masks
are float32, approximately 4 bytes per local response position. Candidate Python
objects are gathered onto every DP rank: memory/communication scale with the
global eligible-pool occurrence count, not vocabulary size; Python serialization
can be expensive on long batches. Counts include every valid response position,
but only configured C+S positions are transmitted. Weight initialization and
normalization are tensor operations; only selected indices receive Fixed4.

Runtime metrics expose `global/occurrence/prepass_seconds` (local elapsed time
including scoring/state restoration/communication/mask construction),
`prepass_forward_count`, and `gathered_candidates`; domain metrics expose
valid_count, budget_count, eligible_count, selected_count, and raw_weight_mean.
GPU throughput/memory measurement is intentionally pending a separately authorized
training or benchmark run.

## Validation

Run locally:

```sh
uv run --no-project --with torch --with pytest --with pyyaml python -m pytest \
  tests/test_occurrence_selection.py tests/test_online_control_selection.py \
  tests/test_per_domain_online_control_modes.py tests/test_per_domain_top_p.py -q
```

Tests cover raw versus corrected loss, duplicate IDs, masked/nonfinite positions,
global DP count budget, strict count gate, deterministic ties, candidate scarcity,
mean-one weights, batch identity, replay/post-step lifetime, configuration
propagation and mixed-mode rejection, and default selector regression. A real
two-process CPU/Gloo test runs the collective used by the scoring prepass with
all winners on one rank. Actor forwards are mocked; no GPU/FSDP training smoke
has been run. The coordinator separately observed three existing fixed-control
fixture failures caused by a missing historical 6GPU YAML; that unrelated
template is not restored by this change.

Recorded results (2026-09-19): implementation-side focused selector/default suite
passed 81 tests in 2.79s. The coordinator's broader final selection passed 124
tests in 2.66s, including real two-rank Gloo. The independent reviewer reported
no blockers and separately verified two microbatches per rank, distinct domain
rows, global budgets and mean-one sums, raw-versus-corrected ordering, masked
NaN exclusion, valid NaN on one rank causing both ranks to fail, and RNG/buffer
restoration. Review interpreter:
`/Users/linghuazhang/.cache/uv/archive-v0/3wBPM0pTyXpA3Xl4UEDLk/bin/python`
(torch 2.5.1). These CPU results do not establish CUDA/NCCL/FSDP performance.
