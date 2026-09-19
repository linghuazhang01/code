# A occurrence sidecar — 2026-09-19

Scope: configuration and documentation only; no implementation/test edits, launch,
remote sync, or commit. Existing dirty worktree belongs to concurrent work.

## Original A and lineage

Source: `../../experiments_records/eval/MOPD_Research_Brief_20260918.md`, row A;
the exact YAML is confirmed by `../../experiments_records/eval/Summary_Detail.md`
(configuration table and run-ID table). Paths below are relative to the code repo:

1. New overlay: `configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_4gpu_A_occurrence_rkl32_fixed4_b528_colocated.yaml`.
2. Direct parent, original A: `configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_a3b_instruct_2507_4gpu_math_code_science_topp_m05_c01_s01_code_structure_only_science_loss_teacherconf_fixed4_b528_colocated.yaml`.
3. A's parent: `configs/token_selection/math_code_science/taxonomy/mopd_qwen1p7b_30b_a3b_instruct_2507_8gpu_math_code_structure_only_toploss_topp05_fixed4_b528.yaml`.
4. Root: `configs/mopd_qwen1p7b_30b_a3b_instruct_2507_5gpu_math_code_science_topk32_baseline_b528.yaml`.

Ancestor filenames do not determine the resolved GPU layout: original A overrides
it to four co-located actor/teacher GPUs. Original A SHA256 at verification:
`ed3f61249d6eefd505774d119c02b2d6cbe6bae6d13fd3a74d353d84841c4901`.
Ancestor SHA256 values, in order:
`4e7d3d86add2aaa7348b4cb0622ae933298bb3dd9e198c67cd39d2fefca05380`,
`836ebd32a590f2cb239538373749c159026deb3fd1e293244314e8c4646dca48`.

The existing `mopd_qwen1p7b_30b_4gpu_m05_c01_s01_cs_occurrence_rkl32_fixed4_b528_colocated.yaml`
is a different C+S/TopLoss experiment, not the A parent; it is untouched.

## Preserved parameters and semantic differences

Only `audit.control_token_online_selection_unit: occurrence` changes a non-identity
YAML field. This preserves A's YAML parameters, but is **not an exact one-variable
ablation**: the occurrence path also changes score provenance and application
timing as described below. Original A remains the default `token_id` mode and is untouched.
The six identity overrides are `runtime.wandb_run_id`, `audit.output_dir`,
`paper_eval.output_dir`, `huggingface_checkpoint.path_prefix`,
`trainer.experiment_name`, and `trainer.default_local_dir`. Their shared new
namespace is `q1p7b-4g-A-occ-mtl05-ctl01-sltc01-f4-b528-coloc-s60`.
`runtime.wandb_resume: never` is inherited unchanged.

| Domain | Unchanged candidates | Unchanged budget | Unchanged score family |
|---|---|---|---|
| Math | 124 Control + 266 Structure | 5% | TopLoss |
| Code | 551 Structure only | 1% | TopLoss |
| Science | 128 Control + 244 Structure | 1% | Loss+TC |

Taxonomy follows `Token.md` section 0. In occurrence mode, eligibility is computed
within each domain across the current complete actor minibatch and all DP ranks;
retain the configured candidate sets and strict token-ID count >20 gate. Rank
individual eligible response positions, so repeated instances of one ID can
receive different weights. The target is `ceil(p * N)`, where N includes all valid
response positions in that domain, including Other and ineligible IDs. Candidate
scarcity caps selection; this is a count budget, not loss mass.

For Science Loss+TC, compute Q2–Q98 separately for detached raw RKL loss and
Teacher log-probability of the actual Student-chosen token, over **globally
eligible occurrences within that domain** (all DP ranks and microbatches).
Clip/min-max normalize each signal using those bounds, then rank by
`L_hat + C_hat + L_hat*C_hat`. Do not first average by token ID, exponentiate
chosen Teacher logp, or use response-local/per-rank normalization. Original ID
mode retains its existing token-ID-mean Q2–Q98 normalization and ranking.
Math/Code retain the TopLoss score family, but the occurrence path ranks detached
raw `selector_token_loss` before weighting/rollout-IS. This differs from original
ID-mode TopLoss and Loss+TC: `audit.observe_completed_step` initializes
`selection_loss_batches_by_domain` only for `kl_entropy`; TopLoss/Loss+TC use
`configured_loss_batches`, whose configured loss includes rollout-IS in
`mopd_verl/full_gradient/actor_loss.py` (lines 385–395 at inspection). Consequently,
the original score source must not be described as uniformly raw pre-IS.

Timing also differs: occurrence selection uses a prepass on the current minibatch
and applies the selected positional masks to that same minibatch. Original ID
selection is updated after the completed step and applies its selected IDs to
the next step. Together with occurrence-level rather than ID-mean normalization,
these differences prevent attributing any future result solely to selection unit.
No old score source, timing, normalization, or ID-mode semantics are changed.
The occurrence implementation and edge-case validation are
owned by the main task; this sidecar does not alter them.

Selected positions have raw weight 4, other valid positions weight 1; retain
per-domain mean-one normalization, dividing by `1 + 3*S/N` for actual selected S.
All RKL32 (teacher support, reverse, temperature 1, no tail bucket), global batch
528, actor minibatch 528, 4GPU co-location, learning rate 5e-6, 60 training steps,
HF step [60]/public, model/data paths, sampling, gates and all other settings are
inherited without overrides.

## Checks performed

- Read repo `AGENTS.md` and all of `Token.md`; session helper is absent.
- Used repository `load_raw_config` to resolve both full inheritance chains;
  flattened leaf diff asserted exactly the seven allowed keys above, with no
  additional parameter changes, including candidate lists and training settings.
- Both original A and overlay passed repository `settings.load_config`.
  Compared complete `dataclasses.asdict` results: exactly the same seven changed
  leaves, with original selection unit resolved to `token_id`.
- Confirmed candidate group counts, four GPUs, both batch sizes, score families,
  fractions, Fixed4, RKL32 and HF settings against resolved original A.
- Interpreter: `/Users/linghuazhang/.cache/uv/archive-v0/3wBPM0pTyXpA3Xl4UEDLk/bin/python`.
  `git diff --check` passed. Main-task validator gained Loss+TC support during
  this audit; successful config loading is not a GPU/runtime validation claim.
- No experiment or evaluation launched; no commit. Only the new overlay and
  this plan were added by this sidecar. Main task retains implementation/testing
  ownership and must complete runtime verification before any separately
  authorized launch.

## Main-task integration verification

The implementation now supports occurrence-level TopLoss and Loss+TC. Combined
local verification passed 155 tests across `test_occurrence_selection.py`,
`test_occurrence_loss_teacher_confidence.py`, `test_online_control_selection.py`,
`test_per_domain_online_control_modes.py`, `test_per_domain_top_p.py`,
`test_domain_gradient_optimization_contracts.py`, and
`test_loss_teacher_confidence_selector.py`. This includes real two-rank Gloo
normalization and collective invalid-input checks; actor forwards are mocked.
Independent code review found no blocker. CUDA/NCCL/FSDP training remains untested.
No remote training was launched or modified. This feature and its required
configuration lineage are scoped for Git publication; unrelated evaluation edits
remain outside the commit.
