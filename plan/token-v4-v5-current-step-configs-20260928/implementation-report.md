# Token V4/V5 Current-Step Implementation Report

## Outcome

Implemented the supplied Token V4/V5 definitions as versioned, hash-locked
runtime artifacts and added the requested Current-Step TopLoss matrix:

- Token versions: V4 and V5
- Math/Code Top-P pairs: 0.05/0.05, 0.05/0.02, and 0.05/0.01
- GPU counts: 3, 4, and 8
- Global train batch and PPO mini-batch: 528
- Public profiles: 2 x 3 x 3 = 18

## Frozen Taxonomy

- V4 Control: Math 121, Code 154
- V5 Control: Math 59, Code 65
- Shared Structure: Math 6, Code 32
- Artifact SHA256:
  `27884fadeea94145df23ede69109c7f8181c866de2f1c9f93d7659786719d817`
- V5 Control is a strict subset of V4 Control; Control and Structure are
  disjoint within every version/domain.

The online selector receives Control candidates only. Position-valid Structure
tokens receive raw weight 4. Control and Structure are then normalized together
exactly once per microbatch/domain.

## Position Semantics

- Math: `Final`/`Answer` only when used as a heading or bold label; boxed and EOS
  markers are valid anywhere.
- Code: only the final closed fenced code block is eligible; fence/language
  markers are restricted to the fence line, code Structure tokens to non-comment
  code lines, and EOS remains eligible anywhere.
- Original generation token boundaries are preserved by decoding each token id
  independently. The runtime never retokenizes the whole response, avoiding
  silent mask loss on noncanonical token sequences.

## GPU Topology

All profiles colocate policy and reference work (`separate_ref_policy: false`).
Actor world sizes are 3, 4, and 8; teacher FSDP sizes match those values.
Therefore the 8-GPU profile uses one student actor rank per GPU and one teacher
model sharded across all eight GPUs. It does not create eight teacher replicas.
Dedicated-teacher-only performance switches remain disabled.

With batch 528, the actor-local sample counts are 176, 132, and 66 for 3/4/8
GPUs. Under the existing 1:1 Math/Code sampling, each domain contributes
88, 66, and 33 samples per actor respectively.

## Main Files

- `mopd_verl/domain_gradient/token_v4_v5.json`
- `mopd_verl/domain_gradient/token_v4_v5_membership.csv`
- `mopd_verl/domain_gradient/token_taxonomy_registry.py`
- `mopd_verl/domain_gradient/structure_positions.py`
- `mopd_verl/domain_gradient/current_step_weights.py`
- `configs/token_selection/math_code/taxonomy/`
- `scripts/finalize_token_v4_v5.py`
- `Token.md`
- `docs/current-step-token-weighting.md`

## Validation

- `uvx ruff format` / `uvx ruff check`: passed for V4/V5-owned Python files.
- Focused pytest suite: 168 passed.
- `git diff --check`: passed.
- Artifact rebuilt from `Qwen/Qwen3-1.7B` and byte-compared successfully.
- Qwen noncanonical token-boundary check confirmed that `[13023, 68]` decodes
  to ` Finale` but retokenizes as `[5649, 1574]`; the new per-token decoding path
  handles this without changing response indices.
- Independent post-fix review found no blocker or high-severity issue; its
  focused subset reported 50 passing tests.

No GPU smoke training was run in this implementation pass. A 1-2 step run is
still needed to measure whether the 8-GPU colocated topology is faster and fits
the target hardware comfortably.
