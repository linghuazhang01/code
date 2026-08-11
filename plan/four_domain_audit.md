# Four-Domain Dynamic OPD Audit Plan

## Goal

Audit the copied dynamic-domain-budgeting implementation, remove only files
proven redundant or superseded, and make the runnable dynamic profile cover
`math`, `code`, `science`, and instruction following (`if`).

## Verification contract

- The four-domain profile loads through `mopd_verl.settings.load_config`.
- Training and validation paths cover exactly the four configured domains.
- IF training rows are routed to the `if` teacher and IFBench reward path.
- The dynamic controller receives IF validation metrics and emits `q`, `p`,
  and `lambda` for all four domains.
- Every configured batch contains the required minimum samples per domain and
  `p_actual * lambda == q` within floating-point tolerance.
- Focused tests, all config-load tests, static compilation, and a dry-run
  command succeed.

## Work items

1. [completed] Build the dependency and data-flow inventory.
2. [completed] Replace the obsolete three-domain dynamic profile with a
   four-domain profile and update docs/tests.
3. [completed] Remove only superseded or demonstrably redundant code/config
   files.
4. [completed] Run focused and broad verification; classify pre-existing
   failures separately.
5. [completed] Run an independent code review and resolve actionable findings.

## Deletion policy

No dataset, experiment output, or pre-existing user change is removed. A file
is deleted only when repository-wide reference search and tests show that it is
superseded or its implementation can be consolidated without changing the
public runtime contract.

## Audit results

- Confirmed existing IF training data: 16,575 rows with
  `data_source=m2rl_ifbench` and domain label `if`.
- Reused the sibling source repository's ignored four-domain assets through
  explicit directory symlinks; no dataset bytes were duplicated or modified.
- Removed the superseded three-domain dynamic profile, the legacy weighted
  sampler API, the standalone persistence helper, and root-launcher debug
  output.
- Reduced the disabled audit/paper-eval config surface from 326 to 230 lines.
- Added fail-fast validation for teacher aliases, rollout multiplicity, and
  corrupt controller checkpoints.
- Empty response rows are skipped only for variance estimation; their batch
  share still participates in exact `q = p * lambda` scaling.
- Broad CPU test suite: 211 passed.
- All 57 ordinary and named configuration references load successfully.
- Four-domain dry-run renders train/validation data, teacher aliases, dynamic
  controller parameters, and IF reward routing.

## Remaining external prerequisites

- Replace the four `teacher_scores: 1.0` calibration placeholders with actual
  Qwen3-30B fixed-probe scores before formal training.
- Materialize or download `../models/Qwen3-4B` and
  `../models/Qwen3-30B-A3B-Instruct-2507` on the training host.
- Activate the repository training environment, which installs
  `verifiable_instructions`; the official IFBench fallback checkout is also
  available under the sibling `temp/IFBench` directory.
- A four-domain Top-32 GPU smoke was not possible on this host.

Independent final review found no remaining P0, P1, or P2 issue.
