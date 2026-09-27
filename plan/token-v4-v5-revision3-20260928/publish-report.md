# Token V4/V5 revision 3 Git delivery

## Scope and index protection

User explicitly authorized commit and normal push through the implementation Agent.
Repository: `https://github.com/linghuazhang01/code`; branch: `main`.
Initial HEAD and fetched `origin/main`: `c3ea0eb988774fed8a2d8173206e6204146751d0`.
A real `git fetch --no-tags origin main` completed; ahead/behind was 0/0.

The initial index had no staged changes. Only the files listed below are selected.
No `git add -A`, stash, reset of the working tree, force-push, PR, remote training
sync or GPU task is part of this delivery.

- Full V4/V5 r3 runtime, artifact, schema/metadata wiring, position masks and tests.
- Required Current-Step dependency implementation, including existing head/tail/TC
  support that the selector/schema/tests share. The 5 existing Current-Step example
  configurations are included as regression fixtures and to keep the documented
  examples reproducible; no new configuration matrix is introduced.
- 44 V4/V5 configuration files, r1/r2 artifact/config snapshots and their historical
  implementation reports preserve provenance.
- `Token.md`, Current-Step documentation and r3 plan/reports.
- `.gitignore`: only 5 token-related exception lines; the 2 LCB exception lines
  remain unstaged.

Excluded work stays local: all eval/scorer changes, `start.sh`, evaluation shell
scripts, project memory, `MANIFEST.md`, `docs/domain-and-token-weighting.md`,
unrelated experiment/queue/resume configurations and runtime outputs.

## Exact files and hunks

Full-file selection (96 files):

```text
Token.md
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v4_fixed4.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v4_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v4_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v4_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v5_fixed4.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v5_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v5_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/_mopd_math_code_current_step_token_v5_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_loss_teacher_confidence_m05_c01_tailhalf_4gpu.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_teacher_confidence_m05_c01_tailhalf_4gpu.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c01_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c02_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v4_toploss_m05_c05_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c01_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c02_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed4_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed4_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed4_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed8_3gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed8_4gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_token_v5_toploss_m05_c05_fixed8_8gpu_colocated.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_toploss_m05_c01_fixed4_4gpu.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_toploss_m05_c01_tailhalf_4gpu.yaml
configs/token_selection/math_code/taxonomy/mopd_math_code_current_step_toploss_m05_c01_tailzero_4gpu.yaml
docs/current-step-token-weighting.md
mopd_verl/domain_gradient/audit.py
mopd_verl/domain_gradient/code_positions.py
mopd_verl/domain_gradient/config.py
mopd_verl/domain_gradient/control_selection_scoring.py
mopd_verl/domain_gradient/current_step_selection.py
mopd_verl/domain_gradient/current_step_weights.py
mopd_verl/domain_gradient/occurrence.py
mopd_verl/domain_gradient/occurrence_config.py
mopd_verl/domain_gradient/structure_positions.py
mopd_verl/domain_gradient/token_source_metrics.py
mopd_verl/domain_gradient/token_taxonomy_history/revision_1/README.md
mopd_verl/domain_gradient/token_taxonomy_history/revision_1/configs.json
mopd_verl/domain_gradient/token_taxonomy_history/revision_1/token_v4_v5.json
mopd_verl/domain_gradient/token_taxonomy_history/revision_1/token_v4_v5_membership.csv
mopd_verl/domain_gradient/token_taxonomy_history/revision_2/README.md
mopd_verl/domain_gradient/token_taxonomy_history/revision_2/configs.json
mopd_verl/domain_gradient/token_taxonomy_history/revision_2/token_v4_v5.json
mopd_verl/domain_gradient/token_taxonomy_history/revision_2/token_v4_v5_membership.csv
mopd_verl/domain_gradient/token_taxonomy_history/revision_2/token_v4_v5_provenance.json
mopd_verl/domain_gradient/token_taxonomy_registry.py
mopd_verl/domain_gradient/token_v4_v5.json
mopd_verl/domain_gradient/token_v4_v5_membership.csv
mopd_verl/domain_gradient/token_v4_v5_provenance.json
mopd_verl/domain_gradient/token_weighting_metrics.py
mopd_verl/launch.py
mopd_verl/settings.py
mopd_verl/verl_audit.py
plan/token-v4-v5-current-step-configs-20260928/implementation-report.md
plan/token-v4-v5-fourbaseline-20260928/implementation-report.md
plan/token-v4-v5-revision3-20260928/implementation-report.md
plan/token-v4-v5-revision3-20260928/notes.md
plan/token-v4-v5-revision3-20260928/publish-report.md
plan/token-v4-v5-revision3-20260928/task_plan.md
scripts/finalize_token_v4_v5.py
tests/test_amplification_sources.py
tests/test_answer_structure_positions.py
tests/test_current_step_config.py
tests/test_current_step_runtime.py
tests/test_current_step_selection.py
tests/test_domain_gradient_optimization_contracts.py
tests/test_structure_positions.py
tests/test_token_v4_v5.py
tests/test_token_v4_v5_runtime.py
third_party/verl/verl/trainer/ppo/ray_trainer.py
third_party/verl/verl/workers/actor/dp_actor.py
```

Partial `.gitignore` selection:

```gitignore
!/scripts/finalize_token_v4_v5.py
!/tests/test_token_v4_v5.py
!/tests/test_token_v4_v5_runtime.py
!/configs/token_selection/math_code/taxonomy/*token_v4*.yaml
!/configs/token_selection/math_code/taxonomy/*token_v5*.yaml
```

## Candidate validation

Exported staged Git tree `6db67ff0850055c864808245ddb9d65aaf959b61` with
`git archive` into an isolated temporary directory. Tests ran against that tree,
with `PYTHONPATH=.` and `/opt/anaconda3/bin/python`, without copying dirty or
ignored source files: **256 passed, 4 deselected in 7.54s**.

```sh
python -m pytest tests/test_token_v4_v5.py tests/test_structure_positions.py \
  tests/test_answer_structure_positions.py tests/test_token_v4_v5_runtime.py \
  tests/test_current_step_config.py tests/test_current_step_selection.py \
  tests/test_current_step_runtime.py tests/test_amplification_sources.py \
  tests/test_audit_occurrence_overrides.py tests/test_config_profiles.py \
  tests/test_domain_gradient_rebuild.py \
  -k 'not real_two_rank and not frozen_taxonomy_matches_canonical_source' -q
```

Only this publish report was finalized after testing; runtime, config, artifact
and test blobs in the final staged tree match the tested tree.

The existing `test_frozen_taxonomy_matches_canonical_source` depends on the
ignored legacy `analysis-output/.../global-taxonomy.csv`; it is excluded from the
clean-tree run. The three existing `real_two_rank` Gloo cases are also excluded,
as planned. V4/V5's own checked-in JSON/CSV/hash/support tests remain included.

Secret/large-artifact checks apply to the selected Git blobs; public taxonomy
token IDs and membership data are not authentication tokens.

- Credential/private-key pattern scan: no matches. No credential files or binary
  experiment outputs are selected. The 97 selected blobs total about 1.27 MB;
  the largest is 143,909 bytes.
- The default staged whitespace check flags exactly three existing raw-data
  details: r1 membership CSV's final blank line, and embedded CRLF in token 21128
  in r2/current membership CSV. These bytes are required by their frozen hashes;
  they are preserved. Scoped whitespace checking excluding only these three CSV
  files passed. No source data was trimmed or rewritten to silence the check.
- The prior working-tree check did not inspect untracked files; this staged
  check explicitly covers the newly tracked artifacts and records their data-only
  whitespace exceptions.

## Commit and push

Planned message: `feat(tokens): adopt revision 3 V4/V5 taxonomy`.
Use the configured Git author identity; no false Claude co-author attribution.
After candidate validation, create the commit, normal-push `main` to
`origin/main`, then compare local HEAD with `git ls-remote origin refs/heads/main`.
The final response records the resulting commit SHA and remote verification.
