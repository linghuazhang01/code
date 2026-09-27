# Token V4/V5 revision 3 migration

## Goal
Adopt the user's third-revision definition with minimal edits to existing configs and necessary validation/position logic. Execute implementation using GPT-6 Sol max.

## Phases
- [x] Read third revision and confirm the existing source JSON has matching counts.
- [x] Sol max: update artifact/config definitions, necessary position rules, and documentation/tests.
- [x] Primary: review and deliver.

## Scope and decisions
- Source: ../plan/code-gap-20260927/structure-budget-context-20260927/; attachment: /Users/linghuazhang/.codex/attachments/19853d7f-0659-426c-a03e-2c64620f96e3/已粘贴的文本.txt.
- Expected V4 C/S Math119/9 Code153/45; V5 Math61/9 Code72/45. C answer words move to shared Structure. V5 sibling additions4/8, V4 intersection57/64.
- Update existing config paths directly; retain Current-Step Math .05/Code .05,.02,.01, batch528, GPU3/4/8 colocated and w4/w8 settings. No new experiment matrix.
- Artifact SHA/profile are locked: update registry/importer and config pins together. Preserve r2 snapshot and distinguish r3 output namespaces to avoid mixed results.
- Minimal runtime change is required: final/answer/conclusion labels in headings or bold labels for both domains; Math additionally permits these words on the same line as literal backslash-boxed. Code answer labels must not be treated as code I/O tokens or variables. Preserve existing Control outside-fence, I/O positions and single normalization.
- Verify source JSON/CSV/NPZ support strictly>20 and C/S disjointness. Add focused boundary tests and validate all36 resolved profiles.
- No refactoring unrelated runtime, no GPU run, remote sync, commit or push.

## Status
Implementation complete: r3 artifact imported with 2,664 CSV/NPZ comparisons and zero mismatches;
38 config files updated in place; r2 snapshot preserved; 257 focused cases passed.
Primary review complete: independently checked source counts, JSON byte identity,
set disjointness, label/boxed rules, config preservation and final diff check.
See `implementation-report.md`.
