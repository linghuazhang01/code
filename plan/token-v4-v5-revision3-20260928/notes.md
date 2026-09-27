# Findings

- Source JSON already matches revision3 counts; current runtime artifact/config remain revision2.
- Existing registry locks r2 SHA, so config-only membership updates fail validation.
- Existing label mask handles final/answer only for Math; r3 requires conclusion and Code answer markers plus Math boxed-line eligibility.
- Reuse prior implementation and testing environment /opt/anaconda3/bin/python. System python lacks NumPy; uv previously panicked. Prior real_two_rank Gloo tests need sandbox-external local CPU execution; no need to repeat unless distributed wiring changes.
- Preserve all unrelated dirty changes and history. Planning-with-files and daily-coding skills already read earlier in this conversation.

## Implementation results

- Imported r3 source JSON/CSV byte-for-byte after strict support and semantic checks: 666 membership rows, 2,664 NPZ comparisons, no mismatch.
- r2 JSON/CSV/provenance and 44 configs preserved under `token_taxonomy_history/revision_2/`; r1 history untouched.
- Updated 2 taxonomy bases and 36 public namespaces to r3; GPU topology bases unchanged.
- Shared answer label detection covers final/answer/conclusion; Math adds same-line literal boxed command, including CRLF; Code labels are excluded inside every fenced block and from I/O weighting.
- New underscore-label regression exposed the old word-boundary mismatch for `__Answer__`; fixed by requiring the closed marker directly.
- 253 focused cases plus 4 new Math/Code × w4/w8 integration cases passed (257 distinct), 3 real-two-rank tests intentionally not rerun because distributed wiring is unchanged.
- Final docs/report completed; primary review pending. No GPU, sync, commit or push.
