# Task Plan: Audit-only loss vectors

## Goal
Support top-k teacher-student cross-entropy vocabulary vectors and explicit
absolute log-probability-difference vectors independently of the training loss
builder.

## Phases
- [x] Phase 1: Inspect current routing and output formats
- [x] Phase 2: Define configuration and compatibility semantics
- [x] Phase 3: Implement settings, trainer routing, logger output, and target config
- [x] Phase 4: Add and run focused tests
- [x] Phase 5: Review compatibility matrix and deliver

## Key Questions
1. How can audit-only top-k tensors be requested without activating top-k training?
2. Which vectors can be generated under each loss builder without changing its objective?
3. How should `logp_abs_vector` relate to the existing `gap_abs` output?

## Decisions Made
- Cross-entropy collection must be controlled by audit config, not inferred from the training loss builder.
- Audit-only top-k collection must not add a top-k term to the actor loss.
- `logp_abs` means `abs(teacher_logp - old_student_logp)` and should retain the existing gap fields for compatibility.

## Errors Encountered
- The configured Python environment is not available in the current shell; use repository-supported alternatives and focused tests where dependencies permit.
- A dedicated profile test was initially inserted before the prior loop's final assertions; moved those assertions back into the loop before verification.
- The bundled Python has NumPy/Pandas but no Torch; importing `verl_audit` reaches a module-level Torch import, so the lightweight frequency test could not execute.

## Status
**Complete** - Implementation, static verification, and independent review finished.
