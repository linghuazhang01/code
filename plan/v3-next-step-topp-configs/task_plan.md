# Task Plan: ExpandedPruned-V3 Next-Step Top-P Configs

## Goal

Add Math-only ExpandedPruned-V3 next-step Top-P overlays for budgets 1%, 2%,
5%, and 10% across the existing 4--8 GPU topology matrix.

## Phases

- [x] Phase 1: Define scope and create the persistent plan.
- [x] Phase 2: Audit existing V3 and Top-P config contracts.
- [x] Phase 3: Generate the 20 overlay configs and tests/documentation.
- [x] Phase 4: Run focused verification and review the resulting diff.

## Key Questions

1. Which fields and filename conventions encode Top-P without changing the V3 pool?
2. Which fields may vary only by GPU topology?
3. How should the obsolete Top-K field be represented when Top-P is active?

## Decisions Made

- Preserve the frozen 115-ID Math V3 unified candidate pool.
- Preserve the four existing untracked 8-GPU Top-P overlays and their
  accompanying test/README edits.
- Use a topology-matched V3 K25 standalone config as the base for each Top-P
  overlay; add only the missing 4--7 GPU variants.
- Treat Top-P as the active runtime budget; K25 remains provenance only, not a
  simultaneous selector constraint.

## Errors Encountered

- The local host does not provide `uv`, `pytest`, `ruff`, or `pyright`, so the
  standard project test/lint commands could not run. Replaced them with Python
  syntax compilation, recursive YAML resolution and contract checks, diff and
  whitespace checks, a focused secret scan, and an independent diff review.

## Status

**Complete** - The 20-profile Top-P matrix, tests, documentation, and focused
verification are finished.
