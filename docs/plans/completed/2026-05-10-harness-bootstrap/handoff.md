# Handoff

## Task ID

`2026-05-10-harness-bootstrap`

## Current State

Initial harness engineering management is installed. Future non-trivial work
should start from `HARNESS.md`, copy templates into `docs/plans/active/`, and
record QA evidence before completion.

## Changed Files

- `AGENTS.md`
- `HARNESS.md`
- `.github/workflows/harness-docs.yml`
- `docs/harness/README.md`
- `docs/harness/context-map.md`
- `docs/harness/operating-model.md`
- `docs/harness/workflow.md`
- `docs/harness/verification.md`
- `docs/harness/invariants.md`
- `docs/harness/prompts/*`
- `docs/harness/templates/*`
- `docs/plans/tech-debt-tracker.md`
- `scripts/check_harness_docs.py`

## Verification Evidence

- Command: `python scripts/check_harness_docs.py`
- Result: passed, `harness-docs: ok`
- Output path: terminal output only

- Command: `python -m py_compile scripts/check_harness_docs.py`
- Result: passed
- Output path: ignored `__pycache__`

## Next Agent Should Read

1. `AGENTS.md`
2. `HARNESS.md`
3. `docs/harness/README.md`
4. `docs/harness/workflow.md`
5. `docs/harness/verification.md`

## Open Follow-Ups

- Follow-up: after one or two implementation tasks, tighten
  `scripts/check_harness_docs.py` to validate active/completed task artifact
  completeness.
- Blocker: none.

## Notes

The parent first-read files named by `AGENTS.md` were not present in this
standalone checkout, so `AGENTS.md` now documents the fallback path.
