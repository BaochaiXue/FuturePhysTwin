# Workflow

This workflow is the default lifecycle for repository work.

## 1. Intake

Create or update `docs/plans/active/<task-id>/task-brief.md`.

Required intake fields:

- Request
- Outcome
- Scope
- Out of scope
- Data or hardware assumptions
- Verification target
- First files to read

Use a short task id such as `2026-05-10-harness-bootstrap` or
`color-stage-debug-viz`.

## 2. Contract

For implementation work, create `sprint-contract.md` before editing source
files. The contract defines what "done" means.

The contract must include:

- User-visible or research-visible behavior
- Files expected to change
- Acceptance criteria
- Verification commands
- Failure modes the evaluator should probe

If the evaluator rejects the contract, revise the contract before implementing.

## 3. Implementation

Implement only the contract. If discovery changes the task, update the contract
before broadening the edit.

Prefer small, independently verifiable changes. For GPU-heavy code, add cheap
static or dry-run checks where possible, then record which full checks still
need data or hardware.

## 4. Evaluation

Create `qa-report.md` before completion.

Evaluation should record:

- Commands run
- Data cases used
- Pass/fail results
- Screenshots, metrics, or output paths when relevant
- Findings and fixes
- Checks skipped with a reason

Do not mark work complete only because code was written.

## 5. Handoff

Create `handoff.md` when work crosses sessions, has follow-ups, or changes
project conventions.

Move finished task folders from `docs/plans/active/` to
`docs/plans/completed/`. If work is intentionally paused, leave it active and
state the next action at the top of the handoff.

## 6. Maintenance

Run the harness docs check before finishing harness changes:

```bash
python scripts/check_harness_docs.py
```

When repeated evaluator findings reveal a durable rule, encode it in
`docs/harness/invariants.md` or a mechanical check.
