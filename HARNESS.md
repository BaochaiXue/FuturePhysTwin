# Harness Engineering Entry Point

This repository is managed as a harness-engineered codebase. The goal is to make
agent work reliable by keeping intent, contracts, verification, and handoff
state in versioned files that future agents can read.

## Read Order

1. `AGENTS.md`
2. `README.md`
3. `docs/harness/README.md`
4. The active task brief or plan under `docs/plans/active/`
5. The files named by the task contract

## Core Rule

Use the repository as the record system. Do not rely on chat-only decisions for
requirements, architecture, verification, or known debt. If a decision matters
after the current session, encode it in `docs/harness/`, `docs/plans/`, or the
nearest relevant source file.

## Operating Loop

1. Planner writes a task brief and, for larger work, an active plan.
2. Generator proposes a sprint contract before implementation.
3. Evaluator reviews the contract before implementation when risk is non-trivial.
4. Generator implements only the agreed scope.
5. Evaluator runs checks, records evidence, and files actionable findings.
6. Generator fixes findings or records a scoped follow-up.
7. Handoff captures changed files, commands run, residual risk, and next tasks.

Small mechanical changes may collapse this into a single task brief plus QA
report, but the same fields still apply.

## Required Artifacts

- Task brief: `docs/harness/templates/task-brief.md`
- Sprint contract: `docs/harness/templates/sprint-contract.md`
- QA report: `docs/harness/templates/qa-report.md`
- Handoff: `docs/harness/templates/handoff.md`
- Decision record: `docs/harness/templates/decision-record.md`

Copy templates into `docs/plans/active/<task-id>/` for active work. When work is
finished, move the folder to `docs/plans/completed/<task-id>/` or record why it
was abandoned.

## PhysTwin Bias

This is the upstream PhysTwin side of the project. Harness artifacts should
describe data processing, reconstruction, Gaussian rendering, simulation,
training, evaluation, and interactive playground behavior in upstream-facing
terms. Do not introduce Newton-bridge conclusions unless the task explicitly
requires that coupling.

## Local Check

Run this before finishing harness-related changes:

```bash
python scripts/check_harness_docs.py
```
