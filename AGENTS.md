# PhysTwin Subtree Rules

This subtree is the upstream PhysTwin side of the project.

## First Reads

- `../AGENTS.md`
- `../docs/phystwin/README.md`
- `HARNESS.md`
- `docs/harness/README.md`
- relevant bridge task page if the work is driven by a bridge need

If this repository is checked out standalone and the parent-level files do not
exist, continue with this file, `HARNESS.md`, and `README.md`.

## Harness Engineering

All non-trivial work in this repository is managed through the harness
engineering system. Treat `HARNESS.md` as the entry map and use
`docs/harness/` for the operating model, contracts, verification rules, and
templates.

Before changing code, create or update a task brief, sprint contract, or active
plan under `docs/plans/active/` unless the change is mechanical and obvious.
Before finishing, attach verification evidence using the QA report template.

## Preferred Bias

Treat PhysTwin edits here as upstream-facing or data/reconstruction-facing work.
Do not mix Newton-bridge conclusions into PhysTwin files unless the coupling is
explicitly the topic.
