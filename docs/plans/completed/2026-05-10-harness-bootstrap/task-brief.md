# Task Brief

## Task ID

`2026-05-10-harness-bootstrap`

## Request

Create the initial harness engineering files for the PhysTwin repository so
future work is managed through versioned agent-readable artifacts.

## Outcome

The repository has a root harness entry point, focused harness documentation,
role prompts, task templates, plan directories, a local validation script, and
CI coverage for harness documentation changes.

## Scope

- In scope: harness docs, templates, plan folders, documentation checks, and
  AGENTS entry-point updates.
- Out of scope: PhysTwin source code, data processing behavior, training logic,
  rendering behavior, and Newton bridge conclusions.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/README.md`

## Data And Hardware Assumptions

- Required data: none.
- Required device: CPU only.
- Expected runtime: under one minute for local checks.
- External services: GitHub Actions only for CI execution.

## Acceptance Criteria

- `HARNESS.md` defines the repository harness entry point.
- `docs/harness/` defines operating model, workflow, context map, invariants,
  verification matrix, role prompts, and templates.
- `docs/plans/active/` and `docs/plans/completed/` exist.
- `scripts/check_harness_docs.py` validates the scaffold.
- GitHub Actions runs the harness docs check for relevant changes.

## Verification Plan

- Static checks: `python -m py_compile scripts/check_harness_docs.py`.
- Functional checks: `python scripts/check_harness_docs.py`.
- Full checks: not applicable; no runtime PhysTwin behavior changed.

## Risks

- Risk: Harness docs become prose-only guidance.
- Mitigation: Add a validation script and CI workflow as the first mechanical
  guardrail.
