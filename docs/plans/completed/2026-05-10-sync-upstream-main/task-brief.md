# Task Brief

## Task ID

`2026-05-10-sync-upstream-main`

## Request

Bring the fork `BaochaiXue/FuturePhysTwin` up to date with upstream
`Jianghanxiao/PhysTwin`, reconcile merge conflicts, and push the updated fork.

## Outcome

Local `main` contains `upstream/main`, preserves fork-specific harness
engineering files, resolves the README conflict using upstream's current public
project documentation, and is pushed to `origin/main`.

## Scope

- In scope: Git remote synchronization, merge conflict resolution, lightweight
  validation, and push to `origin/main`.
- Out of scope: runtime PhysTwin behavior changes, GPU/data validation,
  upstream branch modification, and pull request creation.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/workflow.md`

## Data And Hardware Assumptions

- Required data: none.
- Required device: CPU only.
- Expected runtime: under five minutes excluding network latency.
- External services: GitHub SSH access for fetch and push.

## Acceptance Criteria

- `upstream/main` is fetched.
- Local `main` is not behind `upstream/main`.
- Merge conflicts are resolved without conflict markers.
- Fork-specific harness files remain present.
- `origin/main` is updated by push.

## Verification Plan

- Static checks: `python scripts/check_harness_docs.py`, `git diff --check`.
- Git state checks: `git status --short --branch`,
  `git rev-list --left-right --count main...upstream/main`, and
  `git rev-list --left-right --count origin/main...main`.
- Full checks: not applicable; no runtime code was intentionally changed by
  this task beyond the upstream merge.

## Risks

- Risk: README conflict resolution could drop fork-specific documentation.
- Mitigation: conflict was limited to public upstream update sections; resolved
  README against upstream's current version while retaining fork-specific
  harness docs outside README.
