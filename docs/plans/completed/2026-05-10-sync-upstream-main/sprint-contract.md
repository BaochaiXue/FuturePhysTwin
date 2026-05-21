# Sprint Contract

## Contract ID

`sync-upstream-main`

## Goal

Merge `upstream/main` into local `main`, resolve conflicts, and push the result
to `origin/main`.

## Agreed Scope

Files expected to change:

- `README.md`
- upstream-added assets
- harness task records for this sync

Files expected to read:

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- Git remote and branch state

Out of scope:

- Runtime source edits outside merge results.
- Force-push, reset, or history rewriting.
- PR creation.

## Behavior To Deliver

- Fork `main` contains all commits from `upstream/main`.
- Fork-specific harness engineering scaffold remains in history.
- `origin/main` points at the updated merge result.

## Acceptance Criteria

- Must complete a normal merge, not a reset.
- Must resolve all conflicts.
- Must push `main` to `origin`.
- Must keep local working tree clean after push.

## Evaluator Probes

The evaluator should explicitly check:

- `rg -n "<<<<<<<|=======|>>>>>>>" README.md` finds no conflict markers.
- `git rev-list --left-right --count main...upstream/main` reports zero on the
  right side.
- `git rev-list --left-right --count origin/main...main` reports `0 0` after
  push.
- `python scripts/check_harness_docs.py` passes.

## Verification Commands

```bash
python scripts/check_harness_docs.py
git diff --check
git status --short --branch
git rev-list --left-right --count main...upstream/main
git rev-list --left-right --count origin/main...main
```

## Contract Changes

None.
