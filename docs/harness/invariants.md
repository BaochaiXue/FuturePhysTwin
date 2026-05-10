# Invariants

These rules preserve agent readability and upstream PhysTwin maintainability.

## Repository State

- The repository is the source of truth for durable decisions.
- Keep `AGENTS.md` and `HARNESS.md` short maps, not encyclopedias.
- Put task-specific state under `docs/plans/active/<task-id>/`.
- Move completed task state to `docs/plans/completed/<task-id>/`.

## PhysTwin Scope

- Keep this subtree upstream-facing and data/reconstruction-facing.
- Do not mix bridge-specific conclusions into PhysTwin files unless explicitly
  required by the task.
- Document generated data paths, but do not commit large generated artifacts
  unless they are intentional tracked assets.

## Code Changes

- Prefer existing scripts and patterns over new orchestration layers.
- Keep CLI flags backward-compatible unless the task explicitly changes an
  interface.
- Avoid hidden global state in data, training, and rendering paths.
- Make mask, background, camera, case, and device assumptions explicit.
- Record any new required dependency in setup docs or environment scripts.

## Verification

- No non-trivial change is complete without a QA report.
- If full GPU/data verification is unavailable, record the exact command that
  should be run later.
- Do not claim reconstruction, rendering, or simulation quality improvements
  without metrics, visual evidence, or a clearly marked hypothesis.

## Documentation

- Documentation should point agents to the next file to read.
- Prefer stable file paths and commands over narrative-only guidance.
- When prose becomes a repeated rule, promote it to a check or test.
