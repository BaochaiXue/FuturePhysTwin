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

## Known External Runtime State

- `demo_3_max` has checkpoint-backed SAM 3D Objects runtime validation on the
  local RTX 5090 while preserving Python/Torch/CUDA from `demo_2_max`.
- WSL Hugging Face auth is already valid as `XinjieZhang`; Windows-side
  checkpoint download/copy was not needed for the SAM3D smoke.
- SAM3D checkpoints live outside git at
  `/home/zhangxinjie/external/sam-3d-objects/checkpoints/hf`.
- The completed runtime record is
  `docs/plans/completed/2026-05-10-sam3d-runtime-test/qa-report.md`; read its
  handoff before changing `demo_3_max`, SAM3D checkpoints, or sparse backend
  packages.

## Preferred Bias

Treat PhysTwin edits here as upstream-facing or data/reconstruction-facing work.
Do not mix Newton-bridge conclusions into PhysTwin files unless the coupling is
explicitly the topic.
