# Task Brief

## Task ID

`2026-05-10-sam3d-runtime-test`

## Request

Use `demo_3_max` to test SAM 3D Objects. If Hugging Face credentials are not
available in WSL, use the Windows-side Hugging Face login/cache to obtain the
SAM3D weights and copy them into WSL.

## Outcome

Run a checkpoint-backed SAM 3D Objects smoke test from the existing
`demo_3_max` environment, using the local RTX 5090 stack without downgrading
Python, Torch, or CUDA.

Completed. WSL Hugging Face login worked, official SAM3D checkpoints were
downloaded into the external SAM3D checkout, the CPU-only sparse backend issue
was corrected by replacing generic `spconv` with `spconv-cu121==2.3.8`, and a
low-step checkpoint-backed SAM3D runtime smoke produced a Gaussian splat `.ply`
on the local RTX 5090.

## Scope

- In scope: verifying Hugging Face login, downloading/checking SAM3D
  checkpoints, running official sample inference or the closest feasible smoke,
  and recording results.
- Out of scope: changing the `demo_3_max` core Python/Torch/CUDA stack,
  retraining models, and modifying PhysTwin runtime code.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `docs/plans/completed/2026-05-10-sam3d-demo3-max/qa-report.md`
- `/home/zhangxinjie/external/sam-3d-objects/README.md`

## Acceptance Criteria

- Confirm whether WSL Hugging Face login works.
- Ensure `checkpoints/hf/pipeline.yaml` and required model files exist.
- Run a SAM3D runtime smoke with `demo_3_max` on RTX 5090.
- Record exact command, result, and any blockers.
