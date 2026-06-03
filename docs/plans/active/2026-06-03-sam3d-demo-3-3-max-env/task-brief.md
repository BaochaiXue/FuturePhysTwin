# Task Brief

## Task ID

`2026-06-03-sam3d-demo-3-3-max-env`

## Request

Clone conda environment `demo_3_1_max` as `demo_3_3_max` and install SAM 3D
Objects runtime dependencies while preserving Python 3.12, PyTorch, and CUDA
runtime versions.

## Outcome

Completed: `demo_3_3_max` imports the SAM 3D Objects notebook inference path
and key runtime dependencies without downgrading the inherited
Python/Torch/CUDA stack.

## Scope

- In scope:
  - Audit `demo_3_1_max` for SAM3D dependency availability.
  - Clone `demo_3_1_max` to `demo_3_3_max`.
  - Install SAM3D dependencies into `demo_3_3_max`.
  - Patch dependency source or package choices where needed for Python 3.12 and
    the existing Torch/CUDA stack.
  - Record verification evidence.
- Out of scope:
  - Downgrading Python, PyTorch, torchvision, Torch CUDA, or CUDA runtime.
  - Modifying user worktree changes unrelated to this environment install.
  - Vendoring generated model outputs into git.

## Acceptance Criteria

- `demo_3_3_max` exists as a clone of `demo_3_1_max`.
- Python remains 3.12.x.
- Torch remains `2.11.0+cu130` and Torch CUDA remains `13.0`.
- SAM3D package/import probes pass.
- Sparse CUDA backend probe passes or any blocker is recorded with exact error.
