# Task Brief

## Task ID

`2026-05-24-phystwin-max-env`

## Request

Create a local `phystwin-max` environment for the RTX 4090 machine, using
Python 3.12 and PyTorch CUDA 13.0 wheels installed through `pip3 install torch
torchvision`. Include the full PhysTwin runtime and data-processing dependency
surface where compatible with that stack, including a source-built full Kaolin
rather than a narrow compatibility shim.

## Outcome

The user can activate `phystwin-max` and run PhysTwin/data-processing imports
against Python 3.12, Torch 2.12, Torch CUDA 13.0, and the local RTX 4090 GPUs.

## Scope

- In scope: conda environment creation, pip/conda dependency installation,
  local extension builds when compatible, and smoke-test evidence.
- In scope: recording or patching incompatibilities caused by Python 3.12,
  Torch 2.12, or CUDA 13.0 without silently downgrading those pinned
  requirements.
- Out of scope: changing PhysTwin algorithms or downloading full datasets.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/README.md`
- `env_install/env_install.sh`
- `env_install/5090_env_install.sh`
- `env_install/data_process.sh`
- `docs/plans/completed/2026-05-10-sam3d-runtime-test/handoff.md`

## Data And Hardware Assumptions

- Required data: no full dataset required for environment smoke tests.
- Required device: local NVIDIA RTX 4090 with driver reporting CUDA 13.0.
- Expected runtime: dependency installation may take tens of minutes and may
  include source builds.
- External services: PyTorch wheel index, PyPI, conda channels, and GitHub.

## Acceptance Criteria

- Criterion 1: `phystwin-max` exists with Python 3.12.
- Criterion 2: `torch` and `torchvision` are installed from CUDA 13.0 wheels
  with Torch 2.12 and GPU access.
- Criterion 3: core PhysTwin and data-processing imports succeed where upstream
  packages support Python 3.12/Torch 2.12/CUDA 13.0.
- Criterion 4: any incompatible full-stack dependencies are documented with
  command evidence rather than bypassed silently.
- Criterion 5: full Kaolin, not a minimal shim, imports and runs a CUDA mesh
  operation.

## Verification Plan

- Static checks: `python scripts/check_harness_docs.py`.
- Functional checks: Python/Torch/CUDA import smoke, selected PhysTwin imports,
  and selected data-processing imports.
- Full checks: local extension build/import smokes when builds complete.

## Risks

- Risk: some upstream packages or CUDA extensions may not publish Python 3.12 or
  Torch 2.12/CUDA 13.0 compatible wheels.
- Mitigation: preserve the required Python/Torch/CUDA versions, install
  compatible packages, and document exact blockers for incompatible packages.
