# Task Brief

## Task ID

`2026-05-10-sam3d-demo3-max`

## Request

Clone the existing Demo 2 Max environment, install
`facebookresearch/sam-3d-objects`, keep the current Python, PyTorch, and CUDA
versions, and produce a Demo 3 Max environment compatible with SAM 3D Objects.

## Outcome

`demo_3_max` exists as a conda clone of `demo_2_max`. It keeps Python
3.12.13, PyTorch 2.11.0+cu130, Torch CUDA 13.0, torchvision 0.26.0+cu130, and
the RTX 5090 SM120 build target. SAM 3D Objects is installed editable from
`/home/zhangxinjie/external/sam-3d-objects`; the import path through
`sam3d_objects.pipeline.inference_pipeline_pointmap` and the notebook
`inference` module passes.

## Scope

- In scope: locating Demo 2 Max, cloning it to Demo 3 Max, installing
  SAM 3D Objects dependencies around the existing stack, and recording
  compatibility notes.
- Out of scope: modifying PhysTwin runtime algorithms, retraining models,
  downloading large checkpoints unless required for import validation, and
  changing global system Python/Torch/CUDA versions.

## First Files To Read

- `AGENTS.md`
- `HARNESS.md`
- `README.md`
- `docs/harness/verification.md`

## Data And Hardware Assumptions

- Required data: none for installation/import validation.
- Required device: local RTX 5090 laptop GPU for CUDA import/build validation.
- Expected runtime: dependency installation may take several minutes.
- External services: GitHub clone from `https://github.com/facebookresearch/sam-3d-objects.git`.

## Acceptance Criteria

- Demo 2 Max source is identified before cloning.
- Demo 3 Max is created as a separate environment/folder.
- Python, Torch, and CUDA versions are recorded before and after installation.
- SAM 3D Objects is installed or vendored without changing those core versions.
- At least one SAM 3D Objects import or metadata check passes.

## Verification Plan

- Static checks: record `python --version`, Torch version, Torch CUDA version,
  and `nvcc --version` if available.
- Functional checks: import Torch and at least one SAM 3D Objects module or run
  a package metadata check.
- Full checks: optional GPU model inference if checkpoints/data are available.

## Risks

- Risk: SAM 3D Objects dependency pins may try to install a different Torch or
  CUDA build.
- Mitigation: install dependencies with explicit constraints or `--no-deps`
  where appropriate, then add only missing non-core packages.
- Risk: NVIDIA Kaolin 0.17.0 does not compile against PyTorch 2.11/CUDA 13.
- Mitigation: use a local `kaolin==0.17.0` compatibility package that covers
  the SAM3D imports actually exercised by this environment, while documenting
  that this is not a full Kaolin replacement.
