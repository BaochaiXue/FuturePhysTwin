# Handoff

## Task ID

`2026-05-10-sam3d-demo3-max`

## Current State

`demo_3_max` is installed and validated for SAM 3D Objects imports on the local
RTX 5090 machine. It is a clone of `demo_2_max` and keeps the same Python,
PyTorch, torchvision, and CUDA runtime versions.

## Changed Areas

- Conda env: `/home/zhangxinjie/miniconda3/envs/demo_3_max`
- SAM3D source: `/home/zhangxinjie/external/sam-3d-objects`
- PyTorch3D source patch: `/home/zhangxinjie/external/pytorch3d`
- MoGe source: `/home/zhangxinjie/external/MoGe`
- Kaolin compatibility shim: `/home/zhangxinjie/external/kaolin-compat`
- Conda hooks:
  - `/home/zhangxinjie/miniconda3/envs/demo_3_max/etc/conda/activate.d/demo_3_max.sh`
  - `/home/zhangxinjie/miniconda3/envs/demo_3_max/etc/conda/deactivate.d/demo_3_max.sh`

## Verification Evidence

- Command: torch/CUDA probe in `demo_3_max`
- Result: Python 3.12.13, torch 2.11.0+cu130, Torch CUDA 13.0,
  torchvision 0.26.0+cu130, RTX 5090 available, `TORCH_CUDA_ARCH_LIST=12.0`.

- Command: PyTorch3D/gsplat/spconv/kaolin import probe
- Result: passed. `gsplat` is 1.5.3, PyTorch3D is 0.7.8, `kaolin` resolves to
  `/home/zhangxinjie/external/kaolin-compat`.

- Command: SAM3D import matrix
- Result: passed for `sam3d_objects.pipeline.inference_utils`,
  `sam3d_objects.pipeline.inference_pipeline_pointmap`, and notebook
  `inference`.

- Command: `spconv` CUDA smoke
- Result: passed. A small `SubMConv3d` executed on `cuda:0`, RTX 5090.

## Next Agent Should Read

1. `AGENTS.md`
2. `HARNESS.md`
3. `docs/harness/README.md`
4. `docs/plans/completed/2026-05-10-sam3d-demo3-max/qa-report.md`

## Open Follow-Ups

- Follow-up: run checkpoint-backed inference once SAM3D checkpoints are
  available.
- Follow-up: decide whether to keep `kaolin-compat` as a local shim or port
  Kaolin 0.17.0 to PyTorch 2.11 if broader Kaolin APIs become necessary.
- Follow-up: runtime-test the complete sparse model path under a real SAM3D
  model load.
- Blocker: no blocker for import-level compatibility.

## Notes

Do not install the official cu121 requirement set into this env. The local
machine is RTX 5090, and `demo_3_max` is intentionally held on the cu130
PyTorch stack inherited from `demo_2_max`.
