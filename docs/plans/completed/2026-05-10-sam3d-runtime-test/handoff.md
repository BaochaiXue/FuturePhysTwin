# Handoff

## Task ID

`2026-05-10-sam3d-runtime-test`

## Current State

SAM 3D Objects is now checkpoint-backed and runtime-smoked from `demo_3_max` on
the local RTX 5090. WSL Hugging Face auth worked, so no Windows-side model copy
was required.

The important environment delta from the original `demo_3_max` bootstrap is
that generic `spconv==2.3.8` was replaced with `spconv-cu121==2.3.8` and
`cumm-cu121==0.7.11`. Torch remains `2.11.0+cu130` with Torch CUDA `13.0`.

## Changed Areas

- Conda env package set:
  `/home/zhangxinjie/miniconda3/envs/demo_3_max`
- SAM3D checkpoints:
  `/home/zhangxinjie/external/sam-3d-objects/checkpoints/hf`
- Runtime output:
  `/home/zhangxinjie/outputs/sam3d-runtime-test/kidsroom_mask14_sam3d_smoke_gs.ply`
- Harness records:
  `docs/plans/completed/2026-05-10-sam3d-runtime-test/`

## Verification Evidence

- Command: `conda run --no-capture-output -n demo_3_max hf auth whoami`
- Result: WSL is logged into Hugging Face as `XinjieZhang`.

- Command: model-load smoke with
  `Inference("checkpoints/hf/pipeline.yaml", compile=False)`
- Result: loaded `InferencePipelinePointMap` on `cuda`; spconv backend and sdpa
  attention selected; DINOv2 and SAM3D checkpoints loaded.

- Command: `spconv-cu121` sparse convolution smoke
- Result: `SubMConv3d` executed on `cuda:0` with torch `2.11.0+cu130` on the
  NVIDIA GeForce RTX 5090 Laptop GPU.

- Command: low-step SAM3D official sample smoke
- Result: stage1, stage2, and Gaussian decode completed; output `.ply` size is
  46,901,920 bytes.

## Next Agent Should Read

1. `AGENTS.md`
2. `HARNESS.md`
3. `docs/plans/completed/2026-05-10-sam3d-demo3-max/qa-report.md`
4. `docs/plans/completed/2026-05-10-sam3d-runtime-test/qa-report.md`

## Open Follow-Ups

- Run the default 25-step SAM3D demo only when a quality/performance benchmark
  is needed; the current task validated the runtime path with one step per
  stage.
- If future SAM3D code paths require more Kaolin APIs, extend
  `/home/zhangxinjie/external/kaolin-compat` or revisit a real Kaolin port for
  PyTorch 2.11/CUDA 13.

## Notes

For direct lower-level pipeline calls, pass masks as `uint8` alpha masks with
values 0/255. Bool masks work through the public `Inference.__call__` wrapper
because it performs the alpha conversion itself.
