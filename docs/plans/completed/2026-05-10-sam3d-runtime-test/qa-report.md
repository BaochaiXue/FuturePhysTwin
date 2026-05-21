# QA Report

## Task ID

`2026-05-10-sam3d-runtime-test`

## Summary

Passed for checkpoint-backed runtime smoke. `demo_3_max` kept Python 3.12.13,
torch `2.11.0+cu130`, Torch CUDA `13.0`, and the RTX 5090 target. SAM 3D
Objects loaded the official Hugging Face checkpoints and ran the pointmap
pipeline through stage1, stage2, and Gaussian decoding.

## Environment

- Date: 2026-05-10
- Conda env: `/home/zhangxinjie/miniconda3/envs/demo_3_max`
- Torch: `2.11.0+cu130`
- Torch CUDA: `13.0`
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- CUDA arch target: `TORCH_CUDA_ARCH_LIST=12.0`
- SAM3D source: `/home/zhangxinjie/external/sam-3d-objects`
- Checkpoints: `/home/zhangxinjie/external/sam-3d-objects/checkpoints/hf`
- Output artifact:
  `/home/zhangxinjie/outputs/sam3d-runtime-test/kidsroom_mask14_sam3d_smoke_gs.ply`

## Commands Run

```bash
conda run --no-capture-output -n demo_3_max hf auth whoami
conda run --no-capture-output -n demo_3_max hf download \
  --repo-type model \
  --local-dir /home/zhangxinjie/external/sam-3d-objects/checkpoints/hf-download \
  --max-workers 1 facebook/sam-3d-objects
mv /home/zhangxinjie/external/sam-3d-objects/checkpoints/hf-download/checkpoints \
  /home/zhangxinjie/external/sam-3d-objects/checkpoints/hf
conda run --no-capture-output -n demo_3_max pip uninstall -y spconv cumm
conda run --no-capture-output -n demo_3_max pip install spconv-cu121==2.3.8
```

Runtime smoke:

```bash
conda run --no-capture-output -n demo_3_max env \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_CUDA_ARCH_LIST=12.0 \
  python -
```

The Python smoke loaded `notebook/inference.py`, instantiated
`Inference("checkpoints/hf/pipeline.yaml", compile=False)`, loaded the official
sample image and mask index 14, called `inf._pipeline.run(...)` with
`stage1_inference_steps=1`, `stage2_inference_steps=1`, and
`decode_formats=["gaussian"]`, then saved the Gaussian output as `.ply`.

## Results

- Hugging Face auth: passed in WSL as user `XinjieZhang`; Windows fallback was
  not needed.
- Checkpoint download: passed. Official SAM3D checkpoint files were placed
  under `checkpoints/hf/`, including `ss_generator.ckpt`,
  `slat_generator.ckpt`, `slat_decoder_mesh.ckpt`, `slat_decoder_gs.ckpt`, and
  `ss_decoder.ckpt`.
- Model load: passed. `InferencePipelinePointMap` loaded on `cuda`; initial
  load reserved about 12.7GB of GPU memory.
- Sparse backend correction: passed. Generic `spconv==2.3.8` failed during real
  SAM3D stage2 with `not implemented for CPU ONLY build`; replacing it with
  `spconv-cu121==2.3.8` fixed the real sparse convolution path.
- Runtime smoke: passed. The low-step sample completed in 16.82 seconds after
  model load and produced
  `/home/zhangxinjie/outputs/sam3d-runtime-test/kidsroom_mask14_sam3d_smoke_gs.ply`
  with size 46,901,920 bytes.
- Final smoke memory: about 12.767GB allocated and 16.031GB reserved.

## Findings

- Severity: medium
- Issue: The generic PyPI `spconv==2.3.8` package imports and can pass a shallow
  CUDA probe, but SAM3D stage2 reaches an operator path that is CPU-only in
  that package.
- Evidence: SAM3D failed in `SpconvOps_get_indice_pairs` with `not implemented
  for CPU ONLY build`.
- Status: fixed in `demo_3_max` by installing `spconv-cu121==2.3.8` and
  `cumm-cu121==0.7.11`. This did not change torch, Torch CUDA, Python, or
  torchvision.

- Severity: low
- Issue: Directly calling the lower-level pipeline with a bool mask creates an
  alpha channel of 0/1 rather than 0/255.
- Evidence: the first direct pipeline smoke failed with an empty post-crop mask
  during pointmap normalization.
- Status: fixed in the smoke by passing `(mask.astype("uint8") * 255)`. The
  public `Inference.__call__` wrapper already performs this conversion.

## Acceptance Criteria Results

- Criterion: Confirm WSL Hugging Face login.
  Result: passed.
- Criterion: Ensure checkpoint config and required model files exist.
  Result: passed.
- Criterion: Run SAM3D runtime smoke in `demo_3_max` on RTX 5090.
  Result: passed.
- Criterion: Preserve Python/Torch/CUDA stack.
  Result: passed.

## Skipped Checks

- Full default 25-step SAM3D demo was not run. The completed smoke intentionally
  used one stage1 and one stage2 inference step to validate the runtime path
  quickly without turning this into a quality/performance benchmark.

## Residual Risk

- `spconv-cu121` is a CUDA 12.1 wheel running beside the preserved
  torch `2.11.0+cu130` stack. It passed the local RTX 5090 smoke, but a future
  production run should still retest the exact workload and inference-step
  count it plans to use.
- `kaolin-compat` remains a narrow shim for the notebook imports SAM3D uses
  here, not a full Kaolin replacement.
