# Handoff

## Task ID

`2026-05-24-phystwin-max-env`

## Current State

`phystwin-max` is created and verified on the local RTX 4090 machine with
Python 3.12.13, torch `2.12.0+cu130`, torchvision `0.27.0+cu130`, and Torch
CUDA `13.0`. Full Kaolin `0.18.0` is installed from patched source, not from
the earlier temporary `check_tensor` shim.

The environment includes the PhysTwin runtime surface, TRELLIS data-processing
surface, Grounded-SAM-2/GroundingDINO, PyTorch3D, flash-attn, xformers,
nvdiffrast, diffoctreerast, MipGaussian rasterization, and a CUDA-capable
`spconv-cu121` sparse backend.

## Changed Files

- `gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
- `docs/plans/active/2026-05-24-phystwin-max-env/task-brief.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/sprint-contract.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/qa-report.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/handoff.md`

External source edits:

- `/home/xinjie/external/kaolin-phystwin-max/setup.py`
  - raised `TORCH_MAX_VER` to `2.12.0`
  - changed `python_requires` to `>=3.7,<4`
- `/home/xinjie/external/GroundingDINO-phystwin-max/.../ms_deform_attn_cuda.cu`
  - replaced deprecated Torch dispatch `value.type()` usage with
    `value.scalar_type()`

## Verification Evidence

- Command: `conda run --no-capture-output -n phystwin-max python --version`
  Result: `Python 3.12.13`
- Command: Torch CUDA tensor smoke.
  Result: `2.12.0+cu130 0.27.0+cu130 13.0 NVIDIA GeForce RTX 4090`
- Command: full Kaolin CUDA mesh smoke.
  Result: `kaolin full cuda ok 0.18.0 ...`
- Command: spconv CUDA sparse convolution smoke.
  Result: `spconv cuda ok (4, 3) cuda:0`
- Command: TRELLIS import smoke.
  Result: `trellis imports ok`
- Command: PhysTwin import smoke.
  Result: `phystwin imports ok`
- Command: `conda run --no-capture-output -n phystwin-max python -m pip check`
  Result: `No broken requirements found.`

## Next Agent Should Read

1. `docs/plans/active/2026-05-24-phystwin-max-env/task-brief.md`
2. `docs/plans/active/2026-05-24-phystwin-max-env/sprint-contract.md`
3. `docs/plans/active/2026-05-24-phystwin-max-env/qa-report.md`
4. `/home/xinjie/external/kaolin-phystwin-max/setup.py`
5. `data_process/TRELLIS/setup.sh`
6. `gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`

## Open Follow-Ups

- Follow-up: run the exact real preprocessing workload after selecting the
  input dataset and checkpoint set.
- Follow-up: if a future training run requires `optimizer_type=sparse_adam`,
  add or port `SparseGaussianAdam`; current scripts fall back to standard Adam.

## Notes

The default installed `diff_gaussian_rasterization` package is the
MipGaussian-compatible variant so TRELLIS Gaussian rendering can accept
`kernel_size` and `subpixel_offset`. PhysTwin's primary renderer defaults to
`gsplat`, and the import smoke passes with the built-in standard Adam fallback.
