# QA Report

## Task ID

`2026-06-03-sam3d-demo-3-3-max-env`

## Summary

Completed environment setup for `demo_3_3_max`, cloned from
`demo_3_1_max`, with SAM 3D Objects runtime dependencies installed while
preserving the inherited Python/Torch/CUDA stack:

- Python: `3.12.13`
- torch: `2.11.0+cu130`
- Torch CUDA: `13.0`
- torchvision: `0.26.0+cu130`
- GPU probe: `NVIDIA GeForce RTX 4090`

SAM3D notebook inference imports and checkpoint-backed pipeline initialization
pass in `demo_3_3_max`.

Follow-up Demo 3.3 end-to-end route validation also passed after installing
the additional FuturePhysTwin single-view route dependencies needed beyond the
checkpoint-backed SAM3D initialization smoke.

## Commands Run

- Baseline audit:
  - `conda env list`
  - `conda run --no-capture-output -n demo_3_1_max python - <<'PY' ...`
  - Confirmed `demo_3_1_max` already had Python `3.12.13`, torch
    `2.11.0+cu130`, Torch CUDA `13.0`, and CUDA available.
  - Confirmed `demo_3_1_max` was missing SAM3D runtime packages including
    `spconv`, `cumm`, `gsplat`, `pytorch3d`, `kaolin`, `xatlas`,
    `easydict`, and `lightning`.

- Environment clone:
  - `conda create -y -n demo_3_3_max --clone demo_3_1_max`

- SAM3D source/package setup:
  - `python -m pip install -e /home/xinjie/external/sam-3d-objects --no-deps`
  - Added `/home/xinjie/external/sam-3d-objects/sam3d_objects/init.py` to
    satisfy the package import and set local defaults:
    `SPCONV_ALGO=native`, `ATTN_BACKEND=sdpa`,
    `SPARSE_ATTN_BACKEND=sdpa`.

- Runtime dependency installs:
  - Installed pure/wheel runtime packages including `easydict==1.13`,
    `lightning==2.3.3`, `gradio==5.49.0`, `plyfile==1.1.2`,
    `xatlas==0.0.9`, `pyvista`, `pymeshfix==0.17.0`,
    `igraph==0.11.8`, `seaborn==0.13.2`, `loguru==0.7.2`,
    `fvcore==0.1.5.post20221221`, `roma==1.5.1`, `rootutils==1.0.7`,
    `jsonlines==4.0.0`, `trimesh==4.5.3`, `hydra-core==1.3.2`,
    `einops-exts==0.0.4`, `ftfy==6.2.0`, `scikit-image==0.23.1`,
    `pyrender==0.1.45`, `Rtree==1.3.0`, `OpenEXR==3.3.3`,
    `deprecation==2.1.0`, `optree==0.14.1`, `astor==0.8.1`,
    `jaxtyping`, and Kaolin auxiliary dependencies.
  - Installed `MoGe` and `utils3d` from the official git commits used by
    SAM3D.
  - Installed `spconv-cu121==2.3.8` and `cumm-cu121==0.7.11` instead of
    generic CPU-only `spconv`.
  - Built and installed PyTorch3D `0.7.9` from
    `/home/xinjie/external/pytorch3d-phystwin-max` with
    `CUDA_HOME=/usr/local/cuda-13.0`, `TORCH_CUDA_ARCH_LIST=8.9`, and
    `PYTORCH3D_DISABLE_PULSAR=1`.
  - Built and installed `gsplat==1.5.3` from commit
    `2323de5905d5e90e035f792fe65bad0fedd413e7` with CUDA 13.0 and
    `sm_89`.
  - Built and installed Kaolin `0.18.0` from
    `/home/xinjie/external/kaolin-phystwin-max` with CUDA 13.0 and `sm_89`.
  - Official `point-cloud-utils==0.29.5` failed to build on Python 3.12
    because the sdist did not contain a usable `CMakeLists.txt`; installed
    latest compatible wheel `point-cloud-utils==0.34.0`.
  - For the full Demo 3.3 FuturePhysTwin route, additionally installed
    `diffusers==0.38.0`, `addict`, `yapf`, `timm`, `supervision`,
    `pycocotools`, editable
    `/home/xinjie/external/GroundingDINO-phystwin-max`, editable
    `/home/xinjie/external/nvdiffrast`, and editable
    `/home/xinjie/external/mip-splatting-phystwin-max/submodules/diff-gaussian-rasterization`.
    These were installed with `--no-deps`/`--no-build-isolation` where needed
    to avoid replacing the torch/CUDA stack.

- Final package/core probe:
  - `conda run --no-capture-output -n demo_3_3_max python - <<'PY' ...`
  - Output included:
    - `python 3.12.13`
    - `torch 2.11.0+cu130`
    - `torch_cuda 13.0`
    - `torchvision 0.26.0+cu130`
    - `cuda_available True`
    - `sam3d-objects 0.0.1`
    - `spconv-cu121 2.3.8`
    - `cumm-cu121 0.7.11`
    - `pytorch3d 0.7.9`
    - `gsplat 1.5.3`
    - `kaolin 0.18.0`
    - `point-cloud-utils 0.34.0`
    - `MoGe 1.0.0`
    - `utils3d 0.0.2`

- CUDA smoke:
  - Ran a GPU `spconv.SubMConv3d` sparse convolution.
  - Ran PyTorch3D `knn_points` on CUDA tensors.
  - Ran Kaolin mesh ops and `point_to_mesh_distance` on CUDA tensors.
  - Imported `gsplat.csrc` and `gsplat.rendering`.
  - Output included:
    - `spconv (4, 3) cuda:0`
    - `pytorch3d (1, 8, 1) cuda:0`
    - `kaolin 0.18.0 (1, 4, 3, 3) ...`
    - `gsplat 1.5.3 gsplat.csrc gsplat.rendering`

- SAM3D import and checkpoint-backed initialization:
  - Imported:
    - `sam3d_objects.pipeline.inference_utils`
    - `sam3d_objects.pipeline.inference_pipeline_pointmap`
    - `sam3d_objects.pipeline.inference_pipeline`
    - `sam3d_objects.model.io`
  - Initialized:
    - `Inference('/home/xinjie/external/sam-3d-objects/checkpoints/hf/pipeline.yaml', compile=False)`
  - Output included:
    - `[SPARSE] Backend: spconv, Attention: sdpa`
    - `Loading model weights completed!`
    - `inference init ok InferencePipelinePointMap 12.761`

- Full Demo 3.3 single-environment route:
  - Launched live Demo 3.3 with:
    `conda run --no-capture-output -n demo_3_3_max python
    demo_v3_3/realtime_three_view_litetracker_ffs_dual4090.py ...`
  - Profile:
    `/home/xinjie/proj-QQTT-v2/docs/generated/demo33_single_env_fresh_demo_3_3_max_20260603_45s_profile_shared_runtime.json`
  - Detached completion JSON:
    `/home/xinjie/proj-QQTT-v2/docs/generated/demo33_single_env_fresh_demo_3_3_max_20260603_45s_profile_shape_prior_completion.json`
  - Result: `shape_prior_status = ready`.
  - All route stages used
    `/home/xinjie/miniforge3/envs/demo_3_3_max/bin/python`:
    `image_upscale`, `segment_util_image`, `shape_prior_sam3d`, `align`, and
    `data_process_sample`.
  - Completion timings:
    - `image_upscale`: `17741.6 ms`
    - `segment_util_image`: `7851.0 ms`
    - `shape_prior_sam3d`: `38691.6 ms`
    - `align`: `25166.9 ms`
    - `data_process_sample`: `9020.1 ms`
    - total async elapsed: `98482.7 ms`
  - Final shape prior output:
    - `object_points0 = 9354`
    - `surface_points = 455`
    - `interior_points = 56`
    - coordinate validation: `valid`, reason `ok`
    - final z extent: `0.14132475852966309 m`
  - Artifacts:
    - `final_data.pkl`
    - `track_process_data.pkl`
    - `shape/object.glb`
    - `shape/matching/final_mesh.glb`

- Metadata check:
  - `conda run --no-capture-output -n demo_3_3_max python -m pip check`
  - Expected non-zero result remains because the installed editable SAM3D
    metadata declares the official broad requirement set, including training,
    dev, and cu121-specific packages that were intentionally not installed
    to preserve Python 3.12, torch `2.11.0+cu130`, and Torch CUDA `13.0`.

## Findings

- `demo_3_1_max` did not have SAM 3D Objects runtime dependencies.
- `demo_3_3_max` was created successfully as a clone.
- Python, torch, torchvision, and Torch CUDA were not downgraded.
- Full Kaolin and PyTorch3D were built locally against the target env rather
  than reusing ABI-uncertain wheels from a different env.
- The SAM3D runtime path is usable with `spconv` and `sdpa`; no `xformers` or
  `flash-attn` install was needed.
- The complete Demo 3.3 FuturePhysTwin route now runs in the single
  `demo_3_3_max` environment; no `phystwin-max` launcher is needed for the
  shape-prior route.
- `diff_gaussian_rasterization` must be the mip-splatting-compatible variant
  with `kernel_size` and `subpixel_offset` fields. The plain Gaussian
  Splatting submodule variant is not compatible with SAM3D texture baking.
- `pip check` is not clean due package metadata conflicts and unrelated
  inherited packages, but the runtime acceptance checks pass.

## Residual Notes

- A live run reported one RealSense frame timeout/restart, but the process
  exited successfully and the detached shape-prior completion reached
  `ready`.
- SAM3D logs `[ATTENTION] Using backend: sdpa`; this environment intentionally
  avoids installing `xformers`/`flash-attn` because those routes risk changing
  the preserved Python/Torch/CUDA stack.
