# Handoff

## Task ID

`2026-06-03-sam3d-demo-3-3-max-env`

## Current State

Completed.

`demo_3_3_max` exists and has SAM 3D Objects runtime support installed while
preserving:

- Python `3.12.13`
- torch `2.11.0+cu130`
- Torch CUDA `13.0`
- torchvision `0.26.0+cu130`

Important external state:

- SAM3D source: `/home/xinjie/external/sam-3d-objects`
- SAM3D checkpoints:
  `/home/xinjie/external/sam-3d-objects/checkpoints/hf`
- Local SAM3D patch:
  `/home/xinjie/external/sam-3d-objects/sam3d_objects/init.py`
- PyTorch3D source used:
  `/home/xinjie/external/pytorch3d-phystwin-max`
- Kaolin source used:
  `/home/xinjie/external/kaolin-phystwin-max`

Verification passed:

- Core stack remains Python 3.12 / torch `2.11.0+cu130` / Torch CUDA `13.0`.
- CUDA smokes pass for `spconv`, PyTorch3D, Kaolin, and `gsplat`.
- SAM3D pipeline modules import.
- Checkpoint-backed notebook `Inference` initializes successfully from
  `pipeline.yaml` and reports `InferencePipelinePointMap`.
- Full Demo 3.3 single-environment route reached `shape_prior_status = ready`
  from a fresh live entrypoint run. The live process and all FuturePhysTwin
  route stages used
  `/home/xinjie/miniforge3/envs/demo_3_3_max/bin/python`.

Additional route packages installed after the initial SAM3D smoke:

- `diffusers==0.38.0`
- editable `/home/xinjie/external/GroundingDINO-phystwin-max`
- editable `/home/xinjie/external/nvdiffrast`
- editable
  `/home/xinjie/external/mip-splatting-phystwin-max/submodules/diff-gaussian-rasterization`

Use the mip-splatting-compatible `diff_gaussian_rasterization`; it exposes
`GaussianRasterizationSettings.kernel_size` and `subpixel_offset`, which
SAM3D texture baking requires.

Fresh single-env Demo 3.3 evidence:

- Runtime profile:
  `/home/xinjie/proj-QQTT-v2/docs/generated/demo33_single_env_fresh_demo_3_3_max_20260603_45s_profile_shared_runtime.json`
- Completion JSON:
  `/home/xinjie/proj-QQTT-v2/docs/generated/demo33_single_env_fresh_demo_3_3_max_20260603_45s_profile_shape_prior_completion.json`
- Output case:
  `/home/xinjie/proj-QQTT-v2/result/demo32_ffs_tapnextpp/demo33_shape_prior_warmup/20260603-182928/case`

Known caveat:

- `pip check` still reports expected metadata conflicts because the editable
  SAM3D package advertises broad official requirements, including training,
  dev, and cu121-specific pins. Those pins were intentionally not installed
  because they would conflict with this task's requirement to preserve Python
  3.12 and the torch/CUDA stack. Use the QA runtime evidence rather than
  `pip check` as the acceptance signal for this environment.
- The fresh live validation logged one RealSense frame timeout/restart, but
  the Demo 3.3 process exited with code 0 and the detached shape-prior worker
  completed successfully.

## Next Agent Should Read

1. `AGENTS.md`
2. `HARNESS.md`
3. This task directory
4. `/home/xinjie/external/sam-3d-objects/sam3d_objects/init.py`
