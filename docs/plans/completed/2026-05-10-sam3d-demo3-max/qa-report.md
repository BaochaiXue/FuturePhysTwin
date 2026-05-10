# QA Report

## Task ID

`2026-05-10-sam3d-demo3-max`

## Summary

Passed for environment bootstrap and import validation. `demo_3_max` is cloned
from `demo_2_max`, keeps the requested Python/Torch/CUDA stack, and imports the
SAM 3D Objects pointmap inference path on the local RTX 5090.

## Environment

- Date: 2026-05-10
- Machine: local workspace
- Conda/env: `/home/zhangxinjie/miniconda3/envs/demo_3_max`
- Python: 3.12.13
- Torch: 2.11.0+cu130
- Torch CUDA: 13.0
- torchvision: 0.26.0+cu130
- System CUDA toolkit: `/usr/local/cuda`, nvcc 13.2.78
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- CUDA arch target: `TORCH_CUDA_ARCH_LIST=12.0`

## Commands Run

```bash
conda create -y -n demo_3_max --clone demo_2_max
git clone https://github.com/facebookresearch/sam-3d-objects.git /home/zhangxinjie/external/sam-3d-objects
conda run --no-capture-output -n demo_3_max python -m pip install -e /home/zhangxinjie/external/sam-3d-objects --no-deps
conda run --no-capture-output -n demo_3_max env PYTORCH3D_DISABLE_PULSAR=1 FORCE_CUDA=1 MAX_JOBS=8 python -m pip install --no-deps --no-build-isolation /home/zhangxinjie/external/pytorch3d
conda run --no-capture-output -n demo_3_max env FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 python -m pip install -v --no-deps --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7'
conda run --no-capture-output -n demo_3_max python -m pip install --upgrade-strategy only-if-needed easydict==1.13 lightning==2.3.3 gradio==5.49.0 plyfile==1.1.2 spconv==2.3.8 xatlas==0.0.9 pyvista pymeshfix==0.17.0 igraph==0.11.8
conda run --no-capture-output -n demo_3_max python -m pip install --no-deps 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900'
conda run --no-capture-output -n demo_3_max python -m pip install --no-deps -e /home/zhangxinjie/external/MoGe
conda run --no-capture-output -n demo_3_max python -m pip install --no-deps -e /home/zhangxinjie/external/kaolin-compat
```

Result:

- `sam3d_objects.pipeline.inference_utils` import passed.
- `sam3d_objects.pipeline.inference_pipeline_pointmap` import passed.
- notebook `inference` import passed.
- PyTorch stayed at `2.11.0+cu130`; Torch CUDA stayed at `13.0`.
- `gsplat` built from source with `-gencode=arch=compute_120,code=sm_120`.
- Patched PyTorch3D imports passed with `PYTORCH3D_DISABLE_PULSAR=1`.
- `spconv` CUDA smoke passed on RTX 5090 with a small `SubMConv3d`.

## Source Revisions

- SAM 3D Objects: `81a82373a3a7f4cbb00bd5b32aaf6b4d0f659ddd`
- PyTorch3D: `75ebeeaea0908c5527e7b1e305fbc7681382db47`
- MoGe: `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`
- Kaolin source attempt: `ba9824e394b074099fbae7d5218e68c6362e9ecf` (`v0.17.0`)

## Findings

- Severity: medium
- Issue: SAM3D's official environment pins CUDA 12.1-era packages.
- Evidence: official requirements include `cuda-python==12.1.0`,
  `torchaudio==2.5.1+cu121`, `xformers==0.0.28.post3`,
  `spconv-cu121==2.3.8`, and Kaolin wheels for `torch-2.5.1_cu121`.
- Status: avoided to preserve the RTX 5090 cu130 stack.

- Severity: medium
- Issue: PyTorch3D Pulsar did not link under the CUDA 13/SM120 local build.
- Evidence: Pulsar linker errors during the first PyTorch3D build.
- Status: fixed by a local PyTorch3D patch disabling Pulsar C++ and Python
  imports; SAM3D's needed renderer imports pass.

- Severity: medium
- Issue: NVIDIA Kaolin 0.17.0 source does not compile against PyTorch 2.11.
- Evidence: CUDA compile failed on old `at::DeprecatedTypeProperties` dispatch
  conversions in `kaolin/csrc/ops/spc/*_cuda.cu`.
- Status: local `kaolin-compat` package provides `kaolin.utils.testing` and the
  notebook import names used by SAM3D. It is not a full Kaolin replacement.

## Acceptance Criteria Results

- Criterion: Demo 2 Max source identified.
  Result: passed. Source env was `demo_2_max`.
- Criterion: Demo 3 Max isolated from Demo 2 Max.
  Result: passed. New env is `demo_3_max`.
- Criterion: Python/Torch/CUDA preserved.
  Result: passed.
- Criterion: SAM3D import passes.
  Result: passed.
- Criterion: RTX 5090 target honored.
  Result: passed. CUDA extension builds used SM120.

## Skipped Checks

- Check: full checkpoint-backed SAM3D inference.
- Reason: task was environment compatibility; checkpoint/data access was not
  requested or required for import validation.
- Next command to run: instantiate the configured inference class after the
  SAM3D checkpoints are available locally.

## Residual Risk

- Risk: `spconv==2.3.8` imports and a small CUDA sparse convolution passes,
  but this is the generic package rather than SAM3D's official `spconv-cu121`
  wheel. Test the full sparse model path before production use.
- Risk: `kaolin-compat` is intentionally narrow. If future code uses more
  Kaolin APIs than the current SAM3D imports, either extend the shim or port
  Kaolin 0.17.0 to PyTorch 2.11.
