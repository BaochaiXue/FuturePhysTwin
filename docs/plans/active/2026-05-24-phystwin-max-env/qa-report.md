# QA Report

## Task ID

`2026-05-24-phystwin-max-env`

## Summary

Passed for the requested `phystwin-max` environment on the local RTX 4090
machine. The environment preserves Python 3.12, torch `2.12.0+cu130`,
torchvision `0.27.0+cu130`, and Torch CUDA `13.0`. Full Kaolin was installed
from source after a small metadata patch for Torch 2.12/Python 3.12.

## Environment

- Date: 2026-05-24
- Machine: local WSL/Linux host with 2x NVIDIA GeForce RTX 4090
- Driver: `580.159.03`
- Python: `3.12.13`
- Conda env: `/home/xinjie/miniforge3/envs/phystwin-max`
- Torch: `2.12.0+cu130`
- Torchvision: `0.27.0+cu130`
- Torch CUDA: `13.0`
- CUDA toolkit for extension builds: `/usr/local/cuda-13.0`
- CUDA arch target: `TORCH_CUDA_ARCH_LIST=8.9`
- Kaolin source: `/home/xinjie/external/kaolin-phystwin-max`
- Kaolin commit: `35041a8cb12bdde6da0d0dd6a4e6e9effd5efe70`
- TRELLIS source: `data_process/TRELLIS`
- TRELLIS commit: `442aa1e1afb9014e80681d3bf604e8d728a86ee7`

## Commands Run

Environment and PyTorch:

```bash
conda run --no-capture-output -n phystwin-max python --version
conda run --no-capture-output -n phystwin-max python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print((torch.rand(2, device='cuda') + 1).cpu().tolist())
PY
```

Result: passed.

```text
Python 3.12.13
2.12.0+cu130 0.27.0+cu130 13.0 NVIDIA GeForce RTX 4090
```

Full Kaolin source build:

```bash
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git \
  /home/xinjie/external/kaolin-phystwin-max
conda run --no-capture-output -n phystwin-max bash -lc \
  'export CUDA_HOME=/usr/local/cuda-13.0; \
   export CUDACXX=/usr/local/cuda-13.0/bin/nvcc; \
   export PATH=/usr/local/cuda-13.0/bin:$PATH; \
   export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}; \
   export FORCE_CUDA=1; export TORCH_CUDA_ARCH_LIST=8.9; export MAX_JOBS=8; \
   python -m pip install -v --no-build-isolation --no-deps \
     /home/xinjie/external/kaolin-phystwin-max'
```

Result: passed. Built and installed `kaolin-0.18.0-cp312-cp312-linux_x86_64.whl`.

Full Kaolin CUDA operation smoke:

```bash
conda run --no-capture-output -n phystwin-max python - <<'PY'
import kaolin, torch
from kaolin.utils.testing import check_tensor
from kaolin.ops.mesh import index_vertices_by_faces, sample_points, check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
vertices = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], device='cuda')
faces = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long, device='cuda')
face_vertices = index_vertices_by_faces(vertices, faces)
points = torch.tensor([[[0.2, 0.2, 0.2], [2., 2., 2.]]], device='cuda')
dist = point_to_mesh_distance(points, face_vertices)[0]
sign = check_sign(vertices, faces, points)
samples = sample_points(vertices, faces, 4)[0]
torch.cuda.synchronize()
print('kaolin full cuda ok', kaolin.__version__, check_tensor(vertices, (1, 4, 3), throw=False), tuple(face_vertices.shape), dist.detach().cpu().tolist(), sign.detach().cpu().tolist(), tuple(samples.shape))
PY
```

Result: passed.

```text
kaolin full cuda ok 0.18.0 True (1, 4, 3, 3) [[0.04000000283122063, 8.333332061767578]] [[True, False]] (1, 4, 3)
```

Sparse backend smoke:

```bash
conda run --no-capture-output -n phystwin-max python - <<'PY'
import torch
import spconv.pytorch as spconv
features = torch.randn(4, 2, device='cuda')
indices = torch.tensor([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=torch.int32, device='cuda')
x = spconv.SparseConvTensor(features, indices, spatial_shape=[2, 2, 2], batch_size=1)
conv = spconv.SubMConv3d(2, 3, kernel_size=3, padding=1, bias=False).cuda()
y = conv(x)
torch.cuda.synchronize()
print('spconv cuda ok', tuple(y.features.shape), y.features.device)
PY
```

Result: passed after replacing CPU-only `spconv==2.3.8` with
`spconv-cu121==2.3.8` and `cumm-cu121==0.7.11`.

```text
spconv cuda ok (4, 3) cuda:0
```

TRELLIS imports:

```bash
conda run --no-capture-output -n phystwin-max bash -lc \
  'export PYTHONPATH=/home/xinjie/FuturePhysTwin/data_process/TRELLIS:/home/xinjie/FuturePhysTwin/data_process:$PYTHONPATH; \
   python - <<'"'"'PY'"'"'
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.representations.mesh.flexicubes.flexicubes import FlexiCubes
from trellis.renderers.gaussian_render import GaussianRenderer
print("trellis imports ok")
PY'
```

Result: passed with `[SPARSE] Backend: spconv, Attention: flash_attn`.

PhysTwin imports:

```bash
conda run --no-capture-output -n phystwin-max python - <<'PY'
import gs_train, gs_render, dynamic_fast_canonical, dynamic_fast_color
from gaussian_splatting.scene.gaussian_model import GaussianModel
print('phystwin imports ok')
PY
```

Result: passed. `SparseGaussianAdam` is not available in the installed
TRELLIS-compatible `diff_gaussian_rasterization`; PhysTwin logs its built-in
fallback to standard Adam.

Package consistency:

```bash
conda run --no-capture-output -n phystwin-max python -m pip check
```

Result: passed with `No broken requirements found.`

## Data Cases

- Case: no dataset was required for this environment smoke.
- Input path: not applicable.
- Output path: not applicable.

## Findings

- Severity: medium
- File: `/home/xinjie/external/kaolin-phystwin-max/setup.py`
- Issue: Kaolin source metadata rejected Torch 2.12 and used an overly narrow
  Python requirement expression.
- Evidence: source had `TORCH_MAX_VER = '2.10.0'` and
  `python_requires='~=3.7'`.
- Status: fixed externally by setting `TORCH_MAX_VER = '2.12.0'` and
  `python_requires='>=3.7,<4'`; full Kaolin built and passed CUDA mesh ops.

- Severity: medium
- File: Python package environment
- Issue: generic `spconv==2.3.8` imported but failed real CUDA sparse conv with
  `not implemented for CPU ONLY build`.
- Evidence: `SubMConv3d(...).cuda()` failed in
  `SpconvOps_get_indice_pairs`.
- Status: fixed by installing `spconv-cu121==2.3.8` and
  `cumm-cu121==0.7.11`; the same CUDA sparse conv smoke passed.

- Severity: low
- File: `gaussian_splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
- Issue: CUDA 13 build needed an explicit integer-width header.
- Evidence: `diff_gaussian_rasterization` build needed `uint32_t`/related
  types.
- Status: fixed by adding `#include <cstdint>`.

## Acceptance Criteria Results

- Criterion: `phystwin-max` exists with Python 3.12.
  Result: passed.
- Criterion: `torch` and `torchvision` are installed from CUDA 13.0 wheels with
  Torch 2.12 and GPU access.
  Result: passed.
- Criterion: core PhysTwin and data-processing imports succeed.
  Result: passed.
- Criterion: full Kaolin, not a shim, imports and runs a CUDA mesh op.
  Result: passed.
- Criterion: real sparse backend op runs on CUDA.
  Result: passed with `spconv-cu121` beside Torch CUDA 13.0.
- Criterion: package requirements remain consistent.
  Result: passed with `pip check`.

## Skipped Checks

- Check: full data preprocessing job with real videos/images and checkpoints.
- Reason: the task was environment creation and dependency validation, not a
  long data-processing run.
- Next command to run: invoke the intended preprocessing script with its real
  input data and checkpoints after selecting a concrete dataset case.

## Residual Risk

- Risk: `spconv-cu121` is a CUDA 12.1 wheel used beside Torch CUDA 13.0. It
  passed a real local CUDA sparse convolution smoke, but a full TRELLIS model
  run should still validate the exact workload.
- Owner or follow-up: next runtime/data-preprocessing task.

- Risk: `SparseGaussianAdam` is unavailable in the installed
  TRELLIS-compatible `diff_gaussian_rasterization`; PhysTwin falls back to
  standard Adam, which its scripts already support.
- Owner or follow-up: only needed if a future run explicitly requires
  `optimizer_type=sparse_adam`.
