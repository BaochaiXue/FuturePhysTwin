# Sprint Contract

## Contract ID

`sam3d-demo3-max-bootstrap`

## Goal

Create Demo 3 Max from Demo 2 Max and make it compatible with SAM 3D Objects
without changing the existing Python/Torch/CUDA versions.

## Agreed Scope

Files expected to change:

- Environment clone or directory outside tracked PhysTwin source if Demo 2 Max
  is an environment.
- Task artifacts under `docs/plans/active/2026-05-10-sam3d-demo3-max/`.
- Optional helper docs/scripts if needed to reproduce the install.

Files expected to read:

- Demo 2 Max environment metadata or directory.
- SAM 3D Objects repository metadata.
- Local Python/Torch/CUDA version metadata.

Out of scope:

- Downgrading or upgrading Python, Torch, or CUDA.
- Force reinstalling global packages.
- Committing external SAM 3D Objects source into this repository unless a
  deliberate vendor strategy is documented.

## Behavior To Deliver

- Demo 3 Max is isolated from Demo 2 Max.
- SAM 3D Objects can be imported or inspected from Demo 3 Max.
- Installation choices are reproducible from recorded commands.

## Acceptance Criteria

- Must identify the exact Demo 2 Max source.
- Must record before/after core versions.
- Must avoid dependency commands that replace Torch/CUDA unless explicitly
  blocked and documented.
- Must leave the PhysTwin git worktree clean except task docs or intentional
  helper files.

## Evaluator Probes

The evaluator should explicitly check:

- `python --version` in Demo 3 Max.
- Torch version and `torch.version.cuda` in Demo 3 Max.
- SAM 3D Objects import or package metadata in Demo 3 Max.
- Git status for unintended source changes.

## Verification Commands

```bash
python scripts/check_harness_docs.py
git status --short --branch
conda run --no-capture-output -n demo_3_max python - <<'PY'
import torch, torchvision
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(torchvision.__version__)
PY
conda run --no-capture-output -n demo_3_max python - <<'PY'
import importlib
for name in [
    "sam3d_objects.pipeline.inference_utils",
    "sam3d_objects.pipeline.inference_pipeline_pointmap",
    "inference",
]:
    print(name, importlib.import_module(name).__file__)
PY
```

## Contract Changes

- Official SAM3D CUDA 12.1 package pins were not installed because this
  machine is RTX 5090 and the target env must keep PyTorch cu130/CUDA 13.
- PyTorch3D was installed from the SAM3D-pinned commit with a local
  `PYTORCH3D_DISABLE_PULSAR=1` patch because Pulsar failed to link under the
  CUDA 13/SM120 build.
- NVIDIA Kaolin source build was attempted and failed against PyTorch 2.11
  C++ dispatch APIs; `kaolin-compat` now provides the limited imports required
  by SAM3D in this env.
