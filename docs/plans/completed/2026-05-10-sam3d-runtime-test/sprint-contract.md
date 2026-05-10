# Sprint Contract

## Contract ID

`sam3d-runtime-smoke`

## Goal

Download or locate SAM 3D Objects checkpoints and run a minimal official
runtime smoke in `demo_3_max`.

## Agreed Scope

Files expected to change:

- Task records under `docs/plans/completed/2026-05-10-sam3d-runtime-test/`.
- External checkpoint files under
  `/home/zhangxinjie/external/sam-3d-objects/checkpoints/`.
- Optional external runtime output under `/home/zhangxinjie/outputs/`.

Files expected to read:

- `/home/zhangxinjie/external/sam-3d-objects/README.md`
- `/home/zhangxinjie/external/sam-3d-objects/demo.py`
- Hugging Face CLI login state.

Out of scope:

- Installing official CUDA 12.1 SAM3D requirements.
- Downgrading or changing `demo_3_max` Python/Torch/CUDA.
- Modifying PhysTwin runtime code.

## Behavior To Deliver

- `demo_3_max` can find the SAM3D checkpoint config.
- A runtime smoke either produces an output artifact or reports the exact
  blocker with command evidence.
- RTX 5090/CUDA 13 settings remain intact.

## Verification Commands

```bash
conda run --no-capture-output -n demo_3_max hf auth whoami
conda run --no-capture-output -n demo_3_max python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
```

## Contract Changes

- WSL Hugging Face login was valid, so Windows-side download and copy was not
  needed.
- Generic `spconv==2.3.8` failed the real SAM3D stage2 sparse convolution path
  with a CPU-only operator. The env now uses `spconv-cu121==2.3.8` while
  keeping torch `2.11.0+cu130` and Torch CUDA `13.0`.
- The runtime smoke used one stage1 and one stage2 inference step and decoded
  only the Gaussian output. This validates the real pipeline path without
  treating the run as a reconstruction-quality benchmark.
