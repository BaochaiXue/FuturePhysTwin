# Sprint Contract

## Contract ID

`sam3d-demo-3-3-max-install`

## Goal

Install SAM 3D Objects dependencies into `demo_3_3_max` cloned from
`demo_3_1_max`, preserving Python 3.12 and the existing PyTorch/CUDA stack.

## Agreed Scope

Expected changed areas:

- Conda env: `/home/xinjie/miniforge3/envs/demo_3_3_max`
- External dependency checkouts under `/home/xinjie/external/` as needed
- Harness records under this plan directory

Must not:

- Downgrade Python, torch, torchvision, or Torch CUDA.
- Modify unrelated dirty repository files.

## Verification Commands

```bash
conda run --no-capture-output -n demo_3_3_max python --version
conda run --no-capture-output -n demo_3_3_max python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
conda run --no-capture-output -n demo_3_3_max python - <<'PY'
import sys
sys.path.insert(0, "/home/xinjie/external/sam-3d-objects/notebook")
from inference import Inference
print(Inference)
PY
```

## Result

Completed on 2026-06-03. See `qa-report.md` for command evidence.
