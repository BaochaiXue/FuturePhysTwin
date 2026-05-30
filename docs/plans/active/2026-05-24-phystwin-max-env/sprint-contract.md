# Sprint Contract

## Contract ID

`phystwin-max-env`

## Goal

Build and verify the `phystwin-max` conda environment on the local RTX 4090
machine with Python 3.12 and Torch CUDA 13.0.

## Agreed Scope

Files expected to change:

- `docs/plans/active/2026-05-24-phystwin-max-env/task-brief.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/sprint-contract.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/qa-report.md`
- `docs/plans/active/2026-05-24-phystwin-max-env/handoff.md`

Files expected to read:

- `README.md`
- `env_install/env_install.sh`
- `env_install/5090_env_install.sh`
- `env_install/data_process.sh`
- `docs/plans/completed/2026-05-10-sam3d-runtime-test/handoff.md`

Out of scope:

- Source changes to PhysTwin runtime/training code.
- Dataset downloads or long training/data-processing jobs.

## Behavior To Deliver

- Behavior: `conda activate phystwin-max` selects Python 3.12.
- Behavior: `pip3 install torch torchvision` uses the CUDA 13.0 PyTorch wheel
  index and installs Torch 2.12 with CUDA available.
- Behavior: full PhysTwin/data-processing dependency installation is attempted
  without downgrading Python, Torch, or CUDA below the requested versions.
- Behavior: incompatible dependencies are patched from source where practical or
  left as documented blockers with exact command evidence.
- Behavior: full Kaolin is installed from source for the requested Torch/CUDA
  stack, replacing the temporary `check_tensor` shim.

## Acceptance Criteria

- Must: preserve Python 3.12.
- Must: preserve Torch 2.12 and Torch CUDA 13.0.
- Must: verify GPU availability on RTX 4090.
- Must not: install a conda PyTorch build or older CUDA wheel to satisfy
  downstream dependencies.
- Must: use source patches rather than downgrading Python, Torch, or Torch CUDA
  to make Kaolin install.

## Evaluator Probes

The evaluator should explicitly check:

- Probe: `python --version` inside `phystwin-max`.
- Probe: `torch.__version__`, `torch.version.cuda`, and CUDA tensor creation.
- Probe: import smokes for core runtime and data-processing packages.
- Probe: documented blockers match actual install/build failures.
- Probe: full Kaolin mesh CUDA op runs.

## Verification Commands

```bash
python scripts/check_harness_docs.py
conda run --no-capture-output -n phystwin-max python --version
conda run --no-capture-output -n phystwin-max python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print((torch.rand(2, device='cuda') + 1).cpu().tolist())"
```

## Contract Changes

- 2026-05-24: Initial contract from user request.
- 2026-05-24: User requested forced full Kaolin install with source patches if
  needed.
