# Context Map

This map helps agents find the minimum context needed for PhysTwin work.

## Root Entrypoints

- `AGENTS.md`: repository and subtree instructions.
- `HARNESS.md`: harness engineering entry point.
- `README.md`: official PhysTwin setup, data, training, evaluation, and demo
  instructions.
- `implement_plan.md`: existing working notes for a color-stage debug
  visualization task.

## Main Domains

- `data_process/`: original video processing, segmentation, tracking, point
  cloud, alignment, and shape prior utilities.
- `data_process_sam3d/`: SAM3D-flavored data processing path.
- `gaussian_splatting/`: Gaussian rendering, scene, skinning, and rasterization
  integration.
- `qqtt/`: simulation/model engine package used by the project.
- Root scripts: orchestration entrypoints for processing, optimization,
  training, inference, evaluation, rendering, and visualization.
- `configs/`: simulation and experiment configuration.
- `docker_scripts/` and `env_install/`: environment setup.
- `assets/`: tracked README/demo assets.

## Generated or Heavy Data

The following are normally generated, downloaded, or experiment-local and should
not become harness source-of-truth files:

- `data/`
- `experiments*/`
- `experiments_optimization*/`
- `gaussian_output*/`
- `wandb/`
- `logs/`
- model checkpoints and local test outputs ignored by `.gitignore`

When a task depends on generated data, record the expected path, case name, and
command used to create or fetch it in the task brief.

## Common Workflows

- Data processing: start from `script_process_data.py`, `process_data.py`, and
  `data_process/` or `data_process_sam3d/`.
- First-frame Gaussian data: start from `export_gaussian_data.py`.
- Static Gaussian training: start from `gs_run.sh` or `gs_train.py`.
- Zero-order optimization: start from `script_optimize.py` and
  `optimize_cma.py`.
- First-order optimization: start from `script_train.py` and `train_warp.py`.
- Inference: start from `script_inference.py` and `inference_warp.py`.
- Dynamic rendering and evaluation: start from `gs_run_simulate.py`,
  `export_render_eval_data.py`, `evaluate.py`, and `visualize_render_results.py`.
- Interactive playground: start from `interactive_playground.py`.
- Force/material visualization: start from `visualize_force.py` and
  `visualize_material.py`.

## External Runtime Notes

- SAM 3D Objects for this machine is installed in the external checkout
  `/home/zhangxinjie/external/sam-3d-objects` and tested from conda env
  `demo_3_max`.
- The SAM3D Hugging Face checkpoint directory is
  `/home/zhangxinjie/external/sam-3d-objects/checkpoints/hf`.
- WSL Hugging Face auth worked as `XinjieZhang` during the 2026-05-10 smoke;
  Windows-side fallback was not required.
- Before changing this environment, read
  `docs/plans/completed/2026-05-10-sam3d-demo3-max/qa-report.md` and
  `docs/plans/completed/2026-05-10-sam3d-runtime-test/qa-report.md`.

## Context Boundaries

Keep PhysTwin documentation upstream-facing. Bridge-specific or external system
conclusions belong outside this subtree unless a task explicitly requires that
coupling.
