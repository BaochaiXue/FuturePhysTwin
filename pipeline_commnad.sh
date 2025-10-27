#!/bin/bash

# --- Preparation ---
# Create log directory
mkdir -p logs

# --- Data Processing ---
python script_process_data.py \
  > >(tee logs/script_process_data.out) \
  2> >(tee logs/script_process_data.err >&2)

python dynamic_export_gs_data.py \
  > >(tee logs/dynamic_export_gs_data.out) \
  2> >(tee logs/dynamic_export_gs_data.err >&2)

python export_video_human_mask.py \
  > >(tee logs/export_video_human_mask.out) \
  2> >(tee logs/export_video_human_mask.err >&2)

# --- Optimization Steps ---
# Zero-order Optimization
python script_optimize.py \
  > >(tee logs/script_optimize.out) \
  2> >(tee logs/script_optimize.err >&2)

# First-order Optimization
python script_train.py \
  > >(tee logs/script_train.out) \
  2> >(tee logs/script_train.err >&2)

# Model Inference
python script_inference.py \
  > >(tee logs/script_inference.out) \
  2> >(tee logs/script_inference.err >&2)

# --- Gaussian Training ---
python dynamic_fast_gs.py \
  > >(tee logs/dynamic_fast_gs.out) \
  2> >(tee logs/dynamic_fast_gs.err >&2)
# --- Simulation & Rendering Evaluation ---
bash gs_run_simulate.sh \
  > >(tee logs/gs_run_simulate.out) \
  2> >(tee logs/gs_run_simulate.err >&2)

python export_render_eval_data.py \
  > >(tee logs/export_render_eval_data.out) \
  2> >(tee logs/export_render_eval_data.err >&2)

bash evaluate.sh \
  > >(tee logs/evaluate.out) \
  2> >(tee logs/evaluate.err >&2)

bash gs_run_simulate_white.sh \
  > >(tee logs/gs_run_simulate_white.out) \
  2> >(tee logs/gs_run_simulate_white.err >&2)

python visualize_render_results.py \
  > >(tee logs/visualize_render_results.out) \
  2> >(tee logs/visualize_render_results.err >&2)
