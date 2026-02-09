#!/bin/bash

set -euo pipefail

# --- Preparation ---
# Create log directory
mkdir -p logs

# --- Data Processing ---
python script_process_data.py \
  > >(tee logs/script_process_data.out) \
  2> >(tee logs/script_process_data.err >&2)
  
python export_video_human_mask.py \
  > >(tee logs/export_video_human_mask.out) \
  2> >(tee logs/export_video_human_mask.err >&2)

python dynamic_export_gs_data.py \
  > >(tee logs/dynamic_export_gs_data.out) \
  2> >(tee logs/dynamic_export_gs_data.err >&2)



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
python final_eval.py \
  > >(tee logs/final_eval.out) \
  2> >(tee logs/final_eval.err >&2)
