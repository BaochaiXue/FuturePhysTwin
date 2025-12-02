#!/usr/bin/env bash
# Data-processing env with SAM 3D Objects as the base. Keep a clean env to avoid torch/p3d conflicts.
# Usage: conda create -n data_process python=3.10 -y && conda activate data_process && bash env_install/data_process.sh
set -euo pipefail


#############################################
# Extra deps for data_process scripts
#############################################
# Do NOT reinstall torch/pytorch3d here to avoid version conflicts.
pip install numpy tqdm pillow imageio supervision trimesh diffusers accelerate open3d matplotlib scipy
# Segmentation / grounding (GroundingDINO + SAM2) used by segment_util_image/video.py
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

