"""
Perform model inference on each trained case.

Inputs
------
- Trained checkpoints in ``experiments/<case>/``.
- Supporting optimisation results in ``experiments_optimization/<case>/``.
- Processed case data within ``data/different_types/<case>/``.

Outputs
-------
- Inference artefacts (e.g. ``inference.pkl`` trajectories) saved inside ``experiments/<case>/``.
"""

import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
