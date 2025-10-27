"""
Launch CMA-ES optimisation for each case using the processed datasets.

Inputs
------
- ``data/different_types/<case>/final_data.pkl`` along with calibration and metadata (calibrate.pkl, metadata.json).
- Train/test split metadata in ``data/different_types/<case>/split.json``.

Outputs
-------
- Optimisation logs and ``optimal_params.pkl`` stored under ``experiments_optimization/<case>/``.
"""

import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    os.system(
        f"python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )
