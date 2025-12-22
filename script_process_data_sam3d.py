"""
Process raw multi-view captures into assets required by downstream stages.

Inputs
------
- ``data_config.csv`` describing cases and categories.
- Raw data in ``data/different_types/<case>/`` (RGB, depth, masks, calibrate.pkl, metadata.json, etc.).

Outputs
-------
- Processed artefacts written back to ``data/different_types/<case>/``:
  * Segmented masks, tracked trajectories, point clouds, optional shape priors.
  * ``final_data.pkl`` bundles and ``split.json`` train/test metadata.
  * Timing logs in ``timer.log``.
"""

import os
import csv
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Process all cases or a single case with the SAM3D pipeline."
)
parser.add_argument(
    "--case",
    type=str,
    default=None,
    help="If provided, only process this case name (matches first column in data_config.csv).",
)
args = parser.parse_args()

base_path = "./data/different_types"

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if args.case is not None and case_name != args.case:
            continue

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        if shape_prior.lower() == "true":
            os.system(
                f"conda run -n ps python process_data_sam3d.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
            )
        else:
            os.system(
                f"conda run -n ps python process_data_sam3d.py --base_path {base_path} --case_name {case_name} --category {category}"
            )

print("Data processing complete.")
