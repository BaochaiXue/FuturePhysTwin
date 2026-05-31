# Process to get the masks of the controller and the object
from __future__ import annotations

import glob
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
parser.add_argument("--required_label_min_count", type=int, default=None)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_num = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num
print(f"Processing {case_name}")

script_dir = Path(__file__).resolve().parent
required_label_min_count = args.required_label_min_count
if required_label_min_count is None and case_name.startswith("single_"):
    required_label_min_count = 1
for camera_idx in range(camera_num):
    print(f"Processing {case_name} camera {camera_idx}")
    cmd = [
        sys.executable,
        str(script_dir / "segment_util_video.py"),
        "--base_path",
        base_path,
        "--case_name",
        case_name,
        "--TEXT_PROMPT",
        TEXT_PROMPT,
        "--camera_idx",
        str(camera_idx),
    ]
    if required_label_min_count is not None:
        cmd += ["--required_label_min_count", str(required_label_min_count)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(Path(base_path) / case_name / "tmp_data", ignore_errors=True)
