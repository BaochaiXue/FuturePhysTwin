"""
Generate per-camera human-mask videos used for render overlays.

Inputs
------
- RGB sequences and supporting data in ``data/different_types/<case>/``.
- Depth folders and mask metadata (``mask/mask_info_*.json``) per case.

Outputs
-------
- Foreground human masks saved to ``data/different_types_human_mask/<case>/mask/<camera>/``.
"""

import os
import glob

base_path = "./data/different_types"
output_path = "./data/different_types_human_mask"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    # Process to get the whole human mask for the video

    TEXT_PROMPT = "human"
    camera_num = 3
    assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

    for camera_idx in range(camera_num):
        print(f"Processing {case_name} camera {camera_idx}")
        cmd = (
            f"python ./data_process/segment_util_video.py "
            f"--output_path {output_path}/{case_name} "
            f"--base_path {base_path} "
            f"--case_name {case_name} "
            f"--TEXT_PROMPT {TEXT_PROMPT} "
            f"--camera_idx {camera_idx}"
        )
        mask_info_path = f"{base_path}/{case_name}/mask/mask_info_{camera_idx}.json"
        mask_root_path = f"{base_path}/{case_name}/mask/{camera_idx}"
        if os.path.isfile(mask_info_path) and os.path.isdir(mask_root_path):
            cmd += (
                f" --exclude_mask_info {mask_info_path}"
                f" --exclude_mask_root {mask_root_path}"
            )
        os.system(cmd)
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
