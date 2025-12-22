import os
from argparse import ArgumentParser
import time
import logging
import json
import glob
import subprocess

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
args = parser.parse_args()

# Set the debug flags
PROCESS_SEG = True
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.hand"
CONTROLLER_NAME = "hand"
SHAPE_PRIOR = args.shape_prior
PS_PYTHON = "conda run -n ps python"
SHAPE_PRIOR_PYTHON = "conda run -n psdp python"

logger = None


def setup_logger(log_file="timer.log"):
    global logger

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run_cmd(cmd: str, retries: int = 0, retry_delay: float = 3.0) -> None:
    """Run shell command with optional retry on non-zero exit."""

    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            return
        if attempt > retries:
            raise RuntimeError(f"Command failed (code {result.returncode}): {cmd}")
        logger.warning(
            "Command failed (code %s) on attempt %d/%d: %s. Retrying in %.1fs",
            result.returncode,
            attempt,
            retries,
            cmd,
            retry_delay,
        )
        time.sleep(retry_delay)


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    # Get the mask path for the image
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"

    existDir(f"{base_path}/{case_name}/shape")
    # Get the high-resolution of the image to prepare for the shape prior generation
    with Timer("Image Upscale"):
        # Always regenerate the high-resolution image to avoid stale results.
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
        )

    # Get the masked image of the object
    with Timer("Image Segmentation"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png",
            retries=2,
            retry_delay=2.0,
        )

    with Timer("Shape Prior Generation"):
        run_cmd(
            f"{SHAPE_PRIOR_PYTHON} ./data_process_sam3d/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/dense_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Align the shape prior with partial observation
    with Timer("Alignment"):
        run_cmd(
            f"{PS_PYTHON} ./data_process_sam3d/align.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            run_cmd(
                f"{PS_PYTHON} ./data_process_sam3d/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )
        else:
            run_cmd(
                f"{PS_PYTHON} ./data_process_sam3d/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
