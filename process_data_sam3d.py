from __future__ import annotations

import glob
import json
import logging
import shlex
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Process one case with the SAM3D/MV-SAM3D data pipeline."
    )
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--shape_prior", action="store_true", default=False)
    parser.add_argument(
        "--controller_name",
        type=str,
        default="hand",
        help="Controller label used in mask_info files and segmentation prompts.",
    )
    parser.add_argument(
        "--pipeline_python",
        default=None,
        help=(
            "Python command for segmentation, MV-SAM3D wrapper, tracking, lifting, "
            "alignment, and final sampling subprocesses. Defaults to the current Python."
        ),
    )
    parser.add_argument(
        "--legacy_shape_prior_python",
        default=None,
        help=(
            "Python command for the legacy single-view data_process_sam3d/shape_prior.py "
            "backend. Defaults to the current Python."
        ),
    )
    parser.add_argument(
        "--shape_prior_backend",
        choices=["sam3d", "mvsam3d"],
        default="mvsam3d",
        help="Shape-prior backend. 'sam3d' keeps the legacy single-view path.",
    )
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument("--mvsam3d_python", default=None)
    parser.add_argument("--mvsam3d_run_da3", action="store_true")
    parser.add_argument("--mvsam3d_skip_da3_if_exists", action="store_true")
    parser.add_argument(
        "--mvsam3d_da3_model_path",
        default=None,
        help="Forwarded to shape_prior_mvsam3d.py as --da3_model_path.",
    )
    parser.add_argument("--mvsam3d_force", action="store_true")
    parser.add_argument("--mvsam3d_view_indices", default=None)
    parser.add_argument(
        "--mvsam3d_input_preprocess_backend",
        choices=["legacy_upscale", "raw"],
        default="legacy_upscale",
        help="Preprocess MV-SAM3D inputs. legacy_upscale mirrors the old SAM3D high-res crop path per view.",
    )
    parser.add_argument(
        "--mvsam3d_preprocess_python",
        default=None,
        help="Python command for MV-SAM3D legacy-upscale preprocessing.",
    )
    parser.add_argument("--mvsam3d_merge_da3_glb", action="store_true")
    parser.add_argument("--mvsam3d_run_pose_optimization", action="store_true")
    parser.add_argument(
        "--mvsam3d_max_faces",
        type=int,
        default=50000,
        help="Maximum faces for downstream shape/object.glb; use 0 to keep the full MV-SAM3D mesh.",
    )
    parser.add_argument(
        "--align_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
        help="Alignment backend. auto uses MV-SAM3D multi-view align for the MV-SAM3D shape-prior backend.",
    )
    parser.add_argument("--mvsam3d_align_view_indices", default=None)
    parser.add_argument("--mvsam3d_align_force_rematch", action="store_true")
    parser.add_argument("--mvsam3d_align_max_render_faces", type=int, default=50000)
    parser.add_argument("--mvsam3d_align_silhouette_iters", type=int, default=80)
    parser.add_argument("--mvsam3d_align_depth_weight", type=float, default=0.2)
    parser.add_argument("--mvsam3d_align_silhouette_weight", type=float, default=1.0)
    parser.add_argument("--mvsam3d_align_pcd_weight", type=float, default=0.5)
    parser.add_argument(
        "--shape_prior_sampling_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
    )
    return parser


args = build_arg_parser().parse_args()

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
CONTROLLER_NAME = args.controller_name
TEXT_PROMPT = f"{category}.{CONTROLLER_NAME}"
SHAPE_PRIOR = args.shape_prior
PS_PYTHON_CMD = shlex.split(args.pipeline_python) if args.pipeline_python else [sys.executable]
SHAPE_PRIOR_PYTHON_CMD = (
    shlex.split(args.legacy_shape_prior_python)
    if args.legacy_shape_prior_python
    else [sys.executable]
)

logger: logging.Logger | None = None


def setup_logger(log_file: str = "timer.log") -> None:
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


def existDir(dir_path: str | Path) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_cmd(cmd: str | list[str]) -> str:
    if isinstance(cmd, str):
        return cmd
    return shlex.join(cmd)


def run_cmd(cmd: str | list[str], retries: int = 0, retry_delay: float = 3.0) -> None:
    """Run command with optional retry on non-zero exit."""

    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(cmd, shell=isinstance(cmd, str))
        if result.returncode == 0:
            return
        if attempt > retries:
            raise RuntimeError(
                f"Command failed (code {result.returncode}): {format_cmd(cmd)}"
            )
        assert logger is not None
        logger.warning(
            "Command failed (code %s) on attempt %d/%d: %s. Retrying in %.1fs",
            result.returncode,
            attempt,
            retries,
            format_cmd(cmd),
            retry_delay,
        )
        time.sleep(retry_delay)


class Timer:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __enter__(self) -> None:
        self.start_time = time.time()
        assert logger is not None
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_time = time.time() - self.start_time
        assert logger is not None
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


def legacy_cam0_object_mask_path() -> Path:
    mask_info_path = Path(base_path) / case_name / "mask" / "mask_info_0.json"
    with mask_info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    obj_idx: int | None = None
    for key, value in data.items():
        if str(value).strip().casefold() != CONTROLLER_NAME.casefold():
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    if obj_idx is None:
        raise ValueError(f"No non-controller object detected in {mask_info_path}.")
    return Path(base_path) / case_name / "mask" / "0" / str(obj_idx) / "0.png"


def run_legacy_sam3d_shape_prior() -> None:
    mask_path = legacy_cam0_object_mask_path()
    shape_dir = Path(base_path) / case_name / "shape"
    existDir(shape_dir)

    with Timer("Image Upscale"):
        # Always regenerate the high-resolution image to avoid stale results.
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/image_upscale.py",
                "--img_path",
                str(Path(base_path) / case_name / "color" / "0" / "0.png"),
                "--mask_path",
                str(mask_path),
                "--output_path",
                str(shape_dir / "high_resolution.png"),
                "--category",
                category,
            ]
        )

    with Timer("Image Segmentation"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/segment_util_image.py",
                "--img_path",
                str(shape_dir / "high_resolution.png"),
                "--TEXT_PROMPT",
                category,
                "--output_path",
                str(shape_dir / "masked_image.png"),
            ],
            retries=2,
            retry_delay=2.0,
        )

    with Timer("Shape Prior Generation"):
        run_cmd(
            SHAPE_PRIOR_PYTHON_CMD
            + [
                "./data_process_sam3d/shape_prior.py",
                "--img_path",
                str(shape_dir / "masked_image.png"),
                "--output_dir",
                str(shape_dir),
            ]
        )


def run_mvsam3d_shape_prior() -> None:
    cmd = PS_PYTHON_CMD + [
        "./data_process_sam3d/shape_prior_mvsam3d.py",
        "--base_path",
        base_path,
        "--case_name",
        case_name,
        "--category",
        category,
        "--controller_name",
        CONTROLLER_NAME,
    ]
    if args.mvsam3d_root:
        cmd += ["--mvsam3d_root", args.mvsam3d_root]
    if args.mvsam3d_python:
        cmd += ["--mvsam3d_python", args.mvsam3d_python]
    if args.mvsam3d_view_indices:
        cmd += ["--view_indices", args.mvsam3d_view_indices]
    cmd += ["--input_preprocess_backend", args.mvsam3d_input_preprocess_backend]
    if args.mvsam3d_preprocess_python:
        cmd += ["--preprocess_python", args.mvsam3d_preprocess_python]
    if args.mvsam3d_run_da3:
        cmd.append("--run_da3")
    if args.mvsam3d_skip_da3_if_exists:
        cmd.append("--skip_da3_if_exists")
    if args.mvsam3d_da3_model_path:
        cmd += ["--da3_model_path", args.mvsam3d_da3_model_path]
    if args.mvsam3d_force:
        cmd.append("--force")
    if args.mvsam3d_merge_da3_glb:
        cmd.append("--merge_da3_glb")
    if args.mvsam3d_run_pose_optimization:
        cmd.append("--run_pose_optimization")
    cmd += ["--max_faces", str(args.mvsam3d_max_faces)]

    with Timer("Shape Prior Generation"):
        run_cmd(cmd)


def resolved_align_backend() -> str:
    if args.align_backend != "auto":
        return args.align_backend
    return "mvsam3d" if args.shape_prior_backend == "mvsam3d" else "legacy"


def resolved_sampling_backend() -> str:
    if args.shape_prior_sampling_backend != "auto":
        return args.shape_prior_sampling_backend
    return "mvsam3d" if args.shape_prior_backend == "mvsam3d" else "legacy"


def run_alignment() -> None:
    if resolved_align_backend() == "mvsam3d":
        cmd = PS_PYTHON_CMD + [
            "./data_process_sam3d/align_mvsam3d.py",
            "--base_path",
            base_path,
            "--case_name",
            case_name,
            "--controller_name",
            CONTROLLER_NAME,
            "--max_render_faces",
            str(args.mvsam3d_align_max_render_faces),
            "--silhouette_iters",
            str(args.mvsam3d_align_silhouette_iters),
            "--depth_weight",
            str(args.mvsam3d_align_depth_weight),
            "--silhouette_weight",
            str(args.mvsam3d_align_silhouette_weight),
            "--pcd_weight",
            str(args.mvsam3d_align_pcd_weight),
        ]
        if args.mvsam3d_align_view_indices:
            cmd += ["--view_indices", args.mvsam3d_align_view_indices]
        if args.mvsam3d_align_force_rematch:
            cmd.append("--force_rematch")
    else:
        cmd = PS_PYTHON_CMD + [
            "./data_process_sam3d/align.py",
            "--base_path",
            base_path,
            "--case_name",
            case_name,
            "--controller_name",
            CONTROLLER_NAME,
        ]

    with Timer("Alignment"):
        run_cmd(cmd)


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/segment.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--TEXT_PROMPT",
                TEXT_PROMPT,
            ]
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    if args.shape_prior_backend == "sam3d":
        run_legacy_sam3d_shape_prior()
    elif args.shape_prior_backend == "mvsam3d":
        run_mvsam3d_shape_prior()
    else:  # pragma: no cover - argparse choices prevent this.
        raise ValueError(f"Unknown shape prior backend: {args.shape_prior_backend}")

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/dense_track.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            ]
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/data_process_pcd.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            ]
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/data_process_mask.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--controller_name",
                CONTROLLER_NAME,
            ]
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        run_cmd(
            PS_PYTHON_CMD
            + [
                "./data_process_sam3d/data_process_track.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            ]
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Align the shape prior with partial observation
    run_alignment()

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            run_cmd(
                PS_PYTHON_CMD
                + [
                    "./data_process_sam3d/data_process_sample.py",
                    "--base_path",
                    base_path,
                    "--case_name",
                    case_name,
                    "--shape_prior",
                    "--shape_prior_sampling_backend",
                    resolved_sampling_backend(),
                ]
            )
        else:
            run_cmd(
                PS_PYTHON_CMD
                + [
                    "./data_process_sam3d/data_process_sample.py",
                    "--base_path",
                    base_path,
                    "--case_name",
                    case_name,
                ]
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {
        "frame_len": frame_len,
        "train": [0, int(frame_len * 0.7)],
        "test": [int(frame_len * 0.7), frame_len],
    }
    with open(f"{base_path}/{case_name}/split.json", "w", encoding="utf-8") as handle:
        json.dump(split, handle)
