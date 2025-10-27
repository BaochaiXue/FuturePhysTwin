from __future__ import annotations

"""Create Gaussian-ready assets for exactly one frame of a single case.

Inputs (per case located in ``./data/different_types/<case>``):
    * ``color/<cam>/<frame>.png`` - RGB frame for each of the three cameras.
    * ``mask/mask_info_<cam>.json`` - mapping from mask indices to semantic labels.
    * ``mask/<cam>/<mask_idx>/<frame>.png`` - object mask for every camera.
    * ``depth/<cam>/<frame>.npy`` - depth map in millimetres.
    * ``mask/processed_masks.pkl`` - processed object masks used during filtering.
    * ``pcd/<frame>.npz`` - multi-view point cloud (points and colours).
    * ``calibrate.pkl`` and ``metadata.json`` - camera extrinsics and intrinsics.
    * ``shape/matching/final_mesh.glb`` - Trellis mesh prior (optional).

Outputs (written to ``./data/gaussian_data/<case>`` with the same naming as ``export_gaussian_data.py``):
    * ``{0,1,2}.png`` - RGB images.
    * ``mask_{0,1,2}.png`` - object masks.
    * ``{0,1,2}_high.png`` and ``mask_*_high.png`` - upscaled RGB images and masks (optional).
    * ``mask_human_{0,1,2}.png`` (+ ``_high`` variants when enabled) - human/occluder masks.
    * ``{0,1,2}_depth.npy`` - depth maps.
    * ``camera_meta.pkl`` - camera extrinsics and intrinsics.
    * ``observation.ply`` - fused point cloud for the selected frame.
    * ``shape_prior.glb`` (optional) - Trellis mesh prior when available.

Only one frame is exported per invocation. File names mirror the first-frame convention used by
``export_gaussian_data.py`` so downstream Gaussian training scripts continue to work unchanged.
"""

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

logging.basicConfig(
    level=logging.INFO, format="[export_gaussian_frame] %(levelname)s: %(message)s"
)


# Root directory containing processed data per case (will be overridden by CLI).
DEFAULT_BASE_PATH = Path("./data/different_types")
# Output directory where Gaussian-ready assets will be written (will be overridden by CLI).
DEFAULT_OUTPUT_PATH = Path("./data/gaussian_data")
# Label used in mask metadata to denote controller (hand) pixels.
CONTROLLER_NAME = "hand"


def ensure_dir(path: Path) -> None:
    """Create directory if absent."""

    path.mkdir(parents=True, exist_ok=True)


def run_python_script(
    script: Path,
    args: Iterable[str],
    *,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> None:
    """Execute helper Python scripts with retry logic."""

    cmd = ["python", str(script), *args]
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            logging.error(
                "Helper script failed (%s) attempt %d/%d: %s",
                script,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise
        except Exception as exc:  # pragma: no cover
            logging.exception(
                "Unexpected error running helper script %s attempt %d/%d",
                script,
                attempt,
                max_retries,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise


def copy_frame_assets(
    case_dir: Path,
    output_dir: Path,
    frame_idx: int,
    category: str,
    generate_high_png: bool,
) -> None:
    """Copy RGB/depth/mask assets for a single frame and run auxiliary processing."""

    frame_str = str(frame_idx)  # Reuse the numeric frame ID as a string token.
    for cam_idx in range(3):
        # RGB image
        shutil.copy2(
            case_dir / "color" / str(cam_idx) / f"{frame_str}.png",
            output_dir / f"{cam_idx}.png",
        )

        # Object mask selection
        with (case_dir / "mask" / f"mask_info_{cam_idx}.json").open(
            "r", encoding="utf-8"
        ) as f:
            mask_info = json.load(f)
        object_idx: int | None = None
        for key, value in mask_info.items():
            if value != CONTROLLER_NAME:
                if object_idx is not None:
                    raise ValueError(
                        f"{case_dir.name}: multiple objects for cam {cam_idx}."
                    )
                object_idx = int(key)
        if object_idx is None:
            raise ValueError(f"{case_dir.name}: object mask missing for cam {cam_idx}.")

        mask_path = (
            case_dir / "mask" / str(cam_idx) / str(object_idx) / f"{frame_str}.png"
        )
        shutil.copy2(mask_path, output_dir / f"mask_{cam_idx}.png")

        if generate_high_png:
            # Optional branch that prepares the 4x upsampled RGB/mask assets.
            run_python_script(
                Path("data_process/image_upscale.py"),
                [
                    "--img_path",
                    str(case_dir / "color" / str(cam_idx) / f"{frame_str}.png"),
                    "--output_path",
                    str(output_dir / f"{cam_idx}_high.png"),
                    "--category",
                    category,
                ],
            )

            run_python_script(
                Path("data_process/segment_util_image.py"),
                [
                    "--img_path",
                    str(output_dir / f"{cam_idx}_high.png"),
                    "--TEXT_PROMPT",
                    category,
                    "--output_path",
                    str(output_dir / f"mask_{cam_idx}_high.png"),
                ],
            )

        # Depth map
        shutil.copy2(
            case_dir / "depth" / str(cam_idx) / f"{frame_str}.npy",
            output_dir / f"{cam_idx}_depth.npy",
        )

        # Human mask (low-res and high-res)
        run_python_script(
            Path("data_process/segment_util_image.py"),
            [
                "--img_path",
                str(output_dir / f"{cam_idx}.png"),
                "--TEXT_PROMPT",
                "human",
                "--output_path",
                str(output_dir / f"mask_human_{cam_idx}.png"),
                "--exclude_mask_path",
                str(output_dir / f"mask_{cam_idx}.png"),
            ],
        )
        if generate_high_png:
            run_python_script(
                Path("data_process/segment_util_image.py"),
                [
                    "--img_path",
                    str(output_dir / f"{cam_idx}_high.png"),
                    "--TEXT_PROMPT",
                    "human",
                    "--output_path",
                    str(output_dir / f"mask_human_{cam_idx}_high.png"),
                    "--exclude_mask_path",
                    str(output_dir / f"mask_{cam_idx}_high.png"),
                ],
            )


def save_camera_meta(case_dir: Path, output_dir: Path) -> None:
    """Save extrinsics/intrinsics for Gaussian loader."""

    with (case_dir / "calibrate.pkl").open("rb") as f:
        c2ws = pickle.load(f)
    with (case_dir / "metadata.json").open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)["intrinsics"]

    camera_meta = {"c2ws": c2ws, "intrinsics": intrinsics}
    with (output_dir / "camera_meta.pkl").open("wb") as f:
        pickle.dump(camera_meta, f)


def save_observation_point_cloud(
    case_dir: Path, output_dir: Path, frame_idx: int
) -> None:
    """Fuse masked point clouds for the target frame and write observation.ply."""

    pcd_data = np.load(
        case_dir / "pcd" / f"{frame_idx}.npz"
    )  # World-space point cloud for this frame.
    with (case_dir / "mask" / "processed_masks.pkl").open("rb") as f:
        processed_masks = pickle.load(f)

    # processed_masks may only store the first frame (legacy pipelines) or all frames (newer pipelines).
    mask_frame_idx = frame_idx
    if mask_frame_idx >= len(processed_masks):
        raise IndexError(
            f"processed_masks.pkl does not contain frame index {frame_idx}; "
            "please re-run preprocessing to include this frame."
        )

    obs_points = []
    obs_colors = []
    for cam_idx in range(3):
        points = pcd_data["points"][cam_idx]
        colors = pcd_data["colors"][cam_idx]
        object_mask = processed_masks[mask_frame_idx][cam_idx]["object"]
        obs_points.append(points[object_mask])
        obs_colors.append(colors[object_mask])

    fused_points = np.vstack(obs_points)
    fused_colors = np.vstack(obs_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_points)
    pcd.colors = o3d.utility.Vector3dVector(fused_colors)
    o3d.io.write_point_cloud(str(output_dir / "observation.ply"), pcd)


def maybe_copy_shape_prior(case_dir: Path, output_dir: Path, enabled: bool) -> None:
    """Copy Trellis mesh prior if the dataset requires shape initialization."""

    if not enabled:
        return
    source = case_dir / "shape" / "matching" / "final_mesh.glb"
    if source.exists():
        shutil.copy2(source, output_dir / "shape_prior.glb")
    else:
        print(f"Warning: {source} missing; skipping shape prior.")


def process_case_frame(
    base_path: Path,
    output_path: Path,
    case_name: str,
    category: str,
    shape_prior: str,
    frame_idx: int,
    generate_high_png: bool,
) -> None:
    """Export Gaussian assets for one case and frame index."""

    case_dir = base_path / case_name
    if not case_dir.exists():
        return

    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    output_dir = output_path / case_name  # Match legacy layout (no frame subdirectory).
    ensure_dir(output_dir)

    copy_frame_assets(case_dir, output_dir, frame_idx, category, generate_high_png)
    save_camera_meta(case_dir, output_dir)
    maybe_copy_shape_prior(case_dir, output_dir, shape_prior.lower() == "true")
    save_observation_point_cloud(case_dir, output_dir, frame_idx)


def lookup_case_metadata(case_name: str, config_path: Path) -> tuple[str, str]:
    """Return (category, shape_prior_flag) for the given case by scanning the config CSV."""

    with config_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            name, category, shape_prior = row[:3]
            if name == case_name:
                return category, shape_prior
    raise ValueError(f"Case '{case_name}' not found in {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Gaussian data for a single case and frame."
    )
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="Case name to export (must exist under base_dir).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to export (default: 0).",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Root directory containing cases (default: ./data/different_types).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Directory for Gaussian assets (default: ./data/gaussian_data).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category override; otherwise taken from data_config.csv.",
    )
    parser.add_argument(
        "--shape_prior",
        type=str,
        choices=("true", "false", "True", "False"),
        default=None,
        help="Optional shape_prior override; otherwise taken from data_config.csv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data_config.csv"),
        help="CSV file describing case metadata (default: data_config.csv).",
    )
    parser.add_argument(
        "--generate_high_png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle generation of *_high.png assets (default: enabled).",
    )
    args = parser.parse_args()

    case_name = args.case
    frame_idx = args.frame
    base_path: Path = args.base_dir
    output_path: Path = args.output_dir

    ensure_dir(output_path)

    if args.category is not None and args.shape_prior is not None:
        category = args.category
        shape_prior = args.shape_prior
    else:
        category, shape_prior = lookup_case_metadata(case_name, args.config)

    process_case_frame(
        base_path,
        output_path,
        case_name,
        category,
        shape_prior,
        frame_idx,
        args.generate_high_png,
    )


if __name__ == "__main__":
    main()
