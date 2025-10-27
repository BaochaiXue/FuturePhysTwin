#!/usr/bin/env python3
"""
Python equivalent of the original `gs_run.sh`.

Inputs
------
- Scene directories under ``data/gaussian_data/<scene>`` (or ``--data_dir``), each containing:
    * Gaussian-ready assets (RGB/masks/depth, ``camera_meta.pkl``, ``observation.ply``).
    * Optional ``shape_prior.glb`` used during training.
- ``gaussian_splatting/generate_interp_poses.py`` for creating ``interp_poses.pkl`` per scene.
- Training and rendering scripts: ``gs_train.py``, ``gs_render.py``, ``gaussian_splatting/img2video.py``.

Outputs
-------
- Trained checkpoints and point-cloud snapshots under ``--output_dir/<scene>/<exp_name>/``.
- Rendered evaluation frames in ``--output_dir/<scene>/<exp_name>/test/ours_10000/renders/``.
- Preview videos in ``--video_dir/<scene>/<exp_name>.mp4`` summarising the rendered sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def run_command(cmd: list[str]) -> None:
    """
    Wrapper around subprocess.run to execute external commands and fail fast if any command returns
    a non-zero exit code.
    """

    subprocess.run(cmd, check=True)


def main() -> None:
    """
    Mirror the behavior of gs_run.sh using Python, with detailed inline explanations.

    The script:
        1. Locates the project root (directory containing this script).
        2. Derives all paths relative to that root.
        3. Generates interpolated camera poses for every scene.
        4. Iterates over `data/gaussian_data/<scene>` folders to:
            - run gs_train.py (train Gaussians),
            - run gs_render.py (render evaluation views),
            - convert the renders into a video using img2video.py.
    """

    # Absolute path to the directory that contains this file; serves as project root.
    root: Path = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Python equivalent of gs_run.sh with CLI-controllable paths."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data/gaussian_data"),
        help="Root directory containing per-scene folders (default: ./data/gaussian_data).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./gaussian_output"),
        help="Directory to save trained Gaussians (default: ./gaussian_output).",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("./gaussian_output_video"),
        help="Directory to save output videos (default: ./gaussian_output_video).",
    )
    args = parser.parse_args()

    # Resolve any relative paths with respect to the script location so behaviour matches the shell version.
    relative_data_dir: Path = args.data_dir
    relative_output_dir: Path = args.output_dir
    relative_video_dir: Path = args.video_dir
    # All scenes live under data/gaussian_data/<scene_name>.
    data_dir: Path = (
        relative_data_dir
        if relative_data_dir.is_absolute()
        else (root / relative_data_dir)
    )

    # Output folders for trained Gaussians and preview videos.
    output_dir: Path = (
        relative_output_dir
        if relative_output_dir.is_absolute()
        else (root / relative_output_dir)
    )
    video_dir: Path = (
        relative_video_dir
        if relative_video_dir.is_absolute()
        else (root / relative_video_dir)
    )

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input scene directory not found: {data_dir}")

    # Ensure output directories exist before we start.
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Experiment name is part of the output path; same as gs_run.sh.
    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

    # ----------------------------------------------------------------------
    # Step 1: Precompute interpolated camera poses for every scene directory.
    # Command:
    #   python gaussian_splatting/generate_interp_poses.py --root_dir data/gaussian_data
    # Meaning:
    #   --root_dir <PATH> tells the script where to find per-scene folders so that the
    #   resulting interp_poses.pkl is written alongside each scene.
    # ----------------------------------------------------------------------
    run_command(
        [
            "python",
            str(root / "gaussian_splatting" / "generate_interp_poses.py"),
            "--root_dir",
            str(data_dir),
        ]
    )

    # ----------------------------------------------------------------------
    # Step 2: Loop over each scene folder (e.g., data/gaussian_data/double_lift_cloth_1).
    # ----------------------------------------------------------------------
    for scene_path in sorted(data_dir.glob("*/")):
        # Skip non-directories just in case.
        if not scene_path.is_dir():
            continue

        scene_name = scene_path.name
        print(f"Processing: {scene_name}")

        # Each scene writes results to gaussian_output/<scene>/<exp_name>.
        model_dir = output_dir / scene_name / exp_name

        # ----------------------------------------------------------
        # Step 2a: Train Gaussians (gs_train.py).
        # Command line arguments:
        #   -s <PATH>  : source directory with RGB/Depth/Mask data.
        #   -m <PATH>  : output directory where trained Gaussians/checkpoints go.
        #   --iterations 10000           : number of training iterations.
        #   --lambda_depth 0.001         : weight for depth loss term.
        #   --lambda_normal 0.0          : disable normal loss.
        #   --lambda_anisotropic 0.0     : disable anisotropic regularizer.
        #   --lambda_seg 1.0             : segmentation loss weight.
        #   --use_masks                  : instructs gs_train to read mask_*.png for RGBA blending.
        #   --isotropic                  : enforce isotropic Gaussian splats (spherical).
        #   --gs_init_opt 'hybrid'       : initialize from both observation.ply and shape_prior.glb if present.
        # ----------------------------------------------------------
        train_args = [
            "python",
            str(root / "gs_train.py"),
            "-s",
            str(scene_path),
            "-m",
            str(model_dir),
            "--iterations",
            "10000",
            "--lambda_depth",
            "0.001",
            "--lambda_normal",
            "0.0",
            "--lambda_anisotropic",
            "0.0",
            "--lambda_seg",
            "1.0",
            "--use_masks",
            "--isotropic",
            "--gs_init_opt",
            "hybrid",
        ]
        run_command(train_args)

        # ----------------------------------------------------------
        # Step 2b: Render evaluation views (gs_render.py).
        # Command line arguments:
        #   -s <PATH> : same input folder as training (contains camera_meta.pkl, etc.).
        #   -m <PATH> : path to the trained Gaussians from previous step.
        # ----------------------------------------------------------
        render_args = [
            "python",
            str(root / "gs_render.py"),
            "-s",
            str(scene_path),
            "-m",
            str(model_dir),
        ]
        run_command(render_args)

        # ----------------------------------------------------------
        # Step 2c: Convert rendered images to MP4 (img2video.py).
        # Command line arguments:
        #   --image_folder <PATH> : directory containing rendered PNG frames.
        #   --video_path <PATH>   : where to write the resulting video.
        #       The script also accepts --fps (default 15), but we keep default.
        # ----------------------------------------------------------
        (video_dir / scene_name).mkdir(parents=True, exist_ok=True)
        video_args = [
            "python",
            str(root / "gaussian_splatting" / "img2video.py"),
            "--image_folder",
            str(model_dir / "test" / "ours_10000" / "renders"),
            "--video_path",
            str(video_dir / scene_name / f"{exp_name}.mp4"),
        ]
        run_command(video_args)


if __name__ == "__main__":
    main()
