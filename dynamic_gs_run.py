#!/usr/bin/env python3
"""
Run Gaussian Splatting training and rendering for every frame-specific scene.

Inputs
------
- Frame-scoped scene folders under ``./per_frame_gaussian_data/<frame>/<scene>/`` (or ``--data_dir``),
  each containing the Gaussian-ready assets produced by ``dynamic_export_gs_data.py``:
    * RGB images, masks, depth maps, ``camera_meta.pkl``, ``observation.ply``.
    * Optional ``shape_prior.glb`` mesh prior.
- Helper scripts residing in this repository:
    * ``gaussian_splatting/generate_interp_poses.py`` (creates ``interp_poses.pkl`` per frame/case).
    * ``gs_train.py`` (optimises Gaussian splats).
    * ``gs_render.py`` (renders evaluation views).
    * ``gaussian_splatting/img2video.py`` (converts rendered frames to MP4).

Outputs
-------
- Trained Gaussian checkpoints and point-cloud snapshots in
  ``./per_frame_gaussian_output/<scene>/<frame>/<exp_name>/`` (or ``--output_dir``).
- Rendered images stored under ``.../test/ours_<iterations>/renders/`` within each model directory.
- Preview videos in ``./per_frame_gaussian_output_video/<scene>/<frame>/<exp_name>.mp4`` (or ``--video_dir``).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


def run_command(command: Sequence[str]) -> None:
    """
    Execute an external command and raise immediately if it fails.

    Args:
        command: Full argv list (e.g. ``["python", "script.py", "--flag"]``).
    """

    # ``subprocess.run`` returns a ``CompletedProcess`` object; ``check=True`` makes it
    # raise ``CalledProcessError`` when the exit code is non-zero, preventing silent failures.
    subprocess.run(list(command), check=True)


def ensure_dir(path: Path) -> None:
    """
    Create ``path`` (and any missing parents) if it does not already exist.

    Args:
        path: Directory that should exist after this call.
    """

    # ``exist_ok=True`` avoids errors if the directory has already been created.
    path.mkdir(parents=True, exist_ok=True)


def iter_frame_directories(data_root: Path) -> Iterable[Path]:
    """
    Yield every frame directory (e.g. ``.../per_frame_gaussian_data/0``) in sorted order.

    Args:
        data_root: Base folder supplied via ``--data_dir``.
    """

    # ``glob("*/")`` lists all first-level directories under ``data_root``.
    for frame_dir in sorted(data_root.glob("*/")):
        if frame_dir.is_dir():
            yield frame_dir


def iter_scene_directories(frame_dir: Path) -> Iterable[Path]:
    """
    Yield every case/scene directory inside a given frame folder.

    Args:
        frame_dir: Path such as ``.../per_frame_gaussian_data/0``.
    """

    # Each frame folder can contain several scene subdirectories.
    for scene_dir in sorted(frame_dir.glob("*/")):
        if scene_dir.is_dir():
            yield scene_dir


def resolve_path(base: Path, candidate: Path) -> Path:
    """
    Convert ``candidate`` to an absolute path relative to ``base`` when needed.

    Args:
        base: Reference directory (typically the project root).
        candidate: Potentially relative path supplied via CLI.
    """

    # Joining with ``base`` keeps behaviour consistent regardless of where the script is launched from.
    return candidate if candidate.is_absolute() else base / candidate


def main() -> None:
    """Mirror the behaviour of ``gs_run.sh`` for per-frame exports with extensive inline commentary."""

    # Resolve the project root once so relative paths are robust to the current working directory.
    root: Path = Path(__file__).resolve().parent

    # ------------------------------------------------------------------
    # Construct the CLI parser for flexible directory layouts and settings.
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train and render Gaussian splats for every frame-specific scene."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./per_frame_gaussian_data"),
        help="Root containing frame folders with Gaussian-ready assets.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./per_frame_gaussian_output"),
        help="Directory to store trained Gaussian models (mirrors gs_run.py).",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("./per_frame_gaussian_output_video"),
        help="Directory to store rendered preview videos.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Training iterations passed to gs_train.py (default: 10_000).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
        help="Experiment suffix appended to each model directory.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve CLI paths relative to the project root so they work from any cwd.
    # ------------------------------------------------------------------
    data_dir: Path = resolve_path(root, args.data_dir)
    output_dir: Path = resolve_path(root, args.output_dir)
    video_dir: Path = resolve_path(root, args.video_dir)

    # Guard against typos or missing exports by validating the input directory up front.
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {data_dir}")

    # Ensure output/video directories exist before we launch long-running training jobs.
    ensure_dir(output_dir)
    ensure_dir(video_dir)

    # ------------------------------------------------------------------
    # Iterate over every frame folder (e.g. 0, 5, 10 ...).
    # ------------------------------------------------------------------
    for frame_dir in iter_frame_directories(data_dir):
        frame_name: str = frame_dir.name  # e.g. "0"

        # Generate interpolated camera poses for all scenes within this frame directory.
        run_command(
            [
                "python",
                str(root / "gaussian_splatting" / "generate_interp_poses.py"),
                "--root_dir",
                str(frame_dir),
            ]
        )

        # ------------------------------------------------------------------
        # Iterate over each case/scene for the current frame.
        # ------------------------------------------------------------------
        for scene_dir in iter_scene_directories(frame_dir):
            scene_name: str = scene_dir.name  # e.g. "cloth_scene_01"
            print(f"[Frame {frame_name}] Processing scene: {scene_name}")

            # Construct the output model directory: <output>/<frame>/<scene>/<exp_name>.
            model_dir: Path = output_dir / scene_name / frame_name / args.exp_name
            # Guarantee the parent folder exists (scene + frame).
            ensure_dir(model_dir.parent)

            # ---------------------------
            # Stage 1: Gaussian training
            # ---------------------------
            train_command: list[str] = [
                "python",
                str(root / "gs_train.py"),
                "-s",  # Source directory containing Gaussian-ready data.
                str(scene_dir),
                "-m",  # Destination directory for storing model checkpoints.
                str(model_dir),
                "--iterations",
                str(args.iterations),
                "--lambda_depth",  # Depth-loss weight.
                "0.001",
                "--lambda_normal",  # Normal-loss weight (disabled).
                "0.0",
                "--lambda_anisotropic",  # Anisotropy regulariser weight (disabled).
                "0.0",
                "--lambda_seg",  # Segmentation-mask loss weight.
                "1.0",
                "--use_masks",  # Enable mask-guided training.
                "--isotropic",  # Force isotropic Gaussians (single scale).
                "--gs_init_opt",  # Initialise from both observation and mesh.
                "hybrid",
            ]
            run_command(train_command)

            # ---------------------------
            # Stage 2: Rendering
            # ---------------------------
            render_command: list[str] = [
                "python",
                str(root / "gs_render.py"),
                "-s",  # Scene source path (same as training).
                str(scene_dir),
                "-m",  # Model directory to load the trained Gaussians from.
                str(model_dir),
            ]
            run_command(render_command)

            # ---------------------------
            # Stage 3: Images -> Video
            # ---------------------------
            renders_subdir: Path = (
                model_dir / "test" / f"ours_{args.iterations}" / "renders"
            )
            video_output_path: Path = (
                video_dir / scene_name / frame_name / f"{args.exp_name}.mp4"
            )
            ensure_dir(
                video_output_path.parent
            )  # Ensure <scene>/<frame> directory exists.

            video_command: list[str] = [
                "python",
                str(root / "gaussian_splatting" / "img2video.py"),
                "--image_folder",  # Directory containing rendered PNG frames.
                str(renders_subdir),
                "--video_path",  # Destination MP4 file consolidating the renders.
                str(video_output_path),
            ]
            run_command(video_command)
    # run dynamic_align.py to align per-frame Gaussian models and regenerate videos
    run_command(
        [
            "python",
            str(root / "dynamic_align.py"),
            "--per_frame_root",
            str(output_dir),
            "--output_dir",
            str(root / "gaussian_output"),
            "--video_dir",
            str(root / "gaussian_output_video"),
            "--exp_name",
            args.exp_name,
            "--iterations",
            str(args.iterations),
        ]
    )


if __name__ == "__main__":
    main()
