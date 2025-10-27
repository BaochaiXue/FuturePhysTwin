#!/usr/bin/env python3
"""
Canonical Gaussian pipeline driver.

Inputs
------
- Frame-level Gaussian data at ``./per_frame_gaussian_data/<frame>/<scene>/`` (override via ``--data_dir``) containing RGB, depth, masks, ``camera_meta.pkl``, ``observation.ply``, and optional ``shape_prior.glb``.
- Helper scripts: ``generate_interp_poses.py``, ``dynamic_fast_canonical.py``, ``gs_render.py``, ``img2video.py``.
- Optional motion references (``experiments/<scene>/inference.pkl``, ``data/different_types/<scene>/track_process_data.pkl``). When present they are copied next to the canonical checkpoint and ``prepare_lbs_weights.py`` is invoked to produce ``lbs_data.pt``. Stage B (`dynamic_fast_color.py`) then scans ``per_frame_gaussian_data/*/<scene>/`` via ``--frames_dir`` to aggregate all frame-specific cameras.

Outputs
-------
- ``<output_dir>/<scene>/<frame>/<exp_name>/canonical_gaussians.npz``: canonical geometry/appearance parameters from Stage A.
- ``<output_dir>/<scene>/<frame>/<exp_name>/color_refine/canonical_gaussians_color.npz``: colour-refined parameters generated during Stage B.
- ``<output_dir>/<scene>/<frame>/<exp_name>/point_cloud/iteration_XXXX/point_cloud.ply`` plus allied checkpoint files (``cfg_args``, ``exposure.json``).
- ``<output_dir>/<scene>/<frame>/<exp_name>/lbs/`` and ``lbs_data.pt``: motion references and preprocessed LBS metadata.
- ``<output_dir>/<scene>/<frame>/<exp_name>/test/ours_<iter>/renders/*.png``: canonical render outputs.
- ``<video_dir>/<scene>/<frame>/<exp_name>.mp4``: MP4 preview assembled from the rendered frames.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import shutil
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
        default=Path("./tmp_gaussian_output"),
        help="Directory to store trained Gaussian models (mirrors gs_run.py).",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("./tmp_gaussian_output_video"),
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
    # Only train the canonical (first) frame for every scene.
    # ------------------------------------------------------------------
    frame_directories = list(iter_frame_directories(data_dir))
    if not frame_directories:
        raise FileNotFoundError(f"No frame subdirectories found under {data_dir}")
    frame_count: dict[str, int] = {}
    for frame_dir in frame_directories:
        print(f"[InterpPoses] Generating poses for frame directory: {frame_dir}")
        for scene_dir in iter_scene_directories(frame_dir):
            scene_name = scene_dir.name
            frame_count[scene_name] = frame_count.get(scene_name, 0) + 1
        run_command(
            [
                "python",
                str(root / "gaussian_splatting" / "generate_interp_poses.py"),
                "--root_dir",
                str(frame_dir),
            ]
        )

    canonical_frame_dir: Path = frame_directories[0]
    frame_name: str = canonical_frame_dir.name

    # ------------------------------------------------------------------
    # Iterate over each case/scene for the canonical frame.
    # ------------------------------------------------------------------
    final_models: list[tuple[Path, str, Path]] = []
    for scene_dir in iter_scene_directories(canonical_frame_dir):
        scene_name: str = scene_dir.name  # e.g. "cloth_scene_01"
        print(f"[Frame {frame_name}] Processing scene: {scene_name}")

        # Construct the output model directory: <output>/<scene>/<frame>/<exp_name>.
        model_dir: Path = output_dir / scene_name / frame_name / args.exp_name
        # Guarantee the parent folder exists (scene + frame).
        ensure_dir(model_dir.parent)

        # ---------------------------
        # Stage 1: Gaussian training
        # ---------------------------
        train_command: list[str] = [
            "python",
            str(root / "dynamic_fast_canonical.py"),
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
        ensure_dir(video_output_path.parent)  # Ensure <scene>/<frame> directory exists.

        video_command: list[str] = [
            "python",
            str(root / "gaussian_splatting" / "img2video.py"),
            "--image_folder",  # Directory containing rendered PNG frames.
            str(renders_subdir),
            "--video_path",  # Destination MP4 file consolidating the renders.
            str(video_output_path),
        ]
        run_command(video_command)

        # --------------------------------------------------------------
        # Stage 4: copy motion assets required for downstream LBS.
        # --------------------------------------------------------------
        lbs_dir = model_dir / "lbs"
        ensure_dir(lbs_dir)
        motion_sources = [
            (
                root / "experiments" / scene_name / "inference.pkl",
                "inference.pkl",
            ),
            (
                root
                / "data"
                / "different_types"
                / scene_name
                / "track_process_data.pkl",
                "track_process_data.pkl",
            ),
        ]
        for src_path, target_name in motion_sources:
            if src_path.is_file():
                shutil.copy2(src_path, lbs_dir / target_name)
            else:
                print(f"[LBS] Warning: {src_path} not found; skipping copy.")

        motion_source = lbs_dir / "inference.pkl"
        if motion_source.is_file():
            lbs_command: list[str] = [
                "python",
                str(root / "prepare_lbs_weights.py"),
                "--model_dir",
                str(model_dir),
                "--motion_source",
                str(motion_source),
                "--k_bones",
                "64",
                "--k_skin",
                "8",
            ]
            run_command(lbs_command)
        else:
            print(
                f"[LBS] Warning: {motion_source} missing; skip LBS metadata generation."
            )

        color_command: list[str] = [
            "python",
            str(root / "dynamic_fast_color.py"),
            "--color_only",
            "--freeze_geometry",
            "--lbs_path",
            str(model_dir / "lbs_data.pt"),
            "--source_path",
            str(scene_dir),
            "--model_path",
            str(model_dir),
            "--iterations",
            str(args.iterations * frame_count.get(scene_name, 1) // 10),
            "--use_masks",
        ]
        color_command.extend(["--frames_dir", str(data_dir)])
        run_command(color_command)
        final_models.append((scene_dir, scene_name, model_dir / "color_refine"))

    # --------------------------------------------------------------
    # Final Stage: promote colour-refined results to canonical outputs.
    # --------------------------------------------------------------
    final_output_root = root / "gaussian_output"
    final_video_root = root / "gaussian_output_video"
    ensure_dir(final_output_root)
    ensure_dir(final_video_root)

    for scene_dir, scene_name, color_dir in final_models:
        if not color_dir.exists():
            print(
                f"[Final] Warning: colour refine directory missing for {scene_name}, skipping."
            )
            continue

        dest_model_dir = final_output_root / scene_name / args.exp_name
        if dest_model_dir.exists():
            shutil.rmtree(dest_model_dir)
        shutil.copytree(color_dir, dest_model_dir)

        final_render_command: list[str] = [
            "python",
            str(root / "gs_render.py"),
            "-s",
            str(scene_dir),
            "-m",
            str(dest_model_dir),
        ]
        run_command(final_render_command)

        renders_dir = dest_model_dir / "test" / f"ours_{args.iterations}" / "renders"
        if not renders_dir.exists():
            print(
                f"[Final] Warning: renders directory {renders_dir} missing after rendering; skipping video generation."
            )
            continue

        scene_video_root = final_video_root / scene_name
        ensure_dir(scene_video_root)
        video_path = scene_video_root / f"{args.exp_name}.mp4"
        final_video_command: list[str] = [
            "python",
            str(root / "gaussian_splatting" / "img2video.py"),
            "--image_folder",
            str(renders_dir),
            "--video_path",
            str(video_path),
        ]
        run_command(final_video_command)


if __name__ == "__main__":
    main()
