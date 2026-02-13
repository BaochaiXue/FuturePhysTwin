#!/usr/bin/env python3
"""
Canonical Gaussian pipeline driver.

Inputs
------
- Frame-level Gaussian data at ``./per_frame_gaussian_data/<frame>/<scene>/`` (override via ``--data_dir``) containing RGB, depth, masks, ``camera_meta.pkl``, ``observation.ply``, and optional ``shape_prior.glb``.
- Helper scripts: ``generate_interp_poses.py``, ``dynamic_fast_canonical.py``, ``gs_render.py``, ``img2video.py``.
- Optional motion references (``experiments/<scene>/inference.pkl``, ``data/different_types/<scene>/track_process_data.pkl``). When present they are copied next to the canonical checkpoint and ``precompute_lbs_pose_cache.py`` is invoked to produce ``lbs_pose_cache.pt``. Stage B (`dynamic_fast_color.py`) then scans ``per_frame_gaussian_data/*/<scene>/`` via ``--frames_dir`` to aggregate all frame-specific cameras.

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
import time

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    warn_input_cases_missing_in_config,
)

DEFAULT_COLOR_TRAIN_FRAMES = None
DEFAULT_COLOR_TRAIN_CAMS = None


def run_command(
    command: Sequence[str], max_attempts: int = 3, sleep_time: float = 2.0
) -> None:

    for attempt in range(1, max_attempts + 1):
        try:
            # ``subprocess.run`` returns a ``CompletedProcess`` object; ``check=True`` makes it
            # raise ``CalledProcessError`` when the exit code is non-zero, preventing silent failures.
            subprocess.run(list(command), check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == max_attempts:
                raise

            # Wait a bit before retrying.

            time.sleep(sleep_time)
            print(
                f"[Retry {attempt}/{max_attempts}] Command failed with code {exc.returncode}: {' '.join(command)}"
            )


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
    parser.add_argument(
        "--color_train_frames",
        nargs="+",
        type=int,
        default=DEFAULT_COLOR_TRAIN_FRAMES,
        help=(
            "Frame indices forwarded to dynamic_fast_color via --train_frames "
            "(multiple values are joined with commas)."
        ),
    )
    parser.add_argument(
        "--color_train_cams",
        nargs="+",
        type=int,
        default=DEFAULT_COLOR_TRAIN_CAMS,
        help="Camera indices forwarded to dynamic_fast_color via --train_cams.",
    )
    parser.add_argument(
        "--train_all_parameter",
        action="store_true",
        default=False,
        help=(
            "When set, the colour refinement stage also optimises xyz/scaling/rotation. "
            "Otherwise it only updates colour and opacity."
        ),
    )
    parser.add_argument(
        "--lbs_refresh_interval",
        type=int,
        default=0,
        help=(
            "Rebuild LBS bindings every N iterations during colour refinement when "
            "--train_all_parameter is enabled (0 disables refresh)."
        ),
    )
    parser.add_argument(
        "--mask_pred_with_alpha",
        action="store_true",
        default=False,
        help=(
            "Forward --mask_pred_with_alpha to the colour stage so predicted RGBs are "
            "masked with alpha instead of occlusion maps."
        ),
    )
    parser.add_argument(
        "--lbs_pose_mode",
        type=str,
        choices=("rollout", "absolute"),
        default="rollout",
        help=(
            "Pose cache variant used by dynamic_fast_color. "
            "Defaults to rollout for backward compatibility."
        ),
    )
    parser.add_argument(
        "--lbs_pose_cache_modes",
        nargs="+",
        choices=("rollout", "absolute"),
        default=None,
        help=(
            "Optional pose cache modes forwarded to precompute_lbs_pose_cache.py "
            "via --pose_cache_modes. Default keeps precompute's behavior "
            "(generate both rollout and absolute caches)."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("./data_config.csv"),
        help="Case allowlist CSV path.",
    )
    parser.add_argument(
        "--input-base-path",
        type=Path,
        default=Path("./data/different_types"),
        help="Input case root used with data_config.csv allowlist filtering.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve CLI paths relative to the project root so they work from any cwd.
    # ------------------------------------------------------------------
    data_dir: Path = resolve_path(root, args.data_dir)
    output_dir: Path = resolve_path(root, args.output_dir)
    video_dir: Path = resolve_path(root, args.video_dir)
    config_path: Path = resolve_path(root, args.config_path)
    input_base_path: Path = resolve_path(root, args.input_base_path)

    # Guard against typos or missing exports by validating the input directory up front.
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {data_dir}")
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(input_base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "dynamic_fast_gs", input_base_path, config_path
    )
    allowed_cases = input_cases & config_cases

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
    discovered_scene_names: set[str] = set()
    for frame_dir in frame_directories:
        print(f"[InterpPoses] Generating poses for frame directory: {frame_dir}")
        frame_scene_names = [scene_dir.name for scene_dir in iter_scene_directories(frame_dir)]
        discovered_scene_names.update(frame_scene_names)
        allowed_frame_scene_names = [
            scene_name for scene_name in frame_scene_names if scene_name in allowed_cases
        ]
        for scene_name in allowed_frame_scene_names:
            frame_count[scene_name] = frame_count.get(scene_name, 0) + 1

        if not allowed_frame_scene_names:
            continue

        run_command(
            [
                "python",
                str(root / "gaussian_splatting" / "generate_interp_poses.py"),
                "--root_dir",
                str(frame_dir),
                "--scenes",
                *allowed_frame_scene_names,
            ]
        )
    filter_candidates(
        sorted(discovered_scene_names),
        allowed_cases,
        "dynamic_fast_gs",
        f"{data_dir} (all frames)",
    )

    canonical_frame_dir: Path = frame_directories[0]
    frame_name: str = canonical_frame_dir.name

    # ------------------------------------------------------------------
    # Iterate over each case/scene for the canonical frame.
    # ------------------------------------------------------------------
    final_models: list[tuple[Path, str, Path]] = []
    canonical_scene_dirs = list(iter_scene_directories(canonical_frame_dir))
    scene_dirs_by_name = {scene_dir.name: scene_dir for scene_dir in canonical_scene_dirs}
    filtered_scene_names = filter_candidates(
        [scene_dir.name for scene_dir in canonical_scene_dirs],
        allowed_cases,
        "dynamic_fast_gs",
        str(canonical_frame_dir),
    )
    if not filtered_scene_names:
        print("[dynamic_fast_gs] No allowed cases found after data_config.csv filtering.")
        return
    for scene_name in filtered_scene_names:
        scene_dir = scene_dirs_by_name[scene_name]
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
            "--disable_viewer",
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
        colour_dir = model_dir / "color_refine"
        if colour_dir.exists():
            print(f"[Cleanup] Removing stale colour-refine directory: {colour_dir}")
            shutil.rmtree(colour_dir)

        lbs_dir = model_dir / "lbs"
        ensure_dir(lbs_dir)
        inference_src = root / "experiments" / scene_name / "inference.pkl"
        if inference_src.is_file():
            shutil.copy2(inference_src, lbs_dir / "inference.pkl")
        else:
            raise RuntimeError(
                f"[LBS] Required inference pickle missing: {inference_src}"
            )

        motion_source = lbs_dir / "inference.pkl"
        pose_cache_path = model_dir / "lbs_pose_cache.pt"
        if pose_cache_path.exists():
            print(f"[Cleanup] Removing stale pose cache: {pose_cache_path}")
            pose_cache_path.unlink()
        if motion_source.is_file():
            precompute_command: list[str] = [
                "python",
                str(root / "precompute_lbs_pose_cache.py"),
                "--model_dir",
                str(model_dir),
                "--inference",
                str(motion_source),
                "--output",
                str(pose_cache_path),
                "--K",
                "16",
            ]
            if args.lbs_pose_cache_modes:
                precompute_command.append("--pose_cache_modes")
                precompute_command.extend(args.lbs_pose_cache_modes)
            run_command(precompute_command)
        else:
            print(
                f"[LBS] Warning: {motion_source} missing; skip offline pose cache generation."
            )
        color_iterations = max(
            args.iterations,
            args.iterations * frame_count.get(scene_name, 1) // 10,
        )
        # TODO: delete when real train
        color_iterations = 10000
        color_command: list[str] = [
            "python",
            str(root / "dynamic_fast_color.py"),
            "--color_only",
            "--source_path",
            str(scene_dir),
            "--model_path",
            str(model_dir),
            "--iterations",
            str(color_iterations),
            "--use_masks",
            "--disable_viewer",
        ]
        if args.train_all_parameter:
            color_command.append("--train_all_parameter")
            # Optional: keep geometry anchored during train-all colour refinement.
            # Mirrors the canonical-stage depth supervision weight.
            color_command.extend(["--lambda_depth", "0.1"])
        else:
            color_command.append("--freeze_geometry")
        color_command.extend(["--lbs_pose_mode", args.lbs_pose_mode])
        if pose_cache_path.exists():
            color_command.extend(["--lbs_pose_cache", str(pose_cache_path)])
        else:
            print(
                f"[LBS] Warning: offline pose cache missing at {pose_cache_path}; "
                "colour stage will run without pose deformation."
            )
        color_command.extend(["--frames_dir", str(data_dir)])
        if args.color_train_frames:
            frame_spec = ",".join(str(frame) for frame in args.color_train_frames)
            color_command.extend(["--train_frames", frame_spec])
        if args.color_train_cams:
            color_command.append("--train_cams")
            color_command.extend(str(cam) for cam in args.color_train_cams)
        color_command.extend(["--viz_every", "1000"])
        if args.mask_pred_with_alpha:
            color_command.append("--mask_pred_with_alpha")
        if args.lbs_refresh_interval > 0:
            color_command.extend(
                ["--lbs_refresh_interval", str(args.lbs_refresh_interval)]
            )
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
