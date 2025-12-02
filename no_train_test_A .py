#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
import shutil

from moviepy import ColorClip, CompositeVideoClip, TextClip, VideoFileClip, clips_array


@dataclass(frozen=True)
class PipelineConfig:
    """Single training/evaluation configuration."""

    name_suffix: str
    title_prefix: str
    exp_suffix: str
    train_all_parameter: bool
    mask_pred_with_alpha: bool
    lbs_refresh_interval: int
    video_label: str


VIDEO_DIR = Path("compare_visualization_videos")
GAUSSIAN_OUTPUT_ROOT = Path("gaussian_output")
WHITE_OUTPUT_DIR = Path("gaussian_output_dynamic_white")
FALLBACK_WHITE_OUTPUT_DIR = Path("tmp_gaussian_output_dynamic_white")
DATA_ROOT = Path("data/gaussian_data")
WHITE_VIEWS: tuple[str, ...] = ("0", "1", "2")
BASE_EXP_NAME = "testA"
PIPELINE_CONFIGS: tuple[PipelineConfig, ...] = (
    PipelineConfig(
        name_suffix="trainall_alpha",
        title_prefix="Train-All + Alpha Mask",
        exp_suffix="trainall_alpha",
        train_all_parameter=True,
        mask_pred_with_alpha=True,
        lbs_refresh_interval=1000,
        video_label="TrainAll + Alpha",
    ),
    PipelineConfig(
        name_suffix="trainall_occ",
        title_prefix="Train-All + Occ Mask",
        exp_suffix="trainall_occ",
        train_all_parameter=True,
        mask_pred_with_alpha=False,
        lbs_refresh_interval=1000,
        video_label="TrainAll + Occ",
    ),
    PipelineConfig(
        name_suffix="color_alpha",
        title_prefix="Color/Opacity + Alpha Mask",
        exp_suffix="color_alpha",
        train_all_parameter=False,
        mask_pred_with_alpha=True,
        lbs_refresh_interval=0,
        video_label="Color + Alpha",
    ),
    PipelineConfig(
        name_suffix="color_occ",
        title_prefix="Color/Opacity + Occ Mask",
        exp_suffix="color_occ",
        train_all_parameter=False,
        mask_pred_with_alpha=False,
        lbs_refresh_interval=0,
        video_label="Color + Occ",
    ),
)


def discover_cases(root: Path) -> tuple[str, ...]:
    """Return all case names under ``root``."""

    if not root.exists():
        return ()
    return tuple(
        sorted(path.name for path in root.iterdir() if path.is_dir())
    )


def run_command(
    command: Sequence[str], max_attempts: int = 3, retry_delay: float = 5.0
) -> None:
    """Execute ``command`` with basic retry logic on failure."""

    pretty = " ".join(command)
    for attempt in range(1, max_attempts + 1):
        print(f"[PIPELINE] Running: {pretty} (attempt {attempt}/{max_attempts})")
        try:
            subprocess.run(list(command), check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == max_attempts:
                raise
            print(
                f"[PIPELINE] Command failed with exit code {exc.returncode}. "
                f"Retrying in {retry_delay:.1f}s..."
            )
            time.sleep(retry_delay)


def render_white_video(exp_name: str) -> None:
    """Render white-background videos for ``exp_name`` using gs_run_simulate_white."""

    run_command(
        [
            "python",
            "gs_run_simulate_white.py",
            "--output_dir",
            str(WHITE_OUTPUT_DIR),
            "--views",
            *WHITE_VIEWS,
            "--exp_name",
            exp_name,
            "--data_root",
            str(DATA_ROOT),
            "--gaussian_root",
            str(GAUSSIAN_OUTPUT_ROOT),
        ]
    )


def resolve_white_video(case: str, view: str) -> Path:
    """Locate the freshly rendered white-background video for ``case``/``view``."""

    candidate = WHITE_OUTPUT_DIR / case / f"{view}.mp4"
    if candidate.exists():
        return candidate

    fallback = FALLBACK_WHITE_OUTPUT_DIR / case / f"{view}.mp4"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"White-background video missing; looked for {candidate} and {fallback}"
    )


def rename_video(case: str, exp_name: str, suffix: str, view: str) -> Path:
    """Copy the latest experiment video into compare dir with ``suffix``."""

    src = resolve_white_video(case, view)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = VIDEO_DIR / f"{case}_{suffix}_cam_{view}_{timestamp}.mp4"

    if not src.exists():
        raise FileNotFoundError(
            f"Expected comparison video not found: {src}. "
            "Ensure generate_compare_videos_video.py produced the file."
        )

    if dst.exists():
        print(f"[PIPELINE] Removing stale video: {dst}")
        dst.unlink()

    shutil.copy2(src, dst)
    print(f"[PIPELINE] Stored video as {dst}")
    return dst


def run_pipeline(config: PipelineConfig, cases: Sequence[str]) -> dict[str, dict[str, Path]]:
    """Run the white rendering pipeline and collect videos per case/view."""

    exp_name = f"{BASE_EXP_NAME}_{config.exp_suffix}"
    render_white_video(exp_name)
    outputs: dict[str, dict[str, Path]] = {case: {} for case in cases}
    for case in cases:
        for view in WHITE_VIEWS:
            outputs[case][view] = rename_video(case, exp_name, config.name_suffix, view)
    return outputs


def add_label_overlay(
    clip: VideoFileClip, label: str
) -> CompositeVideoClip | VideoFileClip:
    if not label:
        return clip

    duration = clip.duration
    base_fps = getattr(clip, "fps", 30) or 30
    font_size = max(24, clip.h // 18)
    text_clip = (
        TextClip(
            text=label,
            font_size=font_size,
            color="white",
            method="caption",
            size=(int(clip.w * 0.8), None),
            text_align="left",
        )
        .with_duration(duration)
        .with_position((30, 30))
    )
    bg_clip = (
        ColorClip(
            size=(int(text_clip.w + 40), int(text_clip.h + 40)),
            color=(0, 0, 0),
            duration=duration,
        )
        .with_opacity(0.6)
        .with_position((20, 20))
    )
    composite = (
        CompositeVideoClip([clip, bg_clip, text_clip])
        .with_duration(duration)
        .with_fps(base_fps)
    )
    return composite


def compose_comparison_grid(entries: list[tuple[Path, str]], case: str, view: str) -> None:
    if len(entries) != 4:
        print("[PIPELINE] Skipping comparison grid: expected 4 videos.")
        return

    clips = []
    raw_clips: list[VideoFileClip] = []
    target_size = None
    grid_clip = None
    try:
        for path, label in entries:
            clip = VideoFileClip(str(path))
            raw_clips.append(clip)
            if target_size is None:
                target_size = clip.size
            else:
                clip = clip.resized(new_size=target_size)
            annotated = add_label_overlay(clip, label)
            clips.append(annotated)

        grid_clip = clips_array([[clips[0], clips[1]], [clips[2], clips[3]]])
        output_path = VIDEO_DIR / f"compare_video_2X2_{case}_{view}.mp4"
        grid_clip.write_videofile(
            str(output_path), codec="libx264", audio=False, fps=30
        )
        print(f"[PIPELINE] Saved comparison grid to {output_path}")
    finally:
        if grid_clip is not None:
            try:
                grid_clip.close()
            except Exception:
                pass
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass
        for raw_clip in raw_clips:
            try:
                raw_clip.close()
            except Exception:
                pass


def collect_existing_videos(case: str, view: str) -> list[tuple[Path, str]]:
    """Load the most recent archived videos for each configuration."""

    entries: list[tuple[Path, str]] = []
    for config in PIPELINE_CONFIGS:
        pattern = f"{case}_{config.name_suffix}_cam_{view}_*.mp4"
        matches = sorted(VIDEO_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No archived comparison video matching {pattern} in {VIDEO_DIR}"
            )
        entries.append((matches[-1], config.video_label))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline and compose 2×2 comparison videos."
    )
    parser.add_argument(
        "--compose_only",
        action="store_true",
        help=(
            "Skip running final_eval for each config and only compose the 2x2 grid "
        "using previously archived videos."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Optional list of case names to process; defaults to all cases under data/gaussian_data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = tuple(args.cases) if args.cases else discover_cases(DATA_ROOT)
    if not cases:
        raise FileNotFoundError(f"No case directories found under {DATA_ROOT}")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    collected_by_case: dict[str, dict[str, list[tuple[Path, str]]]] = {
        case: {view: [] for view in WHITE_VIEWS} for case in cases
    }

    if args.compose_only:
        for case in cases:
            for view in WHITE_VIEWS:
                collected_by_case[case][view] = collect_existing_videos(case, view)
    else:
        for config in PIPELINE_CONFIGS:
            print(f"[PIPELINE] === Processing {config.name_suffix} ===")
            outputs = run_pipeline(config, cases)
            for case, views in outputs.items():
                for view, path in views.items():
                    collected_by_case[case][view].append((path, config.video_label))

    for case in cases:
        for view in WHITE_VIEWS:
            compose_comparison_grid(collected_by_case[case][view], case, view)


if __name__ == "__main__":
    main()
