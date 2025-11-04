#!/usr/bin/env python3
"""
Assemble side-by-side comparison videos in a 2×3 grid for each case.

For every case under ``gaussian_output_dynamic_white/<case>/`` the script looks for
camera views ``0.mp4``, ``1.mp4``, and ``2.mp4`` and the matching reference videos
inside ``tmp_gaussian_output_dynamic_white/<case>/``. The output is a single MP4 per
case laid out as:

    row 0: ours view0 | reference view0
    row 1: ours view1 | reference view1
    row 2: ours view2 | reference view2

If one clip is shorter than its counterpart, the remaining duration is filled with a
white frame so both stay aligned. All six tiles are extended to the longest duration
among them before composing the final grid.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from moviepy import (
    ColorClip,
    VideoClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
OURS_ROOT = SCRIPT_ROOT / "gaussian_output_dynamic_white"
REFERENCE_ROOT = SCRIPT_ROOT / "tmp_gaussian_output_dynamic_white"
OUTPUT_ROOT = SCRIPT_ROOT / "compare_visualization_videos"
CAMERA_VIEWS: Tuple[str, ...] = ("0", "1", "2")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("compare_videos_video")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_cases(selected: Optional[Sequence[str]]) -> List[Path]:
    if not OURS_ROOT.exists():
        raise FileNotFoundError(f"Ours root does not exist: {OURS_ROOT.resolve()}")

    all_cases = sorted(p for p in OURS_ROOT.iterdir() if p.is_dir())
    if selected:
        case_dirs: List[Path] = []
        missing: List[str] = []
        for case in selected:
            candidate = OURS_ROOT / case
            if candidate.is_dir():
                case_dirs.append(candidate)
            else:
                missing.append(case)
        if missing:
            raise FileNotFoundError(
                f"Cases not found under {OURS_ROOT}: {', '.join(sorted(missing))}"
            )
        return case_dirs
    return all_cases


def collect_view_pairs(case_dir: Path) -> List[Tuple[str, Path, Path]]:
    case_name = case_dir.name
    reference_dir = REFERENCE_ROOT / case_name
    if not reference_dir.exists():
        LOGGER.warning(
            "Skipping %s: reference case missing (%s)", case_name, reference_dir
        )
        return []

    pairs: List[Tuple[str, Path, Path]] = []
    for view in CAMERA_VIEWS:
        ours_video = case_dir / f"{view}.mp4"
        reference_video = reference_dir / f"{view}.mp4"
        if not ours_video.exists():
            LOGGER.warning(
                "Skipping %s view %s: ours video missing (%s)",
                case_name,
                view,
                ours_video,
            )
            return []
        if not reference_video.exists():
            LOGGER.warning(
                "Skipping %s view %s: reference video missing (%s)",
                case_name,
                view,
                reference_video,
            )
            return []
        pairs.append((view, ours_video, reference_video))
    return pairs


def _clip_fps(clip: VideoClip) -> float:
    fps = getattr(clip, "fps", None)
    if fps is None and hasattr(clip, "reader"):
        fps = getattr(clip.reader, "fps", None)
    return float(fps) if fps else 30.0


def extend_clip(clip: VideoClip, target_duration: float) -> VideoClip:
    if target_duration - clip.duration <= 1e-3:
        return clip if abs(clip.duration - target_duration) <= 1e-3 else clip.subclip(
            0, target_duration
        )
    pad_duration = max(0.0, target_duration - clip.duration)
    if pad_duration <= 1e-3:
        return clip
    fps = _clip_fps(clip)
    color_pad = ColorClip(size=clip.size, color=(255, 255, 255), duration=pad_duration)
    color_pad = color_pad.set_fps(fps)
    padded = concatenate_videoclips([clip, color_pad], method="compose")
    return padded.set_fps(fps)


def close_clip(clip: Optional[VideoClip]) -> None:
    if clip is None:
        return
    closer = getattr(clip, "close", None)
    if callable(closer):
        closer()


def process_case(case_dir: Path, output_dir: Path) -> None:
    case_name = case_dir.name
    pairs = collect_view_pairs(case_dir)
    if not pairs:
        return

    row_clips: List[Tuple[str, VideoClip, VideoClip]] = []
    pair_durations: List[float] = []
    raw_clips: List[VideoFileClip] = []
    grid: Optional[VideoClip] = None
    grid_components: List[VideoClip] = []

    try:
        for view, ours_path, ref_path in pairs:
            ours_clip = VideoFileClip(str(ours_path)).without_audio()
            ref_clip = VideoFileClip(str(ref_path)).without_audio()
            raw_clips.extend([ours_clip, ref_clip])

            target_height = max(ours_clip.h, ref_clip.h)
            if ours_clip.h != target_height:
                ours_clip = ours_clip.resize(height=target_height)
            if ref_clip.h != target_height:
                ref_clip = ref_clip.resize(height=target_height)

            pair_duration = max(ours_clip.duration, ref_clip.duration)
            ours_aligned = extend_clip(ours_clip, pair_duration)
            ref_aligned = extend_clip(ref_clip, pair_duration)

            row_clips.append((view, ours_aligned, ref_aligned))
            pair_durations.append(pair_duration)

        total_duration = max(pair_durations)
        grid_rows = []
        for view, ours_clip, ref_clip in row_clips:
            ours_extended = extend_clip(ours_clip, total_duration)
            ref_extended = extend_clip(ref_clip, total_duration)
            grid_rows.append([ours_extended, ref_extended])
            grid_components.extend([ours_extended, ref_extended])

        grid = clips_array(grid_rows)
        fps = _clip_fps(row_clips[0][1])

        ensure_dir(output_dir)
        output_path = output_dir / f"{case_name}.mp4"
        LOGGER.info("Rendering %s -> %s", case_name, output_path)
        grid.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            fps=fps,
            preset="medium",
            threads=4,
        )
    finally:
        close_clip(grid)
        for clip in grid_components:
            close_clip(clip)
        # Close all open clips to release file handles.
        for _, ours_clip, ref_clip in row_clips:
            close_clip(ours_clip)
            close_clip(ref_clip)
        for clip in raw_clips:
            close_clip(clip)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2x3 comparison videos for each case."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of case names to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to store the composed comparison videos.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    ensure_dir(output_dir)

    case_dirs = iter_cases(args.cases)
    if not case_dirs:
        LOGGER.warning("No cases found under %s", OURS_ROOT)
        return

    for case_dir in case_dirs:
        process_case(case_dir, output_dir)


if __name__ == "__main__":
    main()
