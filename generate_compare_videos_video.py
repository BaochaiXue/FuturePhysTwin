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
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from moviepy import (
    ColorClip,
    CompositeVideoClip,
    TextClip,
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
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        return (
            clip
            if abs(clip.duration - target_duration) <= 1e-3
            else clip.subclipped(0, target_duration)
        )
    pad_duration = max(0.0, target_duration - clip.duration)
    if pad_duration <= 1e-3:
        return clip
    fps = _clip_fps(clip)
    color_pad = ColorClip(
        size=clip.size, color=(255, 255, 255), duration=pad_duration
    ).with_fps(fps)
    padded = concatenate_videoclips([clip, color_pad], method="compose")
    return padded.with_fps(fps)


def annotate_clip(clip: VideoClip, label: str) -> Tuple[VideoClip, List[VideoClip]]:
    """Overlay ``label`` in the top-left corner of ``clip``."""

    if not label:
        return clip, []

    duration = clip.duration
    font_size = max(24, clip.h // 15)
    max_text_width = max(100, int(clip.w - 60))

    text_clip = TextClip(
        text=label,
        font_size=font_size,
        color="white",
        method="caption",
        size=(max_text_width, None),
        text_align="left",
    ).with_duration(duration)

    padding = 12
    background_width = min(int(clip.w - 20), int(text_clip.w + padding * 2))
    background_height = int(text_clip.h + padding * 2)

    background_clip = ColorClip(
        size=(background_width, background_height),
        color=(0, 0, 0),
        duration=duration,
    ).with_opacity(0.6)

    composite = (
        CompositeVideoClip(
            [
                clip,
                background_clip.with_position((15, 15)),
                text_clip.with_position((15 + padding, 15 + padding // 2)),
            ]
        )
        .with_duration(duration)
        .with_fps(_clip_fps(clip))
    )

    return composite, [text_clip, background_clip]


def add_title_bar(
    clip: VideoClip, title: Optional[str]
) -> Tuple[VideoClip, List[VideoClip]]:
    """Create a new clip with ``title`` rendered above the provided ``clip``."""

    if title is None:
        return clip, []

    title_text = title.strip()
    if not title_text:
        return clip, []

    duration = clip.duration
    base_fps = _clip_fps(clip)
    font_size = max(32, clip.h // 20)
    title_area_width = int(clip.w - 40)
    text_clip = TextClip(
        text=title_text,
        font_size=font_size,
        color="black",
        method="caption",
        size=(title_area_width, None),
        text_align="center",
    ).with_duration(duration)

    bar_height = max(int(text_clip.h + 30), font_size + 20)
    canvas = ColorClip(
        size=(clip.w, clip.h + bar_height), color=(255, 255, 255), duration=duration
    )
    text_y = max(0, (bar_height - text_clip.h) // 2)

    composite = (
        CompositeVideoClip(
            [
                canvas,
                clip.with_position((0, bar_height)),
                text_clip.with_position(("center", text_y)),
            ]
        )
        .with_duration(duration)
        .with_fps(base_fps)
    )

    return composite, [text_clip, canvas]


def close_clip(clip: Optional[VideoClip]) -> None:
    if clip is None:
        return
    closer = getattr(clip, "close", None)
    if callable(closer):
        closer()


def process_case(
    case_dir: Path, output_dir: Path, title_template: Optional[str]
) -> None:
    case_name = case_dir.name
    pairs = collect_view_pairs(case_dir)
    if not pairs:
        return

    pair_clips: List[Tuple[str, VideoClip, VideoClip]] = []
    pair_durations: List[float] = []
    raw_clips: List[VideoFileClip] = []
    grid: Optional[VideoClip] = None
    closeables: List[VideoClip] = []
    final_clip: Optional[VideoClip] = None

    try:
        for view, ours_path, ref_path in pairs:
            ours_clip = VideoFileClip(str(ours_path)).without_audio()
            ref_clip = VideoFileClip(str(ref_path)).without_audio()
            raw_clips.extend([ours_clip, ref_clip])

            target_height = max(ours_clip.h, ref_clip.h)
            if ours_clip.h != target_height:
                ours_clip = ours_clip.resized(height=target_height)
            if ref_clip.h != target_height:
                ref_clip = ref_clip.resized(height=target_height)

            pair_duration = max(ours_clip.duration, ref_clip.duration)
            ours_aligned = extend_clip(ours_clip, pair_duration)
            ref_aligned = extend_clip(ref_clip, pair_duration)

            pair_clips.append((view, ours_aligned, ref_aligned))
            pair_durations.append(pair_duration)

        total_duration = max(pair_durations)
        grid_rows = []
        for view, ours_clip, ref_clip in pair_clips:
            ours_extended = extend_clip(ours_clip, total_duration)
            ref_extended = extend_clip(ref_clip, total_duration)

            ours_label = f"View {view} - Ours"
            ref_label = f"View {view} - Reference"
            ours_annotated, ours_aux = annotate_clip(ours_extended, ours_label)
            ref_annotated, ref_aux = annotate_clip(ref_extended, ref_label)

            grid_rows.append([ours_annotated, ref_annotated])
            closeables.extend(
                [ours_extended, ref_extended, ours_annotated, ref_annotated]
            )
            closeables.extend(ours_aux)
            closeables.extend(ref_aux)

        grid = clips_array(grid_rows)
        closeables.append(grid)

        if title_template is None:
            title_text = case_name
        else:
            template = title_template.strip()
            if not template:
                title_text = case_name
            elif "{case" in template:
                try:
                    title_text = template.format(case=case_name)
                except KeyError as exc:
                    raise KeyError(
                        f"Unknown placeholder {{{exc.args[0]}}} in title template: {title_template}"
                    ) from exc
            else:
                title_text = f"{template} {case_name}".strip()
        final_clip, title_aux = add_title_bar(grid, title_text)
        closeables.extend(title_aux)
        if final_clip is not grid:
            closeables.append(final_clip)

        fps = _clip_fps(final_clip)

        ensure_dir(output_dir)
        output_filename = f"{case_name}_{RUN_TIMESTAMP}.mp4"
        output_path = output_dir / output_filename
        LOGGER.info("Rendering %s -> %s", case_name, output_path)
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio=False,
            fps=fps,
            preset="medium",
            threads=4,
        )
    finally:
        cleanup: List[Optional[VideoClip]] = [final_clip, grid, *closeables]
        seen: set[int] = set()
        for clip in cleanup:
            if clip is None:
                continue
            clip_id = id(clip)
            if clip_id in seen:
                continue
            seen.add(clip_id)
            close_clip(clip)
        # Close all open clips to release file handles.
        for _, ours_clip, ref_clip in pair_clips:
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
    parser.add_argument(
        "--title",
        type=str,
        default="Comparison for",
        help=(
            "Title prefix rendered above each comparison video. "
            "Include {case} to control placement explicitly; otherwise the case name is appended."
        ),
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    ensure_dir(output_dir)

    case_dirs = iter_cases(args.cases)
    if not case_dirs:
        LOGGER.warning("No cases found under %s", OURS_ROOT)
        return

    for case_dir in case_dirs:
        process_case(case_dir, output_dir, args.title)


if __name__ == "__main__":
    main()
