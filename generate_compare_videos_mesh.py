#!/usr/bin/env python3
"""
Render turntable videos of shape priors for two cases and compose them side by side.

Steps:
1) Load ``shape_prior.glb`` (fallback ``object.glb``) for each target case.
2) Render an off-screen turntable video of the mesh.
3) Compose the two case videos into one side-by-side clip with labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import open3d as o3d
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
)


CASES = ("double_lift_sloth", "double_lift_sloth_sam3")
BASE_PATH = Path("data/different_types")
OUTPUT_DIR = Path("compare_visualization_videos_mesh")

TURN_FRAMES = 180
TURN_FPS = 30
RENDER_RESOLUTION = (640, 640)
BG_COLOR = (1.0, 1.0, 1.0, 1.0)  # Open3D expects RGBA floats in [0,1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_mesh_path(case_name: str) -> Path:
    """Return a usable mesh path for the case, preferring matching/final_mesh.glb then object.glb."""

    import open3d as o3d

    candidates = [
        BASE_PATH / case_name / "shape" / "matching" / "final_mesh.glb",
    ]

    errors = []
    for cand in candidates:
        if not cand.exists():
            continue
        mesh = o3d.io.read_triangle_mesh(cand)
        if mesh.is_empty():
            errors.append(f"{cand} exists but is empty/unreadable.")
            continue
        return cand

    tried = ", ".join(str(c) for c in candidates)
    msg = f"No valid mesh found for case '{case_name}'. Tried: {tried}."
    if errors:
        msg += " " + " ".join(errors)
    raise FileNotFoundError(msg)


def render_turntable(mesh_path: Path, output_path: Path) -> None:
    """Render a simple turntable of the mesh to MP4 using an off-screen renderer."""

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {mesh_path}")
    mesh.compute_vertex_normals()

    width, height = RENDER_RESOLUTION
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.set_background(BG_COLOR)
    renderer.scene.add_geometry("mesh", mesh, material)

    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = np.max(bounds.get_extent())
    radius = float(extent) * 1.8 if extent > 0 else 1.0
    up = np.array([0.0, 1.0, 0.0])

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        TURN_FPS,
        (width, height),
    )
    if not writer.isOpened():
        del renderer
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for i in range(TURN_FRAMES):
        theta = 2.0 * math.pi * i / TURN_FRAMES
        eye = center + radius * np.array([math.cos(theta), 0.3, math.sin(theta)])
        renderer.scene.camera.look_at(center, eye, up)
        img = renderer.render_to_image()
        frame = np.asarray(img)
        if frame.shape[-1] == 4:  # drop alpha if present
            frame = frame[:, :, :3]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    del renderer


def pad_clip(clip: VideoFileClip, target_duration: float) -> VideoFileClip:
    if clip.duration >= target_duration - 1e-3:
        return clip
    pad_dur = target_duration - clip.duration
    pad_clip = ColorClip(
        size=clip.size, color=(255, 255, 255), duration=pad_dur
    ).with_fps(clip.fps or TURN_FPS)
    return concatenate_videoclips(
        [clip.with_fps(clip.fps or TURN_FPS), pad_clip], method="compose"
    )


def label_clip(clip: VideoFileClip, label: str) -> VideoFileClip:
    if not label:
        return clip
    font_size = max(24, int(min(clip.w, clip.h) * 0.07))
    text = TextClip(
        text=label,
        font_size=font_size,
        color="white",
        method="caption",
        size=(int(clip.w * 0.8), None),
        text_align="left",
    ).with_duration(clip.duration)
    background = ColorClip(
        size=(int(text.w + 24), int(text.h + 16)),
        color=(0, 0, 0),
        duration=clip.duration,
    ).with_opacity(0.65)
    annotated = CompositeVideoClip(
        [
            clip,
            background.with_position((12, 12)),
            text.with_position((20, 16)),
        ]
    ).with_duration(clip.duration)
    return annotated


def compose_side_by_side(
    left: VideoFileClip, right: VideoFileClip, out_path: Path
) -> None:
    target_dur = max(left.duration, right.duration)
    left_padded = pad_clip(left, target_dur)
    right_padded = pad_clip(right, target_dur)
    final = clips_array([[left_padded, right_padded]])
    final.write_videofile(
        str(out_path), fps=left_padded.fps or TURN_FPS, codec="libx264"
    )


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    per_case_videos: dict[str, Path] = {}

    for case in CASES:
        mesh_path = find_mesh_path(case)
        out_path = OUTPUT_DIR / f"{case}_shape.mp4"
        print(f"[info] Rendering turntable for {case} from {mesh_path}")
        render_turntable(mesh_path, out_path)
        per_case_videos[case] = out_path

    left_case, right_case = CASES
    print(f"[info] Composing side-by-side video: {left_case} | {right_case}")
    with VideoFileClip(str(per_case_videos[left_case])) as left_clip, VideoFileClip(
        str(per_case_videos[right_case])
    ) as right_clip:
        labeled_left = label_clip(left_clip, left_case)
        labeled_right = label_clip(right_clip, right_case)
        composite_out = OUTPUT_DIR / "compare_mesh.mp4"
        compose_side_by_side(labeled_left, labeled_right, composite_out)
        print(f"[done] Wrote {composite_out}")


if __name__ == "__main__":
    main()
