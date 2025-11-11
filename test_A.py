#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PipelineConfig:
    """Single training/evaluation configuration."""

    name_suffix: str
    train_cams: tuple[int, ...] | None
    title_prefix: str


SCENE_NAME = "double_lift_cloth_1"
SCENE_FRAME = "115"
VIDEO_DIR = Path("compare_visualization_videos")
PIPELINE_CONFIGS: tuple[PipelineConfig, ...] = (
    PipelineConfig(
        name_suffix="all_cam",
        train_cams=None,
        title_prefix="Comparison for All Cameras",
    ),
    PipelineConfig(
        name_suffix="cam_0", train_cams=(0,), title_prefix="Comparison for Camera 0"
    ),
    PipelineConfig(
        name_suffix="cam_1", train_cams=(1,), title_prefix="Comparison for Camera 1"
    ),
    PipelineConfig(
        name_suffix="cam_2", train_cams=(2,), title_prefix="Comparison for Camera 2"
    ),
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


def rename_video(suffix: str) -> None:
    """Rename the generated comparison video to include ``suffix``."""

    src = VIDEO_DIR / f"{SCENE_NAME}.mp4"
    dst = VIDEO_DIR / f"{SCENE_NAME}_{suffix}.mp4"

    if not src.exists():
        raise FileNotFoundError(
            f"Expected comparison video not found: {src}. "
            "Ensure generate_compare_videos_video.py produced the file."
        )

    if dst.exists():
        print(f"[PIPELINE] Removing stale video: {dst}")
        dst.unlink()

    src.rename(dst)
    print(f"[PIPELINE] Stored video as {dst}")


def build_dynamic_fast_gs_command(config: PipelineConfig) -> list[str]:
    """Return the CLI invocation for dynamic_fast_gs.py."""

    command = [
        "python",
        "dynamic_fast_gs.py",
        "--color_train_frames",
        SCENE_FRAME,
    ]

    if config.train_cams is not None:
        command.append("--color_train_cams")
        command.extend(str(cam) for cam in config.train_cams)

    return command


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full training/evaluation/video pipeline for ``config``."""

    run_command(build_dynamic_fast_gs_command(config))
    run_command(["python", "final_eval.py"])
    run_command(
        [
            "python",
            "generate_compare_videos_video.py",
            "--title",
            config.title_prefix,
        ]
    )
    rename_video(config.name_suffix)


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    for config in PIPELINE_CONFIGS:
        print(f"[PIPELINE] === Processing {config.name_suffix} ===")
        run_pipeline(config)


if __name__ == "__main__":
    main()
