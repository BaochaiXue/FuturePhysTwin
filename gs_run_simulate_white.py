#!/usr/bin/env python3
"""
Python equivalent of ``gs_run_simulate_white.sh`` for rendering with white backgrounds.

Inputs
------
- Colour-stage checkpoints in ``gaussian_output/<scene>/<exp_name>/``.
- Scene definitions within ``data/gaussian_data/<scene>/``.

Outputs
-------
- White-background render frames ``gaussian_output_dynamic_white/<scene>/<view>/``.
- MP4 previews ``gaussian_output_dynamic_white/<scene>/<view>.mp4``.
"""

from __future__ import annotations

import time
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


OUTPUT_DIR = Path("gaussian_output_dynamic_white")
VIEWS: Sequence[str] = ("0", "1", "2")
EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
DATA_ROOT = Path("./data/gaussian_data")
GAUSSIAN_ROOT = Path("./gaussian_output")


def run_command(
    command: Sequence[str], retries: int = 3, sleep_time: float = 2.0
) -> None:
    """
    Execute ``command`` with a simple retry loop.

    Args:
        command: Command to execute (argv style).
        retries: Maximum number of attempts before raising the failure.
        sleep_time: Seconds to wait between attempts.
    """

    attempts = 0
    while True:
        try:
            subprocess.run(list(command), check=True)
            return
        except subprocess.CalledProcessError as exc:
            attempts += 1
            if attempts >= retries:
                raise
            print(
                f"[Retry {attempts}/{retries}] Command failed with code {exc.returncode}: "
                f"{' '.join(command)}"
            )
            time.sleep(sleep_time)


def iter_scene_dirs(root: Path) -> Iterable[Path]:
    """Yield every scene directory located directly under ``root``."""

    if not root.exists():
        return []
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            yield scene_dir


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scene_dir in iter_scene_dirs(DATA_ROOT):
        scene_name = scene_dir.name
        model_dir = GAUSSIAN_ROOT / scene_name / EXP_NAME
        if not model_dir.exists():
            print(f"[Skip] Colour checkpoint missing for {scene_name}: {model_dir}")
            continue

        run_command(
            [
                "python",
                "gs_render_dynamics.py",
                "-s",
                str(scene_dir),
                "-m",
                str(model_dir),
                "--name",
                scene_name,
                "--white_background",
                "--output_dir",
                str(OUTPUT_DIR),
            ]
        )

        for view_name in VIEWS:
            image_folder = OUTPUT_DIR / scene_name / view_name
            if not image_folder.exists():
                print(f"[Skip] Render folder missing for {scene_name}/{view_name}")
                continue
            video_path = OUTPUT_DIR / scene_name / f"{view_name}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    "python",
                    "gaussian_splatting/img2video.py",
                    "--image_folder",
                    str(image_folder),
                    "--video_path",
                    str(video_path),
                ]
            )

            black_folder = OUTPUT_DIR / scene_name / f"{view_name}_black"
            if black_folder.exists():
                run_command(
                    [
                        "python",
                        "gaussian_splatting/img2video.py",
                        "--image_folder",
                        str(black_folder),
                        "--video_path",
                        str(black_folder.with_suffix(".mp4")),
                    ]
                )


if __name__ == "__main__":
    main()
