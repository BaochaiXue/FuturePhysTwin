#!/usr/bin/env python3
"""
Python equivalent of ``gs_run_simulate.sh`` that orchestrates dynamic rendering
and video export for every scene under ``./data/gaussian_data``.

Inputs
------
- Canonical Gaussian checkpoints in ``gaussian_output/<scene>/<exp_name>/``.
- Scene assets in ``data/gaussian_data/<scene>/`` containing evaluation camera paths.

Outputs
-------
- Dynamic render frames in ``gaussian_output_dynamic/<scene>/<view>/``.
- Corresponding MP4 previews ``gaussian_output_dynamic/<scene>/<view>.mp4``.
"""

from __future__ import annotations

import subprocess
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Sequence


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


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Render dynamic views for every scene and export videos."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("gaussian_output_dynamic"),
        help="Destination directory for rendered frames and videos.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=("0", "1", "2"),
        help="List of view identifiers to convert into videos.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
        help="Experiment name used when locating canonical checkpoints.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("./data/gaussian_data"),
        help="Root directory containing per-scene Gaussian-ready assets.",
    )
    parser.add_argument(
        "--gaussian_root",
        type=Path,
        default=Path("./gaussian_output"),
        help="Root directory containing canonical Gaussian checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    views: Sequence[str] = tuple(args.views)
    exp_name: str = args.exp_name
    data_root: Path = args.data_root
    gaussian_root: Path = args.gaussian_root

    output_dir.mkdir(parents=True, exist_ok=True)
    for scene_dir in iter_scene_dirs(data_root):
        scene_name = scene_dir.name
        model_dir = gaussian_root / scene_name / exp_name
        if not model_dir.exists():
            print(f"[Skip] Canonical output missing for {scene_name}: {model_dir}")
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
                "--output_dir",
                str(output_dir),
            ]
        )

        for view_name in views:
            image_folder = output_dir / scene_name / view_name
            if not image_folder.exists():
                print(f"[Skip] Render folder missing for {scene_name}/{view_name}")
                continue
            video_path = output_dir / scene_name / f"{view_name}.mp4"
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


if __name__ == "__main__":
    main()
