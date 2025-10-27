#!/usr/bin/env python3
"""
Construct placeholder alignment outputs by copying first-frame Gaussian results.

Inputs
------
- Per-frame Gaussian outputs under ``./per_frame_gaussian_output/<scene>/<frame>/<exp_name>/``.
- Rendered frames inside each first-frame directory (``test/ours_<iterations>/renders``).

Outputs
-------
- Aggregated first-frame copies in ``--output_dir/<scene>/<exp_name>/``.
- Regenerated videos in ``--video_dir/<scene>/<exp_name>.mp4`` from the copied renders.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence


def run_command(command: Sequence[str]) -> None:
    """Execute a subprocess command and raise on failure."""

    subprocess.run(list(command), check=True)


def ensure_dir(path: Path) -> None:
    """Ensure ``path`` exists, creating parents as needed."""

    path.mkdir(parents=True, exist_ok=True)


def iter_scene_dirs(per_frame_root: Path) -> Iterable[Path]:
    """Yield each scene directory under the per-frame output root."""

    for scene_dir in sorted(per_frame_root.glob("*/")):
        if scene_dir.is_dir():
            yield scene_dir


def find_first_frame(scene_dir: Path) -> Path | None:
    """Return the lowest-index frame directory within ``scene_dir``."""

    frame_dirs = sorted(
        [frame for frame in scene_dir.glob("*/") if frame.is_dir()],
        key=lambda p: p.name,
    )
    return frame_dirs[0] if frame_dirs else None


def copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree from ``src`` to ``dst`` (overwriting if it exists)."""

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder alignment: copy first-frame Gaussians and regenerate videos."
    )
    parser.add_argument(
        "--per_frame_root",
        type=Path,
        default=Path("./per_frame_gaussian_output"),
        help="Directory containing per-frame Gaussian outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./gaussian_output"),
        help="Destination directory for placeholder alignment results.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("./gaussian_output_video"),
        help="Directory to store regenerated videos.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0",
        help="Experiment suffix to copy (matches per-frame training).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Training iteration used when looking up render folders.",
    )
    args = parser.parse_args()

    per_frame_root = args.per_frame_root
    if not per_frame_root.exists():
        raise FileNotFoundError(f"Per-frame root not found: {per_frame_root}")

    ensure_dir(args.output_dir)
    ensure_dir(args.video_dir)

    for scene_dir in iter_scene_dirs(per_frame_root):
        scene_name = scene_dir.name
        first_frame = find_first_frame(scene_dir)
        if first_frame is None:
            print(
                f"Warning: no frame directories found for scene '{scene_name}', skipping."
            )
            continue

        src_model_dir = first_frame / args.exp_name
        if not src_model_dir.exists():
            print(
                f"Warning: expected {src_model_dir} missing, skipping scene '{scene_name}'."
            )
            continue

        dst_model_dir = args.output_dir / scene_name / args.exp_name
        ensure_dir(dst_model_dir.parent)
        copy_tree(src_model_dir, dst_model_dir)

        renders_dir = dst_model_dir / "test" / f"ours_{args.iterations}" / "renders"
        if not renders_dir.exists():
            print(
                f"Warning: renders not found at {renders_dir}, skipping video regeneration."
            )
            continue

        ensure_dir(args.video_dir / scene_name)
        video_path = args.video_dir / scene_name / f"{args.exp_name}.mp4"
        run_command(
            [
                "python",
                "gaussian_splatting/img2video.py",
                "--image_folder",
                str(renders_dir),
                "--video_path",
                str(video_path),
            ]
        )


if __name__ == "__main__":
    main()
