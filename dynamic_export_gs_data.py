#!/usr/bin/env python3
"""
Export Gaussian-ready assets for every frame of every case.

Inputs
------
- Processed case folders in ``data/different_types/<case>/`` containing:
  * ``color/<cam>/<frame>.png`` RGB images.
  * ``depth/<cam>/<frame>.npy`` depth maps.
  * ``mask/**`` segmentation artefacts.
  * ``pcd/<frame>.npz`` fused point clouds and ``mask/processed_masks.pkl``.
  * ``calibrate.pkl`` (camera extrinsics) and ``metadata.json`` (intrinsics, image size).
  * Optional ``shape/matching/final_mesh.glb`` mesh prior.

Outputs
-------
- Legacy first-frame export copied to ``data/gaussian_data/<case>/``.
- Per-frame Gaussian datasets in ``per_frame_gaussian_data/<frame>/<case>/`` including RGB/depth/mask imagery,
  ``camera_meta.pkl``, ``observation.ply``, and ``shape_prior.glb`` when available.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


def run_command(cmd: list[str]) -> None:
    """Execute an external command and raise immediately on failure."""

    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def collect_frame_ids(case_dir: Path) -> list[int]:
    """
    Return all frame indices with a corresponding point cloud npz file.
    """

    pcd_dir = case_dir / "pcd"
    if not pcd_dir.exists():
        print(f"Warning: {pcd_dir} missing; skipping case {case_dir.name}.")
        return []

    frame_ids: list[int] = []
    for npz_file in pcd_dir.glob("*.npz"):
        try:
            frame_ids.append(int(npz_file.stem))
        except ValueError:
            continue
    return sorted(frame_ids)


def required_assets_exist(
    case_dir: Path, frame_id: int, cam_indices: Iterable[int]
) -> bool:
    """
    Check that RGB and depth files exist for the supplied frame/cameras.
    """

    missing: list[Path] = []
    for cam in cam_indices:
        rgb_path = case_dir / "color" / str(cam) / f"{frame_id}.png"
        depth_path = case_dir / "depth" / str(cam) / f"{frame_id}.npy"
        if not rgb_path.exists():
            missing.append(rgb_path)
        if not depth_path.exists():
            missing.append(depth_path)
    if missing:
        print(
            f"Skipping frame {frame_id} in {case_dir.name}; missing: "
            + ", ".join(str(path) for path in missing)
        )
        return False
    return True


def main() -> None:
    root = Path(__file__).resolve().parent

    # Keep the legacy single-frame export to avoid breaking downstream scripts.
    run_command(["python", str(root / "export_gaussian_data.py")])

    per_frame_root = root / "per_frame_gaussian_data"
    ensure_dir(per_frame_root)

    data_root = root / "data" / "different_types"
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    required_cams = [0, 1, 2]

    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        color_dir = case_dir / "color"
        if not color_dir.exists():
            print(f"Warning: {color_dir} missing; skipping case {case_name}.")
            continue
        if any(not (color_dir / str(cam)).exists() for cam in required_cams):
            print(f"Warning: {case_name} missing required camera folders; skipping.")
            continue

        frame_ids = collect_frame_ids(case_dir)
        if not frame_ids:
            continue

        print(f"Processing case {case_name}: {len(frame_ids)} frames detected.")
        for frame_id in frame_ids:
            if not required_assets_exist(case_dir, frame_id, required_cams):
                continue

            per_frame_output_dir = per_frame_root / str(frame_id)
            ensure_dir(per_frame_output_dir)

            run_command(
                [
                    "python",
                    str(root / "export_gaussian_data_one_frame_one_case.py"),
                    "--case",
                    case_name,
                    "--frame",
                    str(frame_id),
                    "--base_dir",
                    str(data_root),
                    "--output_dir",
                    str(per_frame_output_dir),
                    "--no-generate_high_png",
                ]
            )


if __name__ == "__main__":
    main()
