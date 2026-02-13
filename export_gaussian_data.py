#!/usr/bin/env python3
"""Export first-frame assets needed by the canonical Gaussian pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_OUTPUT_PATH = "./data/gaussian_data"
DEFAULT_CONFIG_PATH = "./data_config.csv"
CONTROLLER_NAME = "hand"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_with_retry(cmd: list[str], attempts: int = 5, delay: float = 2.0) -> None:
    """Execute `cmd` with retries to mitigate sporadic crashes."""

    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == attempts:
                raise
            cmd_str = " ".join(str(part) for part in cmd)
            print(
                f"[warn] Command failed (attempt {attempt}/{attempts}): {cmd_str}\n"
                f"       Retrying after {delay:.1f}s..."
            )
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export first-frame Gaussian assets for cases listed in config."
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def iter_config_rows(config_path: Path) -> Iterable[tuple[str, str, str]]:
    with config_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader, start=1):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Malformed row {idx} in {config_path}: expected >=3 columns."
                )
            case_name = row[0].strip()
            category = row[1].strip()
            shape_prior = row[2].strip()
            if not case_name:
                raise ValueError(
                    f"Malformed row {idx} in {config_path}: empty case name."
                )
            yield case_name, category, shape_prior


def process_case(
    *,
    case_name: str,
    category: str,
    shape_prior: str,
    base_path: Path,
    output_path: Path,
    python_bin: str,
) -> None:
    case_dir = base_path / case_name
    dest_case_dir = output_path / case_name
    ensure_dir(dest_case_dir)

    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    for cam_idx in range(3):
        src_rgb = case_dir / "color" / str(cam_idx) / "0.png"
        shutil.copy2(src_rgb, dest_case_dir / f"{cam_idx}.png")

        with (case_dir / "mask" / f"mask_info_{cam_idx}.json").open(
            "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        if obj_idx is None:
            raise ValueError(f"No object mask found for case={case_name}, cam={cam_idx}.")

        mask_path = case_dir / "mask" / str(cam_idx) / str(obj_idx) / "0.png"
        shutil.copy2(mask_path, dest_case_dir / f"mask_{cam_idx}.png")

        run_with_retry(
            [
                python_bin,
                str(ROOT / "data_process" / "image_upscale.py"),
                "--img_path",
                str(src_rgb),
                "--output_path",
                str(dest_case_dir / f"{cam_idx}_high.png"),
                "--category",
                category,
            ]
        )
        run_with_retry(
            [
                python_bin,
                str(ROOT / "data_process" / "segment_util_image.py"),
                "--img_path",
                str(dest_case_dir / f"{cam_idx}_high.png"),
                "--TEXT_PROMPT",
                category,
                "--output_path",
                str(dest_case_dir / f"mask_{cam_idx}_high.png"),
            ]
        )

        shutil.copy2(
            case_dir / "depth" / str(cam_idx) / "0.npy",
            dest_case_dir / f"{cam_idx}_depth.npy",
        )

        run_with_retry(
            [
                python_bin,
                str(ROOT / "data_process" / "segment_util_image.py"),
                "--img_path",
                str(dest_case_dir / f"{cam_idx}.png"),
                "--TEXT_PROMPT",
                "human",
                "--output_path",
                str(dest_case_dir / f"mask_human_{cam_idx}.png"),
                "--exclude_mask_path",
                str(dest_case_dir / f"mask_{cam_idx}.png"),
            ]
        )
        run_with_retry(
            [
                python_bin,
                str(ROOT / "data_process" / "segment_util_image.py"),
                "--img_path",
                str(dest_case_dir / f"{cam_idx}_high.png"),
                "--TEXT_PROMPT",
                "human",
                "--output_path",
                str(dest_case_dir / f"mask_human_{cam_idx}_high.png"),
                "--exclude_mask_path",
                str(dest_case_dir / f"mask_{cam_idx}_high.png"),
            ]
        )

    with (case_dir / "calibrate.pkl").open("rb") as f:
        c2ws = pickle.load(f)
    with (case_dir / "metadata.json").open("r", encoding="utf-8") as f:
        intrinsics = json.load(f)["intrinsics"]
    with (dest_case_dir / "camera_meta.pkl").open("wb") as f:
        pickle.dump({"c2ws": c2ws, "intrinsics": intrinsics}, f)

    if shape_prior.lower() == "true":
        shape_src = case_dir / "shape" / "matching" / "final_mesh.glb"
        if shape_src.exists():
            shutil.copy2(shape_src, dest_case_dir / "shape_prior.glb")

    pcd_path = case_dir / "pcd" / "0.npz"
    processed_mask_path = case_dir / "mask" / "processed_masks.pkl"
    data = np.load(pcd_path)
    with processed_mask_path.open("rb") as f:
        processed_masks = pickle.load(f)

    obs_points = []
    obs_colors = []
    for cam_idx in range(3):
        points = data["points"][cam_idx]
        colors = data["colors"][cam_idx]
        mask = processed_masks[0][cam_idx]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    pcd.colors = o3d.utility.Vector3dVector(obs_colors)
    o3d.io.write_point_cloud(str(dest_case_dir / "observation.ply"), pcd)


def main() -> None:
    args = parse_args()
    python_bin = sys.executable
    base_path = Path(args.base_path).resolve()
    output_path = Path(args.output_path).resolve()
    config_path = Path(args.config_path).resolve()

    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Base path not found or not directory: {base_path}")
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config path not found or not file: {config_path}")

    ensure_dir(output_path)

    processed_case_names: list[str] = []
    seen_cases: set[str] = set()
    for case_name, category, shape_prior in iter_config_rows(config_path):
        if case_name in seen_cases:
            print(
                f"[warn] Duplicate case '{case_name}' in {config_path}; skipping duplicate row."
            )
            continue
        seen_cases.add(case_name)

        case_dir = base_path / case_name
        if not case_dir.exists():
            print(f"[warn] Input case missing on disk, skipping: {case_dir}")
            continue

        process_case(
            case_name=case_name,
            category=category,
            shape_prior=shape_prior,
            base_path=base_path,
            output_path=output_path,
            python_bin=python_bin,
        )
        processed_case_names.append(case_name)

    if not processed_case_names:
        print("[info] No valid cases were exported; skip interp_poses generation.")
        return

    print("[info] Generating interp_poses.pkl")
    run_with_retry(
        [
            python_bin,
            str(ROOT / "gaussian_splatting" / "generate_interp_poses.py"),
            "--root_dir",
            str(output_path),
            "--scenes",
            *processed_case_names,
        ]
    )


if __name__ == "__main__":
    main()
