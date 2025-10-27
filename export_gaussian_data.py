"""Export first-frame assets needed by the canonical Gaussian pipeline.

Inputs
------
- ``data_config.csv`` listing ``case_name, category, shape_prior`` rows.
- Processed case data under ``./data/different_types/<case_name>/`` containing RGB/depth/mask,
  ``calibrate.pkl``, ``metadata.json``, point clouds, and optional shape priors.

Outputs
-------
- Per-case folder ``./data/gaussian_data/<case_name>/`` with RGB/depth/mask copies, upsampled
  variants, human/object masks, camera metadata, observation point cloud, optional shape prior,
  and generated ``interp_poses.pkl`` ready for downstream rendering.
"""

import csv
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

base_path = Path("./data/different_types")
output_path = Path("./data/gaussian_data")
CONTROLLER_NAME = "hand"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


ensure_dir(output_path)
# generate interp_poses.pkl
print(f"[info] Generating interp_poses.pkl")
subprocess.run(
    [
        "python",
        "./gaussian_splatting/generate_interp_poses.py",
        "--root_dir",
        str(output_path),
    ],
    check=True,
)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        case_dir = base_path / case_name
        if not case_dir.exists():
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Create the directory for the case
        dest_case_dir = output_path / case_name
        ensure_dir(dest_case_dir)
        for i in range(3):
            # Copy the original RGB image
            src_rgb = case_dir / "color" / str(i) / "0.png"
            shutil.copy2(src_rgb, dest_case_dir / f"{i}.png")
            # Copy the original mask image
            # Get the mask path for the image
            with (case_dir / "mask" / f"mask_info_{i}.json").open(
                "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            mask_path = case_dir / "mask" / str(i) / str(obj_idx) / "0.png"
            shutil.copy2(mask_path, dest_case_dir / f"mask_{i}.png")
            # Prepare the high-resolution image
            subprocess.run(
                [
                    "python",
                    "./data_process/image_upscale.py",
                    "--img_path",
                    str(src_rgb),
                    "--output_path",
                    str(dest_case_dir / f"{i}_high.png"),
                    "--category",
                    category,
                ],
                check=True,
            )
            # Prepare the segmentation mask of the high-resolution image
            subprocess.run(
                [
                    "python",
                    "./data_process/segment_util_image.py",
                    "--img_path",
                    str(dest_case_dir / f"{i}_high.png"),
                    "--TEXT_PROMPT",
                    category,
                    "--output_path",
                    str(dest_case_dir / f"mask_{i}_high.png"),
                ],
                check=True,
            )

            # Copy the original depth image
            shutil.copy2(
                case_dir / "depth" / str(i) / "0.npy",
                dest_case_dir / f"{i}_depth.npy",
            )

            # Prepare the human mask for the low-resolution image and high-resolution image
            subprocess.run(
                [
                    "python",
                    "./data_process/segment_util_image.py",
                    "--img_path",
                    str(dest_case_dir / f"{i}.png"),
                    "--TEXT_PROMPT",
                    "human",
                    "--output_path",
                    str(dest_case_dir / f"mask_human_{i}.png"),
                    "--exclude_mask_path",
                    str(dest_case_dir / f"mask_{i}.png"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "./data_process/segment_util_image.py",
                    "--img_path",
                    str(dest_case_dir / f"{i}_high.png"),
                    "--TEXT_PROMPT",
                    "human",
                    "--output_path",
                    str(dest_case_dir / f"mask_human_{i}_high.png"),
                    "--exclude_mask_path",
                    str(dest_case_dir / f"mask_{i}_high.png"),
                ],
                check=True,
            )

        # Prepare the intrinsic and extrinsic parameters
        with (case_dir / "calibrate.pkl").open("rb") as f:
            c2ws = pickle.load(f)
        with (case_dir / "metadata.json").open("r", encoding="utf-8") as f:
            intrinsics = json.load(f)["intrinsics"]
        data = {}
        data["c2ws"] = c2ws
        data["intrinsics"] = intrinsics
        with (dest_case_dir / "camera_meta.pkl").open("wb") as f:
            pickle.dump(data, f)

        # Prepare the shape initialization data
        # If with shape prior, then copy the shape prior data
        if shape_prior.lower() == "true":
            shape_src = case_dir / "shape" / "matching" / "final_mesh.glb"
            if shape_src.exists():
                shutil.copy2(shape_src, dest_case_dir / "shape_prior.glb")
        # Save the original pcd data into the world coordinate system
        obs_points = []
        obs_colors = []
        pcd_path = case_dir / "pcd" / "0.npz"
        processed_mask_path = case_dir / "mask" / "processed_masks.pkl"
        data = np.load(pcd_path)
        with processed_mask_path.open("rb") as f:
            processed_masks = pickle.load(f)
        for i in range(3):
            points = data["points"][i]
            colors = data["colors"][i]
            mask = processed_masks[0][i]["object"]
            obs_points.append(points[mask])
            obs_colors.append(colors[mask])

        obs_points = np.vstack(obs_points)
        obs_colors = np.vstack(obs_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([pcd, coordinate])
        o3d.io.write_point_cloud(str(dest_case_dir / "observation.ply"), pcd)
