"""Create interpolated camera trajectories for QQTT scenes.

Inputs
------
- ``--root_dir`` pointing to one or more scene folders (default ``./data/gaussian_data``).
  Each scene must contain ``camera_meta.pkl`` with the original training camera poses.

Outputs
-------
- For every scene directory under ``root_dir`` a new ``interp_poses.pkl`` is written. It
  stores smoothly interpolated camera poses (c2w matrices) used during evaluation/rendering.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.interpolate


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct look-at view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
) -> np.ndarray:
    """
    Create a smooth spline path between input keyframe camera poses.

    Adapted from https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    Poses are (3×4) matrices. The spline operates on (position, look-at, up) tuples.
    """

    def poses_to_points(poses_: np.ndarray, dist: float) -> np.ndarray:
        """Convert pose matrices to (position, lookat, up) representation."""
        pos = poses_[:, :3, -1]
        lookat = poses_[:, :3, -1] - dist * poses_[:, :3, 2]
        up = poses_[:, :3, -1] + dist * poses_[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points: np.ndarray) -> np.ndarray:
        """Convert (position, lookat, up) representation back to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points: np.ndarray, n: int, k: int, s: float) -> np.ndarray:
        """Run multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        return np.reshape(new_points.T, (n, sh[1], sh[2]))

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interpolated camera trajectories for QQTT scenes."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("./data/gaussian_data"),
        help="Root directory containing per-scene folders (default: ./data/gaussian_data).",
    )
    args = parser.parse_args()

    if not args.root_dir.is_dir():
        raise FileNotFoundError(f"Scene root directory not found: {args.root_dir}")

    for scene_dir in sorted(p for p in args.root_dir.iterdir() if p.is_dir()):
        print(f"Processing {scene_dir.name}")
        camera_path = scene_dir / "camera_meta.pkl"
        with camera_path.open("rb") as f:
            camera_meta = pickle.load(f)
        c2ws = camera_meta["c2ws"]

        pose_0, pose_1, pose_2 = c2ws[:3]
        n_interp = 50

        poses_01 = np.stack([pose_0, pose_1], 0)[:, :3, :]
        interp_poses_01 = generate_interpolated_path(poses_01, n_interp)
        poses_12 = np.stack([pose_1, pose_2], 0)[:, :3, :]
        interp_poses_12 = generate_interpolated_path(poses_12, n_interp)
        poses_20 = np.stack([pose_2, pose_0], 0)[:, :3, :]
        interp_poses_20 = generate_interpolated_path(poses_20, n_interp)

        interp_poses = np.concatenate(
            [interp_poses_01, interp_poses_12, interp_poses_20], 0
        )
        output_poses = [
            np.vstack([pose, np.array([0, 0, 0, 1])]) for pose in interp_poses
        ]

        with (scene_dir / "interp_poses.pkl").open("wb") as f:
            pickle.dump(output_poses, f)


if __name__ == "__main__":
    main()
