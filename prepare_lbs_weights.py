#!/usr/bin/env python3
"""
Generate lightweight LBS metadata for dynamic Gaussian colour refinement.

Inputs
------
- ``--model_dir``: canonical model directory containing ``canonical_gaussians.npz``.
- ``--motion_source``: motion trajectory file (defaults to
  ``<model_dir>/lbs/inference.pkl``).
- ``--k_bones``: number of bones/anchors to sample from the canonical Gaussians.
- ``--k_skin``: number of nearest bones for each Gaussian (skinning degree).

Outputs
-------
- ``<model_dir>/lbs_data.pt``: dictionary containing
    * ``bones0`` (k_bones × 3),
    * ``relations`` (k_bones × k_adj),
    * ``skin_indices`` (N × k_skin),
    * ``skin_weights`` (N × k_skin),
    * ``motions`` (T × k_bones × 3).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from gaussian_splatting.skinning import build_skinning_weights


def farthest_point_sample(points: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest point sampling on CPU/GPU."""
    device = points.device
    N = points.shape[0]
    if k >= N:
        return points

    centroids = torch.zeros((k,), dtype=torch.long, device=device)
    distance = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)

    for i in range(k):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance)
    return points[centroids]


def build_relations(bones: torch.Tensor, k_adj: int = 8) -> torch.Tensor:
    """kNN adjacency for bones."""
    dist = torch.cdist(bones, bones)
    idx = dist.topk(k_adj + 1, largest=False).indices[:, 1:]
    return idx


def load_motion(source: Path) -> np.ndarray:
    with source.open("rb") as f:
        data = pickle.load(f)
    return np.asarray(data)


def compute_bone_motions(
    bones0: torch.Tensor,
    motion_source: Path,
) -> torch.Tensor:
    traj = load_motion(motion_source).astype(np.float32)  # (T, N_ctrl, 3)
    if traj.ndim != 3:
        raise ValueError("Motion source must contain a (T, N, 3) array.")
    T, N_ctrl, _ = traj.shape

    # Anchor each bone to the nearest control point (frame 0).
    ctrl0 = traj[0]  # (N_ctrl, 3)
    bones_np = bones0.cpu().numpy()
    dists = np.linalg.norm(bones_np[:, None, :] - ctrl0[None, :, :], axis=-1)
    nearest_idx = dists.argmin(axis=1)  # (B,)
    selected_ctrl = traj[:, nearest_idx, :]  # (T, B, 3)
    motions = selected_ctrl - bones_np[None, :, :]
    return torch.from_numpy(motions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LBS weights and motions for dynamic GS colour refinement."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument(
        "--motion_source",
        type=Path,
        default=None,
        help="Path to motion trajectory (defaults to <model_dir>/lbs/inference.pkl).",
    )
    parser.add_argument("--k_bones", type=int, default=64)
    parser.add_argument("--k_skin", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    motion_source: Path = (
        args.motion_source
        if args.motion_source is not None
        else model_dir / "lbs" / "inference.pkl"
    )
    output_path = args.output or (model_dir / "lbs_data.pt")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    canonical_path = model_dir / "canonical_gaussians.npz"
    if not canonical_path.exists():
        raise FileNotFoundError(f"canonical_gaussians.npz missing at {canonical_path}")
    if not motion_source.exists():
        raise FileNotFoundError(f"Motion source not found: {motion_source}")

    canonical = np.load(canonical_path)
    xyz0 = torch.from_numpy(canonical["xyz"]).float()

    bones0 = farthest_point_sample(xyz0, args.k_bones)
    relations = build_relations(bones0, min(8, args.k_bones - 1))
    skin_indices, skin_weights = build_skinning_weights(xyz0, bones0, args.k_skin)
    motions = compute_bone_motions(bones0, motion_source)

    payload = {
        "bones0": bones0.cpu(),
        "relations": relations.cpu(),
        "skin_indices": skin_indices.cpu(),
        "skin_weights": skin_weights.cpu(),
        "motions": motions.cpu(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[LBS] Saved metadata to {output_path}")


if __name__ == "__main__":
    main()
