#!/usr/bin/env python3
"""
Precompute per-frame LBS poses for Gaussian colour refinement.

This script mirrors the Interactive Playground's LBS deformation logic but runs
it offline so that the colour stage can reuse cached poses instead of solving
KNN/SVD operations on-the-fly.

Inputs
------
- ``--model_dir``: directory containing ``canonical_gaussians.npz`` (and
  optionally ``color_refine/canonical_gaussians_color.npz``).
- ``--inference``: motion trajectory pickle produced by the tracking pipeline.
  It must provide either ``{"x": (T, B, 3)}`` or ``{"x": ..., "prev_x": ...}``
  compatible with the Interactive Playground outputs.

Outputs
-------
- ``--output``: PyTorch file with a dictionary containing:
    * ``relations``     – bone adjacency indices (B, K_adj).
    * ``weights_idx``   – Gaussian-to-bone indices (N, K).
    * ``weights``       – Gaussian skinning weights (N, K).
    * ``pose_caches``   – mapping ``{"rollout","absolute"} -> frame pose dict``.
    * ``pose_cache``    – legacy alias that defaults to rollout cache when available.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from gaussian_splatting.dynamic_utils import (
    calc_weights_vals_from_indices,
    compute_bone_transforms,
    get_topk_indices,
    interpolate_motions,
    knn_weights,
    knn_weights_sparse,
)
from gaussian_splatting.utils.canonical_io import load_canonical_npz, resolve_canonical_npz


def parse_motion_payload(payload: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract x/prev_x trajectories from pickle payloads produced by tracking.
    The function accepts dicts, lists, or raw numpy arrays.
    """

    def _ensure_array(arr_like: Any, name: str) -> np.ndarray:
        arr = np.asarray(arr_like)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (T, N, 3); got {arr.shape}.")
        return arr.astype(np.float32)

    if isinstance(payload, dict):
        if "prev_x" in payload and "x" in payload:
            prev_x = _ensure_array(payload["prev_x"], "prev_x")
            x = _ensure_array(payload["x"], "x")
        elif "x" in payload:
            x = _ensure_array(payload["x"], "x")
            prev_x = np.roll(x, shift=1, axis=0)
            prev_x[0] = x[0]
        elif "traj" in payload:
            x = _ensure_array(payload["traj"], "traj")
            prev_x = np.roll(x, shift=1, axis=0)
            prev_x[0] = x[0]
        else:
            raise KeyError(
                "Motion payload dictionary must contain 'x', optionally with 'prev_x'."
            )
    elif isinstance(payload, (list, tuple)):
        x = _ensure_array(np.stack(payload, axis=0), "motion list")
        prev_x = np.roll(x, shift=1, axis=0)
        prev_x[0] = x[0]
    elif isinstance(payload, np.ndarray):
        x = _ensure_array(payload, "motion array")
        prev_x = np.roll(x, shift=1, axis=0)
        prev_x[0] = x[0]
    else:
        raise TypeError(f"Unsupported motion payload type: {type(payload)!r}")

    if prev_x.shape != x.shape:
        raise ValueError("prev_x and x must share the same shape.")
    return x, prev_x


def build_absolute_pose_cache(
    *,
    xyz0: torch.Tensor,
    quat0: torch.Tensor,
    bones0: torch.Tensor,
    relations: torch.Tensor,
    weights_idx: torch.Tensor,
    weights_vals: torch.Tensor,
    x: torch.Tensor,
    cache_dtype: torch.dtype,
    device: torch.device,
    chunk_size: int = 5_000,
) -> Dict[int, Dict[str, torch.Tensor]]:
    frame_count = x.shape[0]
    pose_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    n_bones = bones0.shape[0]
    n_particles = xyz0.shape[0]
    if weights_idx.shape != weights_vals.shape:
        raise ValueError(
            "weights_idx and weights_vals must have the same shape; "
            f"got {tuple(weights_idx.shape)} vs {tuple(weights_vals.shape)}."
        )
    if weights_idx.ndim != 2:
        raise ValueError(
            f"weights_idx must have shape (N, K); got {tuple(weights_idx.shape)}."
        )
    if weights_idx.shape[0] != n_particles:
        raise ValueError(
            "weights_idx first dim must match xyz0; "
            f"got {weights_idx.shape[0]} vs {n_particles}."
        )
    with torch.no_grad():
        for frame_id in range(frame_count):
            motions = x[frame_id] - bones0
            posed_xyz = torch.empty_like(xyz0)
            posed_quat = torch.empty_like(quat0)
            num_chunks = (n_particles + chunk_size - 1) // chunk_size
            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                end = min((chunk_id + 1) * chunk_size, n_particles)
                xyz_chunk = xyz0[start:end]
                quat_chunk = quat0[start:end]
                idx_chunk = weights_idx[start:end]
                vals_chunk = weights_vals[start:end]
                weights_dense = torch.zeros(
                    (end - start, n_bones), device=device, dtype=torch.float32
                )
                weights_dense[
                    torch.arange(end - start, device=device)[:, None],
                    idx_chunk,
                ] = vals_chunk
                xyz_chunk_new, quat_chunk_new, _ = interpolate_motions(
                    bones=bones0,
                    motions=motions,
                    relations=relations,
                    weights=weights_dense,
                    xyz=xyz_chunk,
                    quat=quat_chunk,
                    device=device,
                    step=frame_id,
                )
                posed_xyz[start:end] = xyz_chunk_new
                posed_quat[start:end] = quat_chunk_new

            pose_cache[int(frame_id)] = {
                "xyz": posed_xyz.detach().to(dtype=cache_dtype, device="cpu"),
                "quat": posed_quat.detach().to(dtype=cache_dtype, device="cpu"),
            }
    return pose_cache


def build_rollout_pose_cache(
    *,
    xyz0: torch.Tensor,
    quat0: torch.Tensor,
    relations: torch.Tensor,
    x: torch.Tensor,
    prev_x: torch.Tensor,
    knn_k: int,
    cache_dtype: torch.dtype,
    device: torch.device,
    chunk_size: int = 5_000,
) -> Dict[int, Dict[str, torch.Tensor]]:
    frame_count = x.shape[0]
    pose_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    all_xyz = xyz0.detach().clone()
    all_quat = quat0.detach().clone()

    with torch.no_grad():
        for frame_id in range(frame_count):
            if frame_id > 0:
                prev_ctrl = prev_x[frame_id].to(device=device, dtype=torch.float32)
                cur_ctrl = x[frame_id].to(device=device, dtype=torch.float32)
                motions = cur_ctrl - prev_ctrl
                num_chunks = (all_xyz.shape[0] + chunk_size - 1) // chunk_size
                for chunk_id in range(num_chunks):
                    start = chunk_id * chunk_size
                    end = min((chunk_id + 1) * chunk_size, all_xyz.shape[0])
                    xyz_chunk = all_xyz[start:end]
                    quat_chunk = all_quat[start:end]
                    weights_dense = knn_weights(prev_ctrl, xyz_chunk, K=knn_k)
                    xyz_chunk_new, quat_chunk_new, _ = interpolate_motions(
                        bones=prev_ctrl,
                        motions=motions,
                        relations=relations,
                        weights=weights_dense,
                        xyz=xyz_chunk,
                        quat=quat_chunk,
                        device=device,
                        step=frame_id,
                    )
                    all_xyz[start:end] = xyz_chunk_new
                    all_quat[start:end] = quat_chunk_new

            pose_cache[int(frame_id)] = {
                "xyz": all_xyz.detach().to(dtype=cache_dtype, device="cpu"),
                "quat": all_quat.detach().to(dtype=cache_dtype, device="cpu"),
            }
    return pose_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute offline LBS pose cache for colour refinement."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--K",
        type=int,
        default=16,
        help="Number of nearest bones per Gaussian.",
    )
    parser.add_argument(
        "--pose_cache_modes",
        nargs="+",
        choices=("rollout", "absolute"),
        default=("rollout", "absolute"),
        help=(
            "Pose cache variants to precompute. "
            "Supported: rollout, absolute. Default: both."
        ),
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Store cached poses in float16 to reduce disk footprint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device for computations (e.g., 'cuda:0' or 'cpu'). "
        "Defaults to CUDA when available.",
    )
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    inference_path: Path = args.inference
    output_path: Path = args.output

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not inference_path.exists():
        raise FileNotFoundError(f"Inference pickle not found: {inference_path}")

    if args.K < 1:
        raise ValueError("--K must be a positive integer.")
    selected_modes = tuple(dict.fromkeys(args.pose_cache_modes))
    if not selected_modes:
        raise ValueError("At least one pose cache mode must be selected.")

    canonical_path = resolve_canonical_npz(model_dir)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[LBS] Using canonical parameters from {canonical_path}")
    print(f"[LBS] Running pose precomputation on device: {device}")
    canonical_payload = load_canonical_npz(canonical_path, device=device)
    xyz0 = canonical_payload["xyz"]
    rot0 = canonical_payload["rotation"]
    if rot0.ndim != 2 or rot0.shape[1] not in (3, 4):
        raise ValueError(
            "Canonical rotation must have shape (N, 4) for quaternions or (N, 3) for axis-angle."
        )
    if rot0.shape[1] == 3:
        angles = torch.norm(rot0, dim=1, keepdim=True).clamp_min(1e-8)
        axis = rot0 / angles
        half_angles = angles * 0.5
        quat0 = torch.empty((rot0.shape[0], 4), device=device, dtype=torch.float32)
        quat0[:, 0] = torch.cos(half_angles).squeeze(1)
        sin_half = torch.sin(half_angles).squeeze(1)
        quat0[:, 1:] = axis * sin_half.unsqueeze(1)
    else:
        quat0 = rot0

    with open(inference_path, "rb") as f:
        motion_payload = pickle.load(f)
    x_np, prev_x_np = parse_motion_payload(motion_payload)

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)  # (T, B, 3)
    prev_x = torch.from_numpy(prev_x_np).to(device=device, dtype=torch.float32)
    T, num_bones, _ = x.shape
    print(f"[LBS] Loaded motion sequence with {T} frames and {num_bones} bones.")

    if num_bones < 1:
        raise ValueError("Motion sequence must contain at least one bone.")

    bones0 = prev_x[0]  # (B, 3)
    if num_bones == 1:
        relations = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        rel_k = min(args.K, num_bones - 1)
        relations = get_topk_indices(bones0, K=rel_k).long()
    knn_k = min(args.K, num_bones)
    if knn_k < 1:
        raise ValueError("K must be >= 1 when bones are present.")
    _, weights_idx = knn_weights_sparse(bones0, xyz0, K=knn_k)
    weights_vals = calc_weights_vals_from_indices(bones0, xyz0, weights_idx)

    bone_transform_cache = torch.zeros((T, num_bones, 4, 4), dtype=torch.float32)
    cache_dtype = torch.float16 if args.half else torch.float32
    with torch.no_grad():
        for frame_id in range(T):
            rel_motions = x[frame_id] - bones0
            transforms_t = compute_bone_transforms(
                bones=bones0,
                motions=rel_motions,
                relations=relations,
                device=device,
                step=frame_id,
            )
            bone_transform_cache[frame_id] = transforms_t.detach().cpu().to(
                dtype=torch.float32
            )
            if (frame_id + 1) % 50 == 0 or frame_id == T - 1:
                print(f"[LBS] Processed frame {frame_id + 1}/{T} for bone transforms")

    pose_caches: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    if "rollout" in selected_modes:
        print("[LBS] Building rollout pose cache")
        pose_caches["rollout"] = build_rollout_pose_cache(
            xyz0=xyz0,
            quat0=quat0,
            relations=relations,
            x=x,
            prev_x=prev_x,
            knn_k=knn_k,
            cache_dtype=cache_dtype,
            device=device,
        )
    if "absolute" in selected_modes:
        print("[LBS] Building absolute pose cache")
        pose_caches["absolute"] = build_absolute_pose_cache(
            xyz0=xyz0,
            quat0=quat0,
            bones0=bones0,
            relations=relations,
            weights_idx=weights_idx,
            weights_vals=weights_vals,
            x=x,
            cache_dtype=cache_dtype,
            device=device,
        )

    default_mode = "rollout" if "rollout" in pose_caches else next(iter(pose_caches))
    legacy_pose_cache = pose_caches[default_mode]

    payload: Dict[str, Any] = {
        "bones0": bones0.detach().cpu().to(dtype=torch.float32),
        "relations": relations.detach().cpu().long(),
        "weights_idx": weights_idx.detach().cpu().long(),
        "weights": weights_vals.detach().cpu().to(dtype=torch.float32),
        "pose_cache": legacy_pose_cache,
        "pose_caches": pose_caches,
        "bone_positions": x.detach().cpu().to(dtype=torch.float32),
        "bone_transforms": bone_transform_cache,
        "meta": {
            "frames": T,
            "bones": num_bones,
            "gaussians": xyz0.shape[0],
            "dtype": str(cache_dtype),
            "available_pose_modes": list(pose_caches.keys()),
            "default_pose_mode": default_mode,
        },
    }
    if "rollout" in pose_caches:
        payload["pose_cache_rollout"] = pose_caches["rollout"]
    if "absolute" in pose_caches:
        payload["pose_cache_absolute"] = pose_caches["absolute"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[LBS] Offline pose cache saved to {output_path}")


if __name__ == "__main__":
    main()
