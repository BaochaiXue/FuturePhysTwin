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
    * ``pose_cache``    – mapping ``frame_id -> {"xyz": tensor, "quat": tensor}``.
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
    get_topk_indices,
    interpolate_motions_speedup,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute offline LBS pose cache for colour refinement."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--K", type=int, default=16, help="Number of nearest bones per Gaussian.")
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
    weights = calc_weights_vals_from_indices(bones0, xyz0, weights_idx)

    pose_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    cache_dtype = torch.float16 if args.half else torch.float32

    torch.set_grad_enabled(False)
    for t in range(T):
        bones_t = prev_x[t]
        motions_t = x[t] - bones_t
        posed_xyz, posed_quat, _ = interpolate_motions_speedup(
            bones=bones_t,
            motions=motions_t,
            relations=relations,
            weights=weights,
            weights_indices=weights_idx,
            xyz=xyz0,
            quat=quat0,
        )
        pose_cache[int(t)] = {
            "xyz": posed_xyz.detach().to(dtype=cache_dtype, device="cpu"),
            "quat": posed_quat.detach().to(dtype=cache_dtype, device="cpu"),
        }
        if (t + 1) % 50 == 0 or t == T - 1:
            print(f"[LBS] Processed frame {t + 1}/{T}")

    payload: Dict[str, Any] = {
        "relations": relations.detach().cpu().long(),
        "weights_idx": weights_idx.detach().cpu().long(),
        "weights": weights.detach().cpu().to(dtype=torch.float32),
        "pose_cache": pose_cache,
        "meta": {
            "frames": T,
            "bones": num_bones,
            "gaussians": xyz0.shape[0],
            "dtype": str(cache_dtype),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[LBS] Offline pose cache saved to {output_path}")


if __name__ == "__main__":
    main()
