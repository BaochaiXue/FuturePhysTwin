from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.optimize import minimize
from scipy.spatial import KDTree, cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from match_pairs import image_pair_matching  # noqa: E402
from utils.align_util import (  # noqa: E402
    as_mesh,
    plot_image_with_points,
    plot_mesh_with_points,
    project_2d_to_3d,
    render_image,
    render_multi_images,
    select_point,
)


MIN_VIEW_INLIERS = 12
MIN_TOTAL_INLIERS = 36
DEFAULT_VIEW_INDICES = ["0", "1", "2"]
DEFAULT_VERTEX_TO_OBS_GATE = 0.035
DEFAULT_OBS_TO_VERTEX_GATE = 0.015
SIM3_SCALE_MULTIPLIERS = (0.85, 0.90, 0.95, 1.00, 1.05)
QUALITY_TOLERANCE = 1e-4


@dataclass
class ViewData:
    view: str
    index: int
    image_path: Path
    mask_path: Path
    raw_img: np.ndarray
    mask_img: np.ndarray
    intrinsic: np.ndarray
    c2w: np.ndarray
    w2c: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    object_mask: np.ndarray
    depth: np.ndarray | None


@dataclass
class ViewMatch:
    view: str
    view_index: int
    best_color: np.ndarray
    best_depth: np.ndarray
    best_pose: np.ndarray
    render_intrinsic: np.ndarray
    match_result: dict[str, np.ndarray]
    bbox: tuple[int, int, int, int]
    mesh_points: np.ndarray
    raw_pixels: np.ndarray
    target_pixels: np.ndarray
    target_world: np.ndarray
    inliers: np.ndarray
    mesh2camera: np.ndarray
    scale: float
    mesh2world: np.ndarray
    reprojection_error: float
    match_count: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align an MV-SAM3D shape prior with multi-view frame-0 evidence."
    )
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--controller_name", required=True)
    parser.add_argument("--view_indices", default=None)
    parser.add_argument("--mesh_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--force_rematch", action="store_true")
    parser.add_argument("--max_render_faces", type=int, default=50000)
    parser.add_argument("--silhouette_iters", type=int, default=80)
    parser.add_argument("--depth_weight", type=float, default=0.2)
    parser.add_argument("--silhouette_weight", type=float, default=1.0)
    parser.add_argument("--pcd_weight", type=float, default=0.5)
    parser.add_argument("--vertex_to_obs_gate", type=float, default=DEFAULT_VERTEX_TO_OBS_GATE)
    parser.add_argument("--obs_to_vertex_gate", type=float, default=DEFAULT_OBS_TO_VERTEX_GATE)
    parser.add_argument("--prune_far_dist", type=float, default=DEFAULT_VERTEX_TO_OBS_GATE)
    parser.add_argument("--disable_ray_arap", action="store_true")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_mean(values: np.ndarray, delta: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    abs_values = np.abs(values)
    quadratic = np.minimum(abs_values, delta)
    linear = abs_values - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))


def parse_view_indices(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def load_manifest_views(case_dir: Path) -> list[str] | None:
    manifest_path = case_dir / "shape" / "mvsam3d" / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    views = manifest.get("view_indices")
    if isinstance(views, list) and views:
        return [str(view) for view in views]
    return None


def resolve_view_indices(case_dir: Path, requested: str | None) -> list[str]:
    views = parse_view_indices(requested)
    if views is None:
        views = load_manifest_views(case_dir)
    if views is None:
        views = DEFAULT_VIEW_INDICES
    missing = [view for view in views if not (case_dir / "color" / view / "0.png").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing frame-0 color view(s): "
            + ", ".join(str(case_dir / "color" / view / "0.png") for view in missing)
        )
    return views


def select_single_object(mask_info_path: Path, controller_name: str) -> tuple[str, str]:
    with mask_info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    candidates: list[tuple[str, str]] = []
    for key, value in data.items():
        label = str(value.get("label", value) if isinstance(value, dict) else value).strip()
        if label.casefold() != controller_name.casefold():
            candidates.append((str(key), label))
    if not candidates:
        raise ValueError(f"No non-controller object found in {mask_info_path}")
    if len(candidates) > 1:
        raise ValueError(f"More than one non-controller object found in {mask_info_path}: {candidates}")
    return candidates[0]


def load_view_data(
    case_dir: Path,
    view_indices: list[str],
    controller_name: str,
) -> tuple[list[ViewData], list[np.ndarray], list[np.ndarray]]:
    with (case_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    intrinsics = np.asarray(metadata["intrinsics"])
    c2ws = pickle.load((case_dir / "calibrate.pkl").open("rb"))
    pcd = np.load(case_dir / "pcd" / "0.npz")
    with (case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)

    views: list[ViewData] = []
    obs_points: list[np.ndarray] = []
    obs_colors: list[np.ndarray] = []
    for view in view_indices:
        view_idx = int(view)
        image_path = case_dir / "color" / view / "0.png"
        raw_img = cv2.imread(str(image_path))
        if raw_img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        object_idx, _ = select_single_object(
            case_dir / "mask" / f"mask_info_{view}.json", controller_name
        )
        mask_path = case_dir / "mask" / view / object_idx / "0.png"
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None or not np.any(mask_img > 0):
            raise ValueError(f"Missing or empty object mask: {mask_path}")

        points = pcd["points"][view_idx]
        colors = pcd["colors"][view_idx]
        object_mask = processed_masks[0][view_idx]["object"]
        obs_points.append(points[object_mask])
        obs_colors.append(colors[object_mask])

        depth_path = case_dir / "depth" / view / "0.npy"
        depth = None
        if depth_path.exists():
            depth = np.load(depth_path) / 1000.0

        c2w = np.asarray(c2ws[view_idx])
        views.append(
            ViewData(
                view=view,
                index=view_idx,
                image_path=image_path,
                mask_path=mask_path,
                raw_img=raw_img,
                mask_img=mask_img,
                intrinsic=intrinsics[view_idx],
                c2w=c2w,
                w2c=np.linalg.inv(c2w),
                points=points,
                colors=colors,
                object_mask=object_mask,
                depth=depth,
            )
        )

    return views, obs_points, obs_colors


def crop_masked_image(raw_img: np.ndarray, mask_img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox_coords = np.argwhere(mask_img > 0.8 * 255)
    if bbox_coords.size == 0:
        raise ValueError("Object mask is empty after thresholding")
    bbox = (
        int(np.min(bbox_coords[:, 1])),
        int(np.min(bbox_coords[:, 0])),
        int(np.max(bbox_coords[:, 1])),
        int(np.max(bbox_coords[:, 0])),
    )
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
    bbox = (
        max(0, int(center[0] - size // 2)),
        max(0, int(center[1] - size // 2)),
        min(raw_img.shape[1], int(center[0] + size // 2)),
        min(raw_img.shape[0], int(center[1] + size // 2)),
    )
    crop_img = raw_img.copy()
    crop_img[mask_img <= 0] = 0
    crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    return crop_img, bbox


def maybe_write_render_mesh(mesh_path: Path, mesh: trimesh.Trimesh, debug_dir: Path, max_faces: int) -> Path:
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh_path
    render_path = debug_dir / "render_mesh.glb"
    if render_path.exists():
        return render_path
    simplified = mesh.simplify_quadric_decimation(max_faces)
    simplified.export(render_path)
    return render_path


def solve_scale(mesh_cam: np.ndarray, target_cam: np.ndarray) -> float:
    denom = float(np.sum(mesh_cam * mesh_cam))
    if denom <= 1e-12:
        return 1.0
    scale = float(np.sum(mesh_cam * target_cam) / denom)
    return max(scale, 1e-6)


def project_world_to_view(points_world: np.ndarray, view: ViewData) -> tuple[np.ndarray, np.ndarray]:
    hom = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    points_cam = (view.w2c @ hom.T).T[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    pixels = np.zeros((points_world.shape[0], 2), dtype=np.float64)
    pixels[:, 0] = points_cam[:, 0] * view.intrinsic[0, 0] / np.maximum(z, 1e-6) + view.intrinsic[0, 2]
    pixels[:, 1] = points_cam[:, 1] * view.intrinsic[1, 1] / np.maximum(z, 1e-6) + view.intrinsic[1, 2]
    valid &= pixels[:, 0] >= 0
    valid &= pixels[:, 1] >= 0
    valid &= pixels[:, 0] < view.raw_img.shape[1]
    valid &= pixels[:, 1] < view.raw_img.shape[0]
    return pixels, valid


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    hom = np.hstack([points, np.ones((points.shape[0], 1))])
    return (matrix @ hom.T).T[:, :3]


def matrix_to_params(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:3, :3]
    scale = float(np.cbrt(max(np.linalg.det(linear), 1e-12)))
    rotation = linear / max(scale, 1e-12)
    rvec, _ = cv2.Rodrigues(rotation.astype(np.float64))
    return np.array(
        [rvec[0, 0], rvec[1, 0], rvec[2, 0], *matrix[:3, 3], math.log(max(scale, 1e-8))],
        dtype=np.float64,
    )


def params_to_matrix(params: np.ndarray) -> np.ndarray:
    rvec = np.asarray(params[:3], dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    scale = math.exp(float(params[6]))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = params[3:6]
    return matrix


def scaled_sim3_matrix(matrix: np.ndarray, scale_multiplier: float) -> np.ndarray:
    scaled = matrix.copy()
    scaled[:3, :3] *= scale_multiplier
    return scaled


def solve_view_match_candidate(
    view: ViewData,
    match_result: dict[str, np.ndarray],
    candidate_idx: int,
    color: np.ndarray,
    depth: np.ndarray,
    camera_pose: np.ndarray,
    render_intrinsic: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[ViewMatch | None, dict[str, Any]]:
    valid_matches = match_result["matches"] > -1
    match_count = int(np.sum(valid_matches))
    record: dict[str, Any] = {
        "candidate_idx": int(candidate_idx),
        "match_count": match_count,
        "status": "started",
    }
    if match_count < MIN_VIEW_INLIERS:
        record["status"] = "too_few_2d_matches"
        return None, record

    render_points = match_result["keypoints0"][valid_matches]
    mesh_points, valid_depth = project_2d_to_3d(
        render_points, depth, render_intrinsic, camera_pose
    )
    raw_pixels_box = match_result["keypoints1"][match_result["matches"][valid_matches]]
    raw_pixels_box = raw_pixels_box[valid_depth]
    raw_pixels = raw_pixels_box + np.array([bbox[0], bbox[1]])
    record["depth_valid_points"] = int(mesh_points.shape[0])
    if mesh_points.shape[0] < MIN_VIEW_INLIERS:
        record["status"] = "too_few_depth_valid_matches"
        return None, record

    target_pixels, target_world = select_point(view.points, raw_pixels, view.object_mask)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float32(mesh_points),
        np.float32(raw_pixels),
        np.float32(view.intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=400,
        reprojectionError=20.0,
        confidence=0.995,
    )
    inlier_count = 0 if inliers is None else int(len(inliers))
    record["pnp_success"] = bool(success)
    record["pnp_inliers"] = inlier_count
    if not success or inliers is None or inlier_count < MIN_VIEW_INLIERS:
        record["status"] = "too_few_pnp_inliers"
        return None, record

    inlier_ids = inliers.reshape(-1)
    cv2.solvePnP(
        np.float32(mesh_points[inlier_ids]),
        np.float32(raw_pixels[inlier_ids]),
        np.float32(view.intrinsic),
        np.zeros(4, dtype=np.float32),
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    projected, _ = cv2.projectPoints(
        np.float32(mesh_points[inlier_ids]),
        rvec,
        tvec,
        view.intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    reprojection_error = float(
        np.linalg.norm(raw_pixels[inlier_ids] - projected.reshape(-1, 2), axis=1).mean()
    )

    rotation, _ = cv2.Rodrigues(rvec)
    mesh2camera = np.eye(4, dtype=np.float64)
    mesh2camera[:3, :3] = rotation
    mesh2camera[:3, 3] = tvec.reshape(-1)

    mesh_cam = transform_points(mesh2camera, mesh_points[inlier_ids])
    target_cam = transform_points(view.w2c, target_world[inlier_ids])
    scale = solve_scale(mesh_cam, target_cam)
    scale_matrix = np.eye(4, dtype=np.float64) * scale
    scale_matrix[3, 3] = 1.0
    mesh2world = view.c2w @ scale_matrix @ mesh2camera

    record.update(
        {
            "status": "accepted",
            "reprojection_error": reprojection_error,
            "scale": float(scale),
        }
    )
    return (
        ViewMatch(
            view=view.view,
            view_index=view.index,
            best_color=color,
            best_depth=depth,
            best_pose=camera_pose,
            render_intrinsic=render_intrinsic,
            match_result=match_result,
            bbox=bbox,
            mesh_points=mesh_points,
            raw_pixels=raw_pixels,
            target_pixels=target_pixels,
            target_world=target_world,
            inliers=inlier_ids,
            mesh2camera=mesh2camera,
            scale=scale,
            mesh2world=mesh2world,
            reprojection_error=reprojection_error,
            match_count=match_count,
        ),
        record,
    )


def render_candidate_matches(
    view: ViewData,
    mesh: trimesh.Trimesh,
    render_mesh_path: Path,
    mesh_hash: str,
    output_dir: Path,
    force_rematch: bool,
) -> ViewMatch | None:
    view_dir = output_dir / f"view_{view.view}"
    ensure_dir(view_dir)
    cache_path = view_dir / "best_match.pkl"
    image_hash = sha1_file(view.image_path)
    mask_hash = sha1_file(view.mask_path)

    if cache_path.exists() and not force_rematch:
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            meta = cached.get("meta", {})
            if (
                meta.get("mesh_sha1") == mesh_hash
                and meta.get("image_sha1") == image_hash
                and meta.get("mask_sha1") == mask_hash
            ):
                payload = cached["payload"]
                if isinstance(payload, ViewMatch):
                    return payload
                return ViewMatch(**payload)
        except Exception:
            pass

    fov = 2 * np.arctan(view.raw_img.shape[1] / (2 * view.intrinsic[0, 0]))
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)
    colors, depths, camera_poses, render_intrinsic = render_multi_images(
        str(render_mesh_path),
        view.raw_img.shape[1],
        view.raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    crop_img, bbox = crop_masked_image(view.raw_img, view.mask_img)
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    best_idx, match_result = image_pair_matching(
        grays,
        crop_img,
        view_dir,
        viz_best=True,
        save=True,
        cache=not force_rematch,
    )
    if hasattr(match_result, "files"):
        match_result = {key: np.asarray(match_result[key]) for key in match_result.files}

    candidate_records: list[dict[str, Any]] = []
    candidate_indices: list[int] = []
    for candidate_path in sorted(view_dir.glob("matches_*.npz")):
        try:
            candidate_indices.append(int(candidate_path.stem.split("_")[-1]))
        except ValueError:
            continue
    if best_idx not in candidate_indices:
        candidate_indices.append(int(best_idx))

    def candidate_rank(candidate_idx: int) -> tuple[int, float]:
        candidate_path = view_dir / f"matches_{candidate_idx}.npz"
        if not candidate_path.exists():
            return (0, 0.0)
        with np.load(candidate_path) as data:
            matches = np.asarray(data["matches"])
            conf = np.asarray(data["match_confidence"])
        valid = matches > -1
        confidence = float(np.mean(conf[valid])) if np.any(valid) else 0.0
        return (int(np.sum(valid)), confidence)

    ranked_candidates = sorted(
        set(candidate_indices),
        key=lambda idx: (*candidate_rank(idx), -abs(idx - int(best_idx))),
        reverse=True,
    )
    best_payload: ViewMatch | None = None
    best_record: dict[str, Any] | None = None
    for candidate_idx in ranked_candidates:
        candidate_path = view_dir / f"matches_{candidate_idx}.npz"
        if candidate_path.exists():
            with np.load(candidate_path) as data:
                candidate_result = {key: np.asarray(data[key]) for key in data.files}
        elif candidate_idx == best_idx:
            candidate_result = match_result
        else:
            continue
        payload, record = solve_view_match_candidate(
            view,
            candidate_result,
            candidate_idx,
            colors[candidate_idx],
            depths[candidate_idx],
            camera_poses[candidate_idx].cpu().numpy(),
            render_intrinsic,
            bbox,
        )
        candidate_records.append(record)
        if payload is None:
            continue
        if best_payload is None:
            best_payload = payload
            best_record = record
            continue
        assert best_record is not None
        current_key = (
            int(record["pnp_inliers"]),
            -float(record["reprojection_error"]),
            int(record["match_count"]),
        )
        best_key = (
            int(best_record["pnp_inliers"]),
            -float(best_record["reprojection_error"]),
            int(best_record["match_count"]),
        )
        if current_key > best_key:
            best_payload = payload
            best_record = record

    with (view_dir / "pnp_candidates.json").open("w") as handle:
        json.dump(candidate_records, handle, indent=2)
    if best_payload is None:
        return None
    payload = best_payload
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "meta": {
                    "mesh_sha1": mesh_hash,
                    "image_sha1": image_hash,
                    "mask_sha1": mask_hash,
                },
                "payload": payload.__dict__,
            },
            handle,
        )
    return payload


def silhouette_depth_score(points_world: np.ndarray, views: list[ViewData]) -> tuple[float, float]:
    iou_losses: list[float] = []
    depth_losses: list[float] = []
    for view in views:
        pixels, valid = project_world_to_view(points_world, view)
        if not np.any(valid):
            iou_losses.append(1.0)
            continue
        valid_pixels = np.floor(pixels[valid]).astype(np.int32)
        valid_pixels[:, 0] = np.clip(valid_pixels[:, 0], 0, view.raw_img.shape[1] - 1)
        valid_pixels[:, 1] = np.clip(valid_pixels[:, 1], 0, view.raw_img.shape[0] - 1)
        proj_mask = np.zeros(view.mask_img.shape, dtype=np.uint8)
        proj_mask[valid_pixels[:, 1], valid_pixels[:, 0]] = 255
        kernel = np.ones((5, 5), np.uint8)
        proj_mask = cv2.dilate(proj_mask, kernel, iterations=2) > 0
        target_mask = view.mask_img > 0
        intersection = np.logical_and(proj_mask, target_mask).sum()
        union = np.logical_or(proj_mask, target_mask).sum()
        iou_losses.append(1.0 - float(intersection / union) if union > 0 else 1.0)

        if view.depth is not None:
            hom = np.hstack([points_world[valid], np.ones((np.sum(valid), 1))])
            cam = (view.w2c @ hom.T).T[:, :3]
            obs_depth = view.depth[valid_pixels[:, 1], valid_pixels[:, 0]]
            keep = (obs_depth > 0.2) & (obs_depth < 1.5) & target_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
            if np.any(keep):
                depth_losses.append(float(np.median(np.abs(cam[keep, 2] - obs_depth[keep]))))
    return float(np.mean(iou_losses)) if iou_losses else 1.0, float(np.mean(depth_losses)) if depth_losses else 0.0


def evaluate_transform(
    matrix: np.ndarray,
    mesh_samples: np.ndarray,
    obs_tree: cKDTree,
    obs_eval_points: np.ndarray,
    views: list[ViewData],
    matches: list[ViewMatch],
    weights: dict[str, float],
    vertex_to_obs_gate: float,
    obs_to_vertex_gate: float,
) -> dict[str, float]:
    samples_world = transform_points(matrix, mesh_samples)
    d_mesh_to_obs, _ = obs_tree.query(samples_world, k=1)
    sample_tree = cKDTree(samples_world)
    d_obs_to_mesh, _ = sample_tree.query(obs_eval_points, k=1)

    keypoint_residuals: list[np.ndarray] = []
    reprojection_residuals: list[np.ndarray] = []
    for match in matches:
        view = views[[view.view for view in views].index(match.view)]
        ids = match.inliers
        mesh_world = transform_points(matrix, match.mesh_points[ids])
        keypoint_residuals.append(np.linalg.norm(mesh_world - match.target_world[ids], axis=1))
        pixels, valid = project_world_to_view(mesh_world, view)
        if np.any(valid):
            reprojection_residuals.append(
                np.linalg.norm(pixels[valid] - match.raw_pixels[ids][valid], axis=1)
            )
    keypoint_loss = robust_mean(np.concatenate(keypoint_residuals), 0.02) if keypoint_residuals else 1.0
    reprojection_loss = (
        robust_mean(np.concatenate(reprojection_residuals) / 25.0, 1.0)
        if reprojection_residuals
        else 1.0
    )
    vertex_to_obs_p95 = float(np.percentile(d_mesh_to_obs, 95))
    obs_to_vertex_p95 = float(np.percentile(d_obs_to_mesh, 95))
    chamfer_loss = robust_mean(d_mesh_to_obs / max(vertex_to_obs_gate, 1e-6), 1.0)
    coverage_loss = robust_mean(d_obs_to_mesh / max(obs_to_vertex_gate, 1e-6), 1.0)
    p95_loss = (
        vertex_to_obs_p95 / max(vertex_to_obs_gate, 1e-6)
        + obs_to_vertex_p95 / max(obs_to_vertex_gate, 1e-6)
    )
    pcd_loss = chamfer_loss + 0.5 * coverage_loss + 0.5 * p95_loss
    silhouette_loss, depth_loss = silhouette_depth_score(samples_world, views)
    total = (
        10.0 * keypoint_loss
        + reprojection_loss
        + weights["pcd"] * pcd_loss
        + weights["silhouette"] * silhouette_loss
        + weights["depth"] * depth_loss
    )
    return {
        "total": float(total),
        "keypoint": float(keypoint_loss),
        "reprojection": float(reprojection_loss),
        "chamfer": float(chamfer_loss),
        "coverage": float(coverage_loss),
        "pcd": float(pcd_loss),
        "vertex_to_obs_p95": vertex_to_obs_p95,
        "obs_to_vertex_p95": obs_to_vertex_p95,
        "silhouette": float(silhouette_loss),
        "depth": float(depth_loss),
    }


def optimize_sim3(
    initial_matrix: np.ndarray,
    mesh_samples: np.ndarray,
    obs_tree: cKDTree,
    obs_eval_points: np.ndarray,
    views: list[ViewData],
    matches: list[ViewMatch],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = {
        "pcd": args.pcd_weight,
        "silhouette": args.silhouette_weight,
        "depth": args.depth_weight,
    }

    def objective(params: np.ndarray) -> float:
        matrix = params_to_matrix(params)
        return evaluate_transform(
            matrix,
            mesh_samples,
            obs_tree,
            obs_eval_points,
            views,
            matches,
            weights,
            args.vertex_to_obs_gate,
            args.obs_to_vertex_gate,
        )["total"]

    initial_params = matrix_to_params(initial_matrix)
    initial_metrics = evaluate_transform(
        initial_matrix,
        mesh_samples,
        obs_tree,
        obs_eval_points,
        views,
        matches,
        weights,
        args.vertex_to_obs_gate,
        args.obs_to_vertex_gate,
    )
    result = minimize(
        objective,
        initial_params,
        method="Powell",
        options={"maxiter": max(args.silhouette_iters, 1), "xtol": 1e-4, "ftol": 1e-4},
    )
    matrix = params_to_matrix(result.x)
    final_metrics = evaluate_transform(
        matrix,
        mesh_samples,
        obs_tree,
        obs_eval_points,
        views,
        matches,
        weights,
        args.vertex_to_obs_gate,
        args.obs_to_vertex_gate,
    )
    return matrix, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(getattr(result, "nit", -1)),
        "initial": initial_metrics,
        "final": final_metrics,
    }


def line_point_distance(p: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = p / np.linalg.norm(p)
    return np.linalg.norm(np.cross(points, p), axis=1) / np.linalg.norm(p)


def get_matching_ray_registration(
    mesh_world: o3d.geometry.TriangleMesh,
    obs_points_world: np.ndarray,
    mesh: trimesh.Trimesh,
    trimesh_indices: np.ndarray,
    c2w: np.ndarray,
    w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obs_points_cam = (w2c @ np.hstack([obs_points_world, np.ones((len(obs_points_world), 1))]).T).T[:, :3]
    vertices_cam = (
        w2c
        @ np.hstack(
            [np.asarray(mesh_world.vertices), np.ones((len(mesh_world.vertices), 1))]
        ).T
    ).T[:, :3]
    obs_kd = KDTree(obs_points_cam)
    new_indices: list[int] = []
    new_targets: list[np.ndarray] = []

    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        norm = np.linalg.norm(vertex)
        if norm <= 1e-8:
            continue
        ray_direction = vertex / norm
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=np.array([[0.0, 0.0, 0.0]]),
            ray_directions=np.array([ray_direction]),
            multiple_hits=False,
        )
        if len(locations) > 0 and np.linalg.norm(locations[0]) < norm - 1e-4:
            continue
        indices = obs_kd.query_ball_point(vertex, 0.02)
        if not indices:
            continue
        distances = line_point_distance(vertex, obs_points_cam[indices])
        closest = indices[int(np.argmin(distances))]
        target = c2w @ np.hstack([obs_points_cam[closest], 1.0])
        new_indices.append(index)
        new_targets.append(target[:3])

    return np.asarray(new_indices, dtype=np.int32), np.asarray(new_targets)


def deform_arap(
    mesh_world: o3d.geometry.TriangleMesh,
    keypoint_mesh_world: np.ndarray,
    keypoint_targets: np.ndarray,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    tree = KDTree(np.asarray(mesh_world.vertices))
    _, indices = tree.query(keypoint_mesh_world)
    indices = np.asarray(indices, dtype=np.int32)
    unique_indices: list[int] = []
    unique_targets: list[np.ndarray] = []
    for index, target in zip(indices, keypoint_targets):
        index = int(index)
        if index in unique_indices:
            unique_targets[unique_indices.index(index)] = target
        else:
            unique_indices.append(index)
            unique_targets.append(target)
    indices = np.asarray(unique_indices, dtype=np.int32)
    if len(indices) == 0:
        return copy.deepcopy(mesh_world), indices
    deformed = mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(indices),
        o3d.utility.Vector3dVector(unique_targets),
        max_iter=1,
    )
    return deformed, indices


def deform_arap_ray_registration(
    mesh_world: o3d.geometry.TriangleMesh,
    obs_points: np.ndarray,
    mesh_for_rays: trimesh.Trimesh,
    trimesh_indices: np.ndarray,
    views: list[ViewData],
    keypoint_indices: np.ndarray,
    keypoint_targets: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    final_indices: list[int] = []
    final_targets: list[np.ndarray] = []
    for index, target in zip(keypoint_indices, keypoint_targets):
        if int(index) not in final_indices:
            final_indices.append(int(index))
            final_targets.append(target)

    for view in views:
        new_indices, new_targets = get_matching_ray_registration(
            mesh_world,
            obs_points,
            mesh_for_rays.copy(),
            trimesh_indices,
            view.c2w,
            view.w2c,
        )
        for index, target in zip(new_indices, new_targets):
            if int(index) not in final_indices:
                final_indices.append(int(index))
                final_targets.append(target)

    return mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )


def write_final_matching_video(
    output_dir: Path,
    final_mesh_world: o3d.geometry.TriangleMesh,
    obs_points: np.ndarray,
    obs_colors: np.ndarray,
) -> None:
    final_mesh_world.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    pcd.colors = o3d.utility.Vector3dVector(obs_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    writer = cv2.VideoWriter(
        str(output_dir / "final_matching.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_dir / 'final_matching.mp4'}")
    vis.add_geometry(pcd)
    vis.add_geometry(final_mesh_world)
    view_control = vis.get_view_control()
    for _ in range(360):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    vis.destroy_window()


def save_projection_overlays(
    debug_dir: Path,
    mesh_samples_world: np.ndarray,
    views: list[ViewData],
) -> None:
    overlay_dir = debug_dir / "overlays"
    ensure_dir(overlay_dir)
    for view in views:
        pixels, valid = project_world_to_view(mesh_samples_world, view)
        image = view.raw_img.copy()
        image[view.mask_img > 0] = (0.6 * image[view.mask_img > 0] + 0.4 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        pts = np.round(pixels[valid]).astype(np.int32)
        for point in pts[:: max(len(pts) // 3000, 1)]:
            cv2.circle(image, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
        cv2.imwrite(str(overlay_dir / f"view_{view.view}_optimized_overlay.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def compute_distance_metrics(mesh_path: Path, obs_points: np.ndarray) -> dict[str, float]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return compute_trimesh_distance_metrics(mesh, obs_points)


def compute_trimesh_distance_metrics(
    mesh: trimesh.Trimesh,
    obs_points: np.ndarray,
) -> dict[str, float]:
    verts = np.asarray(mesh.vertices)
    obs_tree = cKDTree(obs_points)
    vertex_to_obs, _ = obs_tree.query(verts, k=1)
    vertex_tree = cKDTree(verts)
    obs_to_vertex, _ = vertex_tree.query(obs_points, k=1)
    return {
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "vertex_to_obs_median": float(np.median(vertex_to_obs)),
        "vertex_to_obs_p95": float(np.percentile(vertex_to_obs, 95)),
        "obs_to_vertex_median": float(np.median(obs_to_vertex)),
        "obs_to_vertex_p95": float(np.percentile(obs_to_vertex, 95)),
    }


def quality_gate_status(
    metrics: dict[str, float],
    args: argparse.Namespace,
    valid_views: int,
    total_inliers: int,
) -> dict[str, Any]:
    gates = {
        "valid_views": valid_views >= 2,
        "total_inliers": total_inliers >= MIN_TOTAL_INLIERS,
        "vertex_to_obs_p95": metrics["vertex_to_obs_p95"] <= args.vertex_to_obs_gate,
        "obs_to_vertex_p95": metrics["obs_to_vertex_p95"] <= args.obs_to_vertex_gate,
    }
    return {
        "thresholds": {
            "valid_views": 2,
            "total_inliers": MIN_TOTAL_INLIERS,
            "vertex_to_obs_p95": args.vertex_to_obs_gate,
            "obs_to_vertex_p95": args.obs_to_vertex_gate,
        },
        "checks": gates,
        "passed": all(gates.values()),
    }


def mesh_quality_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    vertex_ratio = metrics["vertex_to_obs_p95"] / max(args.vertex_to_obs_gate, 1e-6)
    obs_ratio = metrics["obs_to_vertex_p95"] / max(args.obs_to_vertex_gate, 1e-6)
    median_ratio = metrics["vertex_to_obs_median"] / max(args.vertex_to_obs_gate, 1e-6)
    gate_penalty = max(vertex_ratio - 1.0, 0.0) ** 2 + max(obs_ratio - 1.0, 0.0) ** 2
    return float(vertex_ratio + obs_ratio + 0.25 * median_ratio + 10.0 * gate_penalty)


def sim3_gate_aware_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    vertex_excess = max(metrics["vertex_to_obs_p95"] - args.vertex_to_obs_gate, 0.0)
    obs_excess = max(metrics["obs_to_vertex_p95"] - args.obs_to_vertex_gate, 0.0)
    vertex_ratio = vertex_excess / max(args.vertex_to_obs_gate, 1e-6)
    obs_ratio = obs_excess / max(args.obs_to_vertex_gate, 1e-6)
    # Gate violations decide first; the optimizer objective only breaks ties.
    return float(1000.0 * (vertex_ratio + obs_ratio) + metrics["total"])


def preserves_p95_gates(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> bool:
    return (
        candidate["vertex_to_obs_p95"]
        <= baseline["vertex_to_obs_p95"] + QUALITY_TOLERANCE
        and candidate["obs_to_vertex_p95"]
        <= baseline["obs_to_vertex_p95"] + QUALITY_TOLERANCE
    )


def make_o3d_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    return o3d_mesh


def world_o3d_to_trimesh(
    template: trimesh.Trimesh,
    mesh_world: o3d.geometry.TriangleMesh,
    trimesh_indices: np.ndarray,
) -> trimesh.Trimesh:
    output = template.copy()
    output.vertices = np.asarray(mesh_world.vertices)[trimesh_indices]
    return output


def mask_supported_vertices(vertices: np.ndarray, views: list[ViewData]) -> np.ndarray:
    supported = np.zeros(len(vertices), dtype=bool)
    for view in views:
        pixels, valid = project_world_to_view(vertices, view)
        if not np.any(valid):
            continue
        valid_indices = np.where(valid)[0]
        xy = np.floor(pixels[valid]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, view.mask_img.shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, view.mask_img.shape[0] - 1)
        in_mask = view.mask_img[xy[:, 1], xy[:, 0]] > 0
        supported[valid_indices[in_mask]] = True
    return supported


def prune_far_unsupported_faces(
    mesh: trimesh.Trimesh,
    obs_points: np.ndarray,
    views: list[ViewData],
    max_dist: float,
) -> tuple[trimesh.Trimesh | None, dict[str, Any]]:
    if max_dist <= 0 or len(mesh.faces) == 0:
        return None, {"attempted": False, "reason": "disabled"}

    vertices = np.asarray(mesh.vertices)
    distances, _ = cKDTree(obs_points).query(vertices, k=1)
    supported = mask_supported_vertices(vertices, views)
    far_unsupported = (distances > max_dist) & ~supported
    face_flags = np.sum(far_unsupported[np.asarray(mesh.faces)], axis=1) >= 2
    remove_count = int(np.sum(face_flags))
    if remove_count == 0:
        return None, {"attempted": True, "removed_faces": 0, "reason": "no_far_unsupported_faces"}

    keep_faces = ~face_flags
    if int(np.sum(keep_faces)) < 100:
        return None, {
            "attempted": True,
            "removed_faces": remove_count,
            "reason": "too_few_faces_remaining",
        }

    pruned = mesh.copy()
    pruned.update_faces(keep_faces)
    pruned.remove_unreferenced_vertices()
    components = pruned.split(only_watertight=False)
    if len(components) > 1:
        pruned = max(components, key=lambda component: len(component.faces))
    return pruned, {
        "attempted": True,
        "removed_faces": remove_count,
        "remaining_faces": int(len(pruned.faces)),
        "far_unsupported_vertices": int(np.sum(far_unsupported)),
    }


def choose_observation_eval_points(obs_points: np.ndarray, max_points: int = 6000) -> np.ndarray:
    if len(obs_points) <= max_points:
        return obs_points
    rng = np.random.default_rng(42)
    indices = rng.choice(len(obs_points), size=max_points, replace=False)
    return obs_points[np.sort(indices)]


def main() -> int:
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("align_mvsam3d.py requires CUDA/PyTorch3D for candidate rendering.")

    base_path = Path(args.base_path).expanduser().resolve()
    case_dir = base_path / args.case_name
    mesh_path = Path(args.mesh_path).expanduser().resolve() if args.mesh_path else case_dir / "shape" / "object.glb"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else case_dir / "shape" / "matching"
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else case_dir / "shape" / "mvsam3d" / "align"
    ensure_dir(output_dir)
    ensure_dir(debug_dir)

    view_indices = resolve_view_indices(case_dir, args.view_indices)
    views, obs_by_view, colors_by_view = load_view_data(case_dir, view_indices, args.controller_name)
    obs_points = np.vstack(obs_by_view)
    obs_colors = np.vstack(colors_by_view)
    obs_tree = cKDTree(obs_points)

    mesh = as_mesh(trimesh.load(mesh_path, force="mesh", process=False))
    render_mesh_path = maybe_write_render_mesh(mesh_path, mesh, debug_dir, args.max_render_faces)
    mesh_hash = sha1_file(render_mesh_path)

    matches: list[ViewMatch] = []
    for view in views:
        match = render_candidate_matches(
            view,
            mesh,
            render_mesh_path,
            mesh_hash,
            debug_dir,
            args.force_rematch,
        )
        if match is not None:
            matches.append(match)

    total_inliers = int(sum(len(match.inliers) for match in matches))
    if len(matches) < 2 or total_inliers < MIN_TOTAL_INLIERS:
        raise RuntimeError(
            f"Not enough valid MV align evidence: valid_views={len(matches)}, total_inliers={total_inliers}"
        )

    np.random.seed(42)
    sample_count = min(6000, max(1000, len(mesh.faces)))
    mesh_samples, _ = trimesh.sample.sample_surface(mesh, sample_count)
    obs_eval_points = choose_observation_eval_points(obs_points)
    candidate_scores = []
    optimized_candidates: list[dict[str, Any]] = []
    weights = {
        "pcd": args.pcd_weight,
        "silhouette": args.silhouette_weight,
        "depth": args.depth_weight,
    }
    for match in matches:
        for scale_multiplier in SIM3_SCALE_MULTIPLIERS:
            start_matrix = scaled_sim3_matrix(match.mesh2world, scale_multiplier)
            initial_score = evaluate_transform(
                start_matrix,
                mesh_samples,
                obs_tree,
                obs_eval_points,
                views,
                matches,
                weights,
                args.vertex_to_obs_gate,
                args.obs_to_vertex_gate,
            )
            optimized_matrix, optimizer_metrics = optimize_sim3(
                start_matrix,
                mesh_samples,
                obs_tree,
                obs_eval_points,
                views,
                matches,
                args,
            )
            optimized_score = optimizer_metrics["final"]
            optimizer_loss = float(optimized_score["total"])
            quality_score = sim3_gate_aware_score(optimized_score, args)
            candidate_record = {
                "view": match.view,
                "scale_multiplier": scale_multiplier,
                "initial": initial_score,
                "optimized": optimized_score,
                "quality_score": quality_score,
                "optimizer_loss": optimizer_loss,
                "optimizer": optimizer_metrics,
                "matrix": optimized_matrix,
            }
            optimized_candidates.append(candidate_record)
            candidate_scores.append(
                {
                    "view": match.view,
                    "scale_multiplier": scale_multiplier,
                    "total": optimizer_loss,
                    "gate_aware_score": quality_score,
                    **optimized_score,
                }
            )
    optimized_candidates.sort(key=lambda item: item["quality_score"])
    selected_sim3 = optimized_candidates[0]
    optimized_matrix = selected_sim3["matrix"]
    active_matches = matches

    all_mesh_keypoints = []
    all_targets = []
    for match in active_matches:
        ids = match.inliers
        all_mesh_keypoints.append(transform_points(optimized_matrix, match.mesh_points[ids]))
        all_targets.append(match.target_world[ids])
    keypoint_mesh_world = np.vstack(all_mesh_keypoints)
    keypoint_targets = np.vstack(all_targets)

    initial_mesh_world = make_o3d_mesh(mesh)
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    tree = KDTree(np.asarray(initial_mesh_world.vertices))
    _, trimesh_indices = tree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(optimized_matrix)

    stage_metrics: dict[str, Any] = {}

    def build_stage_candidate(name: str, o3d_mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
        trimesh_candidate = world_o3d_to_trimesh(mesh, o3d_mesh, trimesh_indices)
        distance_metrics = compute_trimesh_distance_metrics(trimesh_candidate, obs_points)
        return {
            "name": name,
            "o3d_mesh": o3d_mesh,
            "trimesh": trimesh_candidate,
            "distance_metrics": distance_metrics,
            "quality_score": mesh_quality_score(distance_metrics, args),
        }

    selected_stage = build_stage_candidate("sim3_only", initial_mesh_world)
    stage_metrics["sim3_only"] = {
        "distance_metrics": selected_stage["distance_metrics"],
        "quality_score": selected_stage["quality_score"],
        "accepted": True,
    }

    keypoint_indices = np.zeros((0,), dtype=np.int32)
    try:
        deformed_keypoint_mesh, keypoint_indices = deform_arap(
            copy.deepcopy(initial_mesh_world),
            keypoint_mesh_world,
            keypoint_targets,
        )
        keypoint_stage = build_stage_candidate("keypoint_arap", deformed_keypoint_mesh)
        keypoint_accepted = preserves_p95_gates(
            keypoint_stage["distance_metrics"], selected_stage["distance_metrics"]
        )
        if keypoint_accepted:
            selected_stage = keypoint_stage
        stage_metrics["keypoint_arap"] = {
            "distance_metrics": keypoint_stage["distance_metrics"],
            "quality_score": keypoint_stage["quality_score"],
            "accepted": keypoint_accepted,
        }
    except RuntimeError as exc:
        stage_metrics["keypoint_arap"] = {
            "accepted": False,
            "error": str(exc),
        }

    mesh_for_rays = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        process=False,
    )
    if args.disable_ray_arap or len(keypoint_indices) == 0:
        stage_metrics["ray_arap"] = {"accepted": False, "skipped": True}
    else:
        try:
            ray_mesh = deform_arap_ray_registration(
                copy.deepcopy(selected_stage["o3d_mesh"]),
                obs_points,
                mesh_for_rays,
                trimesh_indices,
                views,
                keypoint_indices,
                keypoint_targets,
            )
            ray_stage = build_stage_candidate("ray_arap", ray_mesh)
            ray_accepted = preserves_p95_gates(
                ray_stage["distance_metrics"], selected_stage["distance_metrics"]
            )
            if ray_accepted:
                selected_stage = ray_stage
            stage_metrics["ray_arap"] = {
                "distance_metrics": ray_stage["distance_metrics"],
                "quality_score": ray_stage["quality_score"],
                "accepted": ray_accepted,
                "skipped": False,
            }
        except Exception as exc:
            stage_metrics["ray_arap"] = {
                "accepted": False,
                "skipped": False,
                "error": str(exc),
            }

    prune_candidate, prune_info = prune_far_unsupported_faces(
        selected_stage["trimesh"],
        obs_points,
        views,
        args.prune_far_dist,
    )
    if prune_candidate is not None:
        prune_metrics = compute_trimesh_distance_metrics(prune_candidate, obs_points)
        prune_info["distance_metrics"] = prune_metrics
        prune_info["quality_score"] = mesh_quality_score(prune_metrics, args)
        prune_accepted = (
            prune_metrics["vertex_to_obs_p95"]
            < selected_stage["distance_metrics"]["vertex_to_obs_p95"]
            and prune_metrics["obs_to_vertex_p95"]
            <= selected_stage["distance_metrics"]["obs_to_vertex_p95"] + QUALITY_TOLERANCE
        )
        prune_info["accepted"] = prune_accepted
        if prune_accepted:
            selected_stage = {
                "name": "pruned_envelope",
                "o3d_mesh": make_o3d_mesh(prune_candidate),
                "trimesh": prune_candidate,
                "distance_metrics": prune_metrics,
                "quality_score": prune_info["quality_score"],
            }
    else:
        prune_info["accepted"] = False
    stage_metrics["pruned_envelope"] = prune_info

    final_mesh = selected_stage["trimesh"]
    final_mesh_world = selected_stage["o3d_mesh"]
    final_mesh_path = output_dir / "final_mesh.glb"
    final_mesh.export(final_mesh_path)
    write_final_matching_video(output_dir, final_mesh_world, obs_points, obs_colors)
    overlay_samples, _ = trimesh.sample.sample_surface(
        final_mesh, min(6000, max(1000, len(final_mesh.faces)))
    )
    save_projection_overlays(debug_dir, overlay_samples, views)

    distance_metrics = compute_distance_metrics(final_mesh_path, obs_points)
    quality_gates = quality_gate_status(distance_metrics, args, len(matches), total_inliers)

    metrics: dict[str, Any] = {
        "case_name": args.case_name,
        "view_indices": view_indices,
        "valid_views": len(matches),
        "active_keypoint_views": [match.view for match in active_matches],
        "total_inliers": total_inliers,
        "best_initial_view": selected_sim3["view"],
        "selected_scale_multiplier": selected_sim3["scale_multiplier"],
        "selected_stage": selected_stage["name"],
        "candidate_scores": candidate_scores,
        "view_matches": [
            {
                "view": match.view,
                "match_count": match.match_count,
                "inliers": int(len(match.inliers)),
                "reprojection_error": match.reprojection_error,
                "scale": match.scale,
            }
            for match in matches
        ],
        "optimizer": selected_sim3["optimizer"],
        "sim3_optimization": {
            "scale_multipliers": list(SIM3_SCALE_MULTIPLIERS),
            "candidate_count": len(optimized_candidates),
        },
        "stage_metrics": stage_metrics,
        "distance_metrics": distance_metrics,
        "quality_gates": quality_gates,
        "outputs": {
            "final_mesh": str(final_mesh_path),
            "final_matching": str(output_dir / "final_matching.mp4"),
        },
    }
    with (debug_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
