from __future__ import annotations

import argparse
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

    valid_matches = match_result["matches"] > -1
    if int(np.sum(valid_matches)) < MIN_VIEW_INLIERS:
        return None

    render_points = match_result["keypoints0"][valid_matches]
    mesh_points, valid_depth = project_2d_to_3d(
        render_points, depths[best_idx], render_intrinsic, camera_poses[best_idx].cpu().numpy()
    )
    raw_pixels_box = match_result["keypoints1"][match_result["matches"][valid_matches]]
    raw_pixels_box = raw_pixels_box[valid_depth]
    raw_pixels = raw_pixels_box + np.array([bbox[0], bbox[1]])
    if mesh_points.shape[0] < MIN_VIEW_INLIERS:
        return None

    target_pixels, target_world = select_point(view.points, raw_pixels, view.object_mask)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.float32(mesh_points),
        np.float32(raw_pixels),
        np.float32(view.intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=200,
        reprojectionError=15.0,
        confidence=0.99,
    )
    if not success or inliers is None or len(inliers) < MIN_VIEW_INLIERS:
        return None
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

    payload = ViewMatch(
        view=view.view,
        view_index=view.index,
        best_color=colors[best_idx],
        best_depth=depths[best_idx],
        best_pose=camera_poses[best_idx].cpu().numpy(),
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
        match_count=int(np.sum(valid_matches)),
    )
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
    views: list[ViewData],
    matches: list[ViewMatch],
    weights: dict[str, float],
) -> dict[str, float]:
    samples_world = transform_points(matrix, mesh_samples)
    d_mesh_to_obs, _ = obs_tree.query(samples_world, k=1)

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
    chamfer_loss = robust_mean(d_mesh_to_obs, 0.04)
    silhouette_loss, depth_loss = silhouette_depth_score(samples_world, views)
    total = (
        10.0 * keypoint_loss
        + reprojection_loss
        + weights["pcd"] * chamfer_loss
        + weights["silhouette"] * silhouette_loss
        + weights["depth"] * depth_loss
    )
    return {
        "total": float(total),
        "keypoint": float(keypoint_loss),
        "reprojection": float(reprojection_loss),
        "chamfer": float(chamfer_loss),
        "silhouette": float(silhouette_loss),
        "depth": float(depth_loss),
    }


def optimize_sim3(
    initial_matrix: np.ndarray,
    mesh_samples: np.ndarray,
    obs_tree: cKDTree,
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
        return evaluate_transform(matrix, mesh_samples, obs_tree, views, matches, weights)["total"]

    initial_params = matrix_to_params(initial_matrix)
    initial_metrics = evaluate_transform(
        initial_matrix, mesh_samples, obs_tree, views, matches, weights
    )
    result = minimize(
        objective,
        initial_params,
        method="Powell",
        options={"maxiter": max(args.silhouette_iters, 1), "xtol": 1e-4, "ftol": 1e-4},
    )
    matrix = params_to_matrix(result.x)
    final_metrics = evaluate_transform(matrix, mesh_samples, obs_tree, views, matches, weights)
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
    deformed = mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(indices),
        o3d.utility.Vector3dVector(keypoint_targets),
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

    for index in np.where(np.asarray(mesh_world.vertices)[:, 2] > 0)[0]:
        index = int(index)
        target = np.asarray(mesh_world.vertices)[index].copy()
        target[2] = 0
        if index in final_indices:
            final_targets[final_indices.index(index)] = target
        else:
            final_indices.append(index)
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
    candidate_scores = []
    for match in matches:
        score = evaluate_transform(
            match.mesh2world,
            mesh_samples,
            obs_tree,
            views,
            matches,
            {"pcd": args.pcd_weight, "silhouette": args.silhouette_weight, "depth": args.depth_weight},
        )
        candidate_scores.append((score["total"], match.view, match.mesh2world, score))
    candidate_scores.sort(key=lambda item: item[0])
    initial_matrix = candidate_scores[0][2]
    best_score = candidate_scores[0][0]
    trusted_views = {
        view for total, view, _, _ in candidate_scores if total <= best_score * 1.5
    }
    active_matches = [match for match in matches if match.view in trusted_views]
    if not active_matches:
        active_matches = [next(match for match in matches if match.view == candidate_scores[0][1])]

    optimized_matrix, optimizer_metrics = optimize_sim3(
        initial_matrix,
        mesh_samples,
        obs_tree,
        views,
        active_matches,
        args,
    )

    all_mesh_keypoints = []
    all_targets = []
    for match in active_matches:
        ids = match.inliers
        all_mesh_keypoints.append(transform_points(optimized_matrix, match.mesh_points[ids]))
        all_targets.append(match.target_world[ids])
    keypoint_mesh_world = np.vstack(all_mesh_keypoints)
    keypoint_targets = np.vstack(all_targets)

    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    tree = KDTree(np.asarray(initial_mesh_world.vertices))
    _, trimesh_indices = tree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(optimized_matrix)

    deformed_keypoint_mesh, keypoint_indices = deform_arap(
        initial_mesh_world,
        keypoint_mesh_world,
        keypoint_targets,
    )
    mesh_for_rays = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        process=False,
    )
    final_mesh_world = deform_arap_ray_registration(
        deformed_keypoint_mesh,
        obs_points,
        mesh_for_rays,
        trimesh_indices,
        views,
        keypoint_indices,
        keypoint_targets,
    )

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    final_mesh_path = output_dir / "final_mesh.glb"
    mesh.export(final_mesh_path)
    write_final_matching_video(output_dir, final_mesh_world, obs_points, obs_colors)
    save_projection_overlays(debug_dir, transform_points(optimized_matrix, mesh_samples), views)

    metrics: dict[str, Any] = {
        "case_name": args.case_name,
        "view_indices": view_indices,
        "valid_views": len(matches),
        "active_keypoint_views": [match.view for match in active_matches],
        "total_inliers": total_inliers,
        "best_initial_view": candidate_scores[0][1],
        "candidate_scores": [
            {"view": view, "total": float(total), **score}
            for total, view, _, score in candidate_scores
        ],
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
        "optimizer": optimizer_metrics,
        "distance_metrics": compute_distance_metrics(final_mesh_path, obs_points),
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
