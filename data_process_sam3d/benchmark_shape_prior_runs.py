from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


BACKENDS = ("trellis", "mvsam3d")
PRIOR_FAR_THRESHOLD_M = 0.035


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create backend-labeled manifests, observation metrics, and presentation "
            "videos for independent Trellis and MV-SAM3D shape-prior runs."
        )
    )
    parser.add_argument("--trellis_case_dir", required=True)
    parser.add_argument("--mvsam3d_case_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_name", default="single_lift_sloth")
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument("--view_indices", default="0,1,2")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--skip_videos", action="store_true")
    return parser


def parse_view_indices(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str, producer_backend: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "producer_backend": producer_backend,
    }


def copy_artifact(
    source: Path,
    destination: Path,
    role: str,
    producer_backend: str,
) -> dict[str, Any] | None:
    if not source.exists():
        return None
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    return file_record(destination, role, producer_backend)


def load_observed_object_points(case_dir: Path, view_indices: list[int]) -> np.ndarray:
    pcd = np.load(case_dir / "pcd" / "0.npz")
    with (case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
        processed_masks = pickle.load(handle)

    points: list[np.ndarray] = []
    for view_index in view_indices:
        mask = np.asarray(processed_masks[0][view_index]["object"], dtype=bool)
        if "masks" in pcd:
            mask = np.logical_and(mask, np.asarray(pcd["masks"][view_index], dtype=bool))
        points.append(np.asarray(pcd["points"][view_index][mask], dtype=np.float64))
    observed = np.concatenate(points, axis=0)
    return observed[np.isfinite(observed).all(axis=1)]


def load_final_data_points(final_data_path: Path) -> dict[str, np.ndarray]:
    with final_data_path.open("rb") as handle:
        data = pickle.load(handle)
    object_points = np.asarray(data["object_points"][0], dtype=np.float64)
    surface_points = np.asarray(data.get("surface_points", np.zeros((0, 3))), dtype=np.float64)
    interior_points = np.asarray(data.get("interior_points", np.zeros((0, 3))), dtype=np.float64)
    prior_points = np.concatenate([surface_points, interior_points], axis=0)
    final_points = np.concatenate([object_points, prior_points], axis=0)
    return {
        "object_points": object_points,
        "surface_points": surface_points,
        "interior_points": interior_points,
        "prior_points": prior_points,
        "final_points": final_points[np.isfinite(final_points).all(axis=1)],
    }


def nearest_metrics(query: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if query.size == 0 or reference.size == 0:
        return {
            "count": int(len(query)),
            "median": float("inf"),
            "p95": float("inf"),
            "max": float("inf"),
        }
    distances, _ = cKDTree(reference).query(query, k=1)
    return {
        "count": int(len(query)),
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def mesh_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    mesh = trimesh.load(path, force="mesh", process=False)
    return {
        "path": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": np.asarray(mesh.bounds).tolist(),
        "sha256": sha256_file(path),
        "size": int(path.stat().st_size),
    }


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_case(
    backend: str,
    case_dir: Path,
    view_indices: list[int],
) -> dict[str, Any]:
    observed = load_observed_object_points(case_dir, view_indices)
    final_data_path = case_dir / "final_data.pkl"
    points = load_final_data_points(final_data_path)

    prior_to_obs = nearest_metrics(points["prior_points"], observed)
    final_to_obs = nearest_metrics(points["final_points"], observed)
    obs_to_final = nearest_metrics(observed, points["final_points"])
    if points["prior_points"].size:
        prior_distances, _ = cKDTree(observed).query(points["prior_points"], k=1)
        prior_far_fraction = float(np.mean(prior_distances > PRIOR_FAR_THRESHOLD_M))
    else:
        prior_far_fraction = 0.0

    return {
        "backend": backend,
        "case_dir": str(case_dir),
        "observed_object_points": int(len(observed)),
        "point_counts": {
            "object_points": int(len(points["object_points"])),
            "surface_points": int(len(points["surface_points"])),
            "interior_points": int(len(points["interior_points"])),
            "prior_points": int(len(points["prior_points"])),
            "final_points": int(len(points["final_points"])),
        },
        "final_to_observation": final_to_obs,
        "observation_to_final": obs_to_final,
        "prior_to_observation": prior_to_obs,
        "prior_far_threshold_m": PRIOR_FAR_THRESHOLD_M,
        "prior_far_fraction": prior_far_fraction,
        "raw_mesh": mesh_stats(case_dir / "shape" / "object.glb"),
        "aligned_mesh": mesh_stats(case_dir / "shape" / "matching" / "final_mesh.glb"),
        "align_metrics": load_optional_json(
            case_dir / "shape" / "mvsam3d" / "align" / "metrics.json"
        ),
    }


def load_points_for_video(final_data_path: Path) -> np.ndarray:
    points = load_final_data_points(final_data_path)["final_points"]
    return points[np.isfinite(points).all(axis=1)]


def configure_view(vis: o3d.visualization.Visualizer, bounds: np.ndarray) -> None:
    view_control = vis.get_view_control()
    center = bounds.mean(axis=0)
    extent = np.linalg.norm(bounds[1] - bounds[0])
    view_control.set_front([1.0, 0.0, -2.0])
    view_control.set_up([0.0, 0.0, -1.0])
    view_control.set_lookat(center.tolist())
    view_control.set_zoom(0.75 if extent > 0 else 1.0)


def render_points_video(points: np.ndarray, output_path: Path, label: str, frames: int) -> Path:
    ensure_dir(output_path.parent)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.1, 0.45, 0.9])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    bounds = np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())
    configure_view(vis, np.vstack([bounds.min(axis=0), bounds.max(axis=0)]))
    dummy = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    view_control = vis.get_view_control()
    for _ in range(frames):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = cv2.cvtColor(
            (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4)
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        writer.write(frame)
    writer.release()
    vis.destroy_window()
    return output_path


def render_mesh_video(mesh_path: Path, output_path: Path, label: str, frames: int) -> Path:
    ensure_dir(output_path.parent)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.75, 0.75, 0.72])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(o3d_mesh)
    configure_view(vis, np.asarray(mesh.bounds))
    dummy = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy.shape
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    view_control = vis.get_view_control()
    for _ in range(frames):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = cv2.cvtColor(
            (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4)
        cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        writer.write(frame)
    writer.release()
    vis.destroy_window()
    return output_path


def side_by_side_video(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    left_label: str,
    right_label: str,
) -> Path:
    ensure_dir(output_path.parent)
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    ok_l, frame_l = left.read()
    ok_r, frame_r = right.read()
    if not ok_l or not ok_r:
        raise RuntimeError(f"Could not read comparison inputs: {left_path}, {right_path}")

    height = min(frame_l.shape[0], frame_r.shape[0])
    width_l = int(frame_l.shape[1] * height / frame_l.shape[0])
    width_r = int(frame_r.shape[1] * height / frame_r.shape[0])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width_l + width_r, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    while ok_l and ok_r:
        frame_l = cv2.resize(frame_l, (width_l, height))
        frame_r = cv2.resize(frame_r, (width_r, height))
        cv2.putText(frame_l, left_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4)
        cv2.putText(frame_l, left_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        cv2.putText(frame_r, right_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4)
        cv2.putText(frame_r, right_label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
        writer.write(np.concatenate([frame_l, frame_r], axis=1))
        ok_l, frame_l = left.read()
        ok_r, frame_r = right.read()

    writer.release()
    left.release()
    right.release()
    return output_path


def write_checksums(records: list[dict[str, Any]], output_path: Path) -> None:
    lines = []
    for record in records:
        if "sha256" in record:
            lines.append(f"{record['sha256']}  {record['path']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_backend_manifest(
    backend: str,
    case_dir: Path,
    output_dir: Path,
    metrics: dict[str, Any],
    copied_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_mesh = case_dir / "shape" / "object.glb"
    run_hash = sha256_file(raw_mesh)[:12] if raw_mesh.exists() else "missingmesh"
    return {
        "backend": backend,
        "run_id": f"{backend}-{metrics['backend']}-{run_hash}",
        "case_name": case_dir.name,
        "case_dir": str(case_dir),
        "generated_at_unix": time.time(),
        "metrics": metrics,
        "outputs": copied_outputs,
        "lineage_checks": {
            "backend_label_valid": backend in BACKENDS,
            "mvsam3d_paths_avoid_trellis": (
                backend != "mvsam3d"
                or all("trellis" not in record["path"].casefold() for record in copied_outputs)
            ),
            "ambiguous_backup_labels_absent": all(
                "original_shape_backup" not in record["path"].casefold()
                and "backup" not in Path(record["path"]).name.casefold()
                for record in copied_outputs
            ),
        },
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    trellis_case = Path(args.trellis_case_dir).expanduser().resolve()
    mvsam3d_case = Path(args.mvsam3d_case_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifacts_dir = output_dir / "artifacts"
    manifests_dir = output_dir / "manifests"
    videos_dir = output_dir / "videos"
    ensure_dir(artifacts_dir)
    ensure_dir(manifests_dir)
    ensure_dir(videos_dir)

    view_indices = parse_view_indices(args.view_indices)
    case_dirs = {"trellis": trellis_case, "mvsam3d": mvsam3d_case}
    metrics = {
        backend: evaluate_case(backend, case_dir, view_indices)
        for backend, case_dir in case_dirs.items()
    }

    all_records: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for backend, case_dir in case_dirs.items():
        prefix = backend.upper()
        records: list[dict[str, Any]] = []
        for source, dest_name, role in [
            (case_dir / "shape" / "object.glb", f"{prefix}_raw_object.glb", "raw_object_glb"),
            (
                case_dir / "shape" / "matching" / "final_mesh.glb",
                f"{prefix}_aligned_final_mesh.glb",
                "aligned_final_mesh_glb",
            ),
            (case_dir / "final_data.pkl", f"{prefix}_final_data.pkl", "final_data_pkl"),
            (
                case_dir / "shape" / "mvsam3d" / "align" / "metrics.json",
                f"{prefix}_align_metrics.json",
                "align_metrics_json",
            ),
        ]:
            record = copy_artifact(source, artifacts_dir / dest_name, role, backend)
            if record:
                records.append(record)
                all_records.append(record)

        if not args.skip_videos:
            pcd_video = artifacts_dir / f"{prefix}_final_pcd.mp4"
            render_points_video(
                load_points_for_video(case_dir / "final_data.pkl"),
                pcd_video,
                f"{prefix} final_pcd",
                args.frames,
            )
            records.append(file_record(pcd_video, "final_pcd_video", backend))
            all_records.append(records[-1])

            raw_mesh_video = videos_dir / f"{prefix}_raw_mesh.mp4"
            render_mesh_video(
                case_dir / "shape" / "object.glb",
                raw_mesh_video,
                f"{prefix} raw mesh",
                args.frames,
            )
            records.append(file_record(raw_mesh_video, "raw_mesh_video", backend))
            all_records.append(records[-1])

        manifest = build_backend_manifest(
            backend, case_dir, output_dir, metrics[backend], records
        )
        manifest_path = manifests_dir / f"{backend}_run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests[backend] = manifest
        all_records.append(file_record(manifest_path, "backend_run_manifest", backend))

    videos: dict[str, str] = {}
    if not args.skip_videos:
        videos["final_pcd_side_by_side"] = str(
            side_by_side_video(
                artifacts_dir / "TRELLIS_final_pcd.mp4",
                artifacts_dir / "MVSAM3D_final_pcd.mp4",
                artifacts_dir / "LEFT_TRELLIS_RIGHT_MVSAM3D_final_pcd.mp4",
                "LEFT: TRELLIS final_pcd",
                "RIGHT: MVSAM3D final_pcd",
            )
        )
        videos["raw_mesh_side_by_side"] = str(
            side_by_side_video(
                videos_dir / "TRELLIS_raw_mesh.mp4",
                videos_dir / "MVSAM3D_raw_mesh.mp4",
                artifacts_dir / "LEFT_TRELLIS_RIGHT_MVSAM3D_raw_mesh.mp4",
                "LEFT: TRELLIS raw mesh",
                "RIGHT: MVSAM3D raw mesh",
            )
        )
        for path in videos.values():
            all_records.append(file_record(Path(path), "side_by_side_video", "comparison"))

    mv = metrics["mvsam3d"]
    tr = metrics["trellis"]
    mv_align_passed = bool(
        ((mv.get("align_metrics") or {}).get("quality_gates") or {}).get("passed", False)
    )
    mv_final_data_passed = (
        mv["point_counts"]["surface_points"] == 700
        and mv["point_counts"]["interior_points"] == 1000
        and mv["prior_far_fraction"] <= 0.01
    )
    scoreboard = {
        "case_name": args.case_name,
        "comparison_semantics": (
            "MV-SAM3D align is judged only against observations. Trellis is an "
            "independent final-data baseline, never an align reference."
        ),
        "metrics": metrics,
        "mvsam3d_acceptance": {
            "align_quality_gates_passed": mv_align_passed,
            "final_data_counts_passed": mv_final_data_passed,
            "surface_points": mv["point_counts"]["surface_points"],
            "interior_points": mv["point_counts"]["interior_points"],
            "prior_far_fraction": mv["prior_far_fraction"],
            "prior_far_threshold_m": PRIOR_FAR_THRESHOLD_M,
        },
        "mv_beats_trellis": {
            "final_to_observation_p95": (
                mv["final_to_observation"]["p95"] <= tr["final_to_observation"]["p95"]
            ),
            "observation_to_final_p95": (
                mv["observation_to_final"]["p95"] <= tr["observation_to_final"]["p95"]
            ),
        },
        "videos": videos,
        "manifests": {
            backend: str(manifests_dir / f"{backend}_run_manifest.json")
            for backend in BACKENDS
        },
    }
    scoreboard["mv_beats_trellis"]["passed"] = (
        mv_align_passed
        and mv_final_data_passed
        and all(scoreboard["mv_beats_trellis"].values())
    )
    scoreboard_path = output_dir / "scoreboard.json"
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    all_records.append(file_record(scoreboard_path, "scoreboard", "comparison"))
    write_checksums(all_records, artifacts_dir / "checksums.sha256")

    print(json.dumps(scoreboard, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
