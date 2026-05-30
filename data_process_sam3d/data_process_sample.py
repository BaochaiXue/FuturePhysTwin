# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points

import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
import trimesh
import cv2
from utils.align_util import as_mesh
from argparse import ArgumentParser
from pathlib import Path
from scipy.spatial import cKDTree

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.005)
parser.add_argument(
    "--shape_prior_max_dist",
    type=float,
    default=0.05,
    help=(
        "Filter sampled shape-prior points that are too far from observed object points "
        "(meters; set <=0 to disable)."
    ),
)
parser.add_argument(
    "--shape_prior_sampling_backend",
    choices=["legacy", "mvsam3d", "auto"],
    default="auto",
)
parser.add_argument("--target_surface_points", type=int, default=700)
parser.add_argument("--target_interior_points", type=int, default=1000)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size
shape_prior_max_dist = args.shape_prior_max_dist
shape_prior_sampling_backend = args.shape_prior_sampling_backend
target_surface_points = args.target_surface_points
target_interior_points = args.target_interior_points


def filter_points_by_nn_distance(
    points: np.ndarray, reference_points: np.ndarray, max_dist: float
) -> np.ndarray:
    if max_dist <= 0 or points.size == 0 or reference_points.size == 0:
        return points
    tree = cKDTree(reference_points)
    distances, _ = tree.query(points, k=1)
    return points[distances <= max_dist]


def resolve_sampling_backend() -> str:
    if shape_prior_sampling_backend != "auto":
        return shape_prior_sampling_backend
    marker = Path(base_path) / case_name / "shape" / "mvsam3d"
    return "mvsam3d" if marker.exists() else "legacy"


def point_grid_index(
    point: np.ndarray, min_bound: np.ndarray, grid_size: float | None = None
) -> tuple[int, int, int]:
    size = volume_sample_size if grid_size is None else grid_size
    return tuple(np.floor((point - min_bound) / size).astype(int))


def dedupe_points(
    points: np.ndarray,
    min_bound: np.ndarray,
    occupied: set[tuple[int, int, int]] | None = None,
    limit: int | None = None,
    grid_size: float | None = None,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    seen: set[tuple[int, int, int]] = set() if occupied is None else set(occupied)
    selected = []
    for point in points:
        grid_index = point_grid_index(point, min_bound, grid_size=grid_size)
        if grid_index in seen:
            continue
        seen.add(grid_index)
        selected.append(point)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(selected)


def sort_by_reference_distance(points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    tree = cKDTree(reference_points)
    distances, _ = tree.query(points, k=1)
    return points[np.argsort(distances)]


def voxel_interior_candidates(
    trimesh_mesh: trimesh.Trimesh,
    reference_points: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    bounds = trimesh_mesh.bounds
    spacing = max(volume_sample_size, 1e-4)
    axes = [
        np.arange(bounds[0, axis] + spacing * 0.5, bounds[1, axis], spacing)
        for axis in range(3)
    ]
    if any(len(axis) == 0 for axis in axes):
        return np.zeros((0, 3), dtype=np.float32)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if grid.shape[0] > 250000:
        step = int(np.ceil(grid.shape[0] / 250000))
        grid = grid[::step]

    try:
        scene = o3d.t.geometry.RaycastingScene()
        vertices = o3d.core.Tensor(np.asarray(trimesh_mesh.vertices), dtype=o3d.core.Dtype.Float32)
        triangles = o3d.core.Tensor(np.asarray(trimesh_mesh.faces), dtype=o3d.core.Dtype.UInt32)
        scene.add_triangles(vertices, triangles)
        signed = scene.compute_signed_distance(
            o3d.core.Tensor(grid.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        interior = grid[signed < 0]
    except Exception:
        try:
            interior = grid[trimesh_mesh.contains(grid)]
        except Exception:
            interior = np.zeros((0, 3), dtype=np.float32)

    interior = filter_points_by_nn_distance(interior, reference_points, max_dist)
    return sort_by_reference_distance(interior, reference_points)


def sample_mvsam3d_prior_points(
    trimesh_mesh: trimesh.Trimesh,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    min_bound = np.min(reference_points, axis=0)
    prior_grid_size = max(volume_sample_size * 0.4, 1e-4)

    surface_candidates = []
    surface_points = np.zeros((0, 3), dtype=np.float32)
    for count in [max(num_surface_points, 4096), 10000, 50000, 200000]:
        sampled, _ = trimesh.sample.sample_surface(trimesh_mesh, count)
        sampled = filter_points_by_nn_distance(
            sampled, reference_points, shape_prior_max_dist
        )
        sampled = sort_by_reference_distance(sampled, reference_points)
        surface_candidates.append(sampled)
        surface_points = dedupe_points(
            np.vstack(surface_candidates),
            min_bound,
            limit=target_surface_points,
            grid_size=prior_grid_size,
        )
        if len(surface_points) >= target_surface_points:
            break
    if len(surface_points) < target_surface_points:
        for max_dist in [shape_prior_max_dist * 1.5, 0]:
            sampled, _ = trimesh.sample.sample_surface(trimesh_mesh, 200000)
            if max_dist > 0:
                sampled = filter_points_by_nn_distance(sampled, reference_points, max_dist)
            sampled = sort_by_reference_distance(sampled, reference_points)
            surface_candidates.append(sampled)
            surface_points = dedupe_points(
                np.vstack(surface_candidates),
                min_bound,
                limit=target_surface_points,
                grid_size=prior_grid_size,
            )
            if len(surface_points) >= target_surface_points:
                break

    interior_candidates = []
    interior_points = np.zeros((0, 3), dtype=np.float32)
    for count in [10000, 50000, 200000]:
        try:
            sampled = trimesh.sample.volume_mesh(trimesh_mesh, count)
        except Exception:
            sampled = np.zeros((0, 3), dtype=np.float32)
        sampled = filter_points_by_nn_distance(
            sampled, reference_points, shape_prior_max_dist
        )
        sampled = sort_by_reference_distance(sampled, reference_points)
        interior_candidates.append(sampled)
        interior_points = dedupe_points(
            np.vstack(interior_candidates),
            min_bound,
            limit=target_interior_points,
            grid_size=prior_grid_size,
        )
        if len(interior_points) >= target_interior_points:
            break

    if len(interior_points) < target_interior_points:
        fallback = voxel_interior_candidates(
            trimesh_mesh, reference_points, shape_prior_max_dist
        )
        if fallback.size:
            interior_points = dedupe_points(
                np.vstack([*interior_candidates, fallback]),
                min_bound,
                limit=target_interior_points,
                grid_size=prior_grid_size,
            )
    if len(interior_points) < target_interior_points:
        relaxed = voxel_interior_candidates(
            trimesh_mesh, reference_points, shape_prior_max_dist * 1.5
        )
        if relaxed.size:
            interior_points = dedupe_points(
                np.vstack([*interior_candidates, interior_points, relaxed]),
                min_bound,
                limit=target_interior_points,
                grid_size=prior_grid_size,
            )
    if len(interior_points) < target_interior_points:
        unfiltered = voxel_interior_candidates(trimesh_mesh, reference_points, 0)
        if unfiltered.size:
            interior_points = dedupe_points(
                np.vstack([*interior_candidates, interior_points, unfiltered]),
                min_bound,
                limit=target_interior_points,
                grid_size=prior_grid_size,
            )

    return surface_points, interior_points


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def process_unique_points(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Get the unique index in the object points
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    # Make sure all points are above the ground
    object_points[object_points[..., 2] > 0, 2] = 0

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        if resolve_sampling_backend() == "mvsam3d":
            surface_points, interior_points = sample_mvsam3d_prior_points(
                trimesh_mesh, object_points[0]
            )
        else:
            # Sample the surface points
            surface_points, _ = trimesh.sample.sample_surface(
                trimesh_mesh, num_surface_points
            )
            # Sample the interior points
            try:
                interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)
            except Exception:
                interior_points = np.zeros((0, 3), dtype=np.float32)

            # Guard against shape-prior outliers by keeping only points near observed object points.
            surface_points = filter_points_by_nn_distance(
                surface_points, object_points[0], shape_prior_max_dist
            )
            interior_points = filter_points_by_nn_distance(
                interior_points, object_points[0], shape_prior_max_dist
            )

    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    if SHAPE_PRIOR:
        final_surface_points = []
        final_interior_points = []
        use_mvsam3d_sampling = resolve_sampling_backend() == "mvsam3d"
        if use_mvsam3d_sampling:
            final_surface_points = surface_points[:target_surface_points]
            final_interior_points = interior_points[:target_interior_points]
        else:
            prior_grid_size = volume_sample_size
            prior_grid_flag = set(grid_flag)
            interior_grid_flag = prior_grid_flag
            for i in range(surface_points.shape[0]):
                grid_index = point_grid_index(surface_points[i], min_bound, prior_grid_size)
                if grid_index not in prior_grid_flag:
                    prior_grid_flag.add(grid_index)
                    grid_flag[grid_index] = 1
                    final_surface_points.append(surface_points[i])
            for i in range(interior_points.shape[0]):
                grid_index = point_grid_index(interior_points[i], min_bound, prior_grid_size)
                if grid_index not in interior_grid_flag:
                    interior_grid_flag.add(grid_index)
                    grid_flag[grid_index] = 1
                    final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    # Render the final pcd with interior filling as a turntable video
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
    )
    if not video_writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {base_path}/{case_name}/final_pcd.mp4"
        )

    vis.add_geometry(all_pcd)
    # vis.add_geometry(coorindate)
    view_control = vis.get_view_control()
    for j in range(360):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()
    vis.destroy_window()

    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )
    if not video_writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {base_path}/{case_name}/final_data.mp4"
        )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)

    track_data = process_unique_points(track_data)

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)
