# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import numpy as np
import open3d as o3d
import json
import pickle
import imageio
import imageio.v3 as iio
from tqdm import tqdm
import os
from argparse import ArgumentParser
from PIL import Image, ImageDraw

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument(
    "--debug-width",
    type=int,
    default=1280,
    help="Debug rendering width per viewpoint.",
)
parser.add_argument(
    "--debug-height",
    type=int,
    default=720,
    help="Debug rendering height per viewpoint.",
)
parser.add_argument(
    "--debug-panel-mode",
    type=str,
    choices=["camera", "view"],
    default="camera",
    help="camera: cam0/cam1/cam2 raw panels, view: merged cloud with front/side/top.",
)
parser.add_argument(
    "--debug-camera-zoom",
    type=float,
    default=0.11,
    help="Camera panel zoom. Smaller value means closer view.",
)
parser.add_argument(
    "--debug-focus-quantile",
    type=float,
    default=0.90,
    help=(
        "Only for visualization: keep nearest quantile around median center "
        "to suppress far outlier rays."
    ),
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
debug_width = args.debug_width
debug_height = args.debug_height
debug_panel_mode = args.debug_panel_mode
debug_camera_zoom = args.debug_camera_zoom
debug_focus_quantile = args.debug_focus_quantile


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


def getPcdFromDepth(depth, intrinsic):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic).T
    points = points.reshape(H, W, 3)
    return points


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
    total_points = []
    total_colors = []
    total_masks = []
    for i in range(num_cam):
        color = iio.imread(f"{path}/color/{i}/{frame_idx}.png")
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=2)
        if color.shape[-1] == 4:
            color = color[:, :, :3]
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 5.5)
        points_flat = points.reshape(-1, 3)
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)
    # pcd = o3d.geometry.PointCloud()
    # visualize_points = []
    # visualize_colors = []
    # for i in range(num_cam):
    #     visualize_points.append(
    #         total_points[i][total_masks[i]].reshape(-1, 3)
    #     )
    #     visualize_colors.append(
    #         total_colors[i][total_masks[i]].reshape(-1, 3)
    #     )
    # visualize_points = np.concatenate(visualize_points)
    # visualize_colors = np.concatenate(visualize_colors)
    # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
    # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
    # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
    # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
    # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
    # visualize_points = visualize_points[mask]
    # visualize_colors = visualize_colors[mask]

    # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])
    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_raw_points_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
    raw_points_per_cam = []
    raw_colors_per_cam = []
    for i in range(num_cam):
        color = iio.imread(f"{path}/color/{i}/{frame_idx}.png")
        if color.ndim == 2:
            color = np.repeat(color[..., None], 3, axis=2)
        if color.shape[-1] == 4:
            color = color[:, :, :3]
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0

        points = getPcdFromDepth(depth, intrinsic=intrinsics[i])
        valid_raw_mask = np.logical_and(np.isfinite(depth), depth > 0.0)
        points_flat = points.reshape(-1, 3)
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]
        points_world = points_world.reshape(points.shape)

        raw_points = points_world[valid_raw_mask].reshape(-1, 3)
        raw_colors = color[valid_raw_mask].reshape(-1, 3)
        raw_points_per_cam.append(raw_points)
        raw_colors_per_cam.append(raw_colors)

    raw_points_merged = np.concatenate(raw_points_per_cam, axis=0)
    raw_colors_merged = np.concatenate(raw_colors_per_cam, axis=0)
    return raw_points_per_cam, raw_colors_per_cam, raw_points_merged, raw_colors_merged


def set_view(view_control, lookat, front, up, zoom):
    view_control.set_lookat(lookat.tolist())
    view_control.set_front(front)
    view_control.set_up(up)
    view_control.set_zoom(zoom)


def get_focus_points(points, colors, focus_quantile):
    if points.shape[0] == 0:
        return points, colors, np.zeros((3,), dtype=np.float32)
    center = np.median(points, axis=0)
    q = float(np.clip(focus_quantile, 0.05, 1.0))
    if q >= 1.0:
        return points, colors, center

    dist = np.linalg.norm(points - center[None, :], axis=1)
    radius = np.quantile(dist, q)
    keep = dist <= radius
    if keep.sum() < 2000:
        # Avoid over-clipping on sparse frames.
        keep = dist <= np.quantile(dist, 0.95)
    focus_points = points[keep]
    focus_colors = colors[keep]
    focus_center = np.median(focus_points, axis=0)
    return focus_points, focus_colors, focus_center


def draw_view_labels(frame_concat, frame_idx, panel_labels):
    image = Image.fromarray(frame_concat)
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), f"frame={frame_idx}", fill=(255, 255, 255))
    panel_width = frame_concat.shape[1] // len(panel_labels)
    for panel_idx, label in enumerate(panel_labels):
        draw.text((panel_width * panel_idx + 12, 34), label, fill=(255, 255, 255))
    return np.asarray(image)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    frame_num = data["frame_num"]
    print(data["serial_numbers"])

    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    exist_dir(f"{base_path}/{case_name}/pcd")
    debug_dir = f"{base_path}/{case_name}/pcd_debug"
    exist_dir(debug_dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="raw_pcd_debug",
        width=debug_width,
        height=debug_height,
        visible=False,
    )
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.05, 0.05, 0.05])
    render_option.point_size = 4.0

    video_path = f"{debug_dir}/pcd.mp4"
    video_writer = imageio.get_writer(video_path, fps=30, macro_block_size=1)
    if debug_panel_mode == "view":
        panel_labels = ["front", "side", "top"]
        view_presets = (
            ([1.0, 0.0, -2.0], [0.0, 0.0, -1.0], 0.20),  # front-oblique
            ([0.0, 1.0, -2.0], [0.0, 0.0, -1.0], 0.20),  # side-oblique
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], 0.20),  # top-down
        )
    else:
        panel_labels = ["cam0 raw", "cam1 raw", "cam2 raw"]
        view_presets = None

    pcd = None
    for i in tqdm(range(frame_num)):
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )
        (
            raw_points_per_cam,
            raw_colors_per_cam,
            raw_points_merged,
            raw_colors_merged,
        ) = get_raw_points_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )

        if i == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points_merged)
            pcd.colors = o3d.utility.Vector3dVector(raw_colors_merged)
            vis.add_geometry(pcd)
            vis.reset_view_point(True)

            # Dump first-frame raw point clouds for direct inspection.
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(raw_points_merged)
            merged_pcd.colors = o3d.utility.Vector3dVector(raw_colors_merged)
            o3d.io.write_point_cloud(
                f"{debug_dir}/frame0_merged_raw.ply", merged_pcd, write_ascii=False
            )
            for cam_idx in range(num_cam):
                cam_pcd = o3d.geometry.PointCloud()
                cam_pcd.points = o3d.utility.Vector3dVector(raw_points_per_cam[cam_idx])
                cam_pcd.colors = o3d.utility.Vector3dVector(raw_colors_per_cam[cam_idx])
                o3d.io.write_point_cloud(
                    f"{debug_dir}/frame0_cam{cam_idx}_raw.ply",
                    cam_pcd,
                    write_ascii=False,
                )
        else:
            pcd.points = o3d.utility.Vector3dVector(raw_points_merged)
            pcd.colors = o3d.utility.Vector3dVector(raw_colors_merged)
            vis.update_geometry(pcd)

        view_control = vis.get_view_control()
        frames = []
        if debug_panel_mode == "view":
            for front, up, zoom in view_presets:
                render_points, render_colors, render_center = get_focus_points(
                    raw_points_merged, raw_colors_merged, debug_focus_quantile
                )
                pcd.points = o3d.utility.Vector3dVector(render_points)
                pcd.colors = o3d.utility.Vector3dVector(render_colors)
                vis.update_geometry(pcd)
                # Recompute camera bounds from current geometry so zoom works on focused points.
                vis.reset_view_point(True)
                set_view(view_control, render_center, front, up, zoom)
                view_control.unset_constant_z_near()
                view_control.unset_constant_z_far()
                vis.poll_events()
                vis.update_renderer()
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frames.append((frame * 255).astype(np.uint8))
        else:
            for cam_idx in range(num_cam):
                cam_points = raw_points_per_cam[cam_idx]
                cam_colors = raw_colors_per_cam[cam_idx]
                render_points, render_colors, cam_lookat = get_focus_points(
                    cam_points, cam_colors, debug_focus_quantile
                )
                pcd.points = o3d.utility.Vector3dVector(render_points)
                pcd.colors = o3d.utility.Vector3dVector(render_colors)
                vis.update_geometry(pcd)
                # Recompute camera bounds from current geometry so zoom works on focused points.
                vis.reset_view_point(True)
                set_view(
                    view_control,
                    cam_lookat,
                    [1.0, 0.0, -2.0],
                    [0.0, 0.0, -1.0],
                    debug_camera_zoom,
                )
                view_control.unset_constant_z_near()
                view_control.unset_constant_z_far()
                vis.poll_events()
                vis.update_renderer()
                frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                frames.append((frame * 255).astype(np.uint8))
        frame_concat = np.concatenate(frames, axis=1)
        frame_concat = draw_view_labels(frame_concat, i, panel_labels)
        if i == 0:
            iio.imwrite(f"{debug_dir}/frame0_raw_3view.png", frame_concat)
        video_writer.append_data(frame_concat)

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )

    video_writer.close()
    vis.destroy_window()
    print(f"Saved pcd debug video to: {video_path}")
