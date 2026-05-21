# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic CoTracker query-point subsampling.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

num_cam = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == num_cam
device = "cuda"


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    # shape: mask (H, W), boolean foreground.
    return mask


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def subsample_query_pixels(query_pixels, num_queries, seed, camera_idx):
    generator = torch.Generator(device=query_pixels.device)
    generator.manual_seed(int(seed) + int(camera_idx))
    permutation = torch.randperm(
        query_pixels.shape[0], device=query_pixels.device, generator=generator
    )
    return query_pixels[permutation[:num_queries]]


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")
        # Load the video
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # shape: frames (T, H, W, C) -> video (1, T, C, H, W)
        # Load the first-frame mask to get all query points from all masks
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            # shape: current_mask (H, W).
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)
                # shape: mask (H, W), union of all first-frame masks.

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # shape: (Q_all, 2) in (y, x) order.
        # Revert x and y.
        query_pixels = query_pixels[:, ::-1]
        # shape: (Q_all, 2) in (x, y) order.
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )  # shape: (Q_all, 3), columns are (query_frame, x, y).
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        # shape: (Q_all, 3), CoTracker query tensor before subsampling.
        # Randomly select up to 5000 query points.
        num_query_pixels = query_pixels.shape[0]
        target_num_queries = min(num_query_pixels, 5000)
        query_pixels = subsample_query_pixels(
            query_pixels, target_num_queries, args.seed, i
        )  # shape: (Q, 3), where Q <= 5000.

        # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        # pred_tracks, pred_visibility = cotracker(video, queries=query_pixels[None], backward_tracking=True)
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])
        # shape: query_pixels[None] is (1, Q, 3).

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # shape: pred_tracks (1, T_window, Q, 2); pred_visibility (1, T_window, Q).
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        # Save the tracking data into npz
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        # shape: (T_window, Q, 2), converted from (x, y) to (y, x).
        visibility_to_save = pred_visibility[0].cpu().numpy()
        # shape: (T_window, Q), boolean visibility per tracked query.
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
