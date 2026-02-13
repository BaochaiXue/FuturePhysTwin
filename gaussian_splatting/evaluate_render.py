import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from tqdm import tqdm
import torch

# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_filter import (  # noqa: E402
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate render quality metrics.")
    parser.add_argument(
        "--render_path",
        type=str,
        default="./data/render_eval_data",
        help="Directory containing render-evaluation datasets (default: ./data/render_eval_data).",
    )
    parser.add_argument(
        "--human_mask_path",
        type=str,
        default="./data/different_types_human_mask",
        help="Directory containing human masks (default: ./data/different_types_human_mask).",
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default="./data/gaussian_data",
        help="Root directory of Gaussian data (default: ./data/gaussian_data).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gaussian_output_dynamic",
        help="Directory with rendered outputs to evaluate (default: ./gaussian_output_dynamic).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results",
        help="Directory to store evaluation logs (default: ./results).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./data_config.csv",
        help="Case allowlist CSV path (default: ./data_config.csv).",
    )
    parser.add_argument(
        "--input-base-path",
        type=str,
        default="./data/different_types",
        help="Input case root used with data_config.csv allowlist filtering.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_path = str(resolve_path_from_root(REPO_ROOT, args.render_path))
    human_mask_path = str(resolve_path_from_root(REPO_ROOT, args.human_mask_path))
    root_data_dir = str(resolve_path_from_root(REPO_ROOT, args.root_data_dir))
    output_dir = str(resolve_path_from_root(REPO_ROOT, args.output_dir))
    log_dir = str(resolve_path_from_root(REPO_ROOT, args.log_dir))
    config_path = resolve_path_from_root(REPO_ROOT, args.config_path)
    input_base_path = resolve_path_from_root(REPO_ROOT, args.input_base_path)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "output_dynamic.txt")

    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(input_base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "evaluate_render", input_base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    with open(log_file_path, "w") as log_file:

        scene_dirs = sorted(
            p for p in Path(render_path).iterdir() if p.is_dir()
        )
        scene_name = filter_candidates(
            [scene_dir.name for scene_dir in scene_dirs],
            allowed_cases,
            "evaluate_render",
            render_path,
        )

        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = (
            [],
            [],
            [],
            [],
        )
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

        scene_metrics = {}

        for scene in scene_name:

            scene_dir = os.path.join(root_data_dir, scene)
            output_scene_dir = os.path.join(output_dir, scene)
            render_path_dir = os.path.join(render_path, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)

            # Load frame split info
            with open(f"{render_path_dir}/split.json", "r") as f:
                info = json.load(f)
            frame_len = info["frame_len"]
            train_f_idx_range = list(
                range(info["train"][0] + 1, info["train"][1])
            )  # +1 if ignoring the first frame
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            print(
                "train indices range from",
                train_f_idx_range[0],
                "to",
                train_f_idx_range[-1],
            )
            print(
                "test indices range from",
                test_f_idx_range[0],
                "to",
                test_f_idx_range[-1],
            )

            psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            # for view_idx in range(3):
            for view_idx in range(1):  # only consider the first view

                for frame_idx in train_f_idx_range:
                    gt = np.array(
                        Image.open(
                            os.path.join(
                                render_path_dir,
                                "color",
                                str(view_idx),
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    gt_mask = np.array(
                        Image.open(
                            os.path.join(
                                render_path_dir,
                                "mask",
                                str(view_idx),
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    gt_mask = gt_mask.astype(np.float32) / 255.0

                    render = np.array(
                        Image.open(
                            os.path.join(
                                output_scene_dir, str(view_idx), f"{frame_idx:05d}.png"
                            )
                        )
                    )
                    render_mask = (
                        render[:, :, 3]
                        if render.shape[-1] == 4
                        else np.ones_like(render[:, :, 0])
                    )

                    human_mask = np.array(
                        Image.open(
                            os.path.join(
                                human_mask_dir,
                                "mask",
                                str(view_idx),
                                "0",
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    inv_human_mask = (1.0 - human_mask / 255.0).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_train.append(psnr(render_tensor, gt_tensor).item())
                    ssims_train.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_train.append(lpips(render_tensor, gt_tensor).item())
                    ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

                for frame_idx in test_f_idx_range:

                    gt = np.array(
                        Image.open(
                            os.path.join(
                                render_path_dir,
                                "color",
                                str(view_idx),
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    gt_mask = np.array(
                        Image.open(
                            os.path.join(
                                render_path_dir,
                                "mask",
                                str(view_idx),
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    gt_mask = gt_mask.astype(np.float32) / 255.0

                    render = np.array(
                        Image.open(
                            os.path.join(
                                output_scene_dir, str(view_idx), f"{frame_idx:05d}.png"
                            )
                        )
                    )
                    render_mask = (
                        render[:, :, 3]
                        if render.shape[-1] == 4
                        else np.ones_like(render[:, :, 0])
                    )

                    human_mask = np.array(
                        Image.open(
                            os.path.join(
                                human_mask_dir,
                                "mask",
                                str(view_idx),
                                "0",
                                f"{frame_idx}.png",
                            )
                        )
                    )
                    inv_human_mask = (1.0 - human_mask / 255.0).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_test.append(psnr(render_tensor, gt_tensor).item())
                    ssims_test.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_test.append(lpips(render_tensor, gt_tensor).item())
                    ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

            scene_metrics[scene] = {
                "psnr_train": np.mean(psnrs_train),
                "ssim_train": np.mean(ssims_train),
                "lpips_train": np.mean(lpipss_train),
                "iou_train": np.mean(ious_train),
                "psnr_test": np.mean(psnrs_test),
                "ssim_test": np.mean(ssims_test),
                "lpips_test": np.mean(lpipss_test),
                "iou_test": np.mean(ious_test),
            }

            all_psnrs_train.extend(psnrs_train)
            all_ssims_train.extend(ssims_train)
            all_lpipss_train.extend(lpipss_train)
            all_ious_train.extend(ious_train)

            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f"===== Scene: {scene} =====")
            print(f"\t PSNR (train): {np.mean(psnrs_train):.4f}")
            print(f"\t SSIM (train): {np.mean(ssims_train):.4f}")
            print(f"\t LPIPS (train): {np.mean(lpipss_train):.4f}")
            print(f"\t IoU (train): {np.mean(ious_train):.4f}")

            print(f"\t PSNR (test): {np.mean(psnrs_test):.4f}")
            print(f"\t SSIM (test): {np.mean(ssims_test):.4f}")
            print(f"\t LPIPS (test): {np.mean(lpipss_test):.4f}")
            print(f"\t IoU (test): {np.mean(ious_test):.4f}")

        print("===== Overall Results Across All Scenes =====")
        print(f"\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}")
        print(f"\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}")
        print(f"\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}")
        print(f"\t Overall IoU (train): {np.mean(all_ious_train):.4f}")

        print(f"\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}")
        print(f"\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}")
        print(f"\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}")
        print(f"\t Overall IoU (test): {np.mean(all_ious_test):.4f}")

        overall_psnr_train = np.mean(all_psnrs_train)
        overall_ssim_train = np.mean(all_ssims_train)
        overall_lpips_train = np.mean(all_lpipss_train)
        overall_iou_train = np.mean(all_ious_train)

        overall_psnr_test = np.mean(all_psnrs_test)
        overall_ssim_test = np.mean(all_ssims_test)
        overall_lpips_test = np.mean(all_lpipss_test)
        overall_iou_test = np.mean(all_ious_test)

        # Write overall metrics to log file
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES\n")
        log_file.write("=" * 80 + "\n\n")

        log_file.write(f"Overall PSNR (train): {overall_psnr_train:.6f}\n")
        log_file.write(f"Overall SSIM (train): {overall_ssim_train:.6f}\n")
        log_file.write(f"Overall LPIPS (train): {overall_lpips_train:.6f}\n")
        log_file.write(f"Overall IoU (train): {overall_iou_train:.6f}\n\n")

        log_file.write(f"Overall PSNR (test): {overall_psnr_test:.6f}\n")
        log_file.write(f"Overall SSIM (test): {overall_ssim_test:.6f}\n")
        log_file.write(f"Overall LPIPS (test): {overall_lpips_test:.6f}\n")
        log_file.write(f"Overall IoU (test): {overall_iou_test:.6f}\n\n")

        # Create a compact table of all scene metrics
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE\n")
        log_file.write("=" * 80 + "\n\n")

        # Header
        log_file.write(
            f"{'Scene':<50} | {'PSNR-train':<12} | {'SSIM-train':<12} | {'LPIPS-train':<14} | {'IoU-train':<12} | "
        )
        log_file.write(
            f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n"
        )
        log_file.write("-" * 160 + "\n")

        # Scene rows
        for scene in scene_name:
            metrics = scene_metrics[scene]
            log_file.write(f"{scene[:50]:<50} | ")
            log_file.write(f"{metrics['psnr_train']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_train']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_train']:<14.6f} | ")
            log_file.write(f"{metrics['iou_train']:<12.6f} | ")

            log_file.write(f"{metrics['psnr_test']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_test']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_test']:<14.6f} | ")
            log_file.write(f"{metrics['iou_test']:<12.6f}\n")

        # Overall row
        log_file.write("-" * 160 + "\n")
        log_file.write(f"{'OVERALL':<50} | ")
        log_file.write(f"{overall_psnr_train:<12.6f} | ")
        log_file.write(f"{overall_ssim_train:<12.6f} | ")
        log_file.write(f"{overall_lpips_train:<14.6f} | ")
        log_file.write(f"{overall_iou_train:<12.6f} | ")

        log_file.write(f"{overall_psnr_test:<12.6f} | ")
        log_file.write(f"{overall_ssim_test:<12.6f} | ")
        log_file.write(f"{overall_lpips_test:<14.6f} | ")
        log_file.write(f"{overall_iou_test:<12.6f}\n")

        print(f"\nMetrics have been saved to: {log_file_path}")
