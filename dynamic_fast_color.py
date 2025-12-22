from __future__ import annotations

"""
Color-finetuning entry point for Gaussian Splatting with optional LBS-driven pose deformation.

Inputs
------
- Canonical frame data at ``ModelParams.source_path`` (Stage A frame) plus every additional frame directory discovered via ``--frames_dir`` (e.g. ``per_frame_gaussian_data/<frame>/<scene>/``), providing per-frame RGB, depth, masks, ``camera_meta.pkl``, and ``observation.ply``.
- Canonical Gaussian checkpoint ``<model_path>/canonical_gaussians.npz`` (fallback) or ``color_refine/canonical_gaussians_color.npz`` when resuming a previous colour run; this supplies the shared Gaussian kernels (xyz/scaling/rotation/SH/opacity).
- Offline pose cache ``<model_path>/lbs_pose_cache.pt`` produced by
  ``precompute_lbs_pose_cache.py``. This file is now mandatory for colour
  refinement: colour-only mode samples its cached ``posed_xyz/quat`` per frame,
  while ``--train_all_parameter`` additionally reuses the stored bones, weights,
  and motions to perform gradient-friendly LBS deformation.

Outputs
-------
- Updated training metadata (``cfg_args``, ``input.ply``, ``cameras.json``) and standard checkpoints, now stored under ``<model_path>/color_refine/`` when ``--color_only`` is active.
- ``exposure.json`` and TensorBoard logs (when enabled), likewise written to ``color_refine`` for colour runs.
- Refined appearance parameters saved to ``<model_path>/color_refine/canonical_gaussians_color.npz``, preserving the original canonical checkpoint.
"""

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path
from random import randint
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import re
from gaussian_splatting.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gaussian_renderer import render, network_gui
from gaussian_splatting.skinning import LBSDeformer, build_skinning_weights
from gaussian_splatting.dynamic_utils import compute_bone_transforms
from gaussian_splatting.scene import GaussianModel, Scene
from gaussian_splatting.scene.dataset_readers import readQQTTSceneInfo
from gaussian_splatting.utils.canonical_io import dump_gaussian_npz, load_canonical_npz
from gaussian_splatting.utils.camera_utils import cameraList_from_camInfos
from gaussian_splatting.utils.general_utils import get_expon_lr_func, safe_state
from gaussian_splatting.utils.image_utils import psnr  # noqa: F401  # kept for parity
from gaussian_splatting.utils.loss_utils import (
    C1 as SSIM_C1,
    C2 as SSIM_C2,
    anisotropic_loss,
    create_window as ssim_create_window,
    depth_loss,
    l1_loss,
    normal_loss,
    ssim,
)
from tqdm import tqdm
import math

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as SummaryWriterType
else:
    SummaryWriterType = Any

try:
    # Optional logging support; unavailable in minimal installs.
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment]
    TENSORBOARD_FOUND = False
    print("[warn] TensorBoard not available; training logs will not be recorded.")

try:
    # Differentiable SSIM kernel; gracefully falls back to PyTorch version.
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    fused_ssim = None  # type: ignore[assignment]
    FUSED_SSIM_AVAILABLE = False
    print(
        "[warn] fused_ssim not available; falling back to standard PyTorch SSIM implementation."
    )

try:
    # CUDA accelerated optimiser for sparse Gaussian updates.
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SparseGaussianAdam = None  # type: ignore[assignment]
    SPARSE_ADAM_AVAILABLE = False
    print(
        "[warn] SparseGaussianAdam not available; falling back to standard Adam optimiser."
    )


def _tensor_to_numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    return tensor.detach().cpu().float().numpy()


def _to_uint8_img(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    img = tensor.detach().float().cpu()
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 4:
        img = img[:3]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.min() < 0.0 or img.max() > 1.0:
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 1e-6:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = torch.zeros_like(img)
    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).round().byte().numpy().transpose(1, 2, 0)
    return img


def _save_tensor_as_png(
    tensor: Optional[torch.Tensor], path: Path, label: Optional[str]
) -> None:
    array = _to_uint8_img(tensor)
    if array is None:
        return
    _save_array_with_label(array, path, label)


def _annotate_image(img: Image.Image, label: Optional[str]) -> Image.Image:
    if not label:
        return img
    rgba = img.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    font = ImageFont.load_default()
    text = label
    if hasattr(draw, "textbbox"):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        text_w = x1 - x0
        text_h = y1 - y0
    else:
        text_w, text_h = draw.textsize(text, font=font)
    margin = 6
    padding = 4
    rect = (
        margin - padding,
        margin - padding,
        margin + text_w + padding,
        margin + text_h + padding,
    )
    draw.rectangle(rect, fill=(0, 0, 0, 192))
    draw.text((margin, margin), text, font=font, fill=(255, 255, 255, 255))
    return rgba.convert("RGB")


def _save_array_with_label(
    array: Optional[np.ndarray], path: Path, label: Optional[str]
) -> None:
    if array is None:
        return
    img = Image.fromarray(array)
    img = _annotate_image(img, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _sanitize_token(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "untitled"


def _normalize_depth_for_vis(
    depth: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    if depth is None:
        return None
    depth_cpu = depth.detach().float()
    if mask is not None:
        mask_vals = (mask > 0.5).detach()
        if mask_vals.ndim > 2:
            mask_vals = mask_vals.squeeze()
        mask_vals = mask_vals.bool()
        if mask_vals.shape != depth_cpu.shape:
            mask_vals = mask_vals.reshape_as(depth_cpu)
        valid = depth_cpu[mask_vals]
        if valid.numel() > 0:
            min_val = valid.min()
            max_val = valid.max()
        else:
            min_val = depth_cpu.min()
            max_val = depth_cpu.max()
    else:
        min_val = depth_cpu.min()
        max_val = depth_cpu.max()
    if max_val - min_val > 1e-6:
        norm = (depth_cpu - min_val) / (max_val - min_val)
    else:
        norm = torch.zeros_like(depth_cpu)
    return norm.unsqueeze(0).clamp(0.0, 1.0)


def _compute_l1_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(a - b).mean(0, keepdim=True)
    return diff.repeat(3, 1, 1)


def _compute_ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    channel = a.shape[0]
    window_size = 11
    window = ssim_create_window(window_size, channel).to(dtype=a.dtype, device=a.device)
    a4 = a.unsqueeze(0)
    b4 = b.unsqueeze(0)
    mu1 = F.conv2d(a4, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(b4, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(a4 * a4, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(b4 * b4, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(a4 * b4, window, padding=window_size // 2, groups=channel) - mu1_mu2
    )
    c1 = SSIM_C1
    c2 = SSIM_C2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    ssim_map = ssim_map.mean(1, keepdim=False)
    return ssim_map.squeeze(0)


def _make_contact_sheet(
    images: list[tuple[str, Optional[np.ndarray]]],
) -> Optional[np.ndarray]:
    valid: list[tuple[str, np.ndarray]] = [
        (label, img) for label, img in images if img is not None
    ]
    if not valid:
        return None
    pil_images: list[Image.Image] = []
    heights = []
    widths = []
    for label, array in valid:
        img = Image.fromarray(array)
        heights.append(img.height)
        widths.append(img.width)
        pil_images.append(_annotate_image(img, None))
    min_h = min(heights)
    resized = []
    for img in pil_images:
        if img.height != min_h:
            scale = min_h / img.height
            new_w = max(1, int(round(img.width * scale)))
            img = img.resize((new_w, min_h), Image.BILINEAR)
            resized.append(img)
        else:
            resized.append(img)
    n = len(resized)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cell_w = max(img.width for img in resized)
    cell_h = max(img.height for img in resized)
    font = ImageFont.load_default()
    sheet = Image.new(
        "RGB", (cols * cell_w, rows * (cell_h + font.size + 8)), color=(0, 0, 0)
    )
    draw = ImageDraw.Draw(sheet)
    for idx, (orig, img) in enumerate(zip(valid, resized)):
        label, _ = orig
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * (cell_h + font.size + 8)
        sheet.paste(img, (x, y))
        text_y = y + cell_h + 4
        draw.text((x, text_y), label, font=font, fill=(255, 255, 255))
    return np.array(sheet)


def _expand_mask(
    mask: Optional[torch.Tensor], height: int, width: int
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.float()
    mask = mask.expand(-1, height, width)
    return mask


def _make_bg_patch(
    bg: Optional[torch.Tensor], height: int, width: int
) -> Optional[torch.Tensor]:
    if bg is None:
        return None
    return bg.view(3, 1, 1).expand(3, height, width)


def dump_viz(
    iteration: int,
    frame_id: int,
    cam_uid: int,
    image_name: Optional[str],
    root: Path,
    *,
    pred_rgb_raw: torch.Tensor,
    pred_rgb_masked: torch.Tensor,
    pred_seg_raw: torch.Tensor,
    pred_seg_masked: torch.Tensor,
    gt_raw: torch.Tensor,
    gt_masked: torch.Tensor,
    alpha_mask: Optional[torch.Tensor],
    occ_mask: Optional[torch.Tensor],
    inv_occ_mask: Optional[torch.Tensor],
    depth_raw: Optional[torch.Tensor],
    depth_masked: Optional[torch.Tensor],
    normal_raw: Optional[torch.Tensor],
    normal_masked: Optional[torch.Tensor],
    bg_color: Optional[torch.Tensor],
) -> None:
    height, width = pred_rgb_raw.shape[1:]
    pred_rgb_clamped = pred_rgb_raw.clamp(0.0, 1.0)
    pred_rgb_masked_clamped = pred_rgb_masked.clamp(0.0, 1.0)
    gt_raw_clamped = gt_raw.clamp(0.0, 1.0)
    gt_masked_clamped = gt_masked.clamp(0.0, 1.0)
    alpha_vis = _expand_mask(alpha_mask, height, width)
    occ_vis = _expand_mask(occ_mask, height, width)
    inv_occ_vis = _expand_mask(inv_occ_mask, height, width)
    depth_vis = _normalize_depth_for_vis(depth_raw, alpha_mask)
    depth_vis_masked = _normalize_depth_for_vis(depth_masked, alpha_mask)
    normal_vis = (
        normal_raw.permute(2, 0, 1)
        if normal_raw is not None and normal_raw.ndim == 3
        else normal_raw
    )
    normal_masked_vis = (
        normal_masked.permute(2, 0, 1)
        if normal_masked is not None and normal_masked.ndim == 3
        else normal_masked
    )
    if normal_vis is not None:
        normal_vis = (normal_vis + 1.0) / 2.0
    if normal_masked_vis is not None:
        normal_masked_vis = (normal_masked_vis + 1.0) / 2.0
    bg_patch = _make_bg_patch(bg_color, height, width)
    l1_full = _compute_l1_map(pred_rgb_clamped, gt_raw_clamped)
    l1_masked = _compute_l1_map(pred_rgb_masked_clamped, gt_masked_clamped)
    ssim_full = (
        _compute_ssim_map(pred_rgb_clamped, gt_raw_clamped)
        .clamp(0.0, 1.0)
        .repeat(3, 1, 1)
    )
    ssim_masked = (
        _compute_ssim_map(pred_rgb_masked_clamped, gt_masked_clamped)
        .clamp(0.0, 1.0)
        .repeat(3, 1, 1)
    )
    base_token = _sanitize_token(image_name) if image_name else f"frame_{frame_id:05d}"
    base_name = f"i{iteration:07d}_c{cam_uid:04d}_f{frame_id:05d}_{base_token}"
    image_entries: list[tuple[str, Optional[torch.Tensor], str]] = [
        ("pred_rgb_raw", pred_rgb_raw, "Pred RGB (raw)"),
        ("pred_rgb_masked", pred_rgb_masked, "Pred RGB (masked)"),
        ("pred_seg_raw", pred_seg_raw, "Pred Alpha/Seg (raw)"),
        ("pred_seg_masked", pred_seg_masked, "Pred Alpha/Seg (masked)"),
        ("gt_rgb_raw", gt_raw, "GT RGB (raw)"),
        ("gt_rgb_masked", gt_masked, "GT RGB (masked)"),
        ("alpha_mask", alpha_vis, "Alpha Mask"),
        ("occ_mask", occ_vis, "Occlusion Mask"),
        ("inv_occ_mask", inv_occ_vis, "Inverse Occlusion"),
        ("depth", depth_vis, "Depth"),
        ("depth_masked", depth_vis_masked, "Depth (masked)"),
        ("normal", normal_vis, "Normal"),
        ("normal_masked", normal_masked_vis, "Normal (masked)"),
        ("l1", l1_full, "L1 Map"),
        ("l1_masked", l1_masked, "L1 Map (masked)"),
        ("ssim", ssim_full, "SSIM Map"),
        ("ssim_masked", ssim_masked, "SSIM Map (masked)"),
        ("background", bg_patch, "Background Color"),
    ]
    for class_dir, tensor, label in image_entries:
        target_path = root / class_dir / f"{base_name}.png"
        _save_tensor_as_png(tensor, target_path, label)
    contact_images = [
        ("Pred RGB (raw)", _to_uint8_img(pred_rgb_raw)),
        ("Pred RGB (masked)", _to_uint8_img(pred_rgb_masked)),
        ("GT RGB (raw)", _to_uint8_img(gt_raw)),
        ("GT RGB (masked)", _to_uint8_img(gt_masked)),
        ("Alpha Mask", _to_uint8_img(alpha_vis)),
        ("Occlusion Mask", _to_uint8_img(occ_vis)),
        ("L1 Map (masked)", _to_uint8_img(l1_masked)),
        ("SSIM Map (masked)", _to_uint8_img(ssim_masked)),
        ("Background Color", _to_uint8_img(bg_patch)),
    ]
    contact = _make_contact_sheet(contact_images)
    if contact is not None:
        contact_path = root / "contact_sheet" / f"{base_name}.png"
        _save_array_with_label(contact, contact_path, "Contact Sheet")
    npz_payload: dict[str, np.ndarray] = {
        "pred_rgb_raw": _tensor_to_numpy(pred_rgb_raw),
        "pred_rgb_masked": _tensor_to_numpy(pred_rgb_masked),
        "pred_seg_raw": _tensor_to_numpy(pred_seg_raw),
        "pred_seg_masked": _tensor_to_numpy(pred_seg_masked),
        "gt_raw": _tensor_to_numpy(gt_raw),
        "gt_masked": _tensor_to_numpy(gt_masked),
        "l1_full": _tensor_to_numpy(l1_full),
        "l1_masked": _tensor_to_numpy(l1_masked),
        "ssim_full": _tensor_to_numpy(ssim_full),
        "ssim_masked": _tensor_to_numpy(ssim_masked),
    }
    if alpha_mask is not None:
        npz_payload["alpha_mask"] = _tensor_to_numpy(alpha_mask)
    if occ_mask is not None:
        npz_payload["occ_mask"] = _tensor_to_numpy(occ_mask)
    if inv_occ_mask is not None:
        npz_payload["inv_occ_mask"] = _tensor_to_numpy(inv_occ_mask)
    if depth_raw is not None:
        npz_payload["depth_raw"] = _tensor_to_numpy(depth_raw)
    if depth_masked is not None:
        npz_payload["depth_masked"] = _tensor_to_numpy(depth_masked)
    if normal_raw is not None:
        npz_payload["normal_raw"] = _tensor_to_numpy(normal_raw)
    if normal_masked is not None:
        npz_payload["normal_masked"] = _tensor_to_numpy(normal_masked)
    if bg_color is not None:
        npz_payload["bg_color"] = _tensor_to_numpy(bg_color)
    snapshot_dir = root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    np.savez(snapshot_dir / f"{base_name}.npz", **npz_payload)


def load_lbs_data(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"LBS metadata not found: {path}")
    payload = torch.load(path, map_location="cpu")
    device = torch.device("cuda")
    return {
        "bones0": payload["bones0"].float().to(device),
        "relations": payload["relations"].long().to(device),
        "skin_indices": payload["skin_indices"].long().to(device),
        "skin_weights": payload["skin_weights"].float().to(device),
        "motions": payload["motions"].float().to(device),
    }


def assign_canonical_parameters(
    gaussians: GaussianModel, canonical: dict[str, torch.Tensor]
) -> None:
    gaussians._xyz = torch.nn.Parameter(canonical["xyz"].clone().requires_grad_(True))
    scaling_param = gaussians.scaling_inverse_activation(canonical["scaling"])
    gaussians._scaling = torch.nn.Parameter(scaling_param.clone().requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(
        canonical["rotation"].clone().requires_grad_(True)
    )
    opacity_param = gaussians.inverse_opacity_activation(canonical["opacity"])
    gaussians._opacity = torch.nn.Parameter(opacity_param.clone().requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(
        canonical["features_dc"].clone().requires_grad_(True)
    )
    gaussians._features_rest = torch.nn.Parameter(
        canonical["features_rest"].clone().requires_grad_(True)
    )
    gaussians.active_sh_degree = canonical["active_sh_degree"]
    gaussians.max_sh_degree = canonical["max_sh_degree"]
    device = gaussians._xyz.device
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=device)
    gaussians.xyz_gradient_accum = torch.zeros(
        (gaussians.get_xyz.shape[0], 1), device=device
    )
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.snapshot_canonical_pose()


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: Sequence[int],
    saving_iterations: Sequence[int],
    checkpoint_iterations: Sequence[int],
    checkpoint: Optional[str],
    debug_from: int,
    *,
    color_only: bool = False,
    freeze_geometry: bool = False,
    lbs_path: Optional[Path] = None,
    frames_dir: Optional[Path] = None,
    lbs_pose_cache: Optional[Path] = None,
    viz_every: int = 0,
    viz_frames: Optional[set[int]] = None,
    viz_cams: Optional[set[int]] = None,
    viz_root: Optional[Path] = None,
    train_frames: Optional[set[int]] = None,
    train_cams: Optional[set[int]] = None,
    train_all_parameter: bool = False,
    mask_pred_with_alpha: bool = False,
    lbs_refresh_interval: int = 0,
    train_all_bbox_pad: float = 0.25,
    lambda_alpha_leak: float = 0.1,
) -> None:
    """
    Run the full Gaussian optimisation loop for a preprocessed QQTT scene.
    """

    # Refuse to launch with the sparse Adam optimiser when its CUDA extension is missing.
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "Trying to use sparse adam but it is not installed, "
            "please install the correct rasterizer using pip install [3dgs_accel]."
        )

    canonical_dir = Path(dataset.model_path)
    refine_dir = canonical_dir / "color_refine"
    if color_only:
        refine_dir.mkdir(parents=True, exist_ok=True)
        dataset.model_path = str(refine_dir)

    lbs_deformer: Optional[LBSDeformer] = None
    lbs_motions: Optional[torch.Tensor] = None
    pose_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None
    canonical_xyz_ref: Optional[torch.Tensor] = None
    canonical_rot_ref: Optional[torch.Tensor] = None
    bone_transform_cache: Optional[torch.Tensor] = None
    lbs_bones0: Optional[torch.Tensor] = None
    lbs_relations_cache: Optional[torch.Tensor] = None
    bone_positions: Optional[torch.Tensor] = None
    skin_k: Optional[int] = None

    first_iter: int = 0
    # Create the output folder structure and, if available, a TensorBoard writer.
    tb_writer: Optional[SummaryWriterType] = prepare_output_and_logger(dataset)
    # Build the Gaussian container and wrap it in a Scene instance that handles data IO.
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # Resume from a saved checkpoint by restoring the parameters and iteration index.
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Choose background colour based on whether the dataset prefers a white canvas.
    background_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_color, dtype=torch.float32, device="cuda")

    # CUDA events track per-iteration durations for logging.
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    should_freeze_geometry = color_only or freeze_geometry
    if train_all_parameter:
        should_freeze_geometry = False

    if lbs_path is not None:
        print(
            "[LBS] Warning: --lbs_path is deprecated; the colour stage now "
            "requires an offline pose cache and ignores lbs_data.pt"
        )

    if color_only:
        canonical_color_path = (
            canonical_dir / "color_refine" / "canonical_gaussians_color.npz"
        )
        canonical_base_path = canonical_dir / "canonical_gaussians.npz"
        if canonical_color_path.exists():
            try:
                canonical_color_path.unlink()
                print(f"[Colour] Removed stale {canonical_color_path}")
            except OSError as exc:
                print(
                    f"[Colour] Warning: failed to remove {canonical_color_path}: {exc}"
                )
        canonical_path = (
            canonical_color_path
            if canonical_color_path.exists()
            else canonical_base_path
        )
        canonical_data = load_canonical_npz(canonical_path)
        assign_canonical_parameters(gaussians, canonical_data)
        if should_freeze_geometry:
            gaussians.freeze_geometry()
        # When jointly training geometry (--train_all_parameter), reduce all
        # colour-stage learning rates to stabilise optimisation.
        lr_scale = 0.2 if train_all_parameter else 1.0
        feature_lr = opt.feature_lr * lr_scale
        rest_lr = (opt.feature_lr / 20.0) * lr_scale
        param_groups = [
            {"params": [gaussians._features_dc], "lr": feature_lr, "name": "f_dc"},
            {
                "params": [gaussians._features_rest],
                "lr": rest_lr,
                "name": "f_rest",
            },
            {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        ]
        if train_all_parameter:
            xyz_lr = opt.position_lr_init * gaussians.spatial_lr_scale
            param_groups.extend(
                [
                    {
                        "params": [gaussians._xyz],
                        "lr": xyz_lr * lr_scale,
                        "name": "xyz",
                    },
                    {
                        "params": [gaussians._scaling],
                        "lr": opt.scaling_lr * lr_scale,
                        "name": "scaling",
                    },
                    {
                        "params": [gaussians._rotation],
                        "lr": opt.rotation_lr * lr_scale,
                        "name": "rotation",
                    },
                ]
            )
        gaussians.optimizer = torch.optim.Adam(param_groups)
        gaussians.exposure_optimizer = torch.optim.Adam(
            [
                {
                    "params": [gaussians._exposure],
                    "lr": opt.feature_lr * lr_scale,
                    "name": "exposure",
                }
            ]
        )
        pose_cache_path = (
            lbs_pose_cache
            if lbs_pose_cache is not None
            else canonical_dir / "lbs_pose_cache.pt"
        )
        if not pose_cache_path.is_file():
            raise FileNotFoundError(
                "Pose cache required for colour refinement but not found at "
                f"{pose_cache_path}. Pass --lbs_pose_cache or run "
                "precompute_lbs_pose_cache.py to generate it."
            )
        cache_payload = torch.load(pose_cache_path, map_location="cpu")
        cache_dict = cache_payload.get("pose_cache")
        if cache_dict is None:
            raise KeyError(f"{pose_cache_path} does not contain a 'pose_cache' entry.")
        if should_freeze_geometry:
            pose_cache = {
                int(frame_id): (
                    entry["xyz"].to("cuda").float(),
                    entry["quat"].to("cuda").float(),
                )
                for frame_id, entry in cache_dict.items()
            }
            print(
                f"[LBS] Loaded offline pose cache with {len(pose_cache)} frames from "
                f"{pose_cache_path}"
            )
            canonical_xyz_ref = gaussians.get_xyz_canonical().detach().clone()
            canonical_rot_ref = gaussians.get_rotation_canonical().detach().clone()
        cache_bones0 = cache_payload.get("bones0")
        cache_relations = cache_payload.get("relations")
        cache_weights_idx = cache_payload.get("weights_idx")
        cache_weights = cache_payload.get("weights")
        cache_bone_positions = cache_payload.get("bone_positions")
        cache_bone_transforms = cache_payload.get("bone_transforms")
        if cache_bones0 is not None:
            lbs_bones0 = cache_bones0.float().to("cuda")
        if cache_relations is not None:
            lbs_relations_cache = cache_relations.long().to("cuda")
        if cache_weights_idx is not None:
            skin_indices_init = cache_weights_idx.long().to("cuda")
        else:
            skin_indices_init = None
        if cache_weights is not None:
            skin_weights_init = cache_weights.float().to("cuda")
            skin_k = cache_weights.shape[1]
        else:
            skin_weights_init = None
        if cache_bone_positions is not None:
            bone_positions = cache_bone_positions.float().to("cuda")
        if train_all_parameter:
            missing_keys: list[str] = []
            if cache_bones0 is None:
                missing_keys.append("bones0")
            if cache_relations is None:
                missing_keys.append("relations")
            if cache_weights_idx is None:
                missing_keys.append("weights_idx")
            if cache_weights is None:
                missing_keys.append("weights")
            if cache_bone_positions is None:
                missing_keys.append("bone_positions")
            if missing_keys:
                raise KeyError(
                    "Pose cache is missing required entries for train_all_parameter: "
                    + ", ".join(missing_keys)
                )
            if (
                lbs_bones0 is None
                or lbs_relations_cache is None
                or skin_indices_init is None
                or skin_weights_init is None
                or bone_positions is None
            ):
                raise RuntimeError(
                    "Pose cache lacks required tensors to instantiate LBSDeformer."
                )
            lbs_deformer = LBSDeformer(
                lbs_bones0,
                lbs_relations_cache,
                skin_indices_init,
                skin_weights_init,
            )
            lbs_motions = bone_positions - bone_positions[0:1]
            if cache_bone_transforms is not None:
                bone_transform_cache = cache_bone_transforms.float().to("cuda")
            else:
                print(
                    "[LBS] Warning: pose cache missing precomputed bone transforms; "
                    "train_all_parameter will recompute them on-the-fly."
                )
    elif should_freeze_geometry:
        gaussians.freeze_geometry()

    use_sparse_adam = (
        not color_only and opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    )
    # Smoothly anneal the depth supervision coefficient during training.
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    # Build a camera pool spanning the canonical frame and additional frames.
    scene_path = Path(dataset.source_path)
    scene_name = scene_path.name
    try:
        canonical_frame_idx = int(scene_path.parent.name)
    except ValueError:
        canonical_frame_idx = 0

    camera_entries: list[tuple[int, Any]] = []
    seen_paths: set[Path] = set()

    def maybe_refresh_lbs(iteration: int) -> None:
        nonlocal bone_transform_cache
        if (
            lbs_refresh_interval <= 0
            or not train_all_parameter
            or lbs_deformer is None
            or bone_positions is None
            or lbs_bones0 is None
            or lbs_relations_cache is None
            or skin_k is None
        ):
            return
        if iteration % lbs_refresh_interval != 0:
            return
        print(
            f"[LBS] Refreshing skinning weights and bone transforms at iteration {iteration}"
        )
        with torch.no_grad():
            canonical_xyz_live = gaussians.get_xyz.detach()
            new_indices, new_weights = build_skinning_weights(
                canonical_xyz_live,
                lbs_bones0,
                k=skin_k,
            )
            lbs_deformer.skin_indices = new_indices.to(
                device=lbs_bones0.device, dtype=torch.long
            )
            lbs_deformer.skin_weights = new_weights.to(
                device=lbs_bones0.device, dtype=torch.float32
            )
            num_frames = bone_positions.shape[0]
            num_bones = bone_positions.shape[1]
            if (
                bone_transform_cache is None
                or bone_transform_cache.shape[0] != num_frames
                or bone_transform_cache.shape[1] != num_bones
            ):
                bone_transform_cache = torch.zeros(
                    (num_frames, num_bones, 4, 4),
                    device=lbs_bones0.device,
                    dtype=torch.float32,
                )
            for frame_id in range(num_frames):
                motions = bone_positions[frame_id] - bone_positions[0]
                transforms = compute_bone_transforms(
                    lbs_bones0,
                    motions,
                    lbs_relations_cache,
                    device=lbs_bones0.device,
                    step=frame_id,
                )
                bone_transform_cache[frame_id] = transforms

    def ingest_frame_dir(frame_path: Path) -> None:
        resolved = frame_path.resolve()
        if resolved in seen_paths or not frame_path.is_dir():
            return
        try:
            frame_idx_local = int(frame_path.parent.name)
        except ValueError:
            return
        if train_frames is not None and frame_idx_local not in train_frames:
            return
        try:
            scene_info_local = readQQTTSceneInfo(
                str(frame_path),
                dataset.images,
                dataset.depths,
                dataset.eval,
                dataset.train_test_exp,
                dataset.use_masks,
                dataset.gs_init_opt,
                dataset.pts_per_triangles,
                dataset.use_high_res,
            )
        except FileNotFoundError:
            return
        extra_cams_local = cameraList_from_camInfos(
            scene_info_local.train_cameras,
            1.0,
            dataset,
            scene_info_local.is_nerf_synthetic,
            False,
        )
        selected_cams: list[Any] = []
        for cam_local in extra_cams_local:
            cam_uid = getattr(cam_local, "uid", None)
            if train_cams is not None and cam_uid not in train_cams:
                continue
            selected_cams.append(cam_local)
        if selected_cams:
            for cam_local in selected_cams:
                camera_entries.append((frame_idx_local, cam_local))
        seen_paths.add(resolved)

    if frames_dir is not None:
        frames_root = Path(frames_dir)
        for frame_path in sorted(frames_root.glob(f"*/{scene_name}")):
            ingest_frame_dir(frame_path)

    if not camera_entries and (
        train_frames is None or canonical_frame_idx in train_frames
    ):
        ingest_frame_dir(scene_path)

    if not camera_entries:
        raise RuntimeError("No training cameras available for colour stage.")
    frame_summary: dict[int, int] = {}
    for frame_id, _ in camera_entries:
        frame_summary[frame_id] = frame_summary.get(frame_id, 0) + 1
    summary_str = ", ".join(
        f"{fid}: {count} cams" for fid, count in sorted(frame_summary.items())
    )
    print(f"[Frames] Training across frames -> {summary_str}")

    entry_pool = list(range(len(camera_entries)))

    # Rolling averages used solely for smoother console logging.
    ema_loss_for_log: float = 0.0
    ema_l1depth_for_log: float = 0.0
    num_motion_frames = lbs_motions.shape[0] if lbs_motions is not None else 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viz_frames_set = set(viz_frames) if viz_frames is not None else None
    viz_cams_set = set(viz_cams) if viz_cams is not None else None
    viz_root_path = (
        Path(viz_root)
        if viz_root is not None
        else Path(dataset.model_path) / "debug_visualize" / "color_stage"
    )
    for iteration in range(first_iter, opt.iterations + 1):
        # Lazily attach to the interactive viewer so render previews can be streamed.
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes: Optional[memoryview] = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifier,
                ) = network_gui.receive()
                if custom_cam is not None:
                    render_output = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=scaling_modifier,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )
                    net_image = render_output["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        # Start the per-iteration timer once viewer communication has settled.
        iter_start.record()

        # Let the optimiser update its schedule (e.g. cosine LR decay).
        gaussians.update_learning_rate(iteration)

        # Gradually unlock higher SH degrees for a smooth optimisation schedule.
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not entry_pool:
            entry_pool = list(range(len(camera_entries)))
        idx_choice = randint(0, len(entry_pool) - 1)
        entry_idx = entry_pool.pop(idx_choice)
        frame_id, viewpoint_cam = camera_entries[entry_idx]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        cam_uid = getattr(viewpoint_cam, "uid", -1)
        image_token = getattr(viewpoint_cam, "image_name", None)
        should_viz = (
            viz_every > 0
            and iteration % viz_every == 0
            and (viz_frames_set is None or frame_id in viz_frames_set)
            and (viz_cams_set is None or cam_uid in viz_cams_set)
        )

        # Use either random backgrounds (for regularisation) or the configured constant.
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if (
            color_only
            and train_all_parameter
            and lbs_deformer is not None
            and lbs_motions is not None
            and 0 <= frame_id < num_motion_frames
        ):
            # Use the live (trainable) geometry so gradients can flow back when
            # train_all_parameter is enabled.
            xyz_live = gaussians._xyz
            rot_live = gaussians._rotation
            motions_t = lbs_motions[frame_id]
            bone_transform_t: Optional[torch.Tensor] = None
            if (
                bone_transform_cache is not None
                and 0 <= frame_id < bone_transform_cache.shape[0]
            ):
                bone_transform_t = bone_transform_cache[frame_id]
            posed_xyz, posed_rot = lbs_deformer.deform(
                xyz_live,
                rot_live,
                motions_t,
                bone_transforms=bone_transform_t,
            )
            pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
        elif color_only and pose_cache is not None:
            posed = pose_cache.get(int(frame_id))
            if posed is None:
                raise KeyError(
                    f"Frame {frame_id} not found in offline LBS cache "
                    f"({len(pose_cache)} cached frames). Ensure indices align."
                )
            posed_xyz, posed_rot = posed
            if (
                canonical_xyz_ref is not None
                and canonical_rot_ref is not None
                and frame_id != canonical_frame_idx
            ):
                xyz_is_canonical = torch.allclose(
                    posed_xyz, canonical_xyz_ref, atol=1e-5, rtol=1e-4
                )
                rot_is_canonical = torch.allclose(
                    posed_rot, canonical_rot_ref, atol=1e-5, rtol=1e-4
                )
                if xyz_is_canonical and rot_is_canonical:
                    raise RuntimeError(
                        "[LBS] Offline pose cache entry for "
                        f"frame {frame_id} matches the canonical pose "
                        f"({canonical_frame_idx}). The pose cache likely "
                        "does not align with the colour-only training frames."
                    )
            pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
        else:
            pose_ctx = nullcontext()

        with pose_ctx:
            if dataset.disable_sh:
                override_color = gaussians.get_features_dc.squeeze()
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    override_color=override_color,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )
            else:
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                )

            render_rgba = render_pkg["render"]
            depth = render_pkg["depth"]
            normal = render_pkg["normal"]
            viewspace_points = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            # Decompose RGBA(+mask) tensor produced by the renderer.
            pred_seg = render_rgba[3:, ...]
            image = render_rgba[:3, ...]
            gt_image = viewpoint_cam.original_image.cuda()
            viz_pred_rgb_raw: Optional[torch.Tensor] = None
            viz_pred_seg_raw: Optional[torch.Tensor] = None
            viz_gt_raw: Optional[torch.Tensor] = None
            viz_depth_raw: Optional[torch.Tensor] = None
            viz_normal_raw: Optional[torch.Tensor] = None

            if should_viz:
                # Snapshot raw tensors before masks/occlusion adjustments so the
                # debug dump shows exactly what the renderer produced this step.
                viz_pred_rgb_raw = image.detach().clone()
                viz_pred_seg_raw = pred_seg.detach().clone()
                viz_gt_raw = gt_image.detach().clone()
                if depth is not None:
                    viz_depth_raw = depth.detach().clone()
                if normal is not None:
                    viz_normal_raw = normal.detach().clone()

            occ_mask_tensor: Optional[torch.Tensor] = None
            inv_occ_mask_tensor: Optional[torch.Tensor] = None
            if viewpoint_cam.occ_mask is not None:
                # Suppress gradients in regions flagged as occluded by preprocessing.
                occ_mask_tensor = viewpoint_cam.occ_mask.cuda()
                inv_occ_mask_tensor = 1.0 - occ_mask_tensor
                if depth is not None:
                    depth *= inv_occ_mask_tensor
                if normal is not None:
                    normal *= inv_occ_mask_tensor.unsqueeze(-1)

            alpha_mask_tensor: Optional[torch.Tensor] = None
            if viewpoint_cam.alpha_mask is not None:
                # Exclude background pixels when masks are provided.
                alpha_mask_tensor = viewpoint_cam.alpha_mask.cuda()
            image_l1 = image
            gt_image_l1 = gt_image
            if alpha_mask_tensor is not None:
                gt_image_l1 = gt_image_l1 * alpha_mask_tensor

            pred_mask_tensor: Optional[torch.Tensor] = None
            if mask_pred_with_alpha and alpha_mask_tensor is not None:
                pred_mask_tensor = alpha_mask_tensor
            elif inv_occ_mask_tensor is not None:
                pred_mask_tensor = inv_occ_mask_tensor

            if pred_mask_tensor is not None:
                image_l1 = image_l1 * pred_mask_tensor

            alpha_leak: Optional[torch.Tensor] = None
            # Penalise alpha leaking outside the provided alpha mask to suppress
            # stray splats in the background. This does not affect cases without
            # masks or when the weight is zero.
            if (
                alpha_mask_tensor is not None
                and lambda_alpha_leak > 0.0
                and pred_seg is not None
            ):
                alpha_leak = (pred_seg * (1.0 - alpha_mask_tensor)).mean()

            if should_viz:
                pred_rgb_masked = image_l1.detach().clone()
                pred_seg_masked = pred_seg.detach().clone()
                gt_masked = gt_image_l1.detach().clone()
                depth_masked = depth.detach().clone() if depth is not None else None
                normal_masked = normal.detach().clone() if normal is not None else None
                dump_viz(
                    iteration=iteration,
                    frame_id=int(frame_id),
                    cam_uid=int(cam_uid),
                    image_name=image_token,
                    root=viz_root_path,
                    pred_rgb_raw=(
                        viz_pred_rgb_raw
                        if viz_pred_rgb_raw is not None
                        else image.detach().clone()
                    ),
                    pred_rgb_masked=pred_rgb_masked,
                    pred_seg_raw=(
                        viz_pred_seg_raw
                        if viz_pred_seg_raw is not None
                        else pred_seg.detach().clone()
                    ),
                    pred_seg_masked=pred_seg_masked,
                    gt_raw=(
                        viz_gt_raw
                        if viz_gt_raw is not None
                        else gt_image.detach().clone()
                    ),
                    gt_masked=gt_masked,
                    alpha_mask=alpha_mask_tensor,
                    occ_mask=occ_mask_tensor,
                    inv_occ_mask=inv_occ_mask_tensor,
                    depth_raw=viz_depth_raw,
                    depth_masked=depth_masked,
                    normal_raw=viz_normal_raw,
                    normal_masked=normal_masked,
                    bg_color=bg.detach().clone(),
                )

            # Photometric supervision mixes L1 and SSIM according to lambda_dssim.
            Ll1 = l1_loss(image_l1, gt_image_l1)
            if FUSED_SSIM_AVAILABLE and fused_ssim is not None:
                ssim_value = fused_ssim(image_l1.unsqueeze(0), gt_image_l1.unsqueeze(0))
            else:
                ssim_value = ssim(image_l1, gt_image_l1)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim_value
            )
            if alpha_leak is not None:
                loss = loss + lambda_alpha_leak * alpha_leak

            Ll1depth_value: float = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                inv_depth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()

                ll1_depth = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
                depth_term = depth_l1_weight(iteration) * ll1_depth
                loss += depth_term
                Ll1depth_value = depth_term.item()

            loss_seg = torch.tensor(0.0, device="cuda")
            if opt.lambda_seg > 0 and viewpoint_cam.alpha_mask is not None:
                gt_seg = viewpoint_cam.alpha_mask.cuda()
                loss_seg_l1 = l1_loss(pred_seg, gt_seg)
                loss_seg_ssim = ssim(image, gt_image)
                loss_seg = (1.0 - opt.lambda_dssim) * loss_seg_l1 + opt.lambda_dssim * (
                    1.0 - loss_seg_ssim
                )
                loss = loss + opt.lambda_seg * loss_seg

            loss_depth = torch.tensor(0.0, device="cuda")
            if opt.lambda_depth > 0:
                gt_depth = viewpoint_cam.depth.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    loss_depth = depth_loss(depth, gt_depth, alpha_mask)
                else:
                    loss_depth = depth_loss(depth, gt_depth)
                loss = loss + opt.lambda_depth * loss_depth

            loss_normal = torch.tensor(0.0, device="cuda")
            if opt.lambda_normal > 0:
                gt_normal = viewpoint_cam.normal.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    loss_normal = normal_loss(normal, gt_normal, alpha_mask)
                else:
                    loss_normal = normal_loss(normal, gt_normal)
                loss = loss + opt.lambda_normal * loss_normal

            loss_anisotropic = torch.tensor(0.0, device="cuda")
            if opt.lambda_anisotropic > 0:
                loss_anisotropic = anisotropic_loss(gaussians.get_scaling)
                loss = loss + opt.lambda_anisotropic * loss_anisotropic

            # Backpropagate through the renderer and all active loss terms.
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_l1depth_for_log = 0.4 * Ll1depth_value + 0.6 * ema_l1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "L1 Loss": f"{Ll1.item():.5f}",
                        "Depth Loss": f"{loss_depth.item():.5f}",
                        "Normal Loss": f"{loss_normal.item():.5f}",
                        "Seg Loss": f"{loss_seg.item():.5f}",
                        "Anisotropic Loss": f"{loss_anisotropic.item():.5f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Persist metrics and iteration timings to TensorBoard if enabled.
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                iter_start.elapsed_time(iter_end),
            )
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if (not color_only) and iteration < opt.densify_until_iter:
                # Update visibility stats that drive density control heuristics.
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_points,
                    visibility_filter,
                    image.shape[2],
                    image.shape[1],
                    use_gsplat=True,
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    # Periodically clear opacity so redundant splats can reappear or fade.
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                # Step both exposure parameters and geometric parameters.
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                # Explicit checkpoint requested via CLI flag.
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            maybe_refresh_lbs(iteration)

    if color_only:
        color_path = canonical_dir / "color_refine" / "canonical_gaussians_color.npz"
        dump_gaussian_npz(gaussians, color_path)
        print(f"Colour parameters saved to {color_path}")


def prepare_output_and_logger(args: ModelParams) -> Optional[SummaryWriterType]:
    """
    Ensure the output directory exists and create a TensorBoard writer if possible.
    """

    if not args.model_path:
        # Fallback to a unique folder when no explicit model directory is supplied.
        job_id = os.getenv("OAR_JOB_ID")
        unique_str = job_id if job_id else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(
        os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8"
    ) as cfg_log_f:
        # Persist the resolved argument namespace for future audits.
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer: Optional[SummaryWriterType] = None
    if TENSORBOARD_FOUND and SummaryWriter is not None:
        tb_writer = SummaryWriter(args.model_path)
    else:
        # Inform the caller that metric logging will be skipped.
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer: Optional[SummaryWriterType],
    iteration: int,
    Ll1: torch.Tensor,
    loss: torch.Tensor,
    elapsed: float,
) -> None:
    """
    Push scalar metrics to TensorBoard for the current iteration.
    """

    if tb_writer is None:
        # Logging disabled; nothing to do.
        return
    tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
    tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
    tb_writer.add_scalar("iter_time", elapsed, iteration)


def main() -> None:
    """
    Parse CLI arguments and launch the Gaussian training loop.
    """

    parser = ArgumentParser(description="Training script parameters")
    # Register the shared argument groups provided by the original 3DGS codebase.
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--color_only", action="store_true", default=False)
    parser.add_argument("--freeze_geometry", action="store_true", default=False)
    parser.add_argument(
        "--train_all_parameter",
        action="store_true",
        default=False,
        help=(
            "Enable joint optimisation of xyz/scaling/rotation in the colour stage. "
            "When omitted, only colour and opacity are refined."
        ),
    )
    parser.add_argument(
        "--lbs_path",
        type=Path,
        default=None,
        help="Path to lbs_data.pt generated during canonical stage.",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=None,
        help="Optional root containing per-frame data (reserved for future use).",
    )
    parser.add_argument(
        "--train_frames",
        type=str,
        default=None,
        help="Comma-separated list of frame indices to use during colour training.",
    )
    parser.add_argument(
        "--train_cams",
        nargs="+",
        type=int,
        default=None,
        help="List of camera UIDs to use during colour training.",
    )
    parser.add_argument(
        "--lbs_pose_cache",
        type=Path,
        default=None,
        help=(
            "Path to offline LBS cache (.pt) produced by precompute_lbs_pose_cache.py. "
            "Defaults to <model_path>/lbs_pose_cache.pt when omitted."
        ),
    )
    parser.add_argument(
        "--viz_every",
        type=int,
        default=0,
        help="Dump debug visualisations every N iterations (0 disables).",
    )
    parser.add_argument(
        "--viz_frames",
        type=str,
        default=None,
        help="Comma-separated list of frame IDs to visualise.",
    )
    parser.add_argument(
        "--viz_cams",
        type=str,
        default=None,
        help="Comma-separated list of camera UIDs to visualise.",
    )
    parser.add_argument(
        "--viz_out",
        type=str,
        default=None,
        help="Output directory for debug visualisations.",
    )
    parser.add_argument(
        "--mask_pred_with_alpha",
        action="store_true",
        default=False,
        help=(
            "When set, multiply predicted RGB by alpha masks (instead of occlusion masks) "
            "before computing the photometric loss."
        ),
    )
    parser.add_argument(
        "--lbs_refresh_interval",
        type=int,
        default=0,
        help=(
            "Rebuild skinning weights and bone transforms every N iterations when "
            "--train_all_parameter is enabled (0 disables refresh)."
        ),
    )
    parser.add_argument(
        "--train_all_bbox_pad",
        type=float,
        default=0.25,
        help=(
            "Padding ratio (relative to canonical AABB extent) used to clamp trainable "
            "Gaussians when --train_all_parameter is enabled."
        ),
    )
    parser.add_argument(
        "--lambda_alpha_leak",
        type=float,
        default=0.1,
        help=(
            "Weight for penalising predicted alpha outside the provided alpha mask to reduce stray splats."
        ),
    )
    args = parser.parse_args(sys.argv[1:])
    if args.train_all_parameter:
        print(
            "[info] Colour stage will optimise xyz/scaling/rotation in addition to colour/opacity."
        )
    else:
        print("[info] Colour stage restricted to colour + opacity updates.")
    # Always capture the terminal iteration so the final state is stored on disk.
    args.save_iterations.append(args.iterations)

    def _parse_int_set(spec: Optional[Any]) -> Optional[set[int]]:
        if spec is None:
            return None
        if isinstance(spec, (list, tuple, set)):
            values = [str(item).strip() for item in spec if str(item).strip()]
        else:
            values = [item.strip() for item in str(spec).split(",") if item.strip()]
        if not values:
            return None
        return {int(item) for item in values}

    viz_frames = _parse_int_set(args.viz_frames)
    viz_cams = _parse_int_set(args.viz_cams)
    viz_root = Path(args.viz_out).expanduser() if args.viz_out else None
    train_frames = _parse_int_set(args.train_frames)
    train_cams = _parse_int_set(args.train_cams)

    print(f"Optimizing {args.model_path}")

    # Configure deterministic CuBLAS kernels if requested.
    safe_state(args.quiet)

    if not args.disable_viewer:
        # Spawn an IPC server that lets the GUI connect to this training process.
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Kick off optimisation with the fully populated argument sets.
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        color_only=args.color_only,
        freeze_geometry=args.freeze_geometry,
        lbs_path=args.lbs_path,
        frames_dir=args.frames_dir,
        lbs_pose_cache=args.lbs_pose_cache,
        viz_every=args.viz_every,
        viz_frames=viz_frames,
        viz_cams=viz_cams,
        viz_root=viz_root,
        train_frames=train_frames,
        train_cams=train_cams,
        train_all_parameter=args.train_all_parameter,
        mask_pred_with_alpha=args.mask_pred_with_alpha,
        lbs_refresh_interval=args.lbs_refresh_interval,
        train_all_bbox_pad=args.train_all_bbox_pad,
        lambda_alpha_leak=args.lambda_alpha_leak,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
