from __future__ import annotations

"""
Color-finetuning entry point for Gaussian Splatting with optional LBS-driven pose deformation.

Inputs
------
- Canonical frame data at ``ModelParams.source_path`` (Stage A frame) plus every additional frame directory discovered via ``--frames_dir`` (e.g. ``per_frame_gaussian_data/<frame>/<scene>/``), providing per-frame RGB, depth, masks, ``camera_meta.pkl``, and ``observation.ply``.
- Canonical Gaussian checkpoint ``<model_path>/canonical_gaussians.npz`` (fallback) or ``color_refine/canonical_gaussians_color.npz`` when resuming a previous colour run; this supplies the shared Gaussian kernels (xyz/scaling/rotation/SH/opacity).
- Optional LBS bundle ``<model_path>/lbs_data.pt`` with bone anchors, adjacency graph, skin indices/weights, and per-frame bone motions used to deform the canonical kernels to each video frame.

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
from gaussian_splatting.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gaussian_renderer import render, network_gui
from gaussian_splatting.skinning import LBSDeformer
from gaussian_splatting.scene import GaussianModel, Scene
from gaussian_splatting.scene.dataset_readers import readQQTTSceneInfo
from gaussian_splatting.utils.camera_utils import cameraList_from_camInfos
from gaussian_splatting.utils.general_utils import get_expon_lr_func, safe_state
from gaussian_splatting.utils.image_utils import psnr  # noqa: F401  # kept for parity
from gaussian_splatting.utils.loss_utils import (
    anisotropic_loss,
    depth_loss,
    l1_loss,
    normal_loss,
    ssim,
)
from tqdm import tqdm

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

try:
    # Differentiable SSIM kernel; gracefully falls back to PyTorch version.
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    fused_ssim = None  # type: ignore[assignment]
    FUSED_SSIM_AVAILABLE = False

try:
    # CUDA accelerated optimiser for sparse Gaussian updates.
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SparseGaussianAdam = None  # type: ignore[assignment]
SPARSE_ADAM_AVAILABLE = False


def load_canonical_npz(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical parameters not found: {path}")
    data = np.load(path)
    tensors = {}
    device = torch.device("cuda")
    tensors["xyz"] = torch.from_numpy(data["xyz"]).float().to(device)
    tensors["scaling"] = torch.from_numpy(data["scaling"]).float().to(device)
    tensors["rotation"] = torch.from_numpy(data["rotation"]).float().to(device)
    tensors["features_dc"] = torch.from_numpy(data["features_dc"]).float().to(device)
    tensors["features_rest"] = (
        torch.from_numpy(data["features_rest"]).float().to(device)
    )
    tensors["opacity"] = torch.from_numpy(data["opacity"]).float().to(device)
    tensors["active_sh_degree"] = int(data["active_sh_degree"][0])
    tensors["max_sh_degree"] = int(data["max_sh_degree"][0])
    return tensors


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


def dump_color_parameters(gaussians: GaussianModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        payload = {
            "xyz": gaussians.get_xyz.detach().cpu().numpy(),
            "scaling": gaussians.get_scaling.detach().cpu().numpy(),
            "rotation": gaussians.get_rotation.detach().cpu().numpy(),
            "features_dc": gaussians.get_features_dc.detach().cpu().numpy(),
            "features_rest": gaussians.get_features_rest.detach().cpu().numpy(),
            "opacity": gaussians.get_opacity.detach().cpu().numpy(),
            "active_sh_degree": np.array([gaussians.active_sh_degree], dtype=np.int32),
            "max_sh_degree": np.array([gaussians.max_sh_degree], dtype=np.int32),
        }
    np.savez(output_path, **payload)


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

    if color_only:
        canonical_color_path = canonical_dir / "color_refine" / "canonical_gaussians_color.npz"
        canonical_base_path = canonical_dir / "canonical_gaussians.npz"
        if canonical_color_path.exists():
            try:
                canonical_color_path.unlink()
                print(f"[Colour] Removed stale {canonical_color_path}")
            except OSError as exc:
                print(f"[Colour] Warning: failed to remove {canonical_color_path}: {exc}")
        canonical_path = (
            canonical_color_path
            if canonical_color_path.exists()
            else canonical_base_path
        )
        canonical_data = load_canonical_npz(canonical_path)
        assign_canonical_parameters(gaussians, canonical_data)
        if freeze_geometry:
            gaussians.freeze_geometry()
        feature_lr = opt.feature_lr
        rest_lr = opt.feature_lr / 20.0
        param_groups = [
            {"params": [gaussians._features_dc], "lr": feature_lr, "name": "f_dc"},
            {
                "params": [gaussians._features_rest],
                "lr": rest_lr,
                "name": "f_rest",
            },
            {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        ]
        gaussians.optimizer = torch.optim.Adam(param_groups)
        gaussians.exposure_optimizer = torch.optim.Adam(
            [{"params": [gaussians._exposure], "name": "exposure"}]
        )
        if lbs_path is not None:
            lbs_payload = load_lbs_data(lbs_path)
            lbs_deformer = LBSDeformer(
                lbs_payload["bones0"],
                lbs_payload["relations"],
                lbs_payload["skin_indices"],
                lbs_payload["skin_weights"],
            )
            lbs_motions = lbs_payload["motions"]
        else:
            print(
                "[LBS] Warning: --lbs_path not provided; colour stage will run without pose deformation."
            )

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

    base_cameras = list(scene.getTrainCameras())
    camera_entries: list[tuple[int, Any]] = [
        (canonical_frame_idx, cam) for cam in base_cameras
    ]

    if frames_dir is not None:
        frames_root = Path(frames_dir)
        for frame_path in sorted(frames_root.glob(f"*/{scene_name}")):
            if not frame_path.is_dir():
                continue
            try:
                frame_idx = int(frame_path.parent.name)
            except ValueError:
                continue
            if frame_path.resolve() == scene_path.resolve():
                continue
            try:
                scene_info = readQQTTSceneInfo(
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
                continue
            extra_cams = cameraList_from_camInfos(
                scene_info.train_cameras,
                1.0,
                dataset,
                scene_info.is_nerf_synthetic,
                False,
            )
            for cam in extra_cams:
                camera_entries.append((frame_idx, cam))

    if not camera_entries:
        raise RuntimeError("No training cameras available for colour stage.")
    frame_summary: dict[int, int] = {}
    for frame_id, _ in camera_entries:
        frame_summary[frame_id] = frame_summary.get(frame_id, 0) + 1
    summary_str = ", ".join(f"{fid}: {count} cams" for fid, count in sorted(frame_summary.items()))
    print(f"[Frames] Training across frames -> {summary_str}")

    entry_pool = list(range(len(camera_entries)))

    # Rolling averages used solely for smoother console logging.
    ema_loss_for_log: float = 0.0
    ema_l1depth_for_log: float = 0.0
    num_motion_frames = lbs_motions.shape[0] if lbs_motions is not None else 0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
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

        # Use either random backgrounds (for regularisation) or the configured constant.
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if (
            color_only
            and lbs_deformer is not None
            and lbs_motions is not None
            and 0 <= frame_id < num_motion_frames
        ):
            xyz_canonical = gaussians.get_xyz_canonical().to("cuda")
            rot_canonical = gaussians.get_rotation_canonical().to("cuda")
            motions_t = lbs_motions[frame_id]
            posed_xyz, posed_rot = lbs_deformer.deform(
                xyz_canonical, rot_canonical, motions_t
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

            image = render_pkg["render"]
            depth = render_pkg["depth"]
            normal = render_pkg["normal"]
            viewspace_points = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            # Decompose RGBA(+mask) tensor produced by the renderer.
            pred_seg = image[3:, ...]
            image = image[:3, ...]
            gt_image = viewpoint_cam.original_image.cuda()

            if viewpoint_cam.occ_mask is not None:
                # Suppress gradients in regions flagged as occluded by preprocessing.
                occ_mask = viewpoint_cam.occ_mask.cuda()
                inv_occ_mask = 1.0 - occ_mask

                image *= inv_occ_mask.unsqueeze(0)
                pred_seg *= inv_occ_mask.unsqueeze(0)
                depth *= inv_occ_mask
                if normal is not None:
                    normal *= inv_occ_mask.unsqueeze(-1)

            if viewpoint_cam.alpha_mask is not None:
                # Exclude background pixels when masks are provided.
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                gt_image *= alpha_mask

            # Photometric supervision mixes L1 and SSIM according to lambda_dssim.
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE and fused_ssim is not None:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim_value
            )

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

    if color_only:
        color_path = canonical_dir / "color_refine" / "canonical_gaussians_color.npz"
        dump_color_parameters(gaussians, color_path)
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
    args = parser.parse_args(sys.argv[1:])
    # Always capture the terminal iteration so the final state is stored on disk.
    args.save_iterations.append(args.iterations)

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
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
