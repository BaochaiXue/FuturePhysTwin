from __future__ import annotations

import argparse
from pathlib import Path

from process_data_routes import (
    ROOT,
    CommandRunner,
    Timer,
    command_prefix,
    resolve_from_root,
    setup_logger,
    write_route_manifest,
    write_split,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process one case through the experimental MV-SAM3D route. "
            "This route always uses the MV shape prior, MV aligner, and MV sampler."
        )
    )
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--shape_prior", action="store_true", default=False)
    parser.add_argument("--controller_name", type=str, default="hand")
    parser.add_argument(
        "--pipeline_python",
        default=None,
        help="Python command for PhysTwin subprocesses. Defaults to current Python.",
    )
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument("--mvsam3d_python", default=None)
    parser.add_argument("--mvsam3d_run_da3", action="store_true")
    parser.add_argument("--mvsam3d_skip_da3_if_exists", action="store_true")
    parser.add_argument("--mvsam3d_da3_model_path", default=None)
    parser.add_argument("--mvsam3d_force", action="store_true")
    parser.add_argument(
        "--mvsam3d_view_indices",
        default="0,1,2",
        help="Comma-separated frame-0 camera views for MV-SAM3D input.",
    )
    parser.add_argument(
        "--mvsam3d_input_preprocess_backend",
        choices=["legacy_upscale", "raw"],
        default="legacy_upscale",
    )
    parser.add_argument("--mvsam3d_preprocess_python", default=None)
    parser.add_argument("--mvsam3d_merge_da3_glb", action="store_true")
    parser.add_argument("--mvsam3d_run_pose_optimization", action="store_true")
    parser.add_argument("--mvsam3d_max_faces", type=int, default=50000)
    parser.add_argument("--mvsam3d_align_view_indices", default=None)
    parser.add_argument("--mvsam3d_align_force_rematch", action="store_true")
    parser.add_argument("--mvsam3d_align_max_render_faces", type=int, default=50000)
    parser.add_argument("--mvsam3d_align_silhouette_iters", type=int, default=80)
    parser.add_argument("--mvsam3d_align_depth_weight", type=float, default=0.2)
    parser.add_argument("--mvsam3d_align_silhouette_weight", type=float, default=1.0)
    parser.add_argument("--mvsam3d_align_pcd_weight", type=float, default=0.5)
    parser.add_argument("--mvsam3d_align_vertex_to_obs_gate", type=float, default=0.035)
    parser.add_argument("--mvsam3d_align_obs_to_vertex_gate", type=float, default=0.015)
    parser.add_argument("--mvsam3d_align_prune_far_dist", type=float, default=0.035)
    parser.add_argument("--mvsam3d_align_disable_ray_arap", action="store_true")
    return parser


def run_video_segmentation(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    with Timer("Video Segmentation", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process_sam3d" / "segment.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
                "--TEXT_PROMPT",
                f"{args.category}.{args.controller_name}",
            ],
            stage="video_segmentation",
        )


def run_mvsam3d_shape_prior(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    cmd = pipeline_python + [
        str(ROOT / "data_process_sam3d" / "shape_prior_mvsam3d.py"),
        "--base_path",
        str(base_path),
        "--case_name",
        args.case_name,
        "--category",
        args.category,
        "--controller_name",
        args.controller_name,
        "--input_preprocess_backend",
        args.mvsam3d_input_preprocess_backend,
        "--max_faces",
        str(args.mvsam3d_max_faces),
    ]
    if args.mvsam3d_root:
        cmd += ["--mvsam3d_root", args.mvsam3d_root]
    if args.mvsam3d_python:
        cmd += ["--mvsam3d_python", args.mvsam3d_python]
    if args.mvsam3d_view_indices:
        cmd += ["--view_indices", args.mvsam3d_view_indices]
    if args.mvsam3d_preprocess_python:
        cmd += ["--preprocess_python", args.mvsam3d_preprocess_python]
    if args.mvsam3d_run_da3:
        cmd.append("--run_da3")
    if args.mvsam3d_skip_da3_if_exists:
        cmd.append("--skip_da3_if_exists")
    if args.mvsam3d_da3_model_path:
        cmd += ["--da3_model_path", args.mvsam3d_da3_model_path]
    if args.mvsam3d_force:
        cmd.append("--force")
    if args.mvsam3d_merge_da3_glb:
        cmd.append("--merge_da3_glb")
    if args.mvsam3d_run_pose_optimization:
        cmd.append("--run_pose_optimization")

    with Timer("Shape Prior Generation", args.case_name, logger):
        runner.run(cmd, stage="shape_prior_mvsam3d")


def run_tracking_and_lifting(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    with Timer("Dense Tracking", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process_sam3d" / "dense_track.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
            ],
            stage="dense_tracking",
        )

    with Timer("Lift to 3D", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process_sam3d" / "data_process_pcd.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
            ],
            stage="lift_to_3d",
        )

    with Timer("Mask Post-Processing", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process_sam3d" / "data_process_mask.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
                "--controller_name",
                args.controller_name,
            ],
            stage="mask_post_processing",
        )

    with Timer("Data Tracking", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process_sam3d" / "data_process_track.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
            ],
            stage="data_tracking",
        )


def run_mvsam3d_alignment(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    cmd = pipeline_python + [
        str(ROOT / "data_process_sam3d" / "align_mvsam3d.py"),
        "--base_path",
        str(base_path),
        "--case_name",
        args.case_name,
        "--controller_name",
        args.controller_name,
        "--max_render_faces",
        str(args.mvsam3d_align_max_render_faces),
        "--silhouette_iters",
        str(args.mvsam3d_align_silhouette_iters),
        "--depth_weight",
        str(args.mvsam3d_align_depth_weight),
        "--silhouette_weight",
        str(args.mvsam3d_align_silhouette_weight),
        "--pcd_weight",
        str(args.mvsam3d_align_pcd_weight),
        "--vertex_to_obs_gate",
        str(args.mvsam3d_align_vertex_to_obs_gate),
        "--obs_to_vertex_gate",
        str(args.mvsam3d_align_obs_to_vertex_gate),
        "--prune_far_dist",
        str(args.mvsam3d_align_prune_far_dist),
    ]
    if args.mvsam3d_align_view_indices:
        cmd += ["--view_indices", args.mvsam3d_align_view_indices]
    if args.mvsam3d_align_force_rematch:
        cmd.append("--force_rematch")
    if args.mvsam3d_align_disable_ray_arap:
        cmd.append("--disable_ray_arap")

    with Timer("Alignment", args.case_name, logger):
        runner.run(cmd, stage="mvsam3d_alignment")


def run_final_data(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    cmd = pipeline_python + [
        str(ROOT / "data_process_sam3d" / "data_process_sample.py"),
        "--base_path",
        str(base_path),
        "--case_name",
        args.case_name,
    ]
    if args.shape_prior:
        cmd += ["--shape_prior", "--shape_prior_sampling_backend", "mvsam3d"]
    with Timer("Final Data Generation", args.case_name, logger):
        runner.run(cmd, stage="final_data_generation")


def main() -> int:
    args = build_arg_parser().parse_args()
    base_path = resolve_from_root(args.base_path)
    pipeline_python = command_prefix(args.pipeline_python)
    logger = setup_logger()
    runner = CommandRunner(ROOT)

    run_video_segmentation(
        args=args,
        base_path=base_path,
        pipeline_python=pipeline_python,
        runner=runner,
        logger=logger,
    )
    if args.shape_prior:
        run_mvsam3d_shape_prior(
            args=args,
            base_path=base_path,
            pipeline_python=pipeline_python,
            runner=runner,
            logger=logger,
        )
    run_tracking_and_lifting(
        args=args,
        base_path=base_path,
        pipeline_python=pipeline_python,
        runner=runner,
        logger=logger,
    )
    if args.shape_prior:
        run_mvsam3d_alignment(
            args=args,
            base_path=base_path,
            pipeline_python=pipeline_python,
            runner=runner,
            logger=logger,
        )
    run_final_data(
        args=args,
        base_path=base_path,
        pipeline_python=pipeline_python,
        runner=runner,
        logger=logger,
    )
    split = write_split(base_path, args.case_name)
    manifest_path = write_route_manifest(
        base_path=base_path,
        case_name=args.case_name,
        route="mvsam3d",
        backend="mvsam3d",
        category=args.category,
        controller_name=args.controller_name,
        shape_prior=args.shape_prior,
        commands=runner.records,
        split=split,
        extra={
            "mvsam3d_view_indices": args.mvsam3d_view_indices,
            "mvsam3d_input_preprocess_backend": args.mvsam3d_input_preprocess_backend,
            "mvsam3d_root": args.mvsam3d_root,
            "mvsam3d_da3_model_path": args.mvsam3d_da3_model_path,
            "mvsam3d_contract": (
                "experimental MV route; uses MV-SAM3D shape prior, align_mvsam3d.py, "
                "and mvsam3d final sampling."
            ),
        },
    )
    print(f"Route manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
