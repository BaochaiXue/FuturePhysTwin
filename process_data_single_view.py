from __future__ import annotations

import argparse
from pathlib import Path

from process_data_routes import (
    ROOT,
    CommandRunner,
    Timer,
    command_prefix,
    ensure_dir,
    find_cam0_object_mask,
    resolve_from_root,
    setup_logger,
    write_route_manifest,
    write_split,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process one case through the production single-view shape-prior route. "
            "This route follows data_process semantics; only the shape-prior generator "
            "switches between Trellis and SAM3D."
        )
    )
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--shape_prior", action="store_true", default=False)
    parser.add_argument("--controller_name", type=str, default="hand")
    parser.add_argument(
        "--single_view_backend",
        choices=["sam3d", "trellis"],
        default="sam3d",
        help="Single-view shape-prior generator. Downstream alignment/sampling stay legacy.",
    )
    parser.add_argument(
        "--pipeline_python",
        default=None,
        help="Python command for data_process segmentation/tracking/lifting/align/sampling.",
    )
    parser.add_argument(
        "--shape_prior_python",
        default=None,
        help="Python command for Trellis or SAM3D shape-prior generation.",
    )
    parser.add_argument(
        "--sam3d_root",
        default=None,
        help="External SAM3D root forwarded only when --single_view_backend sam3d.",
    )
    parser.add_argument(
        "--sam3d_config",
        default=None,
        help="SAM3D pipeline config forwarded only when --single_view_backend sam3d.",
    )
    parser.add_argument(
        "--force_rematch",
        action="store_true",
        help="Forwarded to legacy data_process/align.py.",
    )
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
                str(ROOT / "data_process" / "segment.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
                "--TEXT_PROMPT",
                f"{args.category}.{args.controller_name}",
            ],
            stage="video_segmentation",
        )


def run_single_view_shape_prior(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    shape_prior_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    mask_path = find_cam0_object_mask(base_path, args.case_name, args.controller_name)
    shape_dir = base_path / args.case_name / "shape"
    ensure_dir(shape_dir)

    high_resolution_path = shape_dir / "high_resolution.png"
    with Timer("Image Upscale", args.case_name, logger):
        if high_resolution_path.exists():
            runner.record_skip(
                "image_upscale",
                f"{high_resolution_path} already exists; matching data_process behavior.",
            )
        else:
            runner.run(
                pipeline_python
                + [
                    str(ROOT / "data_process" / "image_upscale.py"),
                    "--img_path",
                    str(base_path / args.case_name / "color" / "0" / "0.png"),
                    "--mask_path",
                    str(mask_path),
                    "--output_path",
                    str(high_resolution_path),
                    "--category",
                    args.category,
                ],
                stage="image_upscale",
            )

    with Timer("Image Segmentation", args.case_name, logger):
        runner.run(
            pipeline_python
            + [
                str(ROOT / "data_process" / "segment_util_image.py"),
                "--img_path",
                str(high_resolution_path),
                "--TEXT_PROMPT",
                args.category,
                "--output_path",
                str(shape_dir / "masked_image.png"),
            ],
            stage="image_segmentation",
        )

    shape_prior_script = (
        ROOT / "data_process_sam3d" / "shape_prior.py"
        if args.single_view_backend == "sam3d"
        else ROOT / "data_process" / "shape_prior.py"
    )
    cmd = shape_prior_python + [
        str(shape_prior_script),
        "--img_path",
        str(shape_dir / "masked_image.png"),
        "--output_dir",
        str(shape_dir),
    ]
    if args.single_view_backend == "sam3d":
        if args.sam3d_root:
            cmd += ["--sam3d_root", args.sam3d_root]
        if args.sam3d_config:
            cmd += ["--config", args.sam3d_config]

    with Timer("Shape Prior Generation", args.case_name, logger):
        runner.run(cmd, stage=f"shape_prior_{args.single_view_backend}")


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
                str(ROOT / "data_process" / "dense_track.py"),
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
                str(ROOT / "data_process" / "data_process_pcd.py"),
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
                str(ROOT / "data_process" / "data_process_mask.py"),
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
                str(ROOT / "data_process" / "data_process_track.py"),
                "--base_path",
                str(base_path),
                "--case_name",
                args.case_name,
            ],
            stage="data_tracking",
        )


def run_legacy_alignment(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    cmd = pipeline_python + [
        str(ROOT / "data_process" / "align.py"),
        "--base_path",
        str(base_path),
        "--case_name",
        args.case_name,
        "--controller_name",
        args.controller_name,
    ]
    if args.force_rematch:
        cmd.append("--force_rematch")
    with Timer("Alignment", args.case_name, logger):
        runner.run(cmd, stage="legacy_alignment")


def run_final_data(
    *,
    args: argparse.Namespace,
    base_path: Path,
    pipeline_python: list[str],
    runner: CommandRunner,
    logger,
) -> None:
    cmd = pipeline_python + [
        str(ROOT / "data_process" / "data_process_sample.py"),
        "--base_path",
        str(base_path),
        "--case_name",
        args.case_name,
    ]
    if args.shape_prior:
        cmd.append("--shape_prior")
    with Timer("Final Data Generation", args.case_name, logger):
        runner.run(cmd, stage="final_data_generation")


def main() -> int:
    args = build_arg_parser().parse_args()
    base_path = resolve_from_root(args.base_path)
    pipeline_python = command_prefix(args.pipeline_python)
    shape_prior_python = command_prefix(args.shape_prior_python)
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
        run_single_view_shape_prior(
            args=args,
            base_path=base_path,
            pipeline_python=pipeline_python,
            shape_prior_python=shape_prior_python,
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
        run_legacy_alignment(
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
        route="single_view",
        backend=args.single_view_backend,
        category=args.category,
        controller_name=args.controller_name,
        shape_prior=args.shape_prior,
        commands=runner.records,
        split=split,
        extra={
            "single_view_backend": args.single_view_backend,
            "single_view_contract": (
                "data_process-compatible pipeline; only the shape-prior generator "
                "switches between Trellis and SAM3D."
            ),
        },
    )
    print(f"Route manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
