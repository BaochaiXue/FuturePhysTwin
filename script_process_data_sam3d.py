"""Compatibility shim for split single-view and MV-SAM3D batch routes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated batch entrypoint. Use script_process_data_single_view.py "
            "or script_process_data_mvsam3d.py."
        )
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--base-path", default="./data/different_types")
    parser.add_argument("--config-path", default="./data_config.csv")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python command used by the split batch runner for per-case processing.",
    )
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument("--pipeline_python", default=None)
    parser.add_argument(
        "--legacy_shape_prior_python",
        default=None,
        help="Deprecated name; forwarded as --shape_prior_python for single-view SAM3D.",
    )
    parser.add_argument(
        "--shape_prior_backend",
        choices=["sam3d", "mvsam3d"],
        default="sam3d",
        help="Compatibility selector. Default now maps to the single-view SAM3D route.",
    )
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument("--mvsam3d_python", default=None)
    parser.add_argument("--mvsam3d_run_da3", action="store_true")
    parser.add_argument("--mvsam3d_skip_da3_if_exists", action="store_true")
    parser.add_argument("--mvsam3d_da3_model_path", default=None)
    parser.add_argument("--mvsam3d_force", action="store_true")
    parser.add_argument("--mvsam3d_view_indices", default=None)
    parser.add_argument(
        "--mvsam3d_input_preprocess_backend",
        choices=["legacy_upscale", "raw"],
        default="legacy_upscale",
    )
    parser.add_argument("--mvsam3d_preprocess_python", default=None)
    parser.add_argument("--mvsam3d_merge_da3_glb", action="store_true")
    parser.add_argument("--mvsam3d_run_pose_optimization", action="store_true")
    parser.add_argument("--mvsam3d_max_faces", type=int, default=50000)
    parser.add_argument(
        "--align_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
        help="Compatibility-only. Split routes choose alignment internally.",
    )
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
    parser.add_argument(
        "--shape_prior_sampling_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
        help="Compatibility-only. Split routes choose sampling internally.",
    )
    return parser


def add_if_present(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def validate_compatibility(args: argparse.Namespace) -> None:
    if args.shape_prior_backend == "sam3d":
        if args.align_backend == "mvsam3d":
            raise ValueError("single-view SAM3D cannot use --align_backend mvsam3d.")
        if args.shape_prior_sampling_backend == "mvsam3d":
            raise ValueError(
                "single-view SAM3D cannot use --shape_prior_sampling_backend mvsam3d."
            )
    if args.shape_prior_backend == "mvsam3d":
        if args.align_backend == "legacy":
            raise ValueError("MV-SAM3D route is fixed to align_mvsam3d.py.")
        if args.shape_prior_sampling_backend == "legacy":
            raise ValueError("MV-SAM3D route is fixed to mvsam3d final sampling.")


def build_single_view_batch_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "script_process_data_single_view.py"),
        "--base-path",
        args.base_path,
        "--config-path",
        args.config_path,
        "--python",
        args.python,
        "--controller_name",
        args.controller_name,
        "--single_view_backend",
        "sam3d",
    ]
    add_if_present(cmd, "--case", args.case)
    add_if_present(cmd, "--pipeline_python", args.pipeline_python)
    add_if_present(cmd, "--shape_prior_python", args.legacy_shape_prior_python)
    return cmd


def build_mvsam3d_batch_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "script_process_data_mvsam3d.py"),
        "--base-path",
        args.base_path,
        "--config-path",
        args.config_path,
        "--python",
        args.python,
        "--controller_name",
        args.controller_name,
        "--mvsam3d_input_preprocess_backend",
        args.mvsam3d_input_preprocess_backend,
        "--mvsam3d_max_faces",
        str(args.mvsam3d_max_faces),
        "--mvsam3d_align_max_render_faces",
        str(args.mvsam3d_align_max_render_faces),
        "--mvsam3d_align_silhouette_iters",
        str(args.mvsam3d_align_silhouette_iters),
        "--mvsam3d_align_depth_weight",
        str(args.mvsam3d_align_depth_weight),
        "--mvsam3d_align_silhouette_weight",
        str(args.mvsam3d_align_silhouette_weight),
        "--mvsam3d_align_pcd_weight",
        str(args.mvsam3d_align_pcd_weight),
        "--mvsam3d_align_vertex_to_obs_gate",
        str(args.mvsam3d_align_vertex_to_obs_gate),
        "--mvsam3d_align_obs_to_vertex_gate",
        str(args.mvsam3d_align_obs_to_vertex_gate),
        "--mvsam3d_align_prune_far_dist",
        str(args.mvsam3d_align_prune_far_dist),
    ]
    add_if_present(cmd, "--case", args.case)
    add_if_present(cmd, "--pipeline_python", args.pipeline_python)
    add_if_present(cmd, "--mvsam3d_root", args.mvsam3d_root)
    add_if_present(cmd, "--mvsam3d_python", args.mvsam3d_python)
    add_if_present(cmd, "--mvsam3d_da3_model_path", args.mvsam3d_da3_model_path)
    add_if_present(cmd, "--mvsam3d_preprocess_python", args.mvsam3d_preprocess_python)
    add_if_present(cmd, "--mvsam3d_align_view_indices", args.mvsam3d_align_view_indices)
    if args.mvsam3d_view_indices:
        cmd += ["--mvsam3d_view_indices", args.mvsam3d_view_indices]
    if args.mvsam3d_run_da3:
        cmd.append("--mvsam3d_run_da3")
    if args.mvsam3d_skip_da3_if_exists:
        cmd.append("--mvsam3d_skip_da3_if_exists")
    if args.mvsam3d_force:
        cmd.append("--mvsam3d_force")
    if args.mvsam3d_merge_da3_glb:
        cmd.append("--mvsam3d_merge_da3_glb")
    if args.mvsam3d_run_pose_optimization:
        cmd.append("--mvsam3d_run_pose_optimization")
    if args.mvsam3d_align_force_rematch:
        cmd.append("--mvsam3d_align_force_rematch")
    if args.mvsam3d_align_disable_ray_arap:
        cmd.append("--mvsam3d_align_disable_ray_arap")
    return cmd


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        validate_compatibility(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.shape_prior_backend == "sam3d":
        target = "script_process_data_single_view.py --single_view_backend sam3d"
        cmd = build_single_view_batch_command(args)
    else:
        target = "script_process_data_mvsam3d.py"
        cmd = build_mvsam3d_batch_command(args)

    print(
        f"[deprecated] script_process_data_sam3d.py is a compatibility shim; forwarding to {target}.",
        file=sys.stderr,
    )
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("Data processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
