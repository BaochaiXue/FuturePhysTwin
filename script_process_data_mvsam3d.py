"""Batch runner for the experimental MV-SAM3D route."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from case_filter import (
    load_config_cases,
    load_config_rows,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_CONFIG_PATH = "./data_config.csv"
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run process_data_mvsam3d.py for cases listed in data_config.csv."
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python command used to run process_data_mvsam3d.py.",
    )
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument("--pipeline_python", default=None)
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument("--mvsam3d_python", default=None)
    parser.add_argument("--mvsam3d_run_da3", action="store_true")
    parser.add_argument("--mvsam3d_skip_da3_if_exists", action="store_true")
    parser.add_argument("--mvsam3d_da3_model_path", default=None)
    parser.add_argument("--mvsam3d_force", action="store_true")
    parser.add_argument("--mvsam3d_view_indices", default="0,1,2")
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
    return parser.parse_args()


def add_if_present(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_rows = load_config_rows(config_path)
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases,
        config_cases,
        "script_process_data_mvsam3d",
        base_path,
        config_path,
    )
    allowed_cases = input_cases & config_cases

    timer_log = ROOT / "timer.log"
    if timer_log.exists():
        timer_log.unlink()

    process_python = shlex.split(args.python)
    for row in config_rows:
        if len(row) < 3:
            raise ValueError(
                f"Malformed config row in {config_path}: expected >=3 columns, got {row}"
            )
        case_name = row[0].strip()
        category = row[1].strip()
        shape_prior = row[2].strip()

        if args.case is not None and case_name != args.case:
            continue
        if case_name not in allowed_cases:
            continue

        cmd = process_python + [
            str(ROOT / "process_data_mvsam3d.py"),
            "--base_path",
            str(base_path),
            "--case_name",
            case_name,
            "--category",
            category,
            "--controller_name",
            args.controller_name,
            "--mvsam3d_view_indices",
            args.mvsam3d_view_indices,
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
        add_if_present(cmd, "--pipeline_python", args.pipeline_python)
        add_if_present(cmd, "--mvsam3d_root", args.mvsam3d_root)
        add_if_present(cmd, "--mvsam3d_python", args.mvsam3d_python)
        add_if_present(cmd, "--mvsam3d_da3_model_path", args.mvsam3d_da3_model_path)
        add_if_present(cmd, "--mvsam3d_preprocess_python", args.mvsam3d_preprocess_python)
        add_if_present(cmd, "--mvsam3d_align_view_indices", args.mvsam3d_align_view_indices)
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
        if shape_prior.lower() == "true":
            cmd.append("--shape_prior")

        subprocess.run(cmd, check=True, cwd=str(ROOT))

    print("MV-SAM3D data processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
