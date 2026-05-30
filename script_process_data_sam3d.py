"""Process raw multi-view captures with the SAM3D/MV-SAM3D pipeline."""

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
DEFAULT_PROCESS_PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run process_data_sam3d.py for cases listed in data_config.csv."
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="If provided, only process this case name (matches first column in data_config.csv).",
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--python",
        default=DEFAULT_PROCESS_PYTHON,
        help="Python command used to run process_data_sam3d.py.",
    )
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument(
        "--pipeline_python",
        default=None,
        help="Forwarded to process_data_sam3d.py for its internal subprocesses.",
    )
    parser.add_argument(
        "--legacy_shape_prior_python",
        default=None,
        help="Forwarded to process_data_sam3d.py for the legacy single-view shape_prior.py backend.",
    )
    parser.add_argument(
        "--shape_prior_backend",
        choices=["sam3d", "mvsam3d"],
        default="mvsam3d",
    )
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument("--mvsam3d_python", default=None)
    parser.add_argument("--mvsam3d_run_da3", action="store_true")
    parser.add_argument("--mvsam3d_skip_da3_if_exists", action="store_true")
    parser.add_argument(
        "--mvsam3d_da3_model_path",
        default=None,
        help="Forwarded to process_data_sam3d.py for MV-SAM3D DA3 model resolution.",
    )
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
    parser.add_argument(
        "--mvsam3d_max_faces",
        type=int,
        default=50000,
        help="Maximum faces for downstream shape/object.glb; use 0 to keep the full MV-SAM3D mesh.",
    )
    parser.add_argument(
        "--align_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
    )
    parser.add_argument("--mvsam3d_align_view_indices", default=None)
    parser.add_argument("--mvsam3d_align_force_rematch", action="store_true")
    parser.add_argument("--mvsam3d_align_max_render_faces", type=int, default=50000)
    parser.add_argument("--mvsam3d_align_silhouette_iters", type=int, default=80)
    parser.add_argument("--mvsam3d_align_depth_weight", type=float, default=0.2)
    parser.add_argument("--mvsam3d_align_silhouette_weight", type=float, default=1.0)
    parser.add_argument("--mvsam3d_align_pcd_weight", type=float, default=0.5)
    parser.add_argument(
        "--shape_prior_sampling_backend",
        choices=["legacy", "mvsam3d", "auto"],
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_rows = load_config_rows(config_path)
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "script_process_data_sam3d", base_path, config_path
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
            str(ROOT / "process_data_sam3d.py"),
            "--base_path",
            str(base_path),
            "--case_name",
            case_name,
            "--category",
            category,
            "--controller_name",
            args.controller_name,
            "--align_backend",
            args.align_backend,
            "--shape_prior_sampling_backend",
            args.shape_prior_sampling_backend,
        ]
        if args.pipeline_python:
            cmd += ["--pipeline_python", args.pipeline_python]
        if args.legacy_shape_prior_python:
            cmd += ["--legacy_shape_prior_python", args.legacy_shape_prior_python]
        if shape_prior.lower() == "true":
            cmd += ["--shape_prior", "--shape_prior_backend", args.shape_prior_backend]
            if args.shape_prior_backend == "mvsam3d":
                if args.mvsam3d_root:
                    cmd += ["--mvsam3d_root", args.mvsam3d_root]
                if args.mvsam3d_python:
                    cmd += ["--mvsam3d_python", args.mvsam3d_python]
                if args.mvsam3d_view_indices:
                    cmd += ["--mvsam3d_view_indices", args.mvsam3d_view_indices]
                cmd += [
                    "--mvsam3d_input_preprocess_backend",
                    args.mvsam3d_input_preprocess_backend,
                ]
                if args.mvsam3d_preprocess_python:
                    cmd += ["--mvsam3d_preprocess_python", args.mvsam3d_preprocess_python]
                if args.mvsam3d_run_da3:
                    cmd.append("--mvsam3d_run_da3")
                if args.mvsam3d_skip_da3_if_exists:
                    cmd.append("--mvsam3d_skip_da3_if_exists")
                if args.mvsam3d_da3_model_path:
                    cmd += ["--mvsam3d_da3_model_path", args.mvsam3d_da3_model_path]
                if args.mvsam3d_force:
                    cmd.append("--mvsam3d_force")
                if args.mvsam3d_merge_da3_glb:
                    cmd.append("--mvsam3d_merge_da3_glb")
                if args.mvsam3d_run_pose_optimization:
                    cmd.append("--mvsam3d_run_pose_optimization")
                cmd += ["--mvsam3d_max_faces", str(args.mvsam3d_max_faces)]
                if args.mvsam3d_align_view_indices:
                    cmd += ["--mvsam3d_align_view_indices", args.mvsam3d_align_view_indices]
                if args.mvsam3d_align_force_rematch:
                    cmd.append("--mvsam3d_align_force_rematch")
                cmd += [
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
                ]

        subprocess.run(cmd, check=True, cwd=str(ROOT))

    print("Data processing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
