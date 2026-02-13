"""Generate per-camera human-mask videos used for render overlays."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_OUTPUT_PATH = "./data/different_types_human_mask"
DEFAULT_CONFIG_PATH = "./data_config.csv"
DEFAULT_TEXT_PROMPT = "human"
DEFAULT_CAMERA_NUM = 3
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-camera human masks for allowed cases only."
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--camera-num", type=int, default=DEFAULT_CAMERA_NUM)
    args = parser.parse_args()
    if args.camera_num <= 0:
        parser.error("--camera-num must be > 0")
    return args


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    output_path = resolve_path_from_root(ROOT, args.output_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "export_video_human_mask", base_path, config_path
    )
    allowed_cases = input_cases & config_cases
    case_names = filter_candidates(
        sorted(input_cases),
        allowed_cases,
        "export_video_human_mask",
        str(base_path),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    for case_name in case_names:
        case_dir = base_path / case_name
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        case_output_dir = output_path / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        depth_dir = case_dir / "depth"
        depth_camera_dirs = [p for p in depth_dir.glob("*") if p.is_dir()]
        if len(depth_camera_dirs) != args.camera_num:
            raise RuntimeError(
                f"[export_video_human_mask] {case_name} depth camera count="
                f"{len(depth_camera_dirs)} != {args.camera_num}"
            )

        for camera_idx in range(args.camera_num):
            print(f"Processing {case_name} camera {camera_idx}")
            cmd: list[str] = [
                args.python,
                str(ROOT / "data_process" / "segment_util_video.py"),
                "--output_path",
                str(case_output_dir),
                "--base_path",
                str(base_path),
                "--case_name",
                case_name,
                "--TEXT_PROMPT",
                args.text_prompt,
                "--camera_idx",
                str(camera_idx),
            ]
            mask_info_path = case_dir / "mask" / f"mask_info_{camera_idx}.json"
            mask_root_path = case_dir / "mask" / str(camera_idx)
            if mask_info_path.is_file() and mask_root_path.is_dir():
                cmd.extend(
                    [
                        "--exclude_mask_info",
                        str(mask_info_path),
                        "--exclude_mask_root",
                        str(mask_root_path),
                    ]
                )
            subprocess.run(cmd, check=True, cwd=str(ROOT))

            tmp_data_dir = case_dir / "tmp_data"
            if tmp_data_dir.exists():
                shutil.rmtree(tmp_data_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
