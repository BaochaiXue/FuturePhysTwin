"""Assemble RGB frames and object masks for render-space evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from case_filter import (
    load_config_cases,
    load_config_rows,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_OUTPUT_PATH = "./data/render_eval_data"
DEFAULT_CONFIG_PATH = "./data_config.csv"
CONTROLLER_NAME = "hand"
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export render evaluation inputs for allowed cases."
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    output_path = resolve_path_from_root(ROOT, args.output_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_rows = load_config_rows(config_path)
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "export_render_eval_data", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    output_path.mkdir(parents=True, exist_ok=True)

    for row in config_rows:
        if len(row) < 3:
            raise ValueError(
                f"Malformed config row in {config_path}: expected >=3 columns, got {row}"
            )
        case_name = row[0].strip()
        if case_name not in allowed_cases:
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        case_input_dir = base_path / case_name
        case_output_dir = output_path / case_name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        color_src = case_input_dir / "color"
        if not color_src.is_dir():
            raise FileNotFoundError(f"Missing color directory: {color_src}")
        _copy_tree(color_src, case_output_dir / "color")

        mask_output_dir = case_output_dir / "mask"
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        for cam_idx in range(3):
            mask_info_path = case_input_dir / "mask" / f"mask_info_{cam_idx}.json"
            if not mask_info_path.is_file():
                raise FileNotFoundError(f"Missing mask info: {mask_info_path}")
            with mask_info_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            obj_idx: int | None = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError(
                            f"More than one object detected in {mask_info_path}"
                        )
                    obj_idx = int(key)
            if obj_idx is None:
                raise ValueError(f"No object mask detected in {mask_info_path}")

            src_mask_dir = case_input_dir / "mask" / str(cam_idx) / str(obj_idx)
            if not src_mask_dir.is_dir():
                raise FileNotFoundError(f"Missing object mask folder: {src_mask_dir}")
            _copy_tree(src_mask_dir, mask_output_dir / str(cam_idx))

        split_src = case_input_dir / "split.json"
        if not split_src.is_file():
            raise FileNotFoundError(f"Missing split file: {split_src}")
        shutil.copy2(split_src, case_output_dir / "split.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

