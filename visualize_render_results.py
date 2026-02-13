"""
Blend rendered sequences with original footage and masks to create overlays.

Inputs
------
- Dynamic render frames in ``gaussian_output_dynamic_white/<case>/<view>/``.
- Original RGB frames in ``data/different_types/<case>/color/<camera>/``.
- Human masks in ``data/different_types_human_mask/<case>/mask/<camera>/``.
- Object masks in ``data/render_eval_data/<case>/mask/<camera>/``.

Outputs
-------
- Overlay videos saved as ``gaussian_output_dynamic_white/<case>/<camera>_integrate.mp4``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_PREDICTION_DIR = "./gaussian_output_dynamic_white"
DEFAULT_HUMAN_MASK_PATH = "./data/different_types_human_mask"
DEFAULT_OBJECT_MASK_PATH = "./data/render_eval_data"
DEFAULT_CONFIG_PATH = "./data_config.csv"

HEIGHT = 480
WIDTH = 848
FPS = 30
ALPHA = 0.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend white-background renders with original footage and masks."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path(DEFAULT_BASE_PATH),
        help="Directory containing ground-truth RGB frames and splits.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        default=Path(DEFAULT_PREDICTION_DIR),
        help="Directory containing white-background render outputs.",
    )
    parser.add_argument(
        "--human_mask_path",
        type=Path,
        default=Path(DEFAULT_HUMAN_MASK_PATH),
        help="Directory containing human mask PNGs.",
    )
    parser.add_argument(
        "--object_mask_path",
        type=Path,
        default=Path(DEFAULT_OBJECT_MASK_PATH),
        help="Directory containing object mask PNGs.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help="Case allowlist CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import cv2
    import numpy as np

    base_path = resolve_path_from_root(ROOT, args.base_path)
    prediction_dir = resolve_path_from_root(ROOT, args.prediction_dir)
    human_mask_path = resolve_path_from_root(ROOT, args.human_mask_path)
    object_mask_path = resolve_path_from_root(ROOT, args.object_mask_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "visualize_render_results", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    case_dirs = sorted(path for path in base_path.glob("*") if path.is_dir())
    case_dirs_by_name = {case_dir.name: case_dir for case_dir in case_dirs}
    filtered_case_names = filter_candidates(
        [case_dir.name for case_dir in case_dirs],
        allowed_cases,
        "visualize_render_results",
        str(base_path),
    )

    for case_name in filtered_case_names:
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        with (base_path / case_name / "split.json").open("r", encoding="utf-8") as f:
            split = json.load(f)
        frame_len = split["frame_len"]

        for camera_idx in range(3):
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(
                str(prediction_dir / case_name / f"{camera_idx}_integrate.mp4"),
                fourcc,
                FPS,
                (WIDTH, HEIGHT),
            )

            for frame_idx in range(frame_len):
                render_path = (
                    prediction_dir / case_name / str(camera_idx) / f"{frame_idx:05d}.png"
                )
                origin_image_path = (
                    base_path / case_name / "color" / str(camera_idx) / f"{frame_idx}.png"
                )
                human_mask_image_path = (
                    human_mask_path
                    / case_name
                    / "mask"
                    / str(camera_idx)
                    / "0"
                    / f"{frame_idx}.png"
                )
                object_image_path = (
                    object_mask_path
                    / case_name
                    / "mask"
                    / str(camera_idx)
                    / f"{frame_idx}.png"
                )

                render_img = cv2.imread(str(render_path), cv2.IMREAD_UNCHANGED)
                origin_img = cv2.imread(str(origin_image_path))
                human_mask = cv2.imread(str(human_mask_image_path))
                object_mask = cv2.imread(str(object_image_path))
                if (
                    render_img is None
                    or origin_img is None
                    or human_mask is None
                    or object_mask is None
                ):
                    print(
                        f"[Warning] Missing frame assets for {case_name} cam={camera_idx} frame={frame_idx}; skipping frame."
                    )
                    continue

                human_mask_gray = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY) > 0
                object_mask_gray = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY) > 0
                _ = object_mask_gray  # Keep read for parity with prior behavior.

                final_image = origin_img.copy()
                render_mask = np.logical_and(
                    (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
                )
                render_img[~render_mask, 3] = 0

                final_image[:, :, :] = ALPHA * final_image + (1 - ALPHA) * np.array(
                    [255, 255, 255], dtype=np.uint8
                )

                test_alpha = render_img[:, :, 3] / 255
                final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                    :, :, None
                ] + final_image * (1 - test_alpha[:, :, None])

                final_image[human_mask_gray] = ALPHA * origin_img[human_mask_gray] + (
                    1 - ALPHA
                ) * np.array([255, 255, 255], dtype=np.uint8)

                video_writer.write(final_image)

            video_writer.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
