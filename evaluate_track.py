from __future__ import annotations

import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_PREDICTION_PATH = "./experiments"
DEFAULT_OUTPUT_FILE = "results/final_track.csv"
DEFAULT_CONFIG_PATH = "./data_config.csv"


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Evaluate tracking accuracy for predicted trajectories."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path(DEFAULT_BASE_PATH),
        help=f"Directory containing ground-truth tracking data (default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        default=Path(DEFAULT_PREDICTION_PATH),
        help=f"Directory containing predicted trajectories (default: {DEFAULT_PREDICTION_PATH}).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"CSV file to store evaluation results (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help=f"Case allowlist CSV path (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        track_errors.append(track_error)
    return np.mean(track_errors)


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    prediction_path = resolve_path_from_root(ROOT, args.prediction_path)
    output_file = resolve_path_from_root(ROOT, args.output_file)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "evaluate_track", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    case_dirs = sorted(path for path in base_path.glob("*") if path.is_dir())
    case_dirs_by_name = {case_dir.name: case_dir for case_dir in case_dirs}
    filtered_case_names = filter_candidates(
        [case_dir.name for case_dir in case_dirs],
        allowed_cases,
        "evaluate_track",
        str(base_path),
    )

    with output_file.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Case Name",
                "Train Track Error",
                "Test Track Error",
            ]
        )

        for case_name in filtered_case_names:
            print(f"Processing {case_name}!!!!!!!!!!!!!!!")
            case_dir = case_dirs_by_name[case_name]

            split_path = case_dir / "split.json"
            if not split_path.is_file():
                print(f"[warn] Missing split.json for {case_name}; skipping.")
                continue
            with split_path.open("r", encoding="utf-8") as f:
                split = json.load(f)
            train_frame = split["train"][1]
            test_frame = split["test"][1]

            inference_path = prediction_path / case_name / "inference.pkl"
            gt_track_path = case_dir / "gt_track_3d.pkl"
            if not inference_path.is_file():
                print(f"[warn] Missing inference.pkl for {case_name}; skipping.")
                continue
            if not gt_track_path.is_file():
                print(f"[warn] Missing gt_track_3d.pkl for {case_name}; skipping.")
                continue

            with inference_path.open("rb") as f:
                vertices = pickle.load(f)

            with gt_track_path.open("rb") as f:
                gt_track_3d = pickle.load(f)

            mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
            kdtree = KDTree(vertices[0])
            _dis, idx = kdtree.query(gt_track_3d[0][mask])

            train_track_error = evaluate_prediction(
                1, train_frame, vertices, gt_track_3d, idx, mask
            )
            test_track_error = evaluate_prediction(
                train_frame, test_frame, vertices, gt_track_3d, idx, mask
            )
            writer.writerow([case_name, train_track_error, test_track_error])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
