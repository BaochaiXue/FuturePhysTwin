from __future__ import annotations

import csv
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTION_DIR = "./experiments"
DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_OUTPUT_FILE = "results/final_results.csv"
DEFAULT_CONFIG_PATH = "./data_config.csv"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate Chamfer distance for tracked trajectories.")
    parser.add_argument(
        "--prediction_dir",
        type=Path,
        default=Path(DEFAULT_PREDICTION_DIR),
        help=f"Directory containing predicted trajectories (default: {DEFAULT_PREDICTION_DIR}).",
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        default=Path(DEFAULT_BASE_PATH),
        help=f"Directory with ground-truth data (default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f"CSV file to write evaluation results (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help=f"Case allowlist CSV path (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser.parse_args()


def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    import torch
    from pytorch3d.loss import chamfer_distance

    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        current_object_motions_valid = object_motions_valid[frame_idx - 1]
        _ = current_object_motions_valid

        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)
    return {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }


def main() -> int:
    args = parse_args()
    prediction_dir = resolve_path_from_root(ROOT, args.prediction_dir)
    base_path = resolve_path_from_root(ROOT, args.base_path)
    output_file = resolve_path_from_root(ROOT, args.output_file)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "evaluate_chamfer", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    case_dirs = sorted(path for path in prediction_dir.glob("*") if path.is_dir())
    case_dirs_by_name = {case_dir.name: case_dir for case_dir in case_dirs}
    filtered_case_names = filter_candidates(
        [case_dir.name for case_dir in case_dirs],
        allowed_cases,
        "evaluate_chamfer",
        str(prediction_dir),
    )

    with output_file.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Case Name",
                "Train Frame Num",
                "Train Chamfer Error",
                "Test Frame Num",
                "Test Chamfer Error",
            ]
        )

        for case_name in filtered_case_names:
            case_prediction_dir = case_dirs_by_name[case_name]
            print(f"Processing {case_name}")

            inference_path = case_prediction_dir / "inference.pkl"
            case_data_dir = base_path / case_name
            final_data_path = case_data_dir / "final_data.pkl"
            split_path = case_data_dir / "split.json"
            if not inference_path.is_file():
                print(f"[warn] Missing inference.pkl for {case_name}; skipping.")
                continue
            if not final_data_path.is_file():
                print(f"[warn] Missing final_data.pkl for {case_name}; skipping.")
                continue
            if not split_path.is_file():
                print(f"[warn] Missing split.json for {case_name}; skipping.")
                continue

            with inference_path.open("rb") as f:
                vertices = pickle.load(f)

            with final_data_path.open("rb") as f:
                data = pickle.load(f)

            object_points = data["object_points"]
            object_visibilities = data["object_visibilities"]
            object_motions_valid = data["object_motions_valid"]
            num_original_points = object_points.shape[1]
            num_surface_points = num_original_points + data["surface_points"].shape[0]

            with split_path.open("r", encoding="utf-8") as f:
                split = json.load(f)
            train_frame = split["train"][1]
            test_frame = split["test"][1]

            assert (
                test_frame == vertices.shape[0]
            ), f"Test frame {test_frame} != {vertices.shape[0]}"

            results_train = evaluate_prediction(
                1,
                train_frame,
                vertices,
                object_points,
                object_visibilities,
                object_motions_valid,
                num_original_points,
                num_surface_points,
            )
            results_test = evaluate_prediction(
                train_frame,
                test_frame,
                vertices,
                object_points,
                object_visibilities,
                object_motions_valid,
                num_original_points,
                num_surface_points,
            )

            writer.writerow(
                [
                    case_name,
                    results_train["frame_len"],
                    results_train["chamfer_error"],
                    results_test["frame_len"],
                    results_test["chamfer_error"],
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
