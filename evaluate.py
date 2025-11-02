#!/usr/bin/env python3
"""
Python equivalent of ``evaluate.sh`` for running evaluation suites.

Inputs
------
- Predictions in ``experiments/<case>/`` (mesh/trajectory exports).
- Evaluation assets in ``data/different_types/<case>/`` and ``data/render_eval_data/<case>/``.
- Render sequences in ``gaussian_output_dynamic``.

Outputs
-------
- Aggregated quantitative metrics written to ``results/`` (CSV files, logs).
"""

from __future__ import annotations

import time
import subprocess
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "experiments"
DEFAULT_BASE_PATH = PROJECT_ROOT / "data/different_types"
DEFAULT_RENDER_PATH = PROJECT_ROOT / "data/render_eval_data"
DEFAULT_HUMAN_MASK_PATH = PROJECT_ROOT / "data/different_types_human_mask"
DEFAULT_GAUSSIAN_DATA_ROOT = PROJECT_ROOT / "data/gaussian_data"
DEFAULT_GAUSSIAN_OUTPUT_DIR = PROJECT_ROOT / "gaussian_output_dynamic"


EVAL_COMMANDS: Sequence[Sequence[str]] = (
    (
        "python",
        "evaluate_chamfer.py",
        "--prediction_dir",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(RESULTS_DIR / "final_results.csv"),
    ),
    (
        "python",
        "evaluate_track.py",
        "--prediction_path",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(RESULTS_DIR / "final_track.csv"),
    ),
    (
        "python",
        "gaussian_splatting/evaluate_render.py",
        "--render_path",
        str(DEFAULT_RENDER_PATH),
        "--human_mask_path",
        str(DEFAULT_HUMAN_MASK_PATH),
        "--root_data_dir",
        str(DEFAULT_GAUSSIAN_DATA_ROOT),
        "--output_dir",
        str(DEFAULT_GAUSSIAN_OUTPUT_DIR),
        "--log_dir",
        str(RESULTS_DIR),
    ),
)


def run_command(
    command: Sequence[str], retries: int = 3, sleep_time: float = 2.0
) -> None:
    """
    Execute ``command`` with a simple retry loop.

    Args:
        command: Command to execute (argv style).
        retries: Maximum number of attempts before raising the failure.
        sleep_time: Seconds to wait between attempts.
    """

    attempts = 0
    while True:
        try:
            subprocess.run(list(command), check=True)
            return
        except subprocess.CalledProcessError as exc:
            attempts += 1
            if attempts >= retries:
                raise
            print(
                f"[Retry {attempts}/{retries}] Command failed with code {exc.returncode}: "
                f"{' '.join(command)}"
            )
            time.sleep(sleep_time)


def main() -> None:
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    for command in EVAL_COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
