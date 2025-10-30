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


EVAL_COMMANDS: Sequence[Sequence[str]] = (
    ("python", "evaluate_chamfer.py"),
    ("python", "evaluate_track.py"),
    ("python", "gaussian_splatting/evaluate_render.py"),
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
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    for command in EVAL_COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
