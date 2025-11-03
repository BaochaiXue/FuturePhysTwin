#!/usr/bin/env python3
"""
Run the full evaluation pipeline with retry logic and per-step logging.

Stages executed (arguments shown explicitly so defaults stay discoverable):
    1. python gs_run_simulate.py --output_dir gaussian_output_dynamic
       --views 0 1 2 --exp_name init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0
       --data_root data/gaussian_data --gaussian_root gaussian_output
    2. python export_render_eval_data.py
    3. python evaluate_chamfer.py --prediction_dir experiments --base_path data/different_types
       --output_file results/final_results.csv
    4. python evaluate_track.py --prediction_path experiments --base_path data/different_types
       --output_file results/final_track.csv
    5. python gaussian_splatting/evaluate_render.py --render_path data/render_eval_data
       --human_mask_path data/different_types_human_mask --root_data_dir data/gaussian_data
       --output_dir gaussian_output_dynamic --log_dir results
    6. python gs_run_simulate_white.py --output_dir gaussian_output_dynamic_white
       --views 0 1 2 --exp_name init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0
       --data_root data/gaussian_data --gaussian_root gaussian_output
    7. python visualize_render_results.py

Key directories:
    - Canonical checkpoints: gaussian_output/<scene>/<exp_name>/
    - Scene data: data/gaussian_data/<scene>/
    - Differentiated assets: data/different_types/<case>/
    - Render evaluation copies: data/render_eval_data/<case>/
    - White-background renders: gaussian_output_dynamic_white/<case>/<view>/
    - Aggregated metrics and logs: results/, logs/

All paths are resolved relative to the repository root containing this script.

Outputs for each step are captured under ``logs/<step>.out`` and ``logs/<step>.err``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import shutil
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
DEFAULT_VIEWS: Sequence[str] = ("0", "1", "2")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "gaussian_data"
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "experiments"
DEFAULT_BASE_PATH = PROJECT_ROOT / "data" / "different_types"
DEFAULT_RENDER_PATH = PROJECT_ROOT / "data" / "render_eval_data"
DEFAULT_HUMAN_MASK_PATH = PROJECT_ROOT / "data" / "different_types_human_mask"

ORIGINAL_GAUSSIAN_ROOT_NEED_PREPROCESS = PROJECT_ROOT / "tmp_gaussian_output"
ORIGINAL_GAUSSIAN_ROOT = PROJECT_ROOT / "tmp_gaussian_output_FIXED"
ORIGINAL_OUTPUT_DIR = PROJECT_ROOT / "tmp_gaussian_output_dynamic"
ORIGINAL_WHITE_OUTPUT_DIR = PROJECT_ROOT / "tmp_gaussian_output_dynamic_white"
ORIGINAL_RESULTS_DIR = PROJECT_ROOT / "tmp_results"

OURS_GAUSSIAN_ROOT = PROJECT_ROOT / "gaussian_output"
OURS_OUTPUT_DIR = PROJECT_ROOT / "gaussian_output_dynamic"
OURS_WHITE_OUTPUT_DIR = PROJECT_ROOT / "gaussian_output_dynamic_white"
OURS_RESULTS_DIR = PROJECT_ROOT / "results"


def copy_tree(src: Path, dst: Path) -> None:
    """Recursively copy all files from ``src`` to ``dst``."""
    for item in src.rglob("*"):
        relative_path = item.relative_to(src)
        target_path = dst / relative_path
        if item.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(item.read_bytes())


def preprocess_tmp_gaussian_output(
    src_root: Path = ORIGINAL_GAUSSIAN_ROOT_NEED_PREPROCESS,
    dst_root: Path = ORIGINAL_GAUSSIAN_ROOT,
) -> None:
    if dst_root.exists():
        print(f"Removing existing preprocessed directory {dst_root}...")
        shutil.rmtree(dst_root)
    if not src_root.exists():
        print(f"Source directory {src_root} missing; skipping preprocessing.")
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"Preprocessing {src_root} to {dst_root}...")
    for scene_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        scene_name = scene_dir.name
        dst_scene_dir = dst_root / scene_name
        dst_scene_dir.mkdir(parents=True, exist_ok=True)
        experiments = {
            canonical_file.parent
            for canonical_file in scene_dir.rglob("canonical_gaussians.npz")
        }
        if not experiments:
            print(f"[Warn] No canonical outputs found under {scene_dir}; skipping.")
            continue
        for exp_dir in sorted(experiments, key=lambda path: str(path)):
            target_dir = dst_scene_dir / exp_dir.name
            print(f"  Copying {exp_dir} -> {target_dir}")
            copy_tree(exp_dir, target_dir)


PIPELINE_COMMANDS: Sequence[Sequence[str]] = (
    ("python", "export_render_eval_data.py"),
    (
        "python",
        "gs_run_simulate.py",
        "--output_dir",
        str(OURS_OUTPUT_DIR),
        "--views",
        *DEFAULT_VIEWS,
        "--exp_name",
        DEFAULT_EXP_NAME,
        "--data_root",
        str(DEFAULT_DATA_ROOT),
        "--gaussian_root",
        str(OURS_GAUSSIAN_ROOT),
    ),
    (
        "python",
        "evaluate_chamfer.py",
        "--prediction_dir",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(OURS_RESULTS_DIR / "final_results.csv"),
    ),
    (
        "python",
        "evaluate_track.py",
        "--prediction_path",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(OURS_RESULTS_DIR / "final_track.csv"),
    ),
    (
        "python",
        "gaussian_splatting/evaluate_render.py",
        "--render_path",
        str(DEFAULT_RENDER_PATH),
        "--human_mask_path",
        str(DEFAULT_HUMAN_MASK_PATH),
        "--root_data_dir",
        str(DEFAULT_DATA_ROOT),
        "--output_dir",
        str(OURS_OUTPUT_DIR),
        "--log_dir",
        str(OURS_RESULTS_DIR),
    ),
    (
        "python",
        "gs_run_simulate_white.py",
        "--output_dir",
        str(OURS_WHITE_OUTPUT_DIR),
        "--views",
        *DEFAULT_VIEWS,
        "--exp_name",
        DEFAULT_EXP_NAME,
        "--data_root",
        str(DEFAULT_DATA_ROOT),
        "--gaussian_root",
        str(OURS_GAUSSIAN_ROOT),
    ),
    (
        "python",
        "visualize_render_results.py",
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--prediction_dir",
        str(OURS_WHITE_OUTPUT_DIR),
        "--human_mask_path",
        str(DEFAULT_HUMAN_MASK_PATH),
        "--object_mask_path",
        str(DEFAULT_RENDER_PATH),
    ),
    (
        "python",
        "gs_run_simulate.py",
        "--output_dir",
        str(ORIGINAL_OUTPUT_DIR),
        "--views",
        *DEFAULT_VIEWS,
        "--exp_name",
        DEFAULT_EXP_NAME,
        "--data_root",
        str(DEFAULT_DATA_ROOT),
        "--gaussian_root",
        str(ORIGINAL_GAUSSIAN_ROOT),
    ),
    (
        "python",
        "evaluate_chamfer.py",
        "--prediction_dir",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(ORIGINAL_RESULTS_DIR / "final_results.csv"),
    ),
    (
        "python",
        "evaluate_track.py",
        "--prediction_path",
        str(DEFAULT_PREDICTION_DIR),
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--output_file",
        str(ORIGINAL_RESULTS_DIR / "final_track.csv"),
    ),
    (
        "python",
        "gaussian_splatting/evaluate_render.py",
        "--render_path",
        str(DEFAULT_RENDER_PATH),
        "--human_mask_path",
        str(DEFAULT_HUMAN_MASK_PATH),
        "--root_data_dir",
        str(DEFAULT_DATA_ROOT),
        "--output_dir",
        str(ORIGINAL_OUTPUT_DIR),
        "--log_dir",
        str(ORIGINAL_RESULTS_DIR),
    ),
    (
        "python",
        "gs_run_simulate_white.py",
        "--output_dir",
        str(ORIGINAL_WHITE_OUTPUT_DIR),
        "--views",
        *DEFAULT_VIEWS,
        "--exp_name",
        DEFAULT_EXP_NAME,
        "--data_root",
        str(DEFAULT_DATA_ROOT),
        "--gaussian_root",
        str(ORIGINAL_GAUSSIAN_ROOT),
    ),
    (
        "python",
        "visualize_render_results.py",
        "--base_path",
        str(DEFAULT_BASE_PATH),
        "--prediction_dir",
        str(ORIGINAL_WHITE_OUTPUT_DIR),
        "--human_mask_path",
        str(DEFAULT_HUMAN_MASK_PATH),
        "--object_mask_path",
        str(DEFAULT_RENDER_PATH),
    ),
)

DEFAULT_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0


def command_label(command: Sequence[str]) -> str:
    """Return a stable label used for log files."""
    for token in reversed(command):
        if token.endswith(".py"):
            return Path(token).stem
    return Path(command[0]).stem


def write_log(path: Path, text: str, attempt: int) -> None:
    """Append ``text`` to ``path`` while resetting the file on the first attempt."""
    mode = "w" if attempt == 1 else "a"
    with path.open(mode, encoding="utf-8") as handle:
        if attempt > 1:
            handle.write(f"\n=== Attempt {attempt} ===\n")
        if text:
            handle.write(text)


def run_command(
    command: Sequence[str],
    retries: int = DEFAULT_RETRIES,
    sleep_time: float = RETRY_SLEEP_SECONDS,
) -> None:
    """
    Execute ``command`` with retry support, capturing stdout/stderr to log files.
    """

    label = command_label(command)
    stdout_log = LOG_DIR / f"{label}.out"
    stderr_log = LOG_DIR / f"{label}.err"
    attempts = 0

    while True:
        attempts += 1
        print(f"[Run] {' '.join(command)} (attempt {attempts}/{retries})")
        try:
            result = subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            write_log(stdout_log, stdout, attempts)
            write_log(stderr_log, stderr, attempts)
            if stdout:
                sys.stdout.write(stdout)
            if stderr:
                sys.stderr.write(stderr)

            if attempts >= retries:
                print(
                    f"[Fail] Command aborted after {attempts} attempts: {' '.join(command)}",
                    file=sys.stderr,
                )
                raise

            print(
                f"[Retry] Command failed with code {exc.returncode}; retrying in {sleep_time}s…",
                file=sys.stderr,
            )
            time.sleep(sleep_time)
            continue

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        write_log(stdout_log, stdout, attempts)
        write_log(stderr_log, stderr, attempts)
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        print(f"[Done] {' '.join(command)}")
        return


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OURS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preprocess_tmp_gaussian_output()
    for command in PIPELINE_COMMANDS:
        run_command(command)


if __name__ == "__main__":
    main()
