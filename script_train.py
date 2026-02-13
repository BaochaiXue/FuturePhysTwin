"""
Run the first-order inverse-physics training for each case.

Inputs
------
- Optimised parameters from ``experiments_optimization/<case>/optimal_params.pkl``.
- Case data in ``data/different_types/<case>/`` including ``final_data.pkl``, calibration, metadata, and ``split.json``.

Outputs
-------
- Training checkpoints, logs, and metrics written to ``experiments/<case>/``.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from case_filter import (
    filter_candidates,
    load_config_cases,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_CONFIG_PATH = "./data_config.csv"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 3.0
ROOT = Path(__file__).resolve().parent
OPT_RESULTS_ROOT = ROOT / "experiments_optimization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train_warp.py for each case with retry support and fail-fast behavior."
        )
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=f"Base directory containing case folders (default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch train_warp.py.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Case allowlist CSV path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum retries per case after the first attempt "
            f"(default: {DEFAULT_MAX_RETRIES})."
        ),
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=DEFAULT_RETRY_BASE_DELAY,
        help=(
            "Base delay in seconds for exponential backoff between retries "
            f"(default: {DEFAULT_RETRY_BASE_DELAY})."
        ),
    )
    args = parser.parse_args()

    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_base_delay < 0:
        parser.error("--retry-base-delay must be >= 0")
    return args


def _load_train_frame(split_path: Path) -> int:
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)
    if "train" not in split or not isinstance(split["train"], list) or len(split["train"]) < 2:
        raise ValueError(f"Invalid split.json format: {split_path}")

    train_frame = split["train"][1]
    if not isinstance(train_frame, int):
        raise ValueError(f"split['train'][1] is not an int: {split_path}")
    return train_frame


def _signal_name_from_returncode(returncode: int) -> str:
    if returncode >= 0:
        return "N/A"
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


def _validate_case_inputs(case_dir: Path, case_name: str) -> int:
    split_path = case_dir / "split.json"
    final_data_path = case_dir / "final_data.pkl"
    calibrate_path = case_dir / "calibrate.pkl"
    metadata_path = case_dir / "metadata.json"
    optimal_params_path = OPT_RESULTS_ROOT / case_name / "optimal_params.pkl"

    required_files = [
        split_path,
        final_data_path,
        calibrate_path,
        metadata_path,
        optimal_params_path,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    return _load_train_frame(split_path)


def _cleanup_best_checkpoints(case_name: str, attempt: int, max_attempts: int) -> int:
    train_dir = ROOT / "experiments" / case_name / "train"
    if not train_dir.exists():
        print(
            "[Train][{case}] Cleanup best before attempt {attempt}/{total}: deleted=0 "
            "(train dir missing: {path})".format(
                case=case_name,
                attempt=attempt,
                total=max_attempts,
                path=train_dir,
            )
        )
        return 0

    deleted = 0
    for best_ckpt in train_dir.glob("best_*.pth"):
        if best_ckpt.is_file():
            try:
                best_ckpt.unlink()
            except OSError as exc:
                raise OSError(
                    f"[Train][{case_name}] Failed to delete checkpoint: {best_ckpt}"
                ) from exc
            deleted += 1

    print(
        "[Train][{case}] Cleanup best before attempt {attempt}/{total}: deleted={deleted}".format(
            case=case_name,
            attempt=attempt,
            total=max_attempts,
            deleted=deleted,
        )
    )
    return deleted


def _run_case_with_retries(
    *,
    case_name: str,
    base_path: str,
    train_frame: int,
    python_bin: str,
    max_retries: int,
    retry_base_delay: float,
) -> bool:
    max_attempts = max_retries + 1
    cmd = [
        python_bin,
        "train_warp.py",
        "--base_path",
        base_path,
        "--case_name",
        case_name,
        "--train_frame",
        str(train_frame),
    ]

    for attempt in range(1, max_attempts + 1):
        _cleanup_best_checkpoints(
            case_name=case_name,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        print(f"[Train][{case_name}] Attempt {attempt}/{max_attempts}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode == 0:
            print(f"[Train][{case_name}] Success on attempt {attempt}/{max_attempts}.")
            return True

        signal_terminated = completed.returncode < 0
        signal_name = _signal_name_from_returncode(completed.returncode)
        will_retry = attempt < max_attempts
        print(
            "[Train][{case}] Failure: attempt={attempt}/{total}, returncode={rc}, "
            "signal_terminated={sig_term}, signal={sig_name}, will_retry={will_retry}".format(
                case=case_name,
                attempt=attempt,
                total=max_attempts,
                rc=completed.returncode,
                sig_term=signal_terminated,
                sig_name=signal_name,
                will_retry=will_retry,
            )
        )

        if will_retry:
            delay = retry_base_delay * (2 ** (attempt - 1))
            if delay > 0:
                print(f"[Train][{case_name}] Retrying in {delay:.1f}s...")
                time.sleep(delay)

    print(f"[Train][{case_name}] Exhausted retries, marking as failed.")
    return False


def main() -> int:
    args = parse_args()

    base_path = resolve_path_from_root(ROOT, args.base_path)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Base path is not a directory: {base_path}")
    config_path = resolve_path_from_root(ROOT, args.config_path)
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "script_train", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    case_dirs = sorted([path for path in base_path.glob("*") if path.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case directories found under: {base_path}")
    case_dirs_by_name = {case_dir.name: case_dir for case_dir in case_dirs}
    filtered_case_names = filter_candidates(
        [case_dir.name for case_dir in case_dirs],
        allowed_cases,
        "script_train",
        str(base_path),
    )
    case_dirs = [case_dirs_by_name[name] for name in filtered_case_names]
    if not case_dirs:
        print("[Train] No allowed cases found after data_config.csv filtering.")
        print("[Train] Summary: success=0, failed=0")
        return 0

    success_count = 0
    failed_cases: list[str] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        train_frame = _validate_case_inputs(case_dir, case_name)
        success = _run_case_with_retries(
            case_name=case_name,
            base_path=str(base_path),
            train_frame=train_frame,
            python_bin=args.python,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        if success:
            success_count += 1
            continue

        failed_cases.append(case_name)
        print(
            "[Train] Summary: success={success}, failed={failed}".format(
                success=success_count,
                failed=len(failed_cases),
            )
        )
        print(f"[Train] Failed case(s): {', '.join(failed_cases)}")
        return 1

    print(
        "[Train] Summary: success={success}, failed={failed}".format(
            success=success_count,
            failed=len(failed_cases),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
