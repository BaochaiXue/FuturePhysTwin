"""
Launch CMA-ES optimisation for each case using the processed datasets.

Inputs
------
- ``data/different_types/<case>/final_data.pkl`` along with calibration and metadata (calibrate.pkl, metadata.json).
- Train/test split metadata in ``data/different_types/<case>/split.json``.

Outputs
-------
- Optimisation logs and ``optimal_params.pkl`` stored under ``experiments_optimization/<case>/``.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 3.0
ROOT = Path(__file__).resolve().parent
OPT_RESULTS_ROOT = ROOT / "experiments_optimization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run optimize_cma.py for each case with retry support and fail-fast behavior."
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
        help="Python executable used to launch optimize_cma.py.",
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
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=False,
        help=(
            "Skip cases that already have "
            "experiments_optimization/<case>/optimal_params.pkl."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help=(
            "Do not skip existing optimal_params.pkl and force rerun all cases "
            "(default behavior)."
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
        "optimize_cma.py",
        "--base_path",
        base_path,
        "--case_name",
        case_name,
        "--train_frame",
        str(train_frame),
    ]

    for attempt in range(1, max_attempts + 1):
        print(f"[Optimize][{case_name}] Attempt {attempt}/{max_attempts}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode == 0:
            print(f"[Optimize][{case_name}] Success on attempt {attempt}/{max_attempts}.")
            return True

        signal_terminated = completed.returncode < 0
        signal_name = _signal_name_from_returncode(completed.returncode)
        will_retry = attempt < max_attempts
        print(
            "[Optimize][{case}] Failure: attempt={attempt}/{total}, returncode={rc}, "
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
                print(f"[Optimize][{case_name}] Retrying in {delay:.1f}s...")
                time.sleep(delay)

    print(f"[Optimize][{case_name}] Exhausted retries, marking as failed.")
    return False


def main() -> int:
    args = parse_args()
    base_path = Path(args.base_path)
    if not base_path.is_absolute():
        # Resolve relative paths from the repository root to keep behavior stable
        # regardless of the caller's working directory.
        base_path = (ROOT / base_path).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Base path is not a directory: {base_path}")

    case_dirs = sorted([p for p in base_path.glob("*") if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case directories found under: {base_path}")

    success_count = 0
    skipped_count = 0
    failed_cases: list[str] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        split_path = case_dir / "split.json"
        final_data_path = case_dir / "final_data.pkl"
        result_path = OPT_RESULTS_ROOT / case_name / "optimal_params.pkl"

        if not split_path.exists():
            raise FileNotFoundError(f"Missing required file: {split_path}")
        if not final_data_path.exists():
            raise FileNotFoundError(f"Missing required file: {final_data_path}")

        if args.skip_existing and result_path.exists():
            skipped_count += 1
            print(
                f"[Optimize][{case_name}] Skipped because existing result is found: {result_path}"
            )
            continue

        train_frame = _load_train_frame(split_path)
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
            "[Optimize] Summary: success={success}, skipped={skipped}, failed={failed}".format(
                success=success_count,
                skipped=skipped_count,
                failed=len(failed_cases),
            )
        )
        print(f"[Optimize] Failed case(s): {', '.join(failed_cases)}")
        return 1

    print(
        "[Optimize] Summary: success={success}, skipped={skipped}, failed={failed}".format(
            success=success_count,
            skipped=skipped_count,
            failed=len(failed_cases),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Optimize] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
