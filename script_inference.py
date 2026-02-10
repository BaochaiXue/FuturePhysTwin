"""
Perform model inference on each trained case.

Inputs
------
- Trained checkpoints in ``experiments/<case>/``.
- Supporting optimisation results in ``experiments_optimization/<case>/``.
- Processed case data within ``data/different_types/<case>/``.

Outputs
-------
- Inference artefacts (e.g. ``inference.pkl`` trajectories) saved inside ``experiments/<case>/``.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_EXPERIMENTS_PATH = "./experiments"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 3.0
ROOT = Path(__file__).resolve().parent
OPT_RESULTS_ROOT = ROOT / "experiments_optimization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference_warp.py for each case with retry support and fail-fast behavior."
        )
    )
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=f"Base directory containing case folders (default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "--experiments-path",
        default=DEFAULT_EXPERIMENTS_PATH,
        help=(
            "Directory containing experiment case folders used to enumerate "
            f"inference cases (default: {DEFAULT_EXPERIMENTS_PATH})."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch inference_warp.py.",
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


def _signal_name_from_returncode(returncode: int) -> str:
    if returncode >= 0:
        return "N/A"
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


def _validate_case_inputs(case_name: str, base_path: Path) -> None:
    case_data_dir = base_path / case_name
    final_data_path = case_data_dir / "final_data.pkl"
    calibrate_path = case_data_dir / "calibrate.pkl"
    metadata_path = case_data_dir / "metadata.json"
    optimal_params_path = OPT_RESULTS_ROOT / case_name / "optimal_params.pkl"

    required_files = [
        final_data_path,
        calibrate_path,
        metadata_path,
        optimal_params_path,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")


def _run_case_with_retries(
    *,
    case_name: str,
    base_path: str,
    python_bin: str,
    max_retries: int,
    retry_base_delay: float,
) -> bool:
    max_attempts = max_retries + 1
    cmd = [
        python_bin,
        "inference_warp.py",
        "--base_path",
        base_path,
        "--case_name",
        case_name,
    ]

    for attempt in range(1, max_attempts + 1):
        print(f"[Inference][{case_name}] Attempt {attempt}/{max_attempts}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if completed.returncode == 0:
            print(
                f"[Inference][{case_name}] Success on attempt {attempt}/{max_attempts}."
            )
            return True

        signal_terminated = completed.returncode < 0
        signal_name = _signal_name_from_returncode(completed.returncode)
        will_retry = attempt < max_attempts
        print(
            "[Inference][{case}] Failure: attempt={attempt}/{total}, returncode={rc}, "
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
                print(f"[Inference][{case_name}] Retrying in {delay:.1f}s...")
                time.sleep(delay)

    print(f"[Inference][{case_name}] Exhausted retries, marking as failed.")
    return False


def main() -> int:
    args = parse_args()

    base_path = Path(args.base_path)
    if not base_path.is_absolute():
        base_path = (ROOT / base_path).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Base path is not a directory: {base_path}")

    experiments_path = Path(args.experiments_path)
    if not experiments_path.is_absolute():
        experiments_path = (ROOT / experiments_path).resolve()
    if not experiments_path.exists() or not experiments_path.is_dir():
        raise FileNotFoundError(f"Experiments path is not a directory: {experiments_path}")

    case_dirs = sorted([path for path in experiments_path.glob("*") if path.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case directories found under: {experiments_path}")

    success_count = 0
    failed_cases: list[str] = []

    for case_dir in case_dirs:
        case_name = case_dir.name
        _validate_case_inputs(case_name, base_path)
        success = _run_case_with_retries(
            case_name=case_name,
            base_path=str(base_path),
            python_bin=args.python,
            max_retries=args.max_retries,
            retry_base_delay=args.retry_base_delay,
        )
        if success:
            success_count += 1
            continue

        failed_cases.append(case_name)
        print(
            "[Inference] Summary: success={success}, failed={failed}".format(
                success=success_count,
                failed=len(failed_cases),
            )
        )
        print(f"[Inference] Failed case(s): {', '.join(failed_cases)}")
        return 1

    print(
        "[Inference] Summary: success={success}, failed={failed}".format(
            success=success_count,
            failed=len(failed_cases),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Inference] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
