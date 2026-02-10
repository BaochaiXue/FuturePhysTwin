#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT / "logs"
STEPS = [
    ("script_process_data", "script_process_data.py"),
    ("export_video_human_mask", "export_video_human_mask.py"),
    ("dynamic_export_gs_data", "dynamic_export_gs_data.py"),
    ("script_optimize", "script_optimize.py"),
    ("script_train", "script_train.py"),
    ("script_inference", "script_inference.py"),
    ("dynamic_fast_gs", "dynamic_fast_gs.py"),
    ("final_eval", "final_eval.py"),
]


def tee_stream(
    stream, target_stream, log_path: Path, log_mode: str = "w"
) -> threading.Thread:
    def _pump() -> None:
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            for line in iter(stream.readline, ""):
                target_stream.write(line)
                target_stream.flush()
                log_file.write(line)
                log_file.flush()
        stream.close()

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return t


def run_step(step_name: str, script_path: Path, logs_dir: Path, python_bin: str) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [python_bin, str(script_path)]
    print(f"\n[Pipeline] Running {step_name}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    out_thread = tee_stream(proc.stdout, sys.stdout, logs_dir / f"{step_name}.out")
    err_thread = tee_stream(proc.stderr, sys.stderr, logs_dir / f"{step_name}.err")

    return_code = proc.wait()
    out_thread.join()
    err_thread.join()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    print(f"[Pipeline] Finished {step_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline with tee-style logs.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run each step (default: current interpreter).",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for .out/.err logs (default: ./logs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        for step_name, script_file in STEPS:
            run_step(step_name, ROOT / script_file, logs_dir, args.python)
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # fail-fast, same behavior as `set -euo pipefail`
        print(f"\n[Pipeline] Failed: {exc}", file=sys.stderr)
        return 1

    print("\n[Pipeline] All steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
