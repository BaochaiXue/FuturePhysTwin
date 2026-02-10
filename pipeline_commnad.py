#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# START_STAGE accepts stage names from STEPS; None starts from the first stage.
# Mapping to original commands in pipeline_commnad.sh:
#   script_process_data -> python script_process_data.py
#   export_video_human_mask -> python export_video_human_mask.py
#   dynamic_export_gs_data -> python dynamic_export_gs_data.py
#   script_optimize -> python script_optimize.py
#   script_train -> python script_train.py
#   script_inference -> python script_inference.py
#   dynamic_fast_gs -> python dynamic_fast_gs.py
#   final_eval -> python final_eval.py
START_STAGE: str | None = "script_optimize"
ARCHIVE_RESULT_DIR_NAME = "archive_result"
DEFAULT_TASK_NAME = "exp_new_data_and_old_data"
ARCHIVE_OUTPUT_DIRS = (
    "results",
    "gaussian_output_video",
    "gaussian_output_dynamic",
    "gaussian_output_dynamic_white",
)

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


def terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Best-effort terminate the whole process group for `proc` (POSIX).

    This prevents orphaned GPU/worker processes from surviving step failures.
    """

    if os.name != "posix":
        # Fallback: no process-group semantics.
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=grace_seconds)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        return

    pgid = proc.pid  # with start_new_session=True, pgid == pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        # If we can't signal, there's nothing else to do reliably.
        return

    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        return
    try:
        proc.wait(timeout=grace_seconds)
    except Exception:
        return


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


def run_step(
    step_name: str, script_path: Path, logs_dir: Path, python_bin: str
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [python_bin, str(script_path)]
    print(f"\n[Pipeline] Running {step_name}: {' '.join(cmd)}")
    env = os.environ.copy()
    # Force unbuffered stdout/stderr so logs stream in real time even when piped.
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    out_thread = tee_stream(proc.stdout, sys.stdout, logs_dir / f"{step_name}.out")
    err_thread = tee_stream(proc.stderr, sys.stderr, logs_dir / f"{step_name}.err")

    try:
        return_code = proc.wait()
        if return_code != 0:
            # Ensure no subprocesses linger in this step's process group.
            terminate_process_group(proc)
    except KeyboardInterrupt:
        terminate_process_group(proc)
        raise
    except Exception:
        terminate_process_group(proc)
        raise

    out_thread.join()
    err_thread.join()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    print(f"[Pipeline] Finished {step_name}")


def get_archive_task_dir(task_name: str) -> Path:
    archive_task_dir = ROOT / ARCHIVE_RESULT_DIR_NAME / task_name
    if archive_task_dir.exists():
        raise RuntimeError(f"Task archive already exists: {archive_task_dir}")
    return archive_task_dir


def archive_outputs(archive_task_dir: Path) -> None:
    archive_task_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_task_dir.mkdir(parents=False, exist_ok=False)

    for directory_name in ARCHIVE_OUTPUT_DIRS:
        source = ROOT / directory_name
        if not source.exists():
            continue
        shutil.move(str(source), str(archive_task_dir / directory_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pipeline with tee-style logs."
    )
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
    parser.add_argument(
        "--task-name",
        default=DEFAULT_TASK_NAME,
        help="Task name for output archiving (default: exp_new_data_and_old_data).",
    )
    stage_names = [step_name for step_name, _ in STEPS]
    default_start_stage = START_STAGE or stage_names[0]
    if default_start_stage not in stage_names:
        raise ValueError(
            f"Invalid START_STAGE={START_STAGE!r}. Must be one of: {', '.join(stage_names)}"
        )
    parser.add_argument(
        "--start-stage",
        default=default_start_stage,
        choices=stage_names,
        help=(
            "Stage name to start from (default: first stage). "
            "Useful to resume a partially completed run."
        ),
    )
    return parser.parse_args()


def normalize_task_name(task_name: str) -> str:
    normalized = "_".join(task_name.split())
    if not normalized:
        raise ValueError("Task name cannot be empty.")
    return normalized


def main() -> int:
    args = parse_args()
    task_name = normalize_task_name(args.task_name)
    archive_task_dir = get_archive_task_dir(task_name)
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_index = next(
            i for i, (step_name, _) in enumerate(STEPS) if step_name == args.start_stage
        )
        if start_index > 0:
            skipped_steps = ", ".join(step_name for step_name, _ in STEPS[:start_index])
            print(
                (
                    "[Pipeline][Warning] Starting from stage "
                    f"'{args.start_stage}', skipping earlier stage(s): {skipped_steps}"
                ),
                file=sys.stderr,
            )
        for step_name, script_file in STEPS[start_index:]:
            run_step(step_name, ROOT / script_file, logs_dir, args.python)
        archive_outputs(archive_task_dir)
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # fail-fast, same behavior as `set -euo pipefail`
        print(f"\n[Pipeline] Failed: {exc}", file=sys.stderr)
        return 1

    print(f"[Pipeline] Archived outputs to: {archive_task_dir}")
    print("\n[Pipeline] All steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
