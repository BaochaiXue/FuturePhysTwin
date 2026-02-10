#!/usr/bin/env python3
"""
Export Gaussian-ready assets for every frame of every case.

Inputs
------
- Processed case folders in ``data/different_types/<case>/`` containing:
  * ``color/<cam>/<frame>.png`` RGB images.
  * ``depth/<cam>/<frame>.npy`` depth maps.
  * ``mask/**`` segmentation artefacts.
  * ``pcd/<frame>.npz`` fused point clouds and ``mask/processed_masks.pkl``.
  * ``calibrate.pkl`` (camera extrinsics) and ``metadata.json`` (intrinsics, image size).
  * Optional ``shape/matching/final_mesh.glb`` mesh prior.

Outputs
-------
- Legacy first-frame export copied to ``data/gaussian_data/<case>/``.
- Per-frame Gaussian datasets in ``per_frame_gaussian_data/<frame>/<case>/`` including RGB/depth/mask imagery,
  ``camera_meta.pkl``, ``observation.ply``, and ``shape_prior.glb`` when available.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import queue
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

_Task = dict[str, object]
_Result = dict[str, object]


def _configure_line_buffering() -> None:
    # Ensure prompt log flushing even when stdout/stderr are piped.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _frame_export_worker_main(task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    _configure_line_buffering()

    # Import heavy dependencies inside the worker so native failures don't bring down
    # the parent process. This also reduces per-frame startup overhead vs. spawning
    # a brand new process per frame.
    from export_gaussian_data_one_frame_one_case import process_case_frame

    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            base_path = Path(str(task["base_path"]))
            output_path = Path(str(task["output_path"]))
            process_case_frame(
                base_path=base_path,
                output_path=output_path,
                case_name=str(task["case_name"]),
                category=str(task["category"]),
                shape_prior=str(task["shape_prior"]),
                frame_idx=int(task["frame_id"]),
                generate_high_png=False,
            )
        except Exception:
            result_queue.put(
                {
                    "ok": False,
                    "frame_id": int(task.get("frame_id", -1)),
                    "error": traceback.format_exc(),
                }
            )
        else:
            result_queue.put({"ok": True, "frame_id": int(task["frame_id"])})


class FrameExportWorker:
    """A long-lived worker process that exports many frames sequentially.

    If the worker crashes (e.g. SIGSEGV), the parent can restart it and retry the
    frame without killing the whole pipeline.
    """

    def __init__(self, ctx: mp.context.BaseContext) -> None:
        self._ctx = ctx
        self._proc: mp.Process | None = None
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None

    def start(self) -> None:
        if self.is_alive():
            return

        self.stop()
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_frame_export_worker_main,
            args=(self._task_queue, self._result_queue),
        )
        self._proc.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def stop(self) -> None:
        proc = self._proc
        task_queue = self._task_queue
        result_queue = self._result_queue
        self._proc = None
        self._task_queue = None
        self._result_queue = None

        if task_queue is not None:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2.0)

        for q in (task_queue, result_queue):
            if q is None:
                continue
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

    def run_task(self, task: _Task, *, poll: float = 0.2) -> _Result:
        if not self.is_alive():
            raise RuntimeError("worker is not running")
        assert self._task_queue is not None
        assert self._result_queue is not None

        self._task_queue.put(task)
        while True:
            # Detect hard crashes (SIGSEGV etc.) without hanging on queue reads.
            if self._proc is not None and self._proc.exitcode is not None:
                raise RuntimeError(f"worker crashed with exitcode {self._proc.exitcode}")
            try:
                result = self._result_queue.get(timeout=poll)
            except queue.Empty:
                continue
            return dict(result)


def run_command(
    cmd: list[str], *, attempts: int = 5, delay: float = 2.0, echo: bool = True
) -> None:
    """
    Execute an external command with lightweight retry support.

    Args:
        cmd: Command (argv style) to execute.
        attempts: Maximum number of tries before surfacing the failure.
        delay: Seconds to wait between attempts.
        echo: When true, print the command on retry to aid debugging.
    """

    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == attempts:
                raise
            if echo:
                cmd_str = " ".join(str(part) for part in cmd)
                print(
                    f"[warn] Command failed (attempt {attempt}/{attempts}): {cmd_str}\n"
                    f"       Retrying in {delay:.1f}s… ({exc})",
                    flush=True,
                )
            time.sleep(delay)


def run_frame_export(
    *,
    worker: FrameExportWorker,
    base_path: Path,
    output_path: Path,
    case_name: str,
    category: str,
    shape_prior: str,
    frame_id: int,
    attempts: int = 5,
    delay: float = 2.0,
) -> None:
    """
    Execute per-frame export in a persistent worker process with retries.

    This reduces per-frame process start overhead while still providing isolation:
    if the worker crashes (e.g. SIGSEGV), it is restarted and the frame retried.
    """

    for attempt in range(1, attempts + 1):
        task: _Task = {
            "base_path": str(base_path),
            "output_path": str(output_path),
            "case_name": case_name,
            "category": category,
            "shape_prior": shape_prior,
            "frame_id": frame_id,
        }

        try:
            worker.start()
            result = worker.run_task(task)
        except Exception as exc:
            failure_reason = str(exc)
            worker.stop()
        else:
            if bool(result.get("ok", False)):
                return
            error_text = str(result.get("error", "")).strip()
            failure_reason = (
                error_text.splitlines()[-1] if error_text else "unknown error"
            )
            # Reset the worker on Python exceptions too; native libs can remain in a
            # bad state after an error.
            worker.stop()

        if attempt == attempts:
            raise RuntimeError(
                "Frame export failed after retries: "
                f"case={case_name}, frame={frame_id}. Reason: {failure_reason}"
            )

        print(
            f"[warn] Frame export failed (attempt {attempt}/{attempts}): "
            f"case={case_name}, frame={frame_id}\n"
            f"       Retrying in {delay:.1f}s… ({failure_reason})",
            flush=True,
        )
        time.sleep(delay)


def lookup_case_metadata(case_name: str, config_path: Path) -> tuple[str, str]:
    with config_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            name, category, shape_prior = row[:3]
            if name == case_name:
                return category, shape_prior
    raise ValueError(f"Case '{case_name}' not found in {config_path}")


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def collect_frame_ids(case_dir: Path) -> list[int]:
    """
    Return all frame indices with a corresponding point cloud npz file.
    """

    pcd_dir = case_dir / "pcd"
    if not pcd_dir.exists():
        print(
            f"Warning: {pcd_dir} missing; skipping case {case_dir.name}.",
            flush=True,
        )
        return []

    frame_ids: list[int] = []
    for npz_file in pcd_dir.glob("*.npz"):
        try:
            frame_ids.append(int(npz_file.stem))
        except ValueError:
            continue
    return sorted(frame_ids)


def required_assets_exist(
    case_dir: Path, frame_id: int, cam_indices: Iterable[int]
) -> bool:
    """
    Check that RGB and depth files exist for the supplied frame/cameras.
    """

    missing: list[Path] = []
    for cam in cam_indices:
        rgb_path = case_dir / "color" / str(cam) / f"{frame_id}.png"
        depth_path = case_dir / "depth" / str(cam) / f"{frame_id}.npy"
        if not rgb_path.exists():
            missing.append(rgb_path)
        if not depth_path.exists():
            missing.append(depth_path)
    if missing:
        print(
            f"Skipping frame {frame_id} in {case_dir.name}; missing: "
            + ", ".join(str(path) for path in missing),
            flush=True,
        )
        return False
    return True


def main() -> None:
    _configure_line_buffering()
    root = Path(__file__).resolve().parent

    # Keep the legacy single-frame export to avoid breaking downstream scripts.
    run_command(["python", str(root / "export_gaussian_data.py")])

    config_path = root / "data_config.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    per_frame_root = root / "per_frame_gaussian_data"
    ensure_dir(per_frame_root)

    data_root = root / "data" / "different_types"
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    required_cams = [0, 1, 2]

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        # Fallback for platforms where "fork" is unavailable.
        ctx = mp.get_context("spawn")

    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        color_dir = case_dir / "color"
        if not color_dir.exists():
            print(
                f"Warning: {color_dir} missing; skipping case {case_name}.",
                flush=True,
            )
            continue
        if any(not (color_dir / str(cam)).exists() for cam in required_cams):
            print(
                f"Warning: {case_name} missing required camera folders; skipping.",
                flush=True,
            )
            continue

        frame_ids = collect_frame_ids(case_dir)
        if not frame_ids:
            continue

        try:
            category, shape_prior = lookup_case_metadata(case_name, config_path)
        except ValueError:
            print(
                f"Warning: case {case_name} not found in {config_path}; skipping.",
                flush=True,
            )
            continue

        print(
            f"Processing case {case_name}: {len(frame_ids)} frames detected.",
            flush=True,
        )
        worker = FrameExportWorker(ctx)
        try:
            for frame_id in frame_ids:
                if not required_assets_exist(case_dir, frame_id, required_cams):
                    continue

                per_frame_output_dir = per_frame_root / str(frame_id)
                ensure_dir(per_frame_output_dir)

                run_frame_export(
                    worker=worker,
                    base_path=data_root,
                    output_path=per_frame_output_dir,
                    case_name=case_name,
                    category=category,
                    shape_prior=shape_prior,
                    frame_id=frame_id,
                )
        finally:
            worker.stop()


if __name__ == "__main__":
    main()
