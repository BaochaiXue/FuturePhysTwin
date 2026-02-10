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

import multiprocessing as mp
import queue
import subprocess
import time
import traceback
from pathlib import Path
from typing import Iterable

from export_gaussian_data_one_frame_one_case import (
    lookup_case_metadata,
    process_case_frame,
)


def _frame_export_worker(
    error_queue: mp.Queue,
    *,
    base_path: Path,
    output_path: Path,
    case_name: str,
    category: str,
    shape_prior: str,
    frame_id: int,
) -> None:
    try:
        process_case_frame(
            base_path=base_path,
            output_path=output_path,
            case_name=case_name,
            category=category,
            shape_prior=shape_prior,
            frame_idx=frame_id,
            generate_high_png=False,
        )
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


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
                    f"       Retrying in {delay:.1f}s… ({exc})"
                )
            time.sleep(delay)


def run_frame_export(
    *,
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
    Execute per-frame export in an isolated child process with retries.

    Isolation preserves retry behavior when native code crashes (e.g. SIGSEGV).
    """

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        # Fallback for platforms where "fork" is unavailable.
        ctx = mp.get_context("spawn")

    for attempt in range(1, attempts + 1):
        error_queue: mp.Queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_frame_export_worker,
            args=(error_queue,),
            kwargs={
                "base_path": base_path,
                "output_path": output_path,
                "case_name": case_name,
                "category": category,
                "shape_prior": shape_prior,
                "frame_id": frame_id,
            },
        )
        proc.start()
        proc.join()

        error_text: str | None = None
        try:
            error_text = error_queue.get_nowait()
        except queue.Empty:
            error_text = None
        error_queue.close()
        error_queue.join_thread()

        if proc.exitcode == 0 and error_text is None:
            return

        if proc.exitcode is None:
            failure_reason = "child process did not report an exit code"
        elif proc.exitcode < 0:
            failure_reason = f"child process crashed with signal {-proc.exitcode}"
        elif error_text:
            failure_reason = error_text.strip().splitlines()[-1]
        else:
            failure_reason = f"child process exited with code {proc.exitcode}"

        if attempt == attempts:
            raise RuntimeError(
                "Frame export failed after retries: "
                f"case={case_name}, frame={frame_id}. Reason: {failure_reason}"
            )

        print(
            f"[warn] Frame export failed (attempt {attempt}/{attempts}): "
            f"case={case_name}, frame={frame_id}\n"
            f"       Retrying in {delay:.1f}s… ({failure_reason})"
        )
        time.sleep(delay)


def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def collect_frame_ids(case_dir: Path) -> list[int]:
    """
    Return all frame indices with a corresponding point cloud npz file.
    """

    pcd_dir = case_dir / "pcd"
    if not pcd_dir.exists():
        print(f"Warning: {pcd_dir} missing; skipping case {case_dir.name}.")
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
            + ", ".join(str(path) for path in missing)
        )
        return False
    return True


def main() -> None:
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

    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        color_dir = case_dir / "color"
        if not color_dir.exists():
            print(f"Warning: {color_dir} missing; skipping case {case_name}.")
            continue
        if any(not (color_dir / str(cam)).exists() for cam in required_cams):
            print(f"Warning: {case_name} missing required camera folders; skipping.")
            continue

        frame_ids = collect_frame_ids(case_dir)
        if not frame_ids:
            continue

        try:
            category, shape_prior = lookup_case_metadata(case_name, config_path)
        except ValueError:
            print(f"Warning: case {case_name} not found in {config_path}; skipping.")
            continue

        print(f"Processing case {case_name}: {len(frame_ids)} frames detected.")
        for frame_id in frame_ids:
            if not required_assets_exist(case_dir, frame_id, required_cams):
                continue

            per_frame_output_dir = per_frame_root / str(frame_id)
            ensure_dir(per_frame_output_dir)

            run_frame_export(
                base_path=data_root,
                output_path=per_frame_output_dir,
                case_name=case_name,
                category=category,
                shape_prior=shape_prior,
                frame_id=frame_id,
            )


if __name__ == "__main__":
    main()
