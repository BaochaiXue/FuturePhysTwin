from __future__ import annotations

import glob
import json
import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def resolve_from_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def command_prefix(command: str | None) -> list[str]:
    return shlex.split(command) if command else [sys.executable]


def format_cmd(cmd: str | list[str]) -> str:
    if isinstance(cmd, str):
        return cmd
    return shlex.join(str(part) for part in cmd)


def setup_logger(log_file: str = "timer.log") -> logging.Logger:
    logger = logging.getLogger("GlobalLogger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class Timer:
    def __init__(self, task_name: str, case_name: str, logger: logging.Logger):
        self.task_name = task_name
        self.case_name = case_name
        self.logger = logger

    def __enter__(self) -> None:
        self.start_time = time.time()
        self.logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {self.case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


class CommandRunner:
    def __init__(self, cwd: Path = ROOT):
        self.cwd = cwd
        self.records: list[dict[str, Any]] = []

    def record_skip(self, stage: str, reason: str) -> None:
        self.records.append(
            {
                "stage": stage,
                "command": "",
                "cwd": str(self.cwd),
                "status": "skipped",
                "reason": reason,
            }
        )

    def run(
        self,
        cmd: str | list[str],
        *,
        stage: str,
        retries: int = 0,
        retry_delay: float = 3.0,
    ) -> None:
        attempt = 0
        started_at = datetime.now(timezone.utc).isoformat()
        while True:
            attempt += 1
            result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=str(self.cwd))
            if result.returncode == 0:
                self.records.append(
                    {
                        "stage": stage,
                        "command": format_cmd(cmd),
                        "cwd": str(self.cwd),
                        "status": "passed",
                        "attempts": attempt,
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                return
            if attempt > retries:
                self.records.append(
                    {
                        "stage": stage,
                        "command": format_cmd(cmd),
                        "cwd": str(self.cwd),
                        "status": "failed",
                        "attempts": attempt,
                        "returncode": result.returncode,
                        "started_at": started_at,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                raise RuntimeError(
                    f"Command failed (code {result.returncode}): {format_cmd(cmd)}"
                )
            time.sleep(retry_delay)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_cam0_object_mask(
    base_path: str | Path,
    case_name: str,
    controller_name: str,
) -> Path:
    case_dir = Path(base_path) / case_name
    mask_info_path = case_dir / "mask" / "mask_info_0.json"
    with mask_info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    obj_idx: int | None = None
    controller_key = controller_name.strip().casefold()
    for key, value in data.items():
        label = str(value).strip()
        if label.casefold() != controller_key:
            if obj_idx is not None:
                raise ValueError(f"More than one non-controller object in {mask_info_path}.")
            obj_idx = int(key)
    if obj_idx is None:
        raise ValueError(f"No non-controller object detected in {mask_info_path}.")
    return case_dir / "mask" / "0" / str(obj_idx) / "0.png"


def write_split(base_path: str | Path, case_name: str) -> dict[str, Any]:
    case_dir = Path(base_path) / case_name
    frame_len = len(glob.glob(str(case_dir / "pcd" / "*.npz")))
    split = {
        "frame_len": frame_len,
        "train": [0, int(frame_len * 0.7)],
        "test": [int(frame_len * 0.7), frame_len],
    }
    with (case_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle)
    return split


def route_outputs(base_path: str | Path, case_name: str, route: str) -> dict[str, str]:
    case_dir = Path(base_path) / case_name
    outputs = {
        "shape_object_glb": str(case_dir / "shape" / "object.glb"),
        "aligned_final_mesh_glb": str(case_dir / "shape" / "matching" / "final_mesh.glb"),
        "final_data_pkl": str(case_dir / "final_data.pkl"),
        "final_pcd_mp4": str(case_dir / "final_pcd.mp4"),
        "final_data_mp4": str(case_dir / "final_data.mp4"),
    }
    if route == "mvsam3d":
        outputs["mvsam3d_debug_dir"] = str(case_dir / "shape" / "mvsam3d")
        outputs["mvsam3d_align_metrics"] = str(
            case_dir / "shape" / "mvsam3d" / "align" / "metrics.json"
        )
    return outputs


def write_route_manifest(
    *,
    base_path: str | Path,
    case_name: str,
    route: str,
    backend: str,
    category: str,
    controller_name: str,
    shape_prior: bool,
    commands: list[dict[str, Any]],
    split: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    shape_dir = Path(base_path) / case_name / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "route": route,
        "backend": backend,
        "case_name": case_name,
        "category": category,
        "controller_name": controller_name,
        "shape_prior": shape_prior,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commands": commands,
        "outputs": route_outputs(base_path, case_name, route),
        "split": split,
    }
    if extra:
        manifest.update(extra)
    manifest_path = shape_dir / "route_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path
