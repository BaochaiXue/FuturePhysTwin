from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_mvsam3d_scene import (  # noqa: E402
    FRAME_INDEX,
    discover_views,
    parse_view_indices,
    prepare_mvsam3d_scene,
    select_single_object,
)


DEFAULT_MVSAM3D_PYTHON = sys.executable
DEFAULT_PREPROCESS_PYTHON = sys.executable


def command_prefix(command: str | None) -> list[str]:
    return shlex.split(command or DEFAULT_MVSAM3D_PYTHON)


def preprocess_command_prefix(command: str | None) -> list[str]:
    return shlex.split(
        command or os.environ.get("MVSAM3D_PREPROCESS_PYTHON") or DEFAULT_PREPROCESS_PYTHON
    )


def format_command(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def resolve_mvsam3d_root(value: str | None, dry_run: bool) -> Path | None:
    raw = value or os.environ.get("MVSAM3D_ROOT")
    if not raw:
        if dry_run:
            return None
        raise ValueError("MV-SAM3D root is required. Pass --mvsam3d_root or set MVSAM3D_ROOT.")

    root = Path(raw).expanduser()
    if not root.exists():
        if dry_run:
            return root
        raise FileNotFoundError(f"MV-SAM3D root does not exist: {root}")
    return root.resolve()


def require_file(path: Path, description: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def run_logged(
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    log_name: str,
    dry_run: bool,
) -> dict[str, str]:
    stdout_path = logs_dir / f"{log_name}.stdout.log"
    stderr_path = logs_dir / f"{log_name}.stderr.log"
    if dry_run:
        return {
            "command": format_command(command),
            "cwd": str(cwd),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "status": "dry_run",
        }

    logs_dir.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}: {format_command(command)}\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )
    return {
        "command": format_command(command),
        "cwd": str(cwd),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "status": "passed",
    }


def parse_logged_output_dirs(log_paths: list[Path], root: Path) -> list[Path]:
    candidates: list[Path] = []
    pattern = re.compile(r"(?:All output files saved to|Output directory):\s*(.+)$")
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line.strip())
            if not match:
                continue
            raw_path = match.group(1).strip()
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            if path.exists() and path.is_dir():
                candidates.append(path)
    return candidates


def discover_mvsam3d_output_dir(
    mvsam3d_root: Path,
    scene_name: str,
    object_name: str,
    started_at: float,
    log_paths: list[Path],
) -> Path:
    candidates = parse_logged_output_dirs(log_paths, mvsam3d_root)

    search_root = mvsam3d_root / "visualization" / scene_name / object_name
    if search_root.exists():
        for path in search_root.iterdir():
            if path.is_dir() and path.stat().st_mtime >= started_at - 2.0:
                candidates.append(path)

    unique_candidates = sorted(
        {path.resolve() for path in candidates if (path / "result.glb").exists()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if unique_candidates:
        return unique_candidates[0]

    raise FileNotFoundError(
        f"No MV-SAM3D output directory with result.glb found under {search_root} "
        f"or logs after {datetime.fromtimestamp(started_at).isoformat()}."
    )


def copytree_replace(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and force:
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def skipped_record(command: str, cwd: Path) -> dict[str, str]:
    return {
        "command": command,
        "cwd": str(cwd),
        "stdout_log": "",
        "stderr_log": "",
        "status": "skipped",
    }


def prepare_legacy_upscaled_inputs(
    *,
    base_path: Path,
    case_name: str,
    category: str,
    controller_name: str,
    view_indices: str | None,
    work_dir: Path,
    logs_dir: Path,
    preprocess_python: str | None,
    force: bool,
    dry_run: bool,
) -> tuple[dict[str, dict[str, Path]], list[dict[str, str]]]:
    case_dir = base_path / case_name
    views = discover_views(case_dir, parse_view_indices(view_indices))
    python_prefix = preprocess_command_prefix(preprocess_python)
    repo_root = SCRIPT_DIR.parent
    image_upscale_script = SCRIPT_DIR / "image_upscale.py"
    segment_script = SCRIPT_DIR / "segment_util_image.py"
    preprocess_dir = work_dir / "preprocess" / "legacy_upscale"

    source_overrides: dict[str, dict[str, Path]] = {}
    records: list[dict[str, str]] = []
    for view in views:
        object_idx, _ = select_single_object(
            case_dir / "mask" / f"mask_info_{view}.json", controller_name
        )
        source_image_path = case_dir / "color" / view / f"{FRAME_INDEX}.png"
        source_mask_path = case_dir / "mask" / view / object_idx / f"{FRAME_INDEX}.png"
        view_dir = preprocess_dir / view
        high_resolution_path = view_dir / "high_resolution.png"
        masked_image_path = view_dir / "masked_image.png"
        if not dry_run:
            view_dir.mkdir(parents=True, exist_ok=True)

        upscale_command = python_prefix + [
            str(image_upscale_script),
            "--img_path",
            str(source_image_path),
            "--mask_path",
            str(source_mask_path),
            "--output_path",
            str(high_resolution_path),
            "--category",
            category,
        ]
        if dry_run or force or not high_resolution_path.exists():
            records.append(
                run_logged(
                    upscale_command,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    log_name=f"preprocess_view_{view}_upscale",
                    dry_run=dry_run,
                )
            )
        else:
            records.append(
                skipped_record(
                    f"Legacy upscale skipped because {high_resolution_path} already exists",
                    repo_root,
                )
            )

        segment_command = python_prefix + [
            str(segment_script),
            "--img_path",
            str(high_resolution_path),
            "--TEXT_PROMPT",
            category,
            "--output_path",
            str(masked_image_path),
        ]
        if dry_run or force or not masked_image_path.exists():
            records.append(
                run_logged(
                    segment_command,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    log_name=f"preprocess_view_{view}_segment",
                    dry_run=dry_run,
                )
            )
        else:
            records.append(
                skipped_record(
                    f"Legacy segmentation skipped because {masked_image_path} already exists",
                    repo_root,
                )
            )

        source_overrides[view] = {
            "image_path": high_resolution_path,
            "mask_path": masked_image_path,
        }

    return source_overrides, records


def validate_object_glb(path: Path) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required to validate MV-SAM3D object.glb before alignment."
        ) from exc

    mesh = trimesh.load(path, force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"MV-SAM3D object GLB is empty: {path}")
    if len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
        raise ValueError(f"MV-SAM3D object GLB has no vertices/faces: {path}")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
    }


def write_downstream_object_glb(
    source_glb: Path,
    object_glb: Path,
    max_faces: int,
) -> dict[str, Any]:
    source_stats = validate_object_glb(source_glb)
    if max_faces <= 0 or source_stats["faces"] <= max_faces:
        shutil.copy2(source_glb, object_glb)
        output_stats = validate_object_glb(object_glb)
        return {
            "source_object_glb": str(source_glb),
            "source_vertices": source_stats["vertices"],
            "source_faces": source_stats["faces"],
            "object_vertices": output_stats["vertices"],
            "object_faces": output_stats["faces"],
            "max_faces": max_faces,
            "simplified_for_downstream": False,
        }

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required to simplify high-face MV-SAM3D meshes for align.py. "
            "Install open3d or pass --max_faces 0 to keep the full mesh."
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(source_glb))
    if mesh.is_empty():
        raise ValueError(f"Open3D loaded an empty MV-SAM3D mesh: {source_glb}")
    simplified = mesh.simplify_quadric_decimation(max_faces)
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_non_manifold_edges()
    simplified.compute_vertex_normals()

    import numpy as np
    import trimesh

    vertex_colors = None
    if simplified.has_vertex_colors():
        vertex_colors = (np.asarray(simplified.vertex_colors) * 255.0).clip(0, 255).astype(
            np.uint8
        )
    trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles, dtype=np.int64),
        vertex_colors=vertex_colors,
        process=False,
    ).export(object_glb)

    output_stats = validate_object_glb(object_glb)
    return {
        "source_object_glb": str(source_glb),
        "source_vertices": source_stats["vertices"],
        "source_faces": source_stats["faces"],
        "object_vertices": output_stats["vertices"],
        "object_faces": output_stats["faces"],
        "max_faces": max_faces,
        "simplified_for_downstream": True,
    }


def normalize_outputs(
    mvsam3d_output_dir: Path,
    shape_dir: Path,
    work_dir: Path,
    force: bool,
    max_faces: int,
) -> dict[str, Any]:
    shape_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = work_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    debug_output_dir = outputs_dir / mvsam3d_output_dir.name
    copytree_replace(mvsam3d_output_dir, debug_output_dir, force=force)

    result_glb = mvsam3d_output_dir / "result.glb"
    if not result_glb.exists():
        raise FileNotFoundError(
            f"Canonical MV-SAM3D object GLB not found: {result_glb}. "
            "Refusing to substitute another GLB because shape/object.glb must remain "
            "the single reconstructed object mesh expected by align.py."
        )

    object_glb = shape_dir / "object.glb"
    mesh_stats = write_downstream_object_glb(result_glb, object_glb, max_faces=max_faces)

    object_ply: Path | None = None
    source_ply = mvsam3d_output_dir / "result.ply"
    if source_ply.exists():
        object_ply = shape_dir / "object.ply"
        shutil.copy2(source_ply, object_ply)

    visualization_mp4: Path | None = None
    mp4_candidates = sorted(mvsam3d_output_dir.rglob("*.mp4"))
    if mp4_candidates:
        visualization_mp4 = shape_dir / "visualization.mp4"
        shutil.copy2(mp4_candidates[0], visualization_mp4)

    return {
        "mvsam3d_output_dir": str(mvsam3d_output_dir),
        "debug_output_dir": str(debug_output_dir),
        "object_glb": str(object_glb),
        "object_ply": str(object_ply) if object_ply else None,
        "visualization_mp4": str(visualization_mp4) if visualization_mp4 else None,
        "mesh_stats": mesh_stats,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a FuturePhysTwin shape prior with external MV-SAM3D."
    )
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument("--mvsam3d_root", default=None)
    parser.add_argument(
        "--mvsam3d_python",
        default=None,
        help="Python command for MV-SAM3D. Defaults to env MVSAM3D_PYTHON or the current Python executable.",
    )
    parser.add_argument("--view_indices", default=None)
    parser.add_argument(
        "--input_preprocess_backend",
        choices=["legacy_upscale", "raw"],
        default="legacy_upscale",
        help=(
            "MV-SAM3D scene input preprocessing. legacy_upscale runs the old "
            "mask-crop image_upscale.py + segment_util_image.py path for each view; "
            "raw copies the original frame-0 RGB and masks."
        ),
    )
    parser.add_argument(
        "--preprocess_python",
        default=None,
        help=(
            "Python command for image_upscale.py/segment_util_image.py. Defaults to "
            "env MVSAM3D_PREPROCESS_PYTHON or the current Python executable."
        ),
    )
    parser.add_argument("--run_da3", action="store_true")
    parser.add_argument("--skip_da3_if_exists", action="store_true")
    parser.add_argument(
        "--da3_model_path",
        default=None,
        help=(
            "Optional DA3 model path forwarded to MV-SAM3D scripts/run_da3.py. "
            "Useful when the local Hugging Face cache stores weights under a snapshot directory."
        ),
    )
    parser.add_argument("--merge_da3_glb", action="store_true")
    parser.add_argument("--run_pose_optimization", action="store_true")
    parser.add_argument(
        "--max_faces",
        type=int,
        default=50000,
        help=(
            "Maximum faces for downstream shape/object.glb. The full MV-SAM3D "
            "result.glb remains under shape/mvsam3d/outputs/. Use 0 to disable simplification."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    base_path = Path(args.base_path).expanduser().resolve()
    case_dir = base_path / args.case_name
    shape_dir = case_dir / "shape"
    work_dir = shape_dir / "mvsam3d"
    logs_dir = work_dir / "logs"

    source_overrides: dict[str, dict[str, Path]] | None = None
    preprocess_records: list[dict[str, str]] = []
    if args.input_preprocess_backend == "legacy_upscale":
        source_overrides, preprocess_records = prepare_legacy_upscaled_inputs(
            base_path=base_path,
            case_name=args.case_name,
            category=args.category,
            controller_name=args.controller_name,
            view_indices=args.view_indices,
            work_dir=work_dir,
            logs_dir=logs_dir,
            preprocess_python=args.preprocess_python,
            force=args.force,
            dry_run=args.dry_run,
        )

    manifest = prepare_mvsam3d_scene(
        base_path=base_path,
        case_name=args.case_name,
        category=args.category,
        controller_name=args.controller_name,
        view_indices=args.view_indices,
        output_dir=work_dir,
        source_overrides=source_overrides,
        input_preprocess_backend=args.input_preprocess_backend,
        dry_run=args.dry_run,
        force=args.force,
    )

    object_name = str(manifest["object_name"])
    scene_dir = Path(str(manifest["scene_dir"]))
    image_names = ",".join(str(view["view_index"]) for view in manifest["views"])

    mvsam3d_root = resolve_mvsam3d_root(args.mvsam3d_root, dry_run=args.dry_run)
    python_prefix = command_prefix(args.mvsam3d_python or os.environ.get("MVSAM3D_PYTHON"))
    root_for_commands = mvsam3d_root or Path("${MVSAM3D_ROOT}")

    run_da3_script = root_for_commands / "scripts" / "run_da3.py"
    inference_script = root_for_commands / "run_inference_weighted.py"
    require_file(run_da3_script, "MV-SAM3D DA3 runner", dry_run=args.dry_run)
    require_file(inference_script, "MV-SAM3D weighted inference script", dry_run=args.dry_run)

    da3_dir = work_dir / "da3" / args.input_preprocess_backend
    da3_output = da3_dir / "da3_output.npz"
    da3_should_run = args.force or args.run_da3 or not da3_output.exists()
    if da3_output.exists() and args.skip_da3_if_exists and not args.force:
        da3_should_run = False

    command_records: list[dict[str, str]] = []
    if da3_should_run:
        da3_command = python_prefix + [
            "scripts/run_da3.py",
            "--image_dir",
            str(scene_dir / "images"),
            "--output_dir",
            str(da3_dir),
        ]
        if args.da3_model_path:
            da3_command += ["--model_path", args.da3_model_path]
        command_records.append(
            run_logged(
                da3_command,
                cwd=root_for_commands,
                logs_dir=logs_dir,
                log_name="da3",
                dry_run=args.dry_run,
            )
        )
    else:
        command_records.append(
            {
                "command": "DA3 skipped because da3_output.npz already exists",
                "cwd": str(root_for_commands),
                "stdout_log": "",
                "stderr_log": "",
                "status": "skipped",
            }
        )

    inference_command = python_prefix + [
        "run_inference_weighted.py",
        "--input_path",
        str(scene_dir),
        "--mask_prompt",
        object_name,
        "--image_names",
        image_names,
        "--da3_output",
        str(da3_output),
    ]
    if args.merge_da3_glb:
        inference_command.append("--merge_da3_glb")
    if args.run_pose_optimization:
        inference_command.append("--run_pose_optimization")

    started_at = time.time()
    command_records.append(
        run_logged(
            inference_command,
            cwd=root_for_commands,
            logs_dir=logs_dir,
            log_name="mvsam3d_inference",
            dry_run=args.dry_run,
        )
    )

    result: dict[str, Any] = {
        "case_name": args.case_name,
        "category": args.category,
        "controller_name": args.controller_name,
        "object_name": object_name,
        "input_preprocess_backend": args.input_preprocess_backend,
        "scene_dir": str(scene_dir),
        "da3_output": str(da3_output),
        "preprocess_commands": preprocess_records,
        "commands": command_records,
        "dry_run": args.dry_run,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not args.dry_run:
        output_dir = discover_mvsam3d_output_dir(
            mvsam3d_root=mvsam3d_root if mvsam3d_root else root_for_commands,
            scene_name=scene_dir.name,
            object_name=object_name,
            started_at=started_at,
            log_paths=[
                logs_dir / "mvsam3d_inference.stdout.log",
                logs_dir / "mvsam3d_inference.stderr.log",
            ],
        )
        result.update(
            normalize_outputs(
                output_dir,
                shape_dir,
                work_dir,
                force=args.force,
                max_faces=args.max_faces,
            )
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        with (work_dir / "result_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
