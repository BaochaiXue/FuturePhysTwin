from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FRAME_INDEX = "0"


def natural_key(value: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value, value)


def parse_view_indices(view_indices: str | None) -> list[str] | None:
    if view_indices is None or view_indices.strip() == "":
        return None
    parsed = [token.strip() for token in view_indices.split(",") if token.strip()]
    return parsed or None


def sanitize_mask_prompt(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return sanitized or "object"


def normalize_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("label", "name", "class", "category"):
            if key in value:
                return str(value[key]).strip()
    return str(value).strip()


def discover_views(case_dir: Path, requested: list[str] | None) -> list[str]:
    color_dir = case_dir / "color"
    if not color_dir.exists():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")

    if requested is not None:
        missing = [
            view for view in requested if not (color_dir / view / f"{FRAME_INDEX}.png").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Requested view(s) missing frame-0 RGB image: "
                + ", ".join(str(color_dir / view / f"{FRAME_INDEX}.png") for view in missing)
            )
        return requested

    views = [
        path.name
        for path in color_dir.iterdir()
        if path.is_dir() and (path / f"{FRAME_INDEX}.png").exists()
    ]
    views = sorted(views, key=natural_key)
    if not views:
        raise FileNotFoundError(
            f"No frame-0 RGB views found under {color_dir}/<view>/{FRAME_INDEX}.png"
        )
    return views


def load_mask_info(mask_info_path: Path) -> dict[str, Any]:
    if not mask_info_path.exists():
        raise FileNotFoundError(f"Mask info file not found: {mask_info_path}")
    with mask_info_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Mask info must be a JSON object: {mask_info_path}")
    return data


def select_single_object(mask_info_path: Path, controller_name: str) -> tuple[str, str]:
    controller_normalized = controller_name.strip().casefold()
    data = load_mask_info(mask_info_path)
    candidates: list[tuple[str, str]] = []
    for key, value in data.items():
        label = normalize_label(value)
        if label.casefold() != controller_normalized:
            candidates.append((str(key), label))

    if not candidates:
        raise ValueError(
            f"No non-controller object found in {mask_info_path}; "
            f"controller_name={controller_name!r}, labels={list(data.values())!r}"
        )
    if len(candidates) > 1:
        labels = [f"{idx}:{label}" for idx, label in candidates]
        raise ValueError(
            f"More than one non-controller object found in {mask_info_path}: {labels}. "
            "MV-SAM3D v1 integration preserves the single-object downstream contract; "
            "remove extra objects or add explicit multi-object support in a follow-up."
        )
    return candidates[0]


def load_foreground_mask(mask_path: Path) -> np.ndarray:
    import numpy as np
    from PIL import Image

    if not mask_path.exists():
        raise FileNotFoundError(f"Object mask file not found: {mask_path}")
    mask_image = Image.open(mask_path)
    mask_array = np.array(mask_image)

    if mask_image.mode == "RGBA" and mask_array.ndim == 3 and mask_array.shape[2] >= 4:
        foreground = mask_array[..., 3] > 0
    elif mask_array.ndim == 2:
        foreground = mask_array > 0
    elif mask_array.ndim == 3:
        foreground = np.any(mask_array[..., :3] > 0, axis=2)
    else:
        raise ValueError(f"Unsupported mask shape for {mask_path}: {mask_array.shape}")

    if not np.any(foreground):
        raise ValueError(f"Object mask has empty foreground alpha/mask: {mask_path}")
    return foreground


def write_scene_image(
    source_image_path: Path, output_image_path: Path, dry_run: bool
) -> tuple[int, int]:
    if dry_run and not source_image_path.exists():
        return 0, 0

    from PIL import Image

    image = Image.open(source_image_path).convert("RGB")
    width, height = image.size
    if not dry_run:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_image_path)
    return width, height


def write_rgba_mask(
    source_image_path: Path,
    source_mask_path: Path,
    output_mask_path: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    if dry_run and (not source_image_path.exists() or not source_mask_path.exists()):
        return 0, 0, 0

    import numpy as np
    from PIL import Image

    rgb_image = Image.open(source_image_path).convert("RGB")
    rgb = np.array(rgb_image)
    foreground = load_foreground_mask(source_mask_path)
    if foreground.shape != rgb.shape[:2]:
        raise ValueError(
            f"Mask/image shape mismatch for {source_mask_path}: "
            f"mask={foreground.shape}, image={rgb.shape[:2]}"
        )

    alpha = foreground.astype(np.uint8) * 255
    masked_rgb = rgb.copy()
    masked_rgb[~foreground] = 0
    rgba = np.dstack([masked_rgb, alpha])

    if not dry_run:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(output_mask_path)

    return int(foreground.sum()), int(rgb_image.size[0]), int(rgb_image.size[1])


def prepare_mvsam3d_scene(
    base_path: str | Path,
    case_name: str,
    category: str,
    controller_name: str = "hand",
    view_indices: str | list[str] | None = None,
    output_dir: str | Path | None = None,
    source_overrides: dict[str, dict[str, str | Path]] | None = None,
    input_preprocess_backend: str = "raw",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    base = Path(base_path).expanduser().resolve()
    case_dir = base / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    requested_views = (
        parse_view_indices(view_indices)
        if isinstance(view_indices, str) or view_indices is None
        else list(view_indices)
    )
    views = discover_views(case_dir, requested_views)

    shape_work_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else case_dir / "shape" / "mvsam3d"
    )
    scene_dir = shape_work_dir / "input"
    images_dir = scene_dir / "images"

    object_name = sanitize_mask_prompt(category)
    object_dir = scene_dir / object_name

    if force and not dry_run and scene_dir.exists():
        shutil.rmtree(scene_dir)

    manifest: dict[str, Any] = {
        "case_name": case_name,
        "category": category,
        "controller_name": controller_name,
        "object_name": object_name,
        "input_preprocess_backend": input_preprocess_backend,
        "frame_index": int(FRAME_INDEX),
        "view_indices": views,
        "scene_dir": str(scene_dir),
        "images_dir": str(images_dir),
        "object_dir": str(object_dir),
        "manifest_path": str(shape_work_dir / "manifest.json"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "warnings": [],
        "skipped_views": [],
        "views": [],
    }

    labels_seen: set[str] = set()
    for view in views:
        source_image_path = case_dir / "color" / view / f"{FRAME_INDEX}.png"
        mask_info_path = case_dir / "mask" / f"mask_info_{view}.json"
        object_idx, object_label = select_single_object(mask_info_path, controller_name)
        labels_seen.add(object_label)

        source_mask_path = case_dir / "mask" / view / object_idx / f"{FRAME_INDEX}.png"
        if source_overrides and view in source_overrides:
            override = source_overrides[view]
            source_image_path = Path(override["image_path"]).expanduser().resolve()
            source_mask_path = Path(override["mask_path"]).expanduser().resolve()
        generated_image_path = images_dir / f"{view}.png"
        generated_mask_path = object_dir / f"{view}.png"

        image_width, image_height = write_scene_image(
            source_image_path, generated_image_path, dry_run=dry_run
        )
        foreground_pixels, mask_width, mask_height = write_rgba_mask(
            source_image_path, source_mask_path, generated_mask_path, dry_run=dry_run
        )
        if (image_width, image_height) != (mask_width, mask_height):
            raise ValueError(
                f"Generated size mismatch for view {view}: "
                f"image={(image_width, image_height)}, mask={(mask_width, mask_height)}"
            )

        manifest["views"].append(
            {
                "view_index": view,
                "object_index": object_idx,
                "object_label": object_label,
                "source_image_path": str(source_image_path),
                "source_mask_info_path": str(mask_info_path),
                "source_mask_path": str(source_mask_path),
                "generated_image_path": str(generated_image_path),
                "generated_mask_path": str(generated_mask_path),
                "image_size": [image_width, image_height],
                "foreground_pixels": foreground_pixels,
                "source_overridden": bool(source_overrides and view in source_overrides),
            }
        )

    if len(labels_seen) > 1:
        manifest["warnings"].append(
            "Non-controller object labels differ across views: "
            + ", ".join(sorted(labels_seen))
        )

    if not dry_run:
        shape_work_dir.mkdir(parents=True, exist_ok=True)
        with (shape_work_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a FuturePhysTwin case as an MV-SAM3D multi-view scene."
    )
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--controller_name", default="hand")
    parser.add_argument(
        "--view_indices",
        default=None,
        help="Optional comma-separated camera/view indices. Defaults to all color/<view>/0.png folders.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="MV-SAM3D work directory. The scene is written to <output_dir>/input.",
    )
    parser.add_argument(
        "--input_preprocess_backend",
        choices=["raw"],
        default="raw",
        help="Standalone adapter mode only copies raw frame-0 RGB/masks.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    manifest = prepare_mvsam3d_scene(
        base_path=args.base_path,
        case_name=args.case_name,
        category=args.category,
        controller_name=args.controller_name,
        view_indices=args.view_indices,
        output_dir=args.output_dir,
        input_preprocess_backend=args.input_preprocess_backend,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
