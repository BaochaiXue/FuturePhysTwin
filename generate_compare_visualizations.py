#!/usr/bin/env python3
"""
Generate side-by-side comparison grids between two Gaussian output folders.

For every case under `gaussian_output_dynamic_white/<case>/<view>`, the script locates
the last PNG frame and pairs it with the corresponding frame from
`../PhysTwin/gaussian_output_dynamic_white/<case>/<view>`. The result for a case is
an image containing two columns (ours vs original) and one row per camera view.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont


OURS_ROOT = Path("gaussian_output_dynamic_white")
REFERENCE_ROOT = Path("../PhysTwin/gaussian_output_dynamic_white")
OUTPUT_ROOT = Path("compare_visualization")


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("compare_visualization")


@dataclass(frozen=True)
class ViewPair:
    view_name: str
    ours_path: Path
    reference_path: Path


def iter_cases(selected_cases: Iterable[str] | None = None) -> List[Path]:
    """Return case directories to process."""
    if not OURS_ROOT.exists():
        raise FileNotFoundError(f"Cannot find ours root at: {OURS_ROOT.resolve()}")
    all_cases = sorted(p for p in OURS_ROOT.iterdir() if p.is_dir())
    if selected_cases:
        case_dirs: List[Path] = []
        missing: List[str] = []
        for case in selected_cases:
            path = OURS_ROOT / case
            if path.is_dir():
                case_dirs.append(path)
            else:
                missing.append(case)
        if missing:
            raise FileNotFoundError(f"Cases not found under {OURS_ROOT}: {', '.join(missing)}")
        return case_dirs
    return all_cases


def last_frame_png(view_dir: Path) -> Path | None:
    """Return the last PNG frame inside the given view directory."""
    pngs = sorted(p for p in view_dir.glob("*.png") if p.is_file())
    return pngs[-1] if pngs else None


def collect_view_pairs(case_dir: Path) -> List[ViewPair]:
    """Gather matching pairs of (ours, reference) last frames for every view."""
    case_name = case_dir.name
    reference_case_dir = REFERENCE_ROOT / case_name
    if not reference_case_dir.exists():
        LOGGER.warning("Skipping %s: reference case not found at %s", case_name, reference_case_dir)
        return []

    view_dirs = sorted(p for p in case_dir.iterdir() if p.is_dir())
    pairs: List[ViewPair] = []

    for view_dir in view_dirs:
        view_name = view_dir.name
        ours_frame = last_frame_png(view_dir)
        if ours_frame is None:
            LOGGER.warning("Skipping %s view %s: no PNG frames in %s", case_name, view_name, view_dir)
            continue

        reference_view_dir = reference_case_dir / view_name
        if not reference_view_dir.exists():
            LOGGER.warning(
                "Skipping %s view %s: reference view directory missing (%s)",
                case_name,
                view_name,
                reference_view_dir,
            )
            continue

        reference_frame = last_frame_png(reference_view_dir)
        if reference_frame is None:
            LOGGER.warning(
                "Skipping %s view %s: no PNG frames in reference %s",
                case_name,
                view_name,
                reference_view_dir,
            )
            continue

        pairs.append(ViewPair(view_name=view_name, ours_path=ours_frame, reference_path=reference_frame))

    if not pairs:
        LOGGER.warning("Case %s skipped: no valid view pairs found.", case_name)

    return pairs


def build_comparison_image(case_name: str, pairs: List[ViewPair]) -> Image.Image:
    """Create a composite image with two columns and one row per view."""
    ours_images = [Image.open(p.ours_path).convert("RGB") for p in pairs]
    reference_images = [Image.open(p.reference_path).convert("RGB") for p in pairs]

    # Ensure all images are closed after use by keeping references to close later.
    opened_images = list(chain(ours_images, reference_images))

    try:
        font = ImageFont.load_default()
        text_padding = 6
        column_gap = 24
        row_gap = 24
        margin = 32

        left_width = max(img.width for img in ours_images)
        right_width = max(img.width for img in reference_images)
        ascent, descent = font.getmetrics()
        label_height = ascent + descent
        per_row_header = label_height + text_padding * 2

        # Height for each row is the max of the two images plus header space.
        rows_height = [
            per_row_header + max(ours.height, reference.height)
            for ours, reference in zip(ours_images, reference_images)
        ]
        total_height = margin * 2 + sum(rows_height) + row_gap * (len(pairs) - 1 if pairs else 0)
        total_width = margin * 2 + left_width + right_width + column_gap

        canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        y_offset = margin
        for index, (pair, ours_img, ref_img, row_height) in enumerate(
            zip(pairs, ours_images, reference_images, rows_height)
        ):
            header_y = y_offset
            image_y = y_offset + per_row_header

            left_x = margin + (left_width - ours_img.width) // 2
            right_x = margin + left_width + column_gap + (right_width - ref_img.width) // 2

            draw.text((left_x, header_y), f"ours_{pair.view_name}", fill=(0, 0, 0), font=font)
            draw.text((right_x, header_y), f"original_{pair.view_name}", fill=(0, 0, 0), font=font)

            canvas.paste(ours_img, (left_x, image_y))
            canvas.paste(ref_img, (right_x, image_y))

            y_offset += row_height
            if index < len(pairs) - 1:
                y_offset += row_gap

        return canvas
    finally:
        for img in opened_images:
            img.close()


def ensure_output_directory() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison visuals between two Gaussian outputs.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional list of case names to process. Defaults to all cases under ours root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing comparison images. By default existing outputs are skipped.",
    )
    args = parser.parse_args()

    ensure_output_directory()
    case_dirs = iter_cases(args.cases)

    generated = 0
    skipped = 0

    for case_dir in case_dirs:
        case_name = case_dir.name
        output_path = OUTPUT_ROOT / f"{case_name}_compare.png"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping %s: output already exists (use --overwrite to regenerate).", case_name)
            skipped += 1
            continue

        pairs = collect_view_pairs(case_dir)
        if not pairs:
            skipped += 1
            continue

        comparison_image = build_comparison_image(case_name, pairs)
        comparison_image.save(output_path)
        LOGGER.info("Saved %s", output_path)
        generated += 1

    LOGGER.info("Done. Generated %d image(s); skipped %d case(s).", generated, skipped)


if __name__ == "__main__":
    main()
