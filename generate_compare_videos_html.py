#!/usr/bin/env python3
"""
Build an HTML page that displays side-by-side video comparisons.

For every case under ``gaussian_output_dynamic_white/<case>/<view>.mp4`` the script
pairs the corresponding video inside ``../PhysTwin/gaussian_output_dynamic_white``.
Each case section shows three camera views (0, 1, 2) with our output on the left and
the reference output on the right.
"""
from __future__ import annotations

import argparse
import html
import logging
import os
from pathlib import Path
from typing import Iterable, List


SCRIPT_ROOT = Path(__file__).resolve().parent
OURS_ROOT = SCRIPT_ROOT / "gaussian_output_dynamic_white"
REFERENCE_ROOT = SCRIPT_ROOT / "../PhysTwin/gaussian_output_dynamic_white"
OUTPUT_DIR = SCRIPT_ROOT / "compare_visualization"
DEFAULT_OUTPUT_NAME = "comparison_videos.html"
CAMERA_VIEWS = ("0", "1", "2")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("compare_videos_html")


def iter_cases(selected_cases: Iterable[str] | None = None) -> List[Path]:
    """Return the case directories located under OURS_ROOT."""
    if not OURS_ROOT.exists():
        raise FileNotFoundError(f"Ours root does not exist: {OURS_ROOT.resolve()}")

    all_cases = sorted(p for p in OURS_ROOT.iterdir() if p.is_dir())
    if selected_cases:
        case_dirs: List[Path] = []
        missing: List[str] = []
        for case in selected_cases:
            candidate = OURS_ROOT / case
            if candidate.is_dir():
                case_dirs.append(candidate)
            else:
                missing.append(case)
        if missing:
            raise FileNotFoundError(
                f"Cases not found under {OURS_ROOT}: {', '.join(sorted(missing))}"
            )
        return case_dirs
    return all_cases


def relative_video_path(html_dir: Path, video_path: Path) -> str:
    """Return the path from the HTML directory to the given video."""
    return os.path.relpath(video_path, start=html_dir)


def collect_view_pairs(case_dir: Path) -> List[tuple[str, Path, Path]]:
    """Gather paths for each camera view, skipping missing counterparts."""
    case_name = case_dir.name
    reference_dir = REFERENCE_ROOT / case_name
    if not reference_dir.exists():
        LOGGER.warning("Skipping %s: reference case missing (%s)", case_name, reference_dir)
        return []

    pairs: List[tuple[str, Path, Path]] = []
    for view in CAMERA_VIEWS:
        ours_video = case_dir / f"{view}.mp4"
        reference_video = reference_dir / f"{view}.mp4"
        if not ours_video.exists():
            LOGGER.warning("Skipping %s view %s: ours video missing (%s)", case_name, view, ours_video)
            continue
        if not reference_video.exists():
            LOGGER.warning(
                "Skipping %s view %s: reference video missing (%s)", case_name, view, reference_video
            )
            continue
        pairs.append((view, ours_video, reference_video))

    if not pairs:
        LOGGER.warning("Case %s skipped: no matched camera views found.", case_name)
    return pairs


def build_case_section(
    case_name: str, pairs: List[tuple[str, Path, Path]], html_dir: Path
) -> str:
    """Create the HTML markup for a single case."""
    escaped_case = html.escape(case_name)
    rows_markup = []
    for view, ours_video, ref_video in pairs:
        ours_rel = relative_video_path(html_dir, ours_video)
        ref_rel = relative_video_path(html_dir, ref_video)
        escaped_view = html.escape(view)
        row_html = f"""
        <div class="view-row">
            <div class="video-column">
                <h4>ours_{escaped_view}</h4>
                <video controls preload="metadata" muted loop autoplay>
                    <source src="{html.escape(ours_rel)}" type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
            </div>
            <div class="video-column">
                <h4>original_{escaped_view}</h4>
                <video controls preload="metadata" muted loop autoplay>
                    <source src="{html.escape(ref_rel)}" type="video/mp4" />
                    Your browser does not support the video tag.
                </video>
            </div>
        </div>
        """
        rows_markup.append(row_html)

    section_html = f"""
    <section class="case-section" id="{escaped_case}">
        <h2>{escaped_case}</h2>
        {''.join(rows_markup)}
    </section>
    """
    return section_html


def build_html_document(case_sections: List[str]) -> str:
    """Wrap the case sections in a full HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Gaussian Output Comparison</title>
    <style>
        body {{
            font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            background-color: #f5f5f5;
            color: #222222;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 24px 48px;
        }}
        h1 {{
            text-align: center;
            margin: 0 0 36px;
            font-size: 2.1rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }}
        .case-section {{
            background: #ffffff;
            border-radius: 14px;
            padding: 24px 28px 28px;
            margin-bottom: 28px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(15, 23, 42, 0.08);
        }}
        .case-section h2 {{
            margin: 0 0 22px;
            font-size: 1.35rem;
            font-weight: 600;
            color: #1f2937;
        }}
        .view-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 12px 18px;
            margin-bottom: 22px;
        }}
        .view-row:last-of-type {{
            margin-bottom: 0;
        }}
        .video-column {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .video-column h4 {{
            margin: 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: #374151;
            letter-spacing: 0.01em;
        }}
        video {{
            width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background: #000;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Gaussian Output vs Original</h1>
        {''.join(case_sections)}
    </div>
</body>
</html>
"""


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML video comparisons.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of cases to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help="Destination HTML file (default: compare_visualization/comparison_videos.html).",
    )
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output

    if output_path.suffix.lower() != ".html":
        raise ValueError(f"Output file must end with .html, got: {output_path}")

    ensure_output_dir(output_path.parent)
    html_dir = output_path.parent

    case_dirs = iter_cases(args.cases)
    sections: List[str] = []
    for case_dir in case_dirs:
        pairs = collect_view_pairs(case_dir)
        if not pairs:
            continue
        section_html = build_case_section(case_dir.name, pairs, html_dir)
        sections.append(section_html)

    if not sections:
        LOGGER.warning("No case sections generated; nothing to write.")
        return

    document = build_html_document(sections)
    output_path.write_text(document, encoding="utf-8")
    LOGGER.info("Saved HTML comparison to %s", output_path)


if __name__ == "__main__":
    main()
