"""Shared case filtering helpers based on data_config.csv.

This module defines a consistent allowlist policy used by pipeline stages:
1) input case directory must exist under the input base path;
2) case must appear in data_config.csv column 1.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def _is_blank_row(row: list[str]) -> bool:
    return all(not cell.strip() for cell in row)


def load_config_cases(config_path: Path) -> set[str]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {config_path}")

    cases: set[str] = set()
    duplicates: set[str] = set()
    with config_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue
            if _is_blank_row(row):
                continue
            case_name = row[0].strip() if len(row) >= 1 else ""
            if not case_name:
                raise ValueError(
                    f"Malformed config row {row_index} in {config_path}: missing case name"
                )
            if case_name in cases:
                duplicates.add(case_name)
            cases.add(case_name)

    if not cases:
        raise ValueError(f"No valid cases found in config: {config_path}")

    for case_name in sorted(duplicates):
        print(
            f"[CaseFilter][Warning][Shared] Duplicate case '{case_name}' found in {config_path}; using exact-name allowlist semantics."
        )
    return cases


def load_input_cases(input_base_path: Path) -> set[str]:
    input_base_path = Path(input_base_path)
    if not input_base_path.exists():
        raise FileNotFoundError(f"Input base path not found: {input_base_path}")
    if not input_base_path.is_dir():
        raise NotADirectoryError(f"Input base path is not a directory: {input_base_path}")
    return {path.name for path in input_base_path.iterdir() if path.is_dir()}


def compute_allowed_cases(input_base_path: Path, config_path: Path) -> set[str]:
    input_cases = load_input_cases(input_base_path)
    config_cases = load_config_cases(config_path)
    return input_cases & config_cases


def warn_input_cases_missing_in_config(
    input_cases: set[str],
    config_cases: set[str],
    stage_name: str,
    input_base_path: Path,
    config_path: Path,
) -> None:
    missing = sorted(input_cases - config_cases)
    for case_name in missing:
        print(
            "[CaseFilter][Warning][{stage}] Input case '{case}' exists in {input_base} "
            "but is missing in {config}; skipping.".format(
                stage=stage_name,
                case=case_name,
                input_base=Path(input_base_path),
                config=Path(config_path),
            )
        )


def filter_candidates(
    candidates: list[str],
    allowed_cases: set[str],
    stage_name: str,
    source_label: str,
) -> list[str]:
    filtered: list[str] = []
    for case_name in candidates:
        if case_name in allowed_cases:
            filtered.append(case_name)
            continue
        print(
            "[CaseFilter][Info][{stage}] Skip non-allowed case '{case}' discovered in {source}.".format(
                stage=stage_name,
                case=case_name,
                source=source_label,
            )
        )
    return filtered


def resolve_path_from_root(root: Path, candidate: Path | str) -> Path:
    candidate_path = Path(candidate)
    return candidate_path if candidate_path.is_absolute() else (root / candidate_path).resolve()


def load_config_rows(config_path: Path) -> list[list[str]]:
    """Load non-blank rows with basic shape validation.

    Returns rows as raw string lists to let callers interpret extra columns.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    rows: list[list[str]] = []
    with config_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row_index, row in enumerate(reader, start=1):
            if not row or _is_blank_row(row):
                continue
            if len(row) < 1 or not row[0].strip():
                raise ValueError(
                    f"Malformed config row {row_index} in {config_path}: missing case name"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid rows found in config: {config_path}")
    return rows

