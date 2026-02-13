"""Process raw captures into training-ready assets for allowed cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from case_filter import (
    load_config_cases,
    load_config_rows,
    load_input_cases,
    resolve_path_from_root,
    warn_input_cases_missing_in_config,
)


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_CONFIG_PATH = "./data_config.csv"
ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run process_data.py for cases listed in data_config.csv and present on disk."
    )
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = resolve_path_from_root(ROOT, args.base_path)
    config_path = resolve_path_from_root(ROOT, args.config_path)

    config_rows = load_config_rows(config_path)
    config_cases = load_config_cases(config_path)
    input_cases = load_input_cases(base_path)
    warn_input_cases_missing_in_config(
        input_cases, config_cases, "script_process_data", base_path, config_path
    )
    allowed_cases = input_cases & config_cases

    timer_log = ROOT / "timer.log"
    if timer_log.exists():
        timer_log.unlink()

    for row in config_rows:
        if len(row) < 3:
            raise ValueError(
                f"Malformed config row in {config_path}: expected >=3 columns, got {row}"
            )
        case_name = row[0].strip()
        category = row[1].strip()
        shape_prior = row[2].strip()

        if case_name not in allowed_cases:
            continue

        cmd: list[str] = [
            args.python,
            str(ROOT / "process_data.py"),
            "--base_path",
            str(base_path),
            "--case_name",
            case_name,
            "--category",
            category,
        ]
        if shape_prior.lower() == "true":
            cmd.append("--shape_prior")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

