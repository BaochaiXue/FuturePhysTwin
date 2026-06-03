from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_data_process_sample(tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "data_process_sam3d" / "data_process_sample.py"
    script_dir = module_path.parent
    old_argv = list(sys.argv)
    old_path = list(sys.path)
    sys.argv = [
        str(module_path),
        "--base_path",
        str(tmp_path),
        "--case_name",
        "case",
    ]
    try:
        sys.path.insert(0, str(script_dir))
        spec = importlib.util.spec_from_file_location("data_process_sample_under_test", module_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv
        sys.path = old_path


def test_ground_policy_preserve_keeps_positive_z(tmp_path: Path) -> None:
    module = _load_data_process_sample(tmp_path)
    points = np.asarray([[[0.0, 0.0, 0.12], [0.0, 0.0, -0.03]]], dtype=np.float32)

    result = module.apply_ground_policy(points)

    np.testing.assert_allclose(result, points)


def test_ground_policy_clamp_positive_z_is_explicit_legacy(tmp_path: Path) -> None:
    module = _load_data_process_sample(tmp_path)
    module.ground_policy = "clamp-positive-z"
    module.ground_z = 0.0
    points = np.asarray([[[0.0, 0.0, 0.12], [0.0, 0.0, -0.03]]], dtype=np.float32)

    result = module.apply_ground_policy(points)

    expected = np.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -0.03]]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
