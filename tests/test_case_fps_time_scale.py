from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_config_module():
    package_name = "_case_fps_config_test"
    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = [str(ROOT / "qqtt")]
    utils_pkg = types.ModuleType(f"{package_name}.utils")
    utils_pkg.__path__ = [str(ROOT / "qqtt" / "utils")]
    sys.modules[package_name] = root_pkg
    sys.modules[f"{package_name}.utils"] = utils_pkg

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.utils.config",
        ROOT / "qqtt" / "utils" / "config.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cfg = _load_config_module().cfg


@pytest.fixture(autouse=True)
def restore_cfg_timing():
    original = {
        "FPS": cfg.FPS,
        "dt": cfg.dt,
        "num_substeps": cfg.num_substeps,
    }
    yield
    cfg.FPS = original["FPS"]
    cfg.dt = original["dt"]
    cfg.num_substeps = original["num_substeps"]


def test_case_fps_metadata_sets_frame_horizon_to_observation_interval() -> None:
    cfg.FPS = 30
    cfg.dt = 5e-5
    cfg.num_substeps = 667

    cfg.apply_case_timing_from_metadata({"fps": 5})

    assert cfg.FPS == pytest.approx(5.0)
    assert cfg.num_substeps == 4000
    assert cfg.dt == pytest.approx(5e-5)
    assert cfg.dt * cfg.num_substeps == pytest.approx(0.2)


def test_missing_case_fps_uses_configured_fps_with_stable_base_dt() -> None:
    cfg.FPS = 30
    cfg.dt = 5e-5
    cfg.num_substeps = 667

    cfg.apply_case_timing_from_metadata({})

    assert cfg.FPS == pytest.approx(30.0)
    assert cfg.num_substeps == 667
    assert cfg.dt <= 5e-5
    assert cfg.dt * cfg.num_substeps == pytest.approx(1.0 / 30.0)


@pytest.mark.parametrize("fps", [0, -5])
def test_invalid_case_fps_fails_early(fps: int) -> None:
    cfg.FPS = 30
    cfg.dt = 5e-5
    cfg.num_substeps = 667

    with pytest.raises(ValueError, match="metadata fps must be positive"):
        cfg.apply_case_timing_from_metadata({"fps": fps})


@pytest.mark.parametrize(
    "relative_path",
    ["optimize_cma.py", "train_warp.py", "inference_warp.py"],
)
def test_real_case_entrypoints_apply_case_timing_from_metadata(relative_path: str) -> None:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_case_timing_from_metadata"
    ]

    assert len(calls) == 1
