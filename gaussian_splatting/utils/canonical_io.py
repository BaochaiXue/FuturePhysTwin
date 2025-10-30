from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from gaussian_splatting.scene.gaussian_model import GaussianModel

CANONICAL_NPZ_FIELDS = (
    "xyz",
    "scaling",
    "rotation",
    "features_dc",
    "features_rest",
    "opacity",
    "active_sh_degree",
    "max_sh_degree",
)


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_payload(payload: Mapping[str, Any]) -> None:
    missing = [field for field in CANONICAL_NPZ_FIELDS if field not in payload]
    if missing:
        raise KeyError(
            f"Canonical NPZ is missing required fields: {', '.join(sorted(missing))}"
        )


def load_canonical_npz(
    path: Path, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load canonical Gaussian parameters from ``path`` and return torch tensors
    located on ``device`` (defaults to CUDA when available).
    """

    if not path.exists():
        raise FileNotFoundError(f"Canonical parameters not found at {path}")
    data = np.load(path)
    _validate_payload(data)

    device = _resolve_device(device)
    tensors: Dict[str, Any] = {}
    float_fields = {
        "xyz",
        "scaling",
        "rotation",
        "features_dc",
        "features_rest",
        "opacity",
    }
    for key in float_fields:
        tensors[key] = torch.from_numpy(data[key]).float().to(device)
    tensors["active_sh_degree"] = int(np.array(data["active_sh_degree"]).reshape(-1)[0])
    tensors["max_sh_degree"] = int(np.array(data["max_sh_degree"]).reshape(-1)[0])
    return tensors


def dump_gaussian_npz(gaussians: "GaussianModel", output_path: Path) -> None:
    """
    Serialise the full Gaussian state to ``output_path`` using the canonical NPZ layout.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        payload = {
            "xyz": gaussians.get_xyz.detach().cpu().numpy(),
            "scaling": gaussians.get_scaling.detach().cpu().numpy(),
            "rotation": gaussians.get_rotation.detach().cpu().numpy(),
            "features_dc": gaussians.get_features_dc.detach().cpu().numpy(),
            "features_rest": gaussians.get_features_rest.detach().cpu().numpy(),
            "opacity": gaussians.get_opacity.detach().cpu().numpy(),
            "active_sh_degree": np.array([gaussians.active_sh_degree], dtype=np.int32),
            "max_sh_degree": np.array([gaussians.max_sh_degree], dtype=np.int32),
        }
    np.savez(output_path, **payload)


def resolve_canonical_npz(model_dir: Path) -> Path:
    """
    Prefer the colour-refined checkpoint when it exists, otherwise fall back to
    the canonical stage output.
    """

    colour_path = model_dir / "color_refine" / "canonical_gaussians_color.npz"
    canonical_path = model_dir / "canonical_gaussians.npz"
    if colour_path.exists():
        return colour_path
    if canonical_path.exists():
        return canonical_path
    raise FileNotFoundError(
        f"Neither {colour_path} nor {canonical_path} exists; cannot resolve canonical NPZ."
    )


__all__ = [
    "CANONICAL_NPZ_FIELDS",
    "dump_gaussian_npz",
    "load_canonical_npz",
    "resolve_canonical_npz",
]
