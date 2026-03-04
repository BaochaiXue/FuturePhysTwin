#!/usr/bin/env python3
"""Export PhysTwin topology as a standalone sidecar file.

Why:
- Keep topology export decoupled from checkpoint serialization.
- Allow forward-compatible workflows where `best_*.pth` does not carry topology.
- Let inference call the same exporter, while also supporting standalone CLI usage.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger


ROOT = Path(__file__).resolve().parent


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_case_runtime(base_path: Path, case_name: str) -> Path:
    """Prepare cfg for a case and return the experiment directory."""
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml(str(ROOT / "configs" / "cloth.yaml"))
    else:
        cfg.load_from_yaml(str(ROOT / "configs" / "real.yaml"))

    exp_dir = (ROOT / "experiments" / case_name).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    optimal_path = (ROOT / "experiments_optimization" / case_name / "optimal_params.pkl").resolve()
    if not optimal_path.exists():
        raise FileNotFoundError(f"{case_name}: Optimal parameters not found: {optimal_path}")
    with optimal_path.open("rb") as handle:
        optimal_params = pickle.load(handle)
    cfg.set_optimal_params(optimal_params)

    case_dir = (base_path / case_name).resolve()
    with (case_dir / "calibrate.pkl").open("rb") as handle:
        c2ws = pickle.load(handle)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with (case_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = str(case_dir / "color")
    return exp_dir


def dump_topology_from_trainer(
    trainer: InvPhyTrainerWarp,
    case_name: str,
    out_path: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Serialize topology tensors from an initialized trainer."""
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing topology file: {out_path}")

    edges = trainer.init_springs.detach().cpu().numpy().astype(np.int32, copy=False)
    rest = trainer.init_rest_lengths.detach().cpu().numpy().astype(np.float32, copy=False)
    vertices = trainer.init_vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    masses = trainer.init_masses.detach().cpu().numpy().astype(np.float32, copy=False)
    num_object_springs = int(trainer.num_object_springs)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Invalid spring_edges shape: {edges.shape}")
    if rest.ndim != 1 or rest.shape[0] != edges.shape[0]:
        raise ValueError(
            f"Invalid spring_rest_lengths shape: {rest.shape}, expected ({edges.shape[0]},)"
        )
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid init_vertices shape: {vertices.shape}")
    if masses.ndim != 1 or masses.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"Invalid init_masses shape: {masses.shape}, expected ({vertices.shape[0]},)"
        )

    np.savez_compressed(
        out_path,
        topology_version=np.asarray(1, dtype=np.int32),
        case_name=np.asarray(case_name),
        created_at_utc=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        spring_edges=edges,
        spring_rest_lengths=rest,
        init_vertices=vertices,
        init_masses=masses,
        num_object_springs=np.asarray(num_object_springs, dtype=np.int32),
    )

    summary = {
        "case_name": case_name,
        "topology_file": str(out_path),
        "num_vertices": int(vertices.shape[0]),
        "num_springs": int(edges.shape[0]),
        "num_object_springs": num_object_springs,
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(f"Exported topology sidecar: {out_path}")
    print(json.dumps(summary, indent=2))
    return out_path


def export_topology_bundle(
    *,
    base_path: str | Path,
    case_name: str,
    out_path: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Standalone entry for exporting topology without touching training code."""
    set_all_seeds(42)
    base_path = Path(base_path).resolve()
    exp_dir = _load_case_runtime(base_path=base_path, case_name=case_name)
    logger.set_log_file(path=str(exp_dir), name="topology_export_log")

    trainer = InvPhyTrainerWarp(
        data_path=str((base_path / case_name / "final_data.pkl").resolve()),
        base_dir=str(exp_dir),
        pure_inference_mode=True,
    )
    if out_path is None:
        out_path = exp_dir / "topology.npz"
    return dump_topology_from_trainer(
        trainer=trainer,
        case_name=case_name,
        out_path=Path(out_path),
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PhysTwin topology to a standalone topology.npz sidecar."
    )
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output topology .npz. Default: experiments/<case_name>/topology.npz",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_topology_bundle(
        base_path=args.base_path,
        case_name=args.case_name,
        out_path=args.out,
        overwrite=args.overwrite,
    )
