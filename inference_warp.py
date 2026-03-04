from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
import re
from pathlib import Path

from export_topology import dump_topology_from_trainer


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    topology_path = (Path(base_dir) / "topology.npz").resolve()
    dump_topology_from_trainer(
        trainer=trainer,
        case_name=case_name,
        out_path=topology_path,
        overwrite=True,
    )
    logger.info(f"Topology sidecar refreshed: {topology_path}")
    expected_springs = trainer.simulator.n_springs
    expected_edges = trainer.init_springs.detach().cpu()
    expected_rest = trainer.init_rest_lengths.detach().cpu()

    def _extract_epoch(path: str):
        basename = os.path.basename(path)
        match = re.fullmatch(r"best_(\d+)\.pth", basename)
        return int(match.group(1)) if match else None

    candidate_paths = sorted(glob.glob(f"{base_dir}/train/best_*.pth"))
    assert candidate_paths, f"No checkpoint found under {base_dir}/train"

    matching_models = []
    invalid_name_count = 0
    invalid_checkpoint_count = 0
    for path in candidate_paths:
        epoch = _extract_epoch(path)
        if epoch is None:
            invalid_name_count += 1
            logger.warning(f"Skip {path}: filename does not match best_<epoch>.pth")
            continue

        try:
            checkpoint = torch.load(path, map_location=cfg.device)
        except Exception as exc:
            invalid_checkpoint_count += 1
            logger.warning(f"Skip {path}: failed to load checkpoint ({exc})")
            continue

        spring_len = len(checkpoint.get("spring_Y", []))
        if spring_len == expected_springs:
            if "spring_edges" in checkpoint and "spring_rest_lengths" in checkpoint:
                ck_edges = checkpoint["spring_edges"].detach().cpu()
                ck_rest = checkpoint["spring_rest_lengths"].detach().cpu()

                edges_ok = (
                    ck_edges.shape == expected_edges.shape
                    and torch.equal(ck_edges.to(dtype=expected_edges.dtype), expected_edges)
                )
                rest_ok = (
                    ck_rest.shape == expected_rest.shape
                    and torch.allclose(
                        ck_rest.to(dtype=torch.float32),
                        expected_rest.to(dtype=torch.float32),
                        atol=1e-8,
                        rtol=0.0,
                    )
                )
                if not (edges_ok and rest_ok):
                    logger.warning(
                        f"Skip {path}: topology mismatch with current case initialization"
                    )
                    continue
            else:
                logger.warning(
                    f"Checkpoint {path} missing topology fields; "
                    "falling back to spring-count compatibility only"
                )

            mtime = os.path.getmtime(path)
            matching_models.append((epoch, mtime, path))
        else:
            logger.warning(
                f"Skip {path}: checkpoint has {spring_len} springs, expected {expected_springs}"
            )

    assert (
        matching_models
    ), (
        "No checkpoint matches current topology. "
        f"candidates={len(candidate_paths)}, invalid_name={invalid_name_count}, "
        f"invalid_checkpoint={invalid_checkpoint_count}. "
        "Check experiments directory or regenerate models."
    )
    # Pick the checkpoint with the highest epoch and most recent mtime.
    matching_models.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_epoch, best_mtime, best_model_path = matching_models[0]
    logger.info(
        f"Select checkpoint: {best_model_path} (epoch={best_epoch}, mtime={best_mtime})"
    )
    trainer.test(best_model_path)
