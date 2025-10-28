from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
from pathlib import Path

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
    cfg.load_from_yaml("configs/real.yaml")

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
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

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    point_cloud_dir = (
        Path(args.gaussian_path) / case_name / exp_name / "point_cloud"
    )
    if not point_cloud_dir.exists():
        raise FileNotFoundError(
            f"Gaussian point cloud directory not found: {point_cloud_dir}"
        )
    candidate_point_clouds: list[tuple[int, Path]] = []
    for iteration_dir in point_cloud_dir.glob("iteration_*"):
        if not iteration_dir.is_dir():
            continue
        try:
            iteration_id = int(iteration_dir.name.split("_")[-1])
        except ValueError:
            continue
        ply_path = iteration_dir / "point_cloud.ply"
        if ply_path.exists():
            candidate_point_clouds.append((iteration_id, ply_path))
    if not candidate_point_clouds:
        raise FileNotFoundError(
            f"No point_cloud.ply files found under {point_cloud_dir}"
        )
    best_iteration, best_ply_path = max(candidate_point_clouds, key=lambda item: item[0])
    logger.info(
        f"Using Gaussian point cloud from iteration {best_iteration}: {best_ply_path}"
    )
    gaussians_path = str(best_ply_path)

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.visualize_force(
        best_model_path, gaussians_path, args.n_ctrl_parts
    )
