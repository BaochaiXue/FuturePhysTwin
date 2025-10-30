## 重新规划：先离线生成 LBS 形变结果，再在 Color Stage 直接读取

目标：把 Interactive Playground 的“逐帧 LBS 绑定 + 刚体拟合”流程完整移植到颜色微调阶段，但把所有昂贵的 KNN 和 `interpolate_motions_speedup` 运算在 **离线** 完成。这样可以在 Color Stage 内继续随机采样任意帧，同时让 LBS 版本严格遵循 `motions = x - prev_x` 的增量定义。

---
### Relavant information
仓库里与本方案直接相关的现有能力（代码定位）

Playground 里在线 LBS 片段：首次帧算 relations 与 weights/indices，其后每帧
current_pos, current_rot = interpolate_motions_speedup(...) 并写回 gaussians._xyz/_rotation（示例片段）。

Color 训练循环中已有的 LBS 分支：当 color_only 且 lbs_deformer/lbs_motions 存在时，对 xyz0/rot0 应用 LBS，随后进入 gaussians.deform_ctx(...) 做临时姿态替换再渲染求损失——这是我们要“用离线缓存替换在线求解”的精准挂点。

LBS 工具链（你已具备）：
get_topk_indices / knn_weights_sparse / calc_weights_vals_from_indices / interpolate_motions_speedup，在 gaussian_splatting.dynamic_utils、qqtt/engine/trainer_warp.py 中被调用。

LBSDeformer 与权重构建：在 gaussian_splatting/skinning/lbs.py，但本方案离线化后可不再在 Color 阶段实时调用。

高斯模型的“姿态快照与临时变形上下文”：snapshot_canonical_pose()、get_xyz_canonical()、get_rotation_canonical()、freeze_geometry()、deform_ctx()（Color 阶段要用到 freeze_geometry + deform_ctx）。

Color 训练入口的扩展签名（带 color_only/freeze_geometry/lbs_path/frames_dir 等）：是我们扩展 --lbs_pose_cache 的位置。

### 1. 离线预处理脚本（一次性）

新建一个脚本（例如 `precompute_lbs_pose_cache.py`），其主要逻辑：
0. Delete prepare_lbs_weights.py and create precompute_lbs_pose_cache.py（脚本读 canonical_gaussians.npz 与 inference.pkl，仅首帧做绑定，逐帧用 motions = x[t] - prev_x[t] 调 interpolate_motions_speedup 得到该帧的 (posed_xyz, posed_quat)，落盘为 lbs_pose_cache.pt）。姿态缓存以 t -> {xyz, quat} 组织，Color 阶段 O(1) 查表，不再在线做 KNN/SVD。
1. 加载 `canonical_gaussians.npz`（或最新的颜色 checkpoint）得到 canonical 的 `xyz0`、`quat0`。
2. 读取 `inference.pkl`。按照 Playground 的输出，这里应能拿到 `prev_x` 和 `x`（`torch.Tensor` / `np.ndarray` 形状 `(T, N, 3)`）。如果文件只存 `x` 序列，则通过 `np.roll` 手动构造 `prev_x`，并让首帧满足 `prev_x[0] == x[0]`。
3. 仅在 **首帧** 构建绑定：
   - `relations = get_topk_indices(prev_x[0], K=16)`。
   - `weights, weights_idx = knn_weights_sparse(prev_x[0], xyz0, K=16)`。
   - `weights = calc_weights_vals_from_indices(prev_x[0], xyz0, weights_idx)`；这一套正是 Playground 的做法。
   - 将 `relations/weights_idx/weights` 缓存下来供后续所有帧复用。
4. 对于每个时间步 `t`：
   - `bones_t = prev_x[t]`。
   - `motions_t = x[t] - prev_x[t]`（增量位移）。
   - 调用 `interpolate_motions_speedup`：
     ```python
     posed_xyz, posed_quat, _ = interpolate_motions_speedup(
         bones=bones_t,
         motions=motions_t,
         relations=relations,
         weights=weights,
         weights_indices=weights_idx,
         xyz=xyz0,
         quat=quat0,
     )
     ```
   - 把结果写入缓存字典：`pose_cache[t] = {"xyz": posed_xyz.cpu(), "quat": posed_quat.cpu()}`。
5. 结束后保存一个单一的 `lbs_cache.pt`（或 `npz`）：
   ```python
   torch.save(
       {
           "relations": relations.cpu(),
           "weights_idx": weights_idx.cpu(),
           "weights": weights.cpu(),
           "pose_cache": pose_cache,  # 映射 frame_id -> {xyz, quat}
       },
       output_path,
   )
   ```
   如需节省空间，可改为按帧写 `per_frame_pose/<frame_id>.pt`，但统一打包更便于加载。

> 性能提示：`get_topk_indices / knn_weights_sparse` 只在首帧调用一次；`interpolate_motions_speedup` 仍需跑 T 次。对大场景可以分批或使用 `torch.no_grad()` + `float32` 来控制显存。

---

### 2. Color Stage 改动
0. make the input and ouput must be consistent with precompute_lbs_pose_cache.py
1. 在 `dynamic_fast_color.py` CLI 中新增 `--lbs_pose_cache`（指向步骤 1 的输出）。
2. 加载模型时，如果提供了该路径：
   ```python
   pose_payload = torch.load(args.lbs_pose_cache, map_location="cpu")
   pose_cache = {int(k): (v["xyz"].to("cuda"), v["quat"].to("cuda")) for k, v in pose_payload["pose_cache"].items()}
   ```
3. 在训练循环中，把原来“现算 LBS”那段替换成简单查表：
   ```python
   posed = pose_cache.get(frame_id)
   if posed is not None:
       posed_xyz, posed_rot = posed
       pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
   else:
       pose_ctx = nullcontext()
   ```
   保证 canonical 帧（frame_id=0）也在 cache 中；如果不在，就落回原始姿态。
4. 其他逻辑（SH 优化、曝光、监控）保持不变。不再需要实时的 `LBSDeformer` 或 `lbs_motions`，可以删掉相关成员变量。

---

### 3. 产物和命令行清单

```bash
# 预计算姿态缓存
python precompute_lbs_pose_cache.py \
  --model_dir path/to/model \
  --inference path/to/inference.pkl \
  --output path/to/lbs_pose_cache.pt

# 颜色阶段
python dynamic_fast_color.py \
  --model_path path/to/model \
  --frames_dir path/to/per_frame_gaussian_data \
  --color_only --freeze_geometry \
  --lbs_pose_cache path/to/lbs_pose_cache.pt
```

如需继续运行旧流程，直接省略 `--lbs_pose_cache`。

### psudocode
# tools/precompute_lbs_pose_cache.py
from __future__ import annotations
import argparse, pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

# 来自你仓库
from gaussian_splatting.dynamic_utils import (
    get_topk_indices,
    knn_weights_sparse,
    calc_weights_vals_from_indices,
    interpolate_motions_speedup,
)  # :contentReference[oaicite:9]{index=9}

def load_canonical_npz(path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    npz = np.load(path, allow_pickle=True)
    xyz0  = torch.tensor(npz["xyz"],      dtype=torch.float32, device=device)
    quat0 = torch.tensor(npz["rotation"], dtype=torch.float32, device=device)  # Nx4
    return xyz0, quat0

def main():
    p = argparse.ArgumentParser("Precompute LBS pose cache")
    p.add_argument("--model_dir", type=Path, required=True,
                   help="Directory that contains canonical_gaussians.npz")
    p.add_argument("--inference", type=Path, required=True,
                   help="Path to inference.pkl (ndarray, shape (T, N_bones, 3))")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .pt path for the cache (e.g., lbs_pose_cache.pt)")
    p.add_argument("--K", type=int, default=16, help="KNN for bones/skin")
    p.add_argument("--half", action="store_true", help="save half precision payload to reduce size")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cano_path = args.model_dir / "canonical_gaussians.npz"
    if not cano_path.exists():
        raise FileNotFoundError(f"canonical_gaussians.npz not found at {cano_path}")

    # 1) canonical
    xyz0, quat0 = load_canonical_npz(cano_path, device)

    # 2) load inference (x[t] bones positions)
    with open(args.inference, "rb") as f:
        x_np = pickle.load(f)  # numpy.ndarray, (T, N_bones, 3), float32
    assert isinstance(x_np, np.ndarray) and x_np.ndim == 3 and x_np.shape[2] == 3
    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)  # (T, B, 3)
    T, B, _ = x.shape

    # 构造 prev_x：首帧与自身一致
    prev_x = torch.roll(x, shifts=1, dims=0)
    prev_x[0] = x[0]

    # 3) 首帧构建关系与权重（仅一次）
    bones0 = prev_x[0]                        # (B, 3)
    relations = get_topk_indices(bones0, K=args.K)  # (B, K)  :contentReference[oaicite:10]{index=10}
    # 让高斯绑定到“首帧骨点”
    _, weights_idx = knn_weights_sparse(bones0, xyz0, K=args.K)        # (N, K)
    weights = calc_weights_vals_from_indices(bones0, xyz0, weights_idx) # (N, K)

    # 4) 逐帧 LBS
    pose_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for t in range(T):
            motions_t = x[t] - prev_x[t]  # 增量定义
            posed_xyz, posed_quat, _ = interpolate_motions_speedup(
                bones=prev_x[t],
                motions=motions_t,
                relations=relations,
                weights=weights,
                weights_indices=weights_idx,
                xyz=xyz0,
                quat=quat0,
            )  # 与 Playground 用法一致 :contentReference[oaicite:11]{index=11}

            # 保存在 CPU；可选半精度压缩
            if args.half:
                pose_cache[int(t)] = {
                    "xyz": posed_xyz.detach().cpu().half(),
                    "quat": posed_quat.detach().cpu().half(),
                }
            else:
                pose_cache[int(t)] = {
                    "xyz": posed_xyz.detach().cpu().float(),
                    "quat": posed_quat.detach().cpu().float(),
                }

    # 5) 打包缓存
    payload = {
        "relations": relations.detach().cpu(),          # (B, K)
        "weights_idx": weights_idx.detach().cpu(),      # (N, K)
        "weights": weights.detach().cpu(),              # (N, K)
        "pose_cache": pose_cache,                       # t -> {xyz, quat}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[OK] LBS pose cache saved to {args.output}")

if __name__ == "__main__":
    main()

命令行与产物

离线缓存：
In dynamic_gs_run.py, we may call python precompute_lbs_pose_cache.py in proper location.
python precompute_lbs_pose_cache.py \
  --model_dir path/to/<your_model_dir_with_canonical_npz> \
  --inference path/to/inference.pkl \
  --output path/to/lbs_pose_cache.pt \
  --K 16


Color 阶段（仅颜色）训练：

python dynamic_fast_color.py \
  -s path/to/scene \
  -m path/to/<your_model_dir_with_canonical_npz> \
  --color_only --freeze_geometry \
  --lbs_pose_cache path/to/lbs_pose_cache.pt


不提供 --lbs_pose_cache 时会回退到你原先的在线 LBS 分支（如 --lbs_path 存在）。

4) 与现有代码的完全对齐（关键一致性检查）

motions 增量定义：严格用 motions_t = x[t] - prev_x[t]，首帧 prev_x[0]=x[0]。与你的 Playground 一致。

relations/weights：只在首帧用 get_topk_indices(prev_x[0]) 与 knn_weights_sparse(prev_x[0], xyz0) 建立骨邻接与稀疏蒙皮，再用 calc_weights_vals_from_indices 做归一化（稳定）。

刚体拟合：interpolate_motions_speedup(...) 内部采用 SVD/行列式修正思路（与 Kabsch/Umeyama 一致思想），保证旋转矩阵为真旋转。 也契合几何配准的常见做法。
Wikipedia
+1

Color 训练中的姿态应用：仍通过 gaussians.deform_ctx(posed_xyz, posed_rot) 临时替换姿态后渲染，训练只作用于颜色相关参数（你已有此范式）。

冻结几何：调用 freeze_geometry()，彻底避免几何被更新。
Color Stage（训练端）最小改动

在 Color 训练入口（带 color_only/freeze_geometry/lbs_path/frames_dir 的那个文件）增加一个可选参数 --lbs_pose_cache，如提供则优先使用离线缓存替代在线 LBS 分支；未提供时保持现状（向后兼容）。

2.1 加参数（CLI）

在你的 Color 训练脚本 / 模块（含如下签名的 training(..., *, color_only, freeze_geometry, lbs_path, frames_dir)）里扩展一个 lbs_pose_cache: Optional[Path]。

diff：

- def training(..., *, color_only: bool = False, freeze_geometry: bool = False, lbs_path: Optional[Path] = None, frames_dir: Optional[Path] = None) -> None:
+ def training(..., *, color_only: bool = False, freeze_geometry: bool = False,
+              lbs_path: Optional[Path] = None, frames_dir: Optional[Path] = None,
+              lbs_pose_cache: Optional[Path] = None) -> None:


命令行解析处追加：

 parser.add_argument("--color_only", action="store_true")
 parser.add_argument("--freeze_geometry", action="store_true")
 parser.add_argument("--lbs_path", type=Path, default=None)
 parser.add_argument("--frames_dir", type=Path, default=None)
+parser.add_argument("--lbs_pose_cache", type=Path, default=None,
+                    help="Path to offline LBS cache (.pt) produced by precompute_lbs_pose_cache.py")


你的训练入口已经基于 ModelParams/OptimizationParams/PipelineParams，CLI 增参方式与其他参数一致；这里仅展示关键行。

2.2 加载缓存（初始化阶段）
     lbs_deformer: Optional[LBSDeformer] = None
     lbs_motions: Optional[torch.Tensor] = None
+    pose_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None

+    # 优先使用离线缓存（如提供）
+    if lbs_pose_cache is not None:
+        cache_payload = torch.load(lbs_pose_cache, map_location="cpu")
+        # t -> (xyz, quat) on CUDA
+        pose_cache = {
+            int(t): (itm["xyz"].to("cuda").float(), itm["quat"].to("cuda").float())
+            for t, itm in cache_payload["pose_cache"].items()
+        }
+        print(f"[Color] Loaded offline LBS pose cache with {len(pose_cache)} frames from {lbs_pose_cache}")
+    elif lbs_path is not None:
+        # 兼容：保留旧的在线 LBS 流
+        lbs_payload = load_lbs_data(lbs_path)  # 现有函数
+        lbs_deformer = LBSDeformer(
+            lbs_payload["bones0"], lbs_payload["relations"],
+            lbs_payload["skin_indices"], lbs_payload["skin_weights"]
+        )
+        lbs_motions = lbs_payload["motions"]


load_lbs_data/LBSDeformer 为你已有实现。此处仅在未提供 lbs_pose_cache 才继续走旧路径。

2.3 训练循环中替换姿态（查表 + deform_ctx）

找到你 Color 训练循环里原先的在线 LBS 分支（如下判断分支位置）。

-        if (
-            color_only
-            and lbs_deformer is not None
-            and lbs_motions is not None
-            and 0 <= frame_id < num_motion_frames
-        ):
-            xyz_canonical = gaussians.get_xyz_canonical().to("cuda")
-            rot_canonical = gaussians.get_rotation_canonical().to("cuda")
-            motions_t = lbs_motions[frame_id]
-            posed_xyz, posed_rot = lbs_deformer.deform(
-                xyz_canonical, rot_canonical, motions_t
-            )
-            pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
+        if color_only and pose_cache is not None:
+            posed = pose_cache.get(int(frame_id))
+            if posed is None:
+                raise KeyError(f"Frame {frame_id} not found in lbs_pose_cache. "
+                               f"Please ensure frame indices match.")
+            posed_xyz, posed_rot = posed
+            pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
+        elif (
+            color_only
+            and lbs_deformer is not None
+            and lbs_motions is not None
+            and 0 <= frame_id < num_motion_frames
+        ):
+            # 兼容旧分支（在线 LBS）
+            xyz_canonical = gaussians.get_xyz_canonical().to("cuda")
+            rot_canonical = gaussians.get_rotation_canonical().to("cuda")
+            motions_t = lbs_motions[frame_id]
+            posed_xyz, posed_rot = lbs_deformer.deform(
+                xyz_canonical, rot_canonical, motions_t
+            )
+            pose_ctx = gaussians.deform_ctx(posed_xyz, posed_rot)
         else:
             pose_ctx = nullcontext()


渲染/损失/反传逻辑保持不变（仍在 with pose_ctx: 块内渲染、只训练颜色/曝光等外观参数）。你的渲染与损失计算代码块在 Color 循环下方不变。

2.4 冻结几何（确保“只优化颜色/曝光”）

在训练初始化阶段，若 color_only 或 freeze_geometry，显式调用：

if color_only or freeze_geometry:
    gaussians.freeze_geometry()  # 冻结 xyz/rotation/scaling 的梯度


此方法在 GaussianModel 中已实现。
---

### 4. 注意事项

- **帧索引一致性**：`pose_cache` 的键要与 `frames_dir/<frame_id>/<scene>/...` 中的 `frame_id` 值保持一致。，我们在训练时需要注意相关信息，出现异常raise exception with clear message。
- **空间占用**：缓存包含 `N_gaussian × 7`（xyz+quat）× `T`，请使用 `float32` 并在加载后视需要 `.half()`。
- **首帧回退**：若 Color Stage 还需要 canonical 姿态（比如做正则化），把 canonical pose 也保存为 `pose_cache[canonical_frame_id]`。
