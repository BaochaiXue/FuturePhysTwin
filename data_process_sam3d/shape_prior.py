import os
from argparse import ArgumentParser
import math
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
from PIL import Image

# Keep API identical to Trellis version while swapping the backend to SAM3D.
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.

import sys

sys.path.append("sam-3d-objects/notebook")
from inference import Inference  # type: ignore
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils


DEFAULT_CONFIG = Path("sam-3d-objects/checkpoints/hf/pipeline.yaml")
DEFAULT_SEED = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_visualization(
    gaussian,
    mesh,
    output_dir: Path,
    mesh_path: Path,
    num_frames: int = 120,
    fps: int = 30,
) -> None:
    """
    Render a side-by-side turntable video (gaussian color | mesh normal) to match Trellis outputs.
    """

    sam3d_error: Exception | None = None

    # Only use Open3D fallback to render visualization video (skip SAM3D render_video entirely).
    try:
        import cv2
        import open3d as o3d

        mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
        if mesh_o3d.is_empty():
            raise ValueError("Fallback mesh is empty")
        mesh_o3d.compute_vertex_normals()

        width = height = 640
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.add_geometry("mesh", mesh_o3d, material)

        bounds = mesh_o3d.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = np.max(bounds.get_extent())
        radius = float(extent) * 1.8 if extent > 0 else 1.0
        up = np.array([0.0, 1.0, 0.0])

        out_path = output_dir / "visualization.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

        for i in range(num_frames):
            theta = 2.0 * math.pi * i / num_frames
            eye = center + radius * np.array([math.cos(theta), 0.3, math.sin(theta)])
            renderer.scene.camera.look_at(center, eye, up)
            img = renderer.render_to_image()
            frame = np.asarray(img)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        del renderer
        print(f"[info] visualization.mp4 written to {out_path}")
        return
    except Exception as fallback_exc:  # pragma: no cover
        msg = (
            f"Failed to generate visualization.mp4. "
            f"SAM3D render error: {sam3d_error}; fallback error: {fallback_exc}"
        )
        raise RuntimeError(msg) from fallback_exc


def main() -> None:
    parser = ArgumentParser(
        description="Generate shape prior via SAM3D (Trellis-compatible API)"
    )
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="SAM3D pipeline config (default: sam-3d-objects/checkpoints/hf/pipeline.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for SAM3D inference (default: 42)",
    )
    args = parser.parse_args()

    img_path = Path(args.img_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    final_im = Image.open(img_path).convert("RGBA")
    alpha = np.array(final_im)[:, :, 3]
    if np.all(alpha == 255):
        raise ValueError(
            "Image must contain a valid alpha channel for foreground mask."
        )
    # SAM3D expects an alpha-like mask in [0, 255]; keep foreground as 255.
    mask = ((alpha > 0).astype(np.uint8)) * 255
    # SAM3D expects numpy arrays; convert RGB PIL image to uint8 ndarray.
    image_rgb = np.array(final_im.convert("RGB"), dtype=np.uint8)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"SAM3D config not found: {config_path}")

    infer = Inference(str(config_path), compile=False)

    # Follow SAM3D recommended path: call _pipeline.run with baking enabled.
    # Use nvdiffrast for rendering/baking to ensure bake_texture works.
    infer._pipeline.rendering_engine = "nvdiffrast"
    outputs = infer._pipeline.run(
        image_rgb,
        mask,
        seed=args.seed,
        with_mesh_postprocess=True,
        # Enable UV texture baking to produce textured meshes (even when vertex colors are disabled).
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,
    )
    # _pipeline.run already calls postprocess_slat_output; glb should be textured.
    mesh_for_save = outputs.get("glb", outputs.get("mesh", [None])[0])

    def export_mesh(mesh_obj, path: Path) -> None:
        """Export mesh_obj to GLB using available APIs or a trimesh fallback."""

        # If the mesh object exposes a success flag, respect it.
        if hasattr(mesh_obj, "success") and getattr(mesh_obj, "success") is False:
            raise ValueError("Mesh extraction failed (mesh_obj.success is False).")

        # Native export when available
        if hasattr(mesh_obj, "export"):
            mesh_obj.export(path)
            # Validate/export cleanup: strip alpha from vertex colors to keep 3-channel visuals (matches Trellis output expectations).
            import trimesh

            tmp = trimesh.load(path, force="mesh", process=False)
            if tmp.is_empty:
                raise ValueError(f"Exported mesh is empty at {path}")
            if tmp.visual.kind == "vertex" and getattr(tmp.visual, "vertex_colors", None) is not None:
                vc = tmp.visual.vertex_colors
                if vc.shape[1] >= 4:
                    tmp.visual.vertex_colors = vc[:, :3]  # drop alpha channel
                    tmp.export(path)
            return

        # Fallback: handle MeshExtractResult (vertices, faces, optional vertex_attrs).
        import trimesh  # local import

        vertices = getattr(mesh_obj, "vertices", None)
        faces = getattr(mesh_obj, "faces", None)
        vertex_attrs = getattr(mesh_obj, "vertex_attrs", None)
        if vertices is None or faces is None:
            raise AttributeError(
                f"Mesh object missing vertices/faces; cannot export to {path}"
            )
        if hasattr(vertices, "__len__") and len(vertices) == 0:
            raise ValueError("Mesh vertices are empty; cannot export.")
        if hasattr(faces, "__len__") and len(faces) == 0:
            raise ValueError("Mesh faces are empty; cannot export.")
        # Convert tensors to numpy if needed.
        if hasattr(vertices, "detach"):
            vertices = vertices.detach().cpu().numpy()
        if hasattr(faces, "detach"):
            faces = faces.detach().cpu().numpy()
        colors = None
        if vertex_attrs is not None:
            if hasattr(vertex_attrs, "detach"):
                vertex_attrs = vertex_attrs.detach().cpu().numpy()
            if vertex_attrs.shape[-1] >= 3:
                colors = vertex_attrs[..., :3]
                if colors.max() <= 1.0:
                    colors = (colors * 255.0).clip(0, 255)
                colors = colors.astype(np.uint8)
        tm = trimesh.Trimesh(
            vertices=vertices,
            faces=faces.astype(np.int32),
            vertex_colors=colors,
            process=False,
        )
        tm.export(path)
        # Validate export
        tmp = trimesh.load(path, force="mesh", process=False)
        if tmp.is_empty:
            raise ValueError(f"Exported mesh is empty at {path}")

    mesh_path = output_dir / "object.glb"
    export_mesh(mesh_for_save, mesh_path)

    gaussian: Optional[object] = None
    if "gaussian" in outputs and outputs["gaussian"]:
        gaussian = outputs["gaussian"][0]
    elif "gs" in outputs:
        gaussian = outputs["gs"]

    if gaussian is not None:
        gaussian.save_ply(output_dir / "object.ply")
        save_visualization(gaussian, mesh_for_save, output_dir, mesh_path)
    else:
        print(
            "[warn] Gaussian output missing; skipping object.ply and visualization.mp4"
        )


if __name__ == "__main__":
    main()
