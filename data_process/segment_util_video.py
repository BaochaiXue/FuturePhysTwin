"""
Segment an entire RGB video sequence and produce per-frame object/controller masks.

Inputs
------
- ``{base_path}/{case_name}/color/{camera_idx}.mp4``: raw video for each camera.
- ``{base_path}/{case_name}`` auxiliary data (e.g., ``depth/`` directory for consistency checks).

Outputs
-------
- ``{output_path}/mask/mask_info_{camera_idx}.json``: mapping from mask IDs to semantic labels.
- ``{output_path}/mask/{camera_idx}/{mask_id}/{frame_idx}.png``: per-frame binary mask (255 denotes foreground).
- Optional ``--exclude_mask_info``/``--exclude_mask_root`` arguments can zero out known objects before segmentation (useful for human-only masks).
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from tqdm import tqdm

"""
Hyperparam for Ground and Tracking
"""

# Put below base path into args
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect",
)
parser.add_argument("--case_name", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
parser.add_argument("--camera_idx", type=int)
parser.add_argument("--output_path", type=str, default="NONE")
parser.add_argument("--exclude_mask_info", type=str, default=None)
parser.add_argument("--exclude_mask_root", type=str, default=None)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_idx = args.camera_idx
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"
else:
    output_path = args.output_path
exclude_mask_info_path = args.exclude_mask_info
exclude_mask_root = args.exclude_mask_root


def existDir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks of identical shape."""

    if mask_a.shape != mask_b.shape:
        raise ValueError("Mask shapes must match for IoU computation")

    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    union = mask_a_bool.sum() + mask_b_bool.sum() - intersection
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


GROUNDING_DINO_CONFIG = (
    "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT = (
    "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
EXCLUDE_MASK_IOU_THRESHOLD = 0.95
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"
existDir(f"{base_path}/{case_name}/tmp_data")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}")

SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"

"""
Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
"""
# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)


# init sam image predictor and video predictor model
sam2_checkpoint = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=str(source_frames), overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names: List[str] = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image_source, image = load_image(img_path)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes: np.ndarray = box_convert(
    boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
).numpy()
confidences = confidences.numpy().tolist()
class_names: List[str] = labels

exclude_masks: List[np.ndarray] = []
if exclude_mask_info_path and exclude_mask_root:
    try:
        with open(exclude_mask_info_path, "r", encoding="utf-8") as f:
            mask_info = json.load(f)
        for mask_id_str, label in mask_info.items():
            if label == "hand":
                continue
            mask_path = os.path.join(
                exclude_mask_root, str(mask_id_str), f"{ann_frame_idx}.png"
            )
            if not os.path.isfile(mask_path):
                continue
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                continue
            exclude_masks.append(mask_img > 0)
    except FileNotFoundError:
        print(f"[warn] exclude_mask_info not found: {exclude_mask_info_path}")

image_predictor.set_image(image_source)

if exclude_masks:
    keep_indices: List[int] = []
    for idx, (input_box, confidence, class_name) in enumerate(
        zip(input_boxes, confidences, class_names)
    ):
        print(
            f"[debug info]Box: {input_box}, Confidence: {confidence:.4f}, Class: {class_name}"
        )
        try:
            mask_set, _, _ = image_predictor.predict(
                box=input_box, multimask_output=False
            )
        except RuntimeError as exc:
            print(f"[warn] SAM2 prediction failed for box {input_box}: {exc}")
            keep_indices.append(idx)
            continue
        pred_mask: np.ndarray = mask_set[0] > 0
        max_iou = 0.0
        for exclude_mask in exclude_masks:
            print(
                f"[debug info] Comparing with exclude mask of shape {exclude_mask.shape} and pred mask of shape {pred_mask.shape}"
            )
            if exclude_mask.shape != pred_mask.shape:
                resized = cv2.resize(
                    exclude_mask.astype(np.uint8),
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                exclude_bool = resized > 0
            else:
                exclude_bool = exclude_mask
            max_iou = max(max_iou, compute_mask_iou(pred_mask, exclude_bool))
        if max_iou >= EXCLUDE_MASK_IOU_THRESHOLD:
            print(
                f"[info] Skipping box due to overlap with exclude mask (IoU={max_iou:.3f})."
            )
            continue
        keep_indices.append(idx)

    if keep_indices and len(keep_indices) < len(input_boxes):
        input_boxes = input_boxes[keep_indices]
        confidences = [confidences[i] for i in keep_indices]
        class_names = [class_names[i] for i in keep_indices]
    elif not keep_indices:
        print(
            "[warn] All detections matched exclude masks; reverting to original detections."
        )

# prompt SAM image predictor to get the mask for the object

# process the detection results
OBJECTS: List[str] = class_names

print(OBJECTS)

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"

if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
else:
    raise NotImplementedError(
        "SAM 2 video predictor only support point/box/mask prompts"
    )

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments: Dict[int, Dict[int, np.ndarray]] = {}  # per-frame segmentation results
for (
    out_frame_idx,
    out_obj_ids,
    out_mask_logits,
) in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

existDir(f"{output_path}/mask/")
existDir(f"{output_path}/mask/{camera_idx}")

ID_TO_OBJECTS: Dict[int, str] = {i: obj for i, obj in enumerate(OBJECTS)}

# Save the id_to_objects into json
with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
    json.dump(ID_TO_OBJECTS, f)

for frame_idx, masks in video_segments.items():
    for obj_id, mask in masks.items():
        existDir(f"{output_path}/mask/{camera_idx}/{obj_id}")
        # mask is 1 * H * W
        Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
            f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
        )
