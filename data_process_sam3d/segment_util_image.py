"""
Single-image segmentation helper driven by GroundingDINO + SAM2.

Inputs
------
- ``--img_path``: path to the RGB image to segment.
- ``--TEXT_PROMPT``: textual prompt describing the target object.
- GroundingDINO checkpoints under ``./data_process/groundedSAM_checkpoints``.
- SAM 2 checkpoint ``sam2.1_hiera_large.pt`` and config ``configs/sam2.1/sam2.1_hiera_l.yaml``.

Outputs
-------
- RGBA mask image written to ``--output_path`` where foreground pixels inherit RGB from the source image and alpha is the binary mask.
"""

from argparse import ArgumentParser
import logging
import time
import traceback

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert

"""
Hyper parameters
"""

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_path", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
parser.add_argument("--exclude_mask_path", type=str, default=None)
args = parser.parse_args()
EXCLUDE_MASK_PATH: str | None = args.exclude_mask_path
img_path: str = args.img_path
output_path: str = args.output_path
TEXT_PROMPT: str = args.TEXT_PROMPT
EXCLUDE_MASK_IOU_THRESHOLD = 0.95
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 3.0
logging.basicConfig(
    level=logging.INFO,
    format="[segment_util_image] %(levelname)s: %(message)s",
)


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two boolean masks, resizing *mask_b* if needed."""

    mask_a_bool = mask_a.astype(bool)
    print(f"[debug info]mask_a shape: {mask_a.shape}, mask_b shape: {mask_b.shape}")
    if mask_a.shape != mask_b.shape:
        mask_b_resized = cv2.resize(
            mask_b.astype(np.uint8),
            (mask_a.shape[1], mask_a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_b_bool = mask_b_resized.astype(bool)
    else:
        mask_b_bool = mask_b.astype(bool)
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    union = mask_a_bool.sum() + mask_b_bool.sum() - intersection
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def run_segmentation(box_threshold: float = 0.35, text_threshold: float = 0.25) -> None:
    SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = (
        "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT = (
        "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    text = TEXT_PROMPT

    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes: np.ndarray = box_convert(
        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
    ).numpy()

    if isinstance(confidences, torch.Tensor):
        conf_values: list[float] = confidences.detach().cpu().numpy().tolist()
    else:
        conf_values = list(confidences)
    print(
        f"[GroundingDINO Debug] boxes shape={input_boxes.shape}, confidences={conf_values}"
    )
    if EXCLUDE_MASK_PATH is not None:
        exclude_mask = cv2.imread(EXCLUDE_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        if exclude_mask is None:
            raise FileNotFoundError(f"Failed to read exclude mask: {EXCLUDE_MASK_PATH}")
        exclude_mask = (exclude_mask > 0).astype(np.uint8) * 255
        filtered_boxes = []
        for box in input_boxes:
            # use box to get mask from sam2_predictor
            with torch.no_grad():
                masks, _, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            mask_img: np.ndarray = (masks[0] * 255).astype(np.uint8)
            iou = compute_mask_iou(mask_img, exclude_mask)
            print(f"[GroundingDINO Debug] box {box} IoU with exclude mask: {iou}")
            if iou < EXCLUDE_MASK_IOU_THRESHOLD:
                filtered_boxes.append(box)
        if filtered_boxes:
            input_boxes = np.stack(filtered_boxes, axis=0).astype(np.float32)
            print(
                f"[GroundingDINO Debug] {len(input_boxes)} boxes remain after filtering"
            )
        else:
            raise RuntimeError(
                "All detected boxes were filtered out by the exclude mask. "
                "Consider lowering EXCLUDE_MASK_IOU_THRESHOLD or updating the mask."
            )
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    print(f"Detected {len(masks)} objects")

    raw_img: np.ndarray | None = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    mask_img: np.ndarray = (masks[0] * 255).astype(np.uint8)

    ref_img = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask_img > 0
    ref_img[mask_bool, :3] = raw_img[mask_bool]
    ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
    cv2.imwrite(output_path, ref_img)
    logging.info("Saved mask to %s", output_path)


def main() -> None:
    last_exc: Exception | None = None
    box_threshold = 0.35
    text_threshold = 0.25
    for attempt in range(1, MAX_RETRIES + 1):
        if torch.cuda.is_available():
            try:
                stats = torch.cuda.memory_stats()
                allocated_mb = stats["allocated_bytes.all.current"] / (1024**2)
                reserved_mb = stats["reserved_bytes.all.current"] / (1024**2)
                logging.info(
                    "Attempt %d/%d: GPU memory allocated %.2f MB, reserved %.2f MB",
                    attempt,
                    MAX_RETRIES,
                    allocated_mb,
                    reserved_mb,
                )
            except Exception:  # pragma: no cover - stats may fail on some devices
                logging.debug(
                    "Unable to query torch.cuda.memory_stats()", exc_info=True
                )
        else:
            logging.info(
                "Attempt %d/%d: CUDA unavailable, running on CPU", attempt, MAX_RETRIES
            )
        try:
            run_segmentation(box_threshold=box_threshold, text_threshold=text_threshold)
            return
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            logging.error(
                "Segmentation attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                img_path,
                exc,
            )
            logging.debug("Stack trace:\n%s", traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            box_threshold *= 0.95
            text_threshold *= 0.95
            print(
                f"[Warning] Segmentation attempt {attempt} failed, with box_threshold={box_threshold}, text_threshold={text_threshold}"
            )
    raise SystemExit(
        f"Segmentation failed after {MAX_RETRIES} attempts; last error: {last_exc}"
    ) from last_exc


if __name__ == "__main__":
    main()
