"""
inference_visualize.py

Run inference with a trained Swin-S + Mask R-CNN model on the TEST SET
and save visualized segmentation outputs.

The script reads the test annotation JSON (produced by train.py's auto-split)
to know exactly which images belong to the test set. 

Each output image shows:
  - Predicted bounding boxes with confidence scores
  - Instance segmentation masks (colour-coded per word)
  - Ground-truth bounding boxes in green (from the annotation JSON)

Outputs are saved to:
  <output_dir>/
    <image_name>_pred.jpg     (one per test image)
    inference_log.json        (run summary)

Example usage:

    python inference_visualize.py \\
        --model       outputs/training/model_final.pth \\
        --images      /path/to/all_images \\
        --annotations outputs/training/splits/test_annotations.json \\
        --output_dir  outputs/visualizations \\
        --swint_repo  SwinT_detectron2

"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference + visualization on the test set"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model weights (.pth)"
    )
    parser.add_argument(
        "--images", required=True,
        help="Folder that contains ALL 435 images"
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Test-set COCO annotation JSON "
             "(outputs/training/splits/test_annotations.json)"
    )
    parser.add_argument(
        "--output_dir", default="outputs/visualizations",
        help="Directory to save visualized outputs (default: outputs/visualizations)"
    )
    parser.add_argument(
        "--swint_repo", default="SwinT_detectron2",
        help="Path to SwinT_detectron2 repository (default: SwinT_detectron2)"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.5,
        help="Minimum confidence score to display (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu' (default: cuda if available)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_swint_repo(swint_repo):
    """Add SwinT_detectron2 to sys.path to register the Swin backbone."""
    if not os.path.isdir(swint_repo):
        raise FileNotFoundError(
            f"SwinT_detectron2 repo not found at: {swint_repo}\n"
            "Please clone: git clone https://github.com/xiaohu2015/SwinT_detectron2.git"
        )
    if swint_repo not in sys.path:
        sys.path.insert(0, swint_repo)
    import importlib
    for mod in ["models", "swint"]:
        mod_path = os.path.join(swint_repo, mod + ".py")
        if os.path.exists(mod_path):
            spec = importlib.util.spec_from_file_location(mod, mod_path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)


def build_cfg(swint_repo, model_path, score_threshold, device):
    cfg = get_cfg()

    config_file = os.path.join(
        swint_repo, "configs", "SwinT", "mask_rcnn_swins_S_FPN_3x.yaml"
    )
    if not os.path.exists(config_file):
        config_file = os.path.join(
            swint_repo, "configs", "SwinT", "mask_rcnn_swint_T_FPN_3x.yaml"
        )
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Swin config not found at: {config_file}\n"
            "Run train.py first — it generates the Swin-S config automatically."
        )

    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS                     = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.SWINT.DEPTHS                = (2, 2, 18, 2)
    cfg.INPUT.MAX_SIZE_TEST               = 1333
    cfg.INPUT.MIN_SIZE_TEST               = 800
    cfg.MODEL.DEVICE                      = device
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Load test set from annotation JSON
# ---------------------------------------------------------------------------

def load_test_set(annotations_json, images_dir):
    """
    Read the test annotation JSON and return:
      - image_list  : list of dicts {id, filename, filepath, width, height}
      - gt_lookup   : dict  filename → list of [x1,y1,x2,y2] GT boxes

    Only images whose file actually exists in images_dir are included.
    Missing files are reported as warnings.
    """
    with open(annotations_json) as f:
        coco = json.load(f)

    gt_lookup = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        gt_lookup.setdefault(img_id, [])
        x, y, w, h = ann["bbox"]
        gt_lookup[img_id].append([x, y, x + w, y + h])

    image_list = []
    missing    = []

    for img_info in coco["images"]:
        fname    = os.path.basename(img_info["file_name"])
        filepath = os.path.join(images_dir, fname)

        if not os.path.exists(filepath):
            missing.append(fname)
            continue

        image_list.append({
            "id":       img_info["id"],
            "filename": fname,
            "filepath": filepath,
            "width":    img_info.get("width",  0),
            "height":   img_info.get("height", 0),
            "gt_boxes": gt_lookup.get(img_info["id"], []),
        })

    if missing:
        print(f"\n[warn] {len(missing)} image(s) listed in JSON but not found "
              f"in {images_dir}:")
        for m in missing:
            print(f"        {m}")

    return image_list


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

# Colour palette — one colour per predicted word instance
_PALETTE = [
    (255,  82,  82), (255, 165,   0), ( 50, 205,  50), ( 30, 144, 255),
    (238, 130, 238), (255, 215,   0), (  0, 206, 209), (255, 127,  80),
    (127, 255,   0), (  0, 191, 255), (255,  20, 147), (173, 255,  47),
    (255, 160, 122), (135, 206, 235), (240, 230, 140), (152, 251, 152),
    (175, 238, 238), (216, 191, 216), (255, 222, 173), (224, 255, 255),
    (210, 180, 140), (128, 128,   0), (  0, 128, 128), (128,   0, 128),
    (  0,   0, 128), (128,   0,   0), (  0, 128,   0), (192, 192, 192),
]

MASK_ALPHA  = 0.38
BOX_THICK   = 2
GT_COLOR    = (0, 210, 0)    # bright green for GT boxes
GT_THICK    = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX


def draw_predictions(image, instances, gt_boxes):
    """
    Overlay predicted masks, boxes, scores, and GT boxes on the image.

    Parameters
    ----------
    image     : BGR numpy array (H, W, 3)
    instances : Detectron2 Instances (CPU)
    gt_boxes  : list of [x1,y1,x2,y2]  (ground truth, drawn in green)

    Returns
    -------
    Annotated BGR numpy array
    """
    canvas  = image.copy()
    overlay = image.copy()

    n           = len(instances)
    has_boxes   = instances.has("pred_boxes")
    has_scores  = instances.has("scores")
    has_masks   = instances.has("pred_masks")

    # ── Draw masks ────────────────────────────────────────────────────────
    if has_masks:
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            mask  = instances.pred_masks[i].numpy().astype(bool)
            overlay[mask] = color
        cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)

    # ── Draw predicted boxes and score labels ─────────────────────────────
    if has_boxes:
        for i in range(n):
            color        = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = [int(v) for v in instances.pred_boxes.tensor[i]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, BOX_THICK)

            if has_scores:
                score = float(instances.scores[i])
                label = f"{score:.2f}"
                (tw, th), bl = cv2.getTextSize(label, FONT, 0.42, 1)
                cv2.rectangle(canvas,
                              (x1, y1 - th - bl - 4),
                              (x1 + tw + 2, y1),
                              color, -1)
                cv2.putText(canvas, label,
                            (x1 + 1, y1 - bl - 2),
                            FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Draw ground-truth boxes ───────────────────────────────────────────
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GT_COLOR, GT_THICK)

    # ── Legend (top-right) ────────────────────────────────────────────────
    _draw_legend(canvas, n)

    return canvas


def _draw_legend(img, n_pred):
    h, w     = img.shape[:2]
    entries  = [
        (_PALETTE[0], f"Predictions ({n_pred})"),
        (GT_COLOR,    "Ground truth"),
    ]
    pad      = 8
    box_sz   = 13
    line_h   = box_sz + 7
    leg_h    = len(entries) * line_h + 2 * pad
    leg_w    = 175
    x0, y0   = w - leg_w - 8, 8

    # Semi-transparent dark background
    roi = img[y0:y0 + leg_h, x0:x0 + leg_w]
    cv2.addWeighted(np.zeros_like(roi), 0.5, roi, 0.5, 0, roi)
    img[y0:y0 + leg_h, x0:x0 + leg_w] = roi

    for i, (color, text) in enumerate(entries):
        ey = y0 + pad + i * line_h
        cv2.rectangle(img,
                      (x0 + pad, ey),
                      (x0 + pad + box_sz, ey + box_sz),
                      color, -1)
        cv2.putText(img, text,
                    (x0 + pad + box_sz + 5, ey + box_sz - 1),
                    FONT, 0.40, (240, 240, 240), 1, cv2.LINE_AA)


def _draw_stats_bar(img, fname, n_pred, n_gt):
    """Thin info bar appended to the bottom of the image."""
    h, w  = img.shape[:2]
    bar   = np.zeros((26, w, 3), dtype=np.uint8)
    text  = f"{fname}   |   GT words: {n_gt}   Predicted: {n_pred}"
    cv2.putText(bar, text, (8, 18), FONT, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate paths
    for label, path in [
        ("--model",       args.model),
        ("--images",      args.images),
        ("--annotations", args.annotations),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Load SwinT repo
    load_swint_repo(args.swint_repo)

    # Build predictor
    print(f"\nLoading model  : {args.model}")
    cfg       = build_cfg(args.swint_repo, args.model,
                          args.score_threshold, args.device)
    predictor = DefaultPredictor(cfg)
    print("Model loaded.\n")

    # Load test set from annotation JSON
    print(f"Reading test set from: {args.annotations}")
    image_list = load_test_set(args.annotations, args.images)

    if not image_list:
        raise RuntimeError(
            "No test images found. Check that --images points to the folder "
            "containing all 435 images and --annotations points to "
            "outputs/training/splits/test_annotations.json"
        )

    print(f"Test images found : {len(image_list)}")
    print(f"Output folder     : {args.output_dir}\n")
    print(f"{'─'*65}")

    total_preds = 0
    total_gt    = 0

    for idx, item in enumerate(image_list, 1):
        img = cv2.imread(item["filepath"])
        if img is None:
            print(f"  [{idx:>3}/{len(image_list)}] SKIP (unreadable): {item['filename']}")
            continue

        # Inference
        outputs   = predictor(img)
        instances = outputs["instances"].to("cpu")
        n_pred    = len(instances)
        n_gt      = len(item["gt_boxes"])

        total_preds += n_pred
        total_gt    += n_gt

        # Draw and save
        canvas   = draw_predictions(img, instances, item["gt_boxes"])
        canvas   = _draw_stats_bar(canvas, item["filename"], n_pred, n_gt)

        stem     = os.path.splitext(item["filename"])[0]
        out_path = os.path.join(args.output_dir, f"{stem}_pred.jpg")
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  [{idx:>3}/{len(image_list)}]  {item['filename']:<38}  "
              f"GT: {n_gt:>3}  Pred: {n_pred:>3}  →  saved")

    # Summary
    print(f"{'─'*65}")
    print(f"\n  Done.")
    print(f"  Test images processed : {len(image_list)}")
    print(f"  Total GT words        : {total_gt}")
    print(f"  Total predictions     : {total_preds}")
    print(f"  Outputs saved to      : {args.output_dir}\n")

    # Save run log
    log = {
        "model":            args.model,
        "annotations":      args.annotations,
        "images_dir":       args.images,
        "score_threshold":  args.score_threshold,
        "n_test_images":    len(image_list),
        "total_gt_words":   total_gt,
        "total_predictions": total_preds,
        "output_dir":       args.output_dir,
    }
    log_path = os.path.join(args.output_dir, "inference_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Run log saved to      : {log_path}\n")


if __name__ == "__main__":
    main()
