"""

Download the trained model from link shared in READme file. Run the trained word segmentation model on sample images (provided in Sample_input folder)

then save visualized predictions to an output folder.

This is the quickest way to verify the model works on any machine.
No training require for testing our trained model - just the model and images.

Each output image shows:
  - Segmentation color mask
  - Bounding boxes with confidence scores

Outputs are saved to:
  outputs/demo/
    <image_name>_pred.jpg   (one per input image)

Example usage:

    python demo.py \\
        --model   models/final_model.pth \\
        --images  sample_input/ \\
        --swint_repo SwinT_detectron2

    # Optionally change output folder or score threshold:
        --output_dir  outputs/demo
        --score_threshold 0.5
"""



import argparse
import importlib
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
        description="Run Bengali word segmentation on sample images"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model weights (.pth)  "
             "e.g. models/final_model.pth"
    )
    parser.add_argument(
        "--images", required=True,
        help="Folder containing sample images to test"
    )
    parser.add_argument(
        "--output_dir", default="outputs/predictions",
        help="Folder to save output images (default: outputs/predictions)"
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

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_swint_repo(swint_repo):
    """Add SwinT_detectron2 to sys.path to register the Swin backbone."""
    if not os.path.isdir(swint_repo):
        raise FileNotFoundError(
            f"SwinT_detectron2 repo not found at: {swint_repo}\n"
            "Please clone it:\n"
            "  git clone https://github.com/xiaohu2015/SwinT_detectron2.git"
        )
    if swint_repo not in sys.path:
        sys.path.insert(0, swint_repo)
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
# Drawing
# ---------------------------------------------------------------------------

_PALETTE = [
    (255,  82,  82), (255, 165,   0), ( 50, 205,  50), ( 30, 144, 255),
    (238, 130, 238), (255, 215,   0), (  0, 206, 209), (255, 127,  80),
    (127, 255,   0), (  0, 191, 255), (255,  20, 147), (173, 255,  47),
    (255, 160, 122), (135, 206, 235), (240, 230, 140), (152, 251, 152),
    (175, 238, 238), (216, 191, 216), (255, 222, 173), (224, 255, 255),
    (210, 180, 140), (128, 128,   0), (  0, 128, 128), (128,   0, 128),
    (  0,   0, 128), (128,   0,   0), (  0, 128,   0), (192, 192, 192),
]

MASK_ALPHA = 0.38
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def draw_predictions(image, instances):
    """
    Draw predicted masks, bounding boxes, and confidence scores on the image.

    Parameters
    ----------
    image     : BGR numpy array (H, W, 3)
    instances : Detectron2 Instances (CPU)

    Returns
    -------
    Annotated BGR numpy array
    """
    canvas  = image.copy()
    overlay = image.copy()

    n          = len(instances)
    has_boxes  = instances.has("pred_boxes")
    has_scores = instances.has("scores")
    has_masks  = instances.has("pred_masks")

    # Draw masks
    if has_masks:
        for i in range(n):
            color = _PALETTE[i % len(_PALETTE)]
            mask  = instances.pred_masks[i].numpy().astype(bool)
            overlay[mask] = color
        cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1 - MASK_ALPHA, 0, canvas)

    # Draw bounding boxes and score labels
    if has_boxes:
        for i in range(n):
            color           = _PALETTE[i % len(_PALETTE)]
            x1, y1, x2, y2 = [int(v) for v in instances.pred_boxes.tensor[i]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

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

    # Stats bar at the bottom
    h, w  = canvas.shape[:2]
    bar   = np.zeros((26, w, 3), dtype=np.uint8)
    cv2.putText(bar, f"Words detected: {n}", (10, 18),
                FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    canvas = np.vstack([canvas, bar])

    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Download the trained model from the link in README.md and "
            "place it at models/final_model.pth"
        )
    if not os.path.isdir(args.images):
        raise FileNotFoundError(
            f"Images folder not found: {args.images}\n"
            "Point --images to the folder containing the sample images."
        )

    # Collect image files
    image_files = sorted([
        f for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])
    if not image_files:
        raise RuntimeError(
            f"No supported images found in: {args.images}\n"
            f"Supported formats: {', '.join(SUPPORTED_EXT)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    load_swint_repo(args.swint_repo)

    print(f"\nModel       : {args.model}")
    print(f"Device      : {args.device}")
    print(f"Threshold   : {args.score_threshold}")
    print(f"Images      : {args.images}  ({len(image_files)} found)")
    print(f"Output      : {args.output_dir}")
    print(f"\nLoading model ...")

    predictor = DefaultPredictor(
        build_cfg(args.swint_repo, args.model,
                  args.score_threshold, args.device)
    )
    print("Model loaded.\n")
    print(f"{'─'*55}")

    # Run inference on each image
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(args.images, fname)
        img      = cv2.imread(img_path)

        if img is None:
            print(f"  [{idx:>3}/{len(image_files)}]  SKIP (unreadable): {fname}")
            continue

        outputs   = predictor(img)
        instances = outputs["instances"].to("cpu")
        n_words   = len(instances)

        canvas   = draw_predictions(img, instances)
        stem     = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"{stem}_pred.jpg")
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  [{idx:>3}/{len(image_files)}]  {fname:<35}  "
              f"words detected: {n_words:>3}  →  saved")

    print(f"{'─'*55}")
    print(f"\nDone.  Results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
