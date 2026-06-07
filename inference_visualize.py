"""
inference_visualize.py

Generate prediction visualizations for Bengali handwritten word segmentation
using Swin Transformer + Mask R-CNN.

This script:
- Loads the trained model
- Runs inference on validation images
- Draws predicted masks and bounding boxes
- Saves visualizations to outputs/predictions/

"""

import argparse
import glob
import importlib.util
import io
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
 
import cv2
import torch
 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and save word segmentation visualizations"
    )
    parser.add_argument("--model",       default="models/final_model.pth")
    parser.add_argument("--images",      default="sample_data/images")
    parser.add_argument("--output",      default="outputs/predictions")
    parser.add_argument("--swint_repo",  default="SwinT_detectron2")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    return parser.parse_args()
 
 
def load_train_net(swint_repo):
    train_net_path = os.path.join(swint_repo, "train_net.py")
    if not os.path.exists(train_net_path):
        raise FileNotFoundError(
            f"Could not find train_net.py at: {train_net_path}\n"
            "Please place/clone the SwinT_detectron2 folder in the repository root."
        )
    sys.path.insert(0, swint_repo)
    spec = importlib.util.spec_from_file_location("train_net", train_net_path)
    train_net = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(train_net)
    return train_net
 
 
def build_config(args, train_net):
    config_path = os.path.join(
        args.swint_repo, "configs", "SwinT", "mask_rcnn_swins_S_FPN_3x.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Swin-S config not found at: {config_path}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model weights not found: {args.model}\n"
            "Place the trained model at models/final_model.pth"
        )
 
    cfg = get_cfg()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        train_net.add_swint_config(cfg)
    cfg.merge_from_file(config_path)
 
    cfg.MODEL.SWINT.DEPTHS                    = [2, 2, 18, 2]
    cfg.MODEL.WEIGHTS                         = args.model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES           = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST     = args.score_thresh
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION  = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.TEST.DETECTIONS_PER_IMAGE             = 300  # dense Bengali pages
    cfg.INPUT.MIN_SIZE_TEST                   = 800
    cfg.INPUT.MAX_SIZE_TEST                   = 1333
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg
 
 
def collect_images(image_dir):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(image_paths)
 
 
def main():
    args = parse_args()
 
    logging.disable(logging.CRITICAL)
    os.environ["DETECTRON2_SILENT"] = "1"
    os.makedirs(args.output, exist_ok=True)
 
    train_net = load_train_net(args.swint_repo)
    cfg = build_config(args, train_net)
 
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        predictor = DefaultPredictor(cfg)
 
    image_paths = collect_images(args.images)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in: {args.images}\n"
            "Supported formats: jpg, jpeg, png, tif, tiff, bmp"
        )
 
    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[SKIP] Could not read: {os.path.basename(image_path)}")
            continue
 
        outputs   = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")
 
        visualizer = Visualizer(
            image_bgr[:, :, ::-1],
            scale=1.0,
            instance_mode=ColorMode.IMAGE
        )
        vis_bgr = visualizer.draw_instance_predictions(instances).get_image()[:, :, ::-1]
 
        base_name   = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output, f"{base_name}_pred.png")
        cv2.imwrite(output_path, vis_bgr)
 
        print(f"{os.path.basename(image_path):50s} -> {len(instances):4d} words detected")
 
    print(f"\nVisualizations saved to: {args.output}")
 
 
if __name__ == "__main__":
    main()
