"""
evaluate.py

Reviewer-friendly evaluation script for Bengali handwritten word segmentation
using Swin Transformer (Swin-S) + Mask R-CNN.

This script reports:
  - Bounding Box AP50 / AP75
  - Segmentation Mask AP50 / AP75
  - Word-level Precision / Recall / F1-score at IoU >= 0.50

Training summary:
  - Architecture  : Mask R-CNN + Swin-S backbone
  - Dataset       : 326 Bengali handwritten answer-book pages (COCO format)
  - Optimizer     : SGD (momentum=0.9, weight_decay=0.0001)
  - Iterations    : 5,000
  - Val set       : 30 annotated images

Expected repository structure:

    bengali-word-segmentation/
    ├── models/
    │   └── final_model.pth
    ├── sample_data/
    │   ├── images/          ← 30 validation images
    │   └── annotations/
    │       └── val_annotations.json
    ├── scripts/
    │   └── evaluate.py
    └── SwinT_detectron2/
        └── train_net.py

Example usage:

    python evaluate.py \
        --model   models/final_model.pth \
        --images  sample_data/images \
        --annotations sample_data/annotations/val_annotations.json \
        --output  outputs/evaluation \
        --swint_repo SwinT_detectron2
"""

import argparse
import importlib.util
import io
import logging
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Bengali handwritten word segmentation model"
    )
    parser.add_argument("--model",       default="models/final_model.pth")
    parser.add_argument("--images",      default="sample_data/images")
    parser.add_argument("--annotations", default="sample_data/annotations/val_annotations.json")
    parser.add_argument("--output",      default="outputs/evaluation")
    parser.add_argument("--swint_repo",  default="SwinT_detectron2")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    return parser.parse_args()


def silence():
    return redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())


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
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    cfg = get_cfg()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        train_net.add_swint_config(cfg)
    cfg.merge_from_file(config_path)

    cfg.MODEL.SWINT.DEPTHS                    = [2, 2, 18, 2]
    cfg.MODEL.WEIGHTS                         = args.model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES           = 1
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION  = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST     = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE             = 300    # dense Bengali pages
    cfg.INPUT.MIN_SIZE_TEST                   = 800
    cfg.INPUT.MAX_SIZE_TEST                   = 1333
    cfg.DATALOADER.NUM_WORKERS                = 2
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def register_validation_dataset(dataset_name, annotation_file, image_dir):
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)
    register_coco_instances(dataset_name, {}, annotation_file, image_dir)


def compute_word_detection_metrics(coco_gt, predictions_file, score_thresh=0.5):
    if not os.path.exists(predictions_file):
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            "This is generated automatically by COCOEvaluator during inference."
        )

    predictions = torch.load(predictions_file, map_location="cpu")

    coco_dt_list = []
    for pred in predictions:
        for inst in pred["instances"]:
            x, y, w, h = inst["bbox"]
            coco_dt_list.append({
                "image_id":    pred["image_id"],
                "category_id": 1,
                "bbox":        [float(x), float(y), float(w), float(h)],
                "score":       float(inst["score"])
            })

    coco_dt = coco_gt.loadRes(coco_dt_list) if coco_dt_list else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.50])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()

    tp, fp, fn = 0, 0, 0
    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        if list(eval_img["aRng"]) != [0, 10000000000.0]:
            continue

        dt_scores  = np.array(eval_img["dtScores"])
        dt_matches = np.array(eval_img["dtMatches"])[0]
        dt_ignore  = np.array(eval_img["dtIgnore"])[0].astype(bool)
        gt_ignore  = np.array(eval_img["gtIgnore"]).astype(bool)
        gt_ids     = eval_img["gtIds"]

        keep           = dt_scores >= score_thresh
        kept_matches   = dt_matches[keep]
        kept_ignore    = dt_ignore[keep]

        tp += int(np.sum((kept_matches > 0) & (~kept_ignore)))
        fp += int(np.sum((kept_matches == 0) & (~kept_ignore)))

        matched_gt_ids = set(kept_matches[kept_matches > 0].astype(int).tolist())
        valid_gt_ids   = [int(g) for g, ign in zip(gt_ids, gt_ignore) if not ign]
        fn += sum(1 for g in valid_gt_ids if g not in matched_gt_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn


def main():
    args = parse_args()

    # Suppress all Detectron2 / library output
    logging.disable(logging.CRITICAL)
    os.environ["DETECTRON2_SILENT"] = "1"
    os.makedirs(args.output, exist_ok=True)

    dataset_name = "bengali_word_val_30"

    train_net = load_train_net(args.swint_repo)
    register_validation_dataset(dataset_name, args.annotations, args.images)
    cfg = build_config(args, train_net)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()

    evaluator  = COCOEvaluator(dataset_name, output_dir=args.output)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        results = inference_on_dataset(model, val_loader, evaluator)

    bbox_ap50 = results["bbox"].get("AP50", 0.0)
    bbox_ap75 = results["bbox"].get("AP75", 0.0)
    segm_ap50 = results["segm"].get("AP50", 0.0)
    segm_ap75 = results["segm"].get("AP75", 0.0)

    coco_gt          = COCO(args.annotations)
    predictions_file = os.path.join(args.output, "instances_predictions.pth")

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        precision, recall, f1, tp, fp, fn = compute_word_detection_metrics(
            coco_gt=coco_gt,
            predictions_file=predictions_file,
            score_thresh=args.score_thresh
        )

    total_gt       = len(coco_gt.getAnnIds())
    total_detected = tp + fp

    # ── Print only clean results ───────────────────────────────────────────────
    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Validation Images      : {len(coco_gt.getImgIds())}")
    print(f"  Total Word Annotations : {total_gt}")

    print("\n" + "=" * 60)
    print("  MODEL PERFORMANCE  (COCO metrics, %)")
    print("=" * 60)
    print(f"  {'Metric':<10} {'Bounding Box':>15} {'Segmentation':>15}")
    print(f"  {'-'*42}")
    print(f"  {'AP50':<10} {bbox_ap50:>15.2f} {segm_ap50:>15.2f}")
    print(f"  {'AP75':<10} {bbox_ap75:>15.2f} {segm_ap75:>15.2f}")

    print("\n" + "=" * 60)
    print(f"  WORD DETECTION METRICS  (IoU >= 0.50, score >= {args.score_thresh})")
    print("=" * 60)
    print(f"  {'Precision':<15} {precision * 100:.2f}%")
    print(f"  {'Recall':<15} {recall * 100:.2f}%")
    print(f"  {'F1 Score':<15} {f1 * 100:.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
