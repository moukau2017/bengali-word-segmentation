"""
evaluate.py

Reviewer-friendly evaluation script for Bengali handwritten word segmentation
using Swin Transformer + Mask R-CNN.

This script reports:
- Bounding Box AP50 / AP75
- Segmentation Mask AP50 / AP75
- Word-level Precision / Recall / F1-score at IoU >= 0.50

Expected repository structure:

bengali-word-segmentation/
├── models/
│   └── final_model.pth
├── sample_data/
│   ├── images/
│   └── annotations/
│       └── val_annotations.json
├── scripts/
│   └── evaluate.py
└── SwinT_detectron2/
    └── train_net.py
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

    parser.add_argument(
        "--model",
        default="validation_images",
        help="Path to trained model weights"
    )

    parser.add_argument(
        "--images",
        default="sample_data/images",
        help="Path to validation images"
    )

    parser.add_argument(
        "--annotations",
        default="val_annotations.json",
        help="Path to COCO validation annotation file"
    )

    parser.add_argument(
        "--output",
        default="outputs/evaluation",
        help="Directory to save evaluation outputs"
    )

    parser.add_argument(
        "--swint_repo",
        default="SwinT_detectron2",
        help="Path to SwinT Detectron2 repository/folder"
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold for word-level Precision/Recall/F1 calculation"
    )

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

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        spec.loader.exec_module(train_net)

    return train_net


def build_config(args, train_net):
    config_path = os.path.join(
        args.swint_repo,
        "configs",
        "SwinT",
        "mask_rcnn_swins_S_FPN_3x.yaml"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Could not find Swin-S config file at: {config_path}"
        )

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Download the trained model from the README link and place it at models/final_model.pth"
        )

    cfg = get_cfg()

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        train_net.add_swint_config(cfg)

    cfg.merge_from_file(config_path)

    # Model configuration
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 18, 2]
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh

    # Same inference configuration used for reported validation results
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def register_validation_dataset(dataset_name, annotation_file, image_dir):
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
        MetadataCatalog.remove(dataset_name)

    register_coco_instances(
        dataset_name,
        {},
        annotation_file,
        image_dir
    )


def compute_word_detection_metrics(coco_gt, predictions_file, score_thresh=0.5):
    predictions = torch.load(predictions_file, map_location="cpu")

    coco_dt_list = []

    for pred in predictions:
        image_id = pred["image_id"]

        for inst in pred["instances"]:
            x, y, w, h = inst["bbox"]

            coco_dt_list.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(inst["score"])
            })

    coco_dt = coco_gt.loadRes(coco_dt_list) if coco_dt_list else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.50])

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        coco_eval.evaluate()
        coco_eval.accumulate()

    tp_count = 0
    fp_count = 0
    fn_count = 0

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue

        # Use only area=all to avoid duplicate counting across area ranges
        if list(eval_img["aRng"]) != [0, 10000000000.0]:
            continue

        dt_scores = np.array(eval_img["dtScores"])
        dt_matches = np.array(eval_img["dtMatches"])[0]
        dt_ignore = np.array(eval_img["dtIgnore"])[0].astype(bool)
        gt_ignore = np.array(eval_img["gtIgnore"]).astype(bool)
        gt_ids = eval_img["gtIds"]

        keep = dt_scores >= score_thresh

        kept_matches = dt_matches[keep]
        kept_ignore = dt_ignore[keep]

        tp_count += int(np.sum((kept_matches > 0) & (~kept_ignore)))
        fp_count += int(np.sum((kept_matches == 0) & (~kept_ignore)))

        matched_gt_ids = set(
            kept_matches[kept_matches > 0].astype(int).tolist()
        )

        valid_gt_ids = [
            int(gt_id)
            for gt_id, ignore in zip(gt_ids, gt_ignore)
            if not ignore
        ]

        fn_count += sum(
            1 for gt_id in valid_gt_ids
            if gt_id not in matched_gt_ids
        )

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return precision, recall, f1_score, tp_count, fp_count, fn_count


def main():
    args = parse_args()

    logging.disable(logging.CRITICAL)
    os.environ["DETECTRON2_SILENT"] = "1"
    os.makedirs(args.output, exist_ok=True)

    dataset_name = "bengali_word_validation"

    train_net = load_train_net(args.swint_repo)
    register_validation_dataset(dataset_name, args.annotations, args.images)
    cfg = build_config(args, train_net)

    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model = train_net.Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()

    evaluator = COCOEvaluator(dataset_name, output_dir=args.output)
    val_loader = build_detection_test_loader(cfg, dataset_name)

    with redirect_stdout(sink), redirect_stderr(sink):
        results = inference_on_dataset(model, val_loader, evaluator)

    bbox_ap50 = results["bbox"].get("AP50", 0.0)
    bbox_ap75 = results["bbox"].get("AP75", 0.0)
    segm_ap50 = results["segm"].get("AP50", 0.0)
    segm_ap75 = results["segm"].get("AP75", 0.0)

    coco_gt = COCO(args.annotations)
    predictions_file = os.path.join(args.output, "instances_predictions.pth")

    precision, recall, f1_score, tp, fp, fn = compute_word_detection_metrics(
        coco_gt=coco_gt,
        predictions_file=predictions_file,
        score_thresh=args.score_thresh
    )

    total_gt = len(coco_gt.getAnnIds())
    total_detected = tp + fp

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Validation Images      : {len(coco_gt.getImgIds())}")
    print(f"Total Word Annotations : {total_gt}")

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"{'Metric':<10}{'Bounding Box':>15}{'Segmentation':>15}")
    print("-" * 40)
    print(f"{'AP50':<10}{bbox_ap50:>15.2f}{segm_ap50:>15.2f}")
    print(f"{'AP75':<10}{bbox_ap75:>15.2f}{segm_ap75:>15.2f}")

    print("\n" + "=" * 60)
    print("WORD DETECTION METRICS")
    print("=" * 60)
    print(f"{'Metric':<15}{'Value'}")
    print("-" * 30)
    print(f"{'Precision':<15}{precision * 100:.2f}%")
    print(f"{'Recall':<15}{recall * 100:.2f}%")
    print(f"{'F1 Score':<15}{f1_score * 100:.2f}%")

    print("\n" + "=" * 60)
    print("WORD COUNT STATISTICS")
    print("=" * 60)
    print(f"Total GT Words       : {total_gt}")
    print(f"Total Detected Words : {total_detected}")
    print(f"Difference           : {total_detected - total_gt:+d}")


if __name__ == "__main__":
    main()
