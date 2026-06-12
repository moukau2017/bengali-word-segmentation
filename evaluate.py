"""
evaluate.py

Evaluate a trained Swin-S + Mask R-CNN model on the validation set and the
test set for Bengali handwritten word segmentation.

Reports per-set:
  - COCO bbox  AP50, AP75
  - COCO segm  AP50, AP75
  - Word-level Precision, Recall, F1-score  (IoU threshold configurable,
    default 0.5)


    python evaluate.py \\
        --model       outputs/training/model_final.pth \\
        --images      /path/to/all/images \\
        --val_json    outputs/training/splits/val_annotations.json \\
        --test_json   outputs/training/splits/test_annotations.json \\
        --output_dir  outputs/evaluation \\
        --swint_repo  SwinT_detectron2

    
"""

 
import argparse
import importlib.util
import os
import sys
 
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Swin-S Mask R-CNN on val and test sets"
    )
    parser.add_argument("--model",     required=True,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--images",    required=True,
                        help="Folder containing all images")
    parser.add_argument("--val_json",  required=True,
                        help="COCO annotation JSON for the validation set")
    parser.add_argument("--test_json", required=True,
                        help="COCO annotation JSON for the test set")
    parser.add_argument("--swint_repo", default="SwinT_detectron2",
                        help="Path to SwinT_detectron2 repository")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Minimum prediction confidence score (default: 0.5)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for word-level metrics (default: 0.5)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cuda or cpu (default: cuda if available)")
    return parser.parse_args()
 
 
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
 
def load_swint_repo(swint_repo):
    """Add SwinT repo to sys.path and register the Swin backbone with detectron2."""
 
    if not os.path.isdir(swint_repo):
        raise FileNotFoundError(
            f"SwinT_detectron2 repo not found at: {swint_repo}\n"
            "Please clone: git clone https://github.com/xiaohu2015/SwinT_detectron2.git"
        )
 
    if swint_repo not in sys.path:
        sys.path.insert(0, swint_repo)
 
    # The swint backbone is a package (swint/__init__.py), not a flat .py file.
    # It must be imported explicitly so it registers itself with detectron2's
    # model registry — without this, build_cfg / DefaultPredictor will crash.
    swint_init = os.path.join(swint_repo, "swint", "__init__.py")
    if not os.path.exists(swint_init):
        raise FileNotFoundError(
            f"swint package not found at: {swint_init}\n"
            "Make sure SwinT_detectron2 is properly cloned."
        )
 
    spec = importlib.util.spec_from_file_location(
        "swint", swint_init,
        submodule_search_locations=[os.path.join(swint_repo, "swint")]
    )
    swint_mod = importlib.util.module_from_spec(spec)
    sys.modules["swint"] = swint_mod
    spec.loader.exec_module(swint_mod)
    print(f"[setup] swint backbone registered from: {swint_repo}")
 
 
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
# Inference
# ---------------------------------------------------------------------------
 
def run_inference(predictor, coco_gt, images_dir):
    results = []
    img_ids = coco_gt.getImgIds()
    print(f"  Running inference on {len(img_ids)} images...")
 
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        fname    = os.path.basename(img_info["file_name"])
        img_path = os.path.join(images_dir, fname)
 
        if not os.path.exists(img_path):
            print(f"  [warn] image not found, skipping: {fname}")
            continue
 
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [warn] could not read image: {fname}")
            continue
 
        outputs   = predictor(img)
        instances = outputs["instances"].to("cpu")
 
        if not instances.has("pred_boxes"):
            continue
 
        boxes  = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        masks  = instances.pred_masks.numpy() if instances.has("pred_masks") else []
 
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            result = {
                "image_id":    img_id,
                "category_id": 1,
                "bbox":        [float(x1), float(y1),
                                float(x2 - x1), float(y2 - y1)],
                "score":       float(scores[i]),
            }
            if len(masks) > 0:
                from pycocotools import mask as mask_util
                rle = mask_util.encode(
                    np.asfortranarray(masks[i].astype(np.uint8))
                )
                rle["counts"] = rle["counts"].decode("utf-8")
                result["segmentation"] = rle
            results.append(result)
 
    return results
 
 
# ---------------------------------------------------------------------------
# COCO metrics  (AP50, AP75 only)
# ---------------------------------------------------------------------------
 
def compute_coco_metrics(coco_gt, results):
    if not results:
        return {"bbox_AP50": 0.0, "bbox_AP75": 0.0,
                "segm_AP50": 0.0, "segm_AP75": 0.0}
 
    coco_dt  = coco_gt.loadRes(results)
    metrics  = {}
    has_segm = any("segmentation" in r for r in results)
 
    for iou_type in ["bbox", "segm"]:
        if iou_type == "segm" and not has_segm:
            metrics["segm_AP50"] = 0.0
            metrics["segm_AP75"] = 0.0
            continue
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # stats[1] = AP@IoU=0.50,  stats[2] = AP@IoU=0.75
        metrics[f"{iou_type}_AP50"] = float(ev.stats[1])
        metrics[f"{iou_type}_AP75"] = float(ev.stats[2])
 
    return metrics
 
 
# ---------------------------------------------------------------------------
# Word-level Precision / Recall / F1
# ---------------------------------------------------------------------------
 
def _box_iou(a, b):
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0
 
 
def compute_word_metrics(coco_gt, results, iou_threshold):
    preds_by_image = {}
    for r in results:
        preds_by_image.setdefault(r["image_id"], []).append(r)
    for img_id in preds_by_image:
        preds_by_image[img_id].sort(key=lambda x: x["score"], reverse=True)
 
    total_tp = total_fp = total_fn = 0
 
    for img_id in coco_gt.getImgIds():
        anns     = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_boxes = [[a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2],
                     a["bbox"][1] + a["bbox"][3]] for a in anns]
        matched  = [False] * len(gt_boxes)
 
        preds = preds_by_image.get(img_id, [])
        pred_boxes = [[p["bbox"][0], p["bbox"][1],
                       p["bbox"][0] + p["bbox"][2],
                       p["bbox"][1] + p["bbox"][3]] for p in preds]
 
        tp = fp = 0
        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if matched[gi]:
                    continue
                iou = _box_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched[best_idx] = True
            else:
                fp += 1
 
        total_tp += tp
        total_fp += fp
        total_fn += matched.count(False)
 
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
 
    return round(precision, 4), round(recall, 4), round(f1, 4)
 
 
# ---------------------------------------------------------------------------
# Evaluate one split and print results
# ---------------------------------------------------------------------------
 
def evaluate_split(split_name, images_dir, json_path,
                   predictor, iou_threshold):
 
    print(f"\n[eval] Starting {split_name} set evaluation...")
    coco_gt = COCO(json_path)
    results = run_inference(predictor, coco_gt, images_dir)
    coco_m  = compute_coco_metrics(coco_gt, results)
    precision, recall, f1 = compute_word_metrics(
        coco_gt, results, iou_threshold
    )
 
    print(f"\n{'='*45}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*45}")
    print(f"  BBox  AP50  :  {coco_m['bbox_AP50']:.4f}")
    print(f"  BBox  AP75  :  {coco_m['bbox_AP75']:.4f}")
    print(f"  Segm  AP50  :  {coco_m['segm_AP50']:.4f}")
    print(f"  Segm  AP75  :  {coco_m['segm_AP75']:.4f}")
    print(f"  {'─'*35}")
    print(f"  Precision   :  {precision:.4f}")
    print(f"  Recall      :  {recall:.4f}")
    print(f"  F1-score    :  {f1:.4f}")
    print(f"{'='*45}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
 
    for label, path in [
        ("--model",     args.model),
        ("--images",    args.images),
        ("--val_json",  args.val_json),
        ("--test_json", args.test_json),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
 
    load_swint_repo(args.swint_repo)
 
    print(f"\nLoading model: {args.model}")
    predictor = DefaultPredictor(
        build_cfg(args.swint_repo, args.model,
                  args.score_threshold, args.device)
    )
    print("Model loaded.")
 
    evaluate_split("validation", args.images, args.val_json,
                   predictor, args.iou_threshold)
 
    evaluate_split("test", args.images, args.test_json,
                   predictor, args.iou_threshold)
 
 
if __name__ == "__main__":
    main()
