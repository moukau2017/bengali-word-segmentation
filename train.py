"""
train.py

Training script for Bengali handwritten word segmentation using
Swin Transformer (Swin-S) + Mask R-CNN.

This script automatically splits a single annotated dataset into
train / val / test subsets (83% / 10% / 7% by default) before training.

Expected dataset format:
- Images folder
- Single COCO-format annotation JSON
- Single category: "word"

Example usage (auto-split from one annotated set):

    python train.py \
        --images      /path/to/all/images \
        --json        /path/to/annotations.json \
        --output_dir  outputs/training \
        --swint_repo  SwinT_detectron2 \
        --pretrained_weights models/swin_small_patch4_window7_224_d2.pth

"""

import argparse
import importlib.util
import json
import os
import pathlib
import random
import re
import sys

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Swin-S + Mask R-CNN for Bengali handwritten word segmentation"
    )

    split_group = parser.add_argument_group("Auto-split mode")
    split_group.add_argument("--images",     default=None)
    split_group.add_argument("--json",       default=None)
    split_group.add_argument("--val_ratio",  type=float, default=0.10)
    split_group.add_argument("--test_ratio", type=float, default=0.07)
    split_group.add_argument("--seed",       type=int,   default=42)

    manual_group = parser.add_argument_group("Manual / pre-split mode")
    manual_group.add_argument("--train_images", default=None)
    manual_group.add_argument("--train_json",   default=None)
    manual_group.add_argument("--val_images",   default=None)
    manual_group.add_argument("--val_json",     default=None)

    parser.add_argument("--output_dir",          default="outputs/training")
    parser.add_argument("--swint_repo",          default="SwinT_detectron2")
    parser.add_argument("--pretrained_weights",  default="models/swin_small_patch4_window7_224_d2.pth")
    parser.add_argument("--num_gpus",            type=int, default=1)
    parser.add_argument("--resume",              action="store_true")

    # Training schedule — exposed as CLI args so nothing is hardcoded
    parser.add_argument("--max_iter",         type=int, default=5000)
    parser.add_argument("--checkpoint_every", type=int, default=2500)
    parser.add_argument("--eval_every",       type=int, default=2500)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_coco_annotations(images_dir, coco_json_path, output_dir,
                            val_ratio=0.10, test_ratio=0.07, seed=42):
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0"
        )

    with open(coco_json_path) as f:
        coco = json.load(f)

    all_images = coco["images"]
    all_anns   = coco["annotations"]
    categories = coco.get("categories", [])
    info       = coco.get("info",       {})
    licenses   = coco.get("licenses",   [])

    rng = random.Random(seed)
    shuffled = all_images[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_val   = max(1, round(n * val_ratio))
    n_test  = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Dataset too small ({n} images) to create non-empty train split."
        )

    val_imgs   = shuffled[:n_val]
    test_imgs  = shuffled[n_val:n_val + n_test]
    train_imgs = shuffled[n_val + n_test:]

    print(f"\n[split] Total: {n}  →  train: {len(train_imgs)}  |  "
          f"val: {len(val_imgs)}  |  test: {len(test_imgs)}")

    id_to_split = {}
    for img in train_imgs: id_to_split[img["id"]] = "train"
    for img in val_imgs:   id_to_split[img["id"]] = "val"
    for img in test_imgs:  id_to_split[img["id"]] = "test"

    train_anns, val_anns, test_anns = [], [], []
    for ann in all_anns:
        s = id_to_split.get(ann["image_id"])
        if   s == "train": train_anns.append(ann)
        elif s == "val":   val_anns.append(ann)
        elif s == "test":  test_anns.append(ann)

    def _make_coco(imgs, anns):
        return {"info": info, "licenses": licenses,
                "categories": categories, "images": imgs, "annotations": anns}

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "train": os.path.join(output_dir, "train_annotations.json"),
        "val":   os.path.join(output_dir, "val_annotations.json"),
        "test":  os.path.join(output_dir, "test_annotations.json"),
    }

    with open(paths["train"], "w") as f: json.dump(_make_coco(train_imgs, train_anns), f)
    with open(paths["val"],   "w") as f: json.dump(_make_coco(val_imgs,   val_anns),   f)
    with open(paths["test"],  "w") as f: json.dump(_make_coco(test_imgs,  test_anns),  f)

    summary = {
        "seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio,
        "n_total": n, "n_train": len(train_imgs),
        "n_val": len(val_imgs), "n_test": len(test_imgs),
        "train_image_ids": [i["id"] for i in train_imgs],
        "val_image_ids":   [i["id"] for i in val_imgs],
        "test_image_ids":  [i["id"] for i in test_imgs],
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[split] JSONs written to: {output_dir}\n")
    return paths["train"], paths["val"], paths["test"]


# ---------------------------------------------------------------------------
# SwinT helpers
# ---------------------------------------------------------------------------

def load_train_net(swint_repo):
    train_net_path = os.path.join(swint_repo, "train_net.py")
    if not os.path.exists(train_net_path):
        raise FileNotFoundError(f"train_net.py not found at: {train_net_path}")

    sys.path.insert(0, swint_repo)
    spec = importlib.util.spec_from_file_location("train_net", train_net_path)
    train_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_net)
    return train_net


def ensure_swins_config(swint_repo):
    src = os.path.join(swint_repo, "configs", "SwinT", "mask_rcnn_swint_T_FPN_3x.yaml")
    dst = os.path.join(swint_repo, "configs", "SwinT", "mask_rcnn_swins_S_FPN_3x.yaml")

    if os.path.exists(dst):
        return dst

    if not os.path.exists(src):
        raise FileNotFoundError(f"Swin-T config not found at: {src}")

    text = pathlib.Path(src).read_text()
    text = re.sub(
        r"(DEPTHS\s*:\s*\[)\s*2\s*,\s*2\s*,\s*6\s*,\s*2\s*(\])",
        r"\g<1>2, 2, 18, 2\g<2>",
        text,
    )
    pathlib.Path(dst).write_text(text)

    # Specific check — "18" alone could match anything in the YAML
    if "2, 2, 18, 2" not in pathlib.Path(dst).read_text():
        raise RuntimeError(
            "Swin-S config generation failed — DEPTHS substitution did not apply.\n"
            f"Please manually set DEPTHS: [2, 2, 18, 2] in: {dst}"
        )

    return dst


# ---------------------------------------------------------------------------
# Dataset registration
# ---------------------------------------------------------------------------

def register_datasets(train_images, train_json, val_images, val_json):
    train_name = "hw_bengali_train"
    val_name   = "hw_bengali_val"

    for name in [train_name, val_name]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name,   {}, val_json,   val_images)
    return train_name, val_name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    auto_split_mode = args.images is not None and args.json is not None
    manual_mode     = all([args.train_images, args.train_json,
                           args.val_images,   args.val_json])

    if not auto_split_mode and not manual_mode:
        raise ValueError(
            "Provide either --images + --json  OR  "
            "--train_images + --train_json + --val_images + --val_json"
        )

    if auto_split_mode and manual_mode:
        print("[warn] Both modes supplied — manual (pre-split) paths take precedence.")
        auto_split_mode = False

    if auto_split_mode:
        split_dir = os.path.join(args.output_dir, "splits")
        train_json, val_json, _ = split_coco_annotations(
            images_dir=args.images, coco_json_path=args.json,
            output_dir=split_dir,   val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, seed=args.seed,
        )
        train_images_dir = val_images_dir = args.images
    else:
        train_images_dir = args.train_images
        train_json       = args.train_json
        val_images_dir   = args.val_images
        val_json         = args.val_json

    train_net   = load_train_net(args.swint_repo)
    config_file = ensure_swins_config(args.swint_repo)

    train_name, val_name = register_datasets(
        train_images_dir, train_json, val_images_dir, val_json
    )

    if not args.resume and not os.path.exists(args.pretrained_weights):
        raise FileNotFoundError(
            f"Pretrained weights not found: {args.pretrained_weights}"
        )

    # Scale LR steps relative to max_iter — never hardcoded
    steps  = f"({int(args.max_iter * 0.80)}, {int(args.max_iter * 0.95)})"
    warmup = str(max(10, int(args.max_iter * 0.05)))

    # ── Flags first, then config overrides ────────────────────────────────
    cli_args = ["--config-file", config_file, "--num-gpus", str(args.num_gpus)]
    if args.resume:
        cli_args.append("--resume")

    cli_args += [
        "OUTPUT_DIR",    args.output_dir,

        "MODEL.SWINT.DEPTHS", "(2, 2, 18, 2)",

        # Fix: double quotes inside the tuple string
        "DATASETS.TRAIN", f'("{train_name}",)',
        "DATASETS.TEST",  f'("{val_name}",)',

        "MODEL.ROI_HEADS.NUM_CLASSES", "1",

        "INPUT.MAX_SIZE_TRAIN", "1333",
        "INPUT.MIN_SIZE_TRAIN", "(640, 672, 704, 736, 768, 800)",
        "INPUT.MAX_SIZE_TEST",  "1333",
        "INPUT.MIN_SIZE_TEST",  "800",

        "SOLVER.IMS_PER_BATCH", "2",
        "SOLVER.BASE_LR",       "0.0001",
        "SOLVER.MAX_ITER",      str(args.max_iter),
        "SOLVER.STEPS",         steps,
        "SOLVER.GAMMA",         "0.1",
        "SOLVER.WARMUP_ITERS",  warmup,
        "SOLVER.WARMUP_FACTOR", "0.001",

        "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION",  "14",
        "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION", "14",

        "SOLVER.CHECKPOINT_PERIOD", str(args.checkpoint_every),
        "TEST.EVAL_PERIOD",         str(args.eval_every),

        "DATALOADER.NUM_WORKERS", "2",
    ]

    if not args.resume:
        # Insert after OUTPUT_DIR — before config overrides that need weights
        cli_args.insert(cli_args.index("MODEL.SWINT.DEPTHS"),
                        "MODEL.WEIGHTS")
        cli_args.insert(cli_args.index("MODEL.SWINT.DEPTHS"),
                        args.pretrained_weights)

    d2_args = train_net.default_argument_parser().parse_args(cli_args)

    launch(
        train_net.main,
        d2_args.num_gpus,
        num_machines=d2_args.num_machines,
        machine_rank=d2_args.machine_rank,
        dist_url=d2_args.dist_url,
        args=(d2_args,),
    )


if __name__ == "__main__":
    main()
