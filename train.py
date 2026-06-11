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

    # ── Auto-split mode (recommended) ──────────────────────────────────────
    split_group = parser.add_argument_group(
        "Auto-split mode",
        "Provide a single images folder + annotation JSON; the script will "
        "randomly split them into train / val / test."
    )
    split_group.add_argument(
        "--images",
        default=None,
        help="Path to the folder containing ALL images (auto-split mode)"
    )
    split_group.add_argument(
        "--json",
        default=None,
        help="Path to the single COCO annotation JSON (auto-split mode)"
    )
    split_group.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Fraction of images used for validation (default: 0.10)"
    )
    split_group.add_argument(
        "--test_ratio",
        type=float,
        default=0.07,
        help="Fraction of images used for testing (default: 0.07)"
    )
    split_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )

    # ── Manual / pre-split mode ────────────────────────────────────────────
    manual_group = parser.add_argument_group(
        "Manual / pre-split mode",
        "Supply your own pre-split image folders and annotation JSONs."
    )
    manual_group.add_argument("--train_images", default=None,
                              help="Path to training images")
    manual_group.add_argument("--train_json",   default=None,
                              help="Path to training COCO annotation JSON")
    manual_group.add_argument("--val_images",   default=None,
                              help="Path to validation images")
    manual_group.add_argument("--val_json",     default=None,
                              help="Path to validation COCO annotation JSON")

    # ── Shared options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        default="outputs/training",
        help="Directory to save checkpoints, logs, and split JSONs"
    )
    parser.add_argument(
        "--swint_repo",
        default="SwinT_detectron2",
        help="Path to SwinT Detectron2 repository/folder"
    )
    parser.add_argument(
        "--pretrained_weights",
        default="models/swin_small_patch4_window7_224_d2.pth",
        help="Path to converted Swin-S Detectron2 pretrained weights"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in output_dir"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_coco_annotations(
    images_dir: str,
    coco_json_path: str,
    output_dir: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.07,
    seed: int = 42,
):
    """
    Randomly split a COCO-format annotation file into train / val / test subsets.

    Splitting is done at the *image* level so that no image appears in more
    than one subset.  All annotations that belong to an image follow it into
    the same subset.

    Parameters
    ----------
    images_dir     : folder that contains all the image files
    coco_json_path : path to the original COCO annotation JSON
    output_dir     : folder where the three split JSONs will be written
    val_ratio      : fraction of images for validation  (e.g. 0.10)
    test_ratio     : fraction of images for testing     (e.g. 0.07)
    seed           : random seed for reproducibility

    Returns
    -------
    (train_json, val_json, test_json) – absolute paths to the written JSONs
    """

    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0"
        )

    with open(coco_json_path) as f:
        coco = json.load(f)

    all_images = coco["images"]          # list of image dicts
    all_anns   = coco["annotations"]     # list of annotation dicts
    categories = coco.get("categories", [])
    info       = coco.get("info",       {})
    licenses   = coco.get("licenses",   [])

    # Shuffle images with a fixed seed for reproducibility
    rng = random.Random(seed)
    shuffled = all_images[:]
    rng.shuffle(shuffled)

    n          = len(shuffled)
    n_val      = max(1, round(n * val_ratio))
    n_test     = max(1, round(n * test_ratio))
    n_train    = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Dataset too small ({n} images) to create non-empty train split "
            f"with val_ratio={val_ratio} and test_ratio={test_ratio}."
        )

    val_imgs   = shuffled[:n_val]
    test_imgs  = shuffled[n_val:n_val + n_test]
    train_imgs = shuffled[n_val + n_test:]

    print(
        f"\n[split] Total: {n} images  →  "
        f"train: {len(train_imgs)}  |  "
        f"val: {len(val_imgs)}  |  "
        f"test: {len(test_imgs)}"
    )

    # Build a lookup: image_id → subset name (for filtering annotations)
    id_to_split = {}
    for img in train_imgs: id_to_split[img["id"]] = "train"
    for img in val_imgs:   id_to_split[img["id"]] = "val"
    for img in test_imgs:  id_to_split[img["id"]] = "test"

    # Split annotations
    train_anns, val_anns, test_anns = [], [], []
    for ann in all_anns:
        split = id_to_split.get(ann["image_id"])
        if split == "train": train_anns.append(ann)
        elif split == "val":   val_anns.append(ann)
        elif split == "test":  test_anns.append(ann)

    def _make_coco(imgs, anns):
        return {
            "info":        info,
            "licenses":    licenses,
            "categories":  categories,
            "images":      imgs,
            "annotations": anns,
        }

    os.makedirs(output_dir, exist_ok=True)

    train_json_path = os.path.join(output_dir, "train_annotations.json")
    val_json_path   = os.path.join(output_dir, "val_annotations.json")
    test_json_path  = os.path.join(output_dir, "test_annotations.json")

    with open(train_json_path, "w") as f:
        json.dump(_make_coco(train_imgs, train_anns), f)

    with open(val_json_path, "w") as f:
        json.dump(_make_coco(val_imgs, val_anns), f)

    with open(test_json_path, "w") as f:
        json.dump(_make_coco(test_imgs, test_anns), f)

    # Also write a small split-summary for traceability
    summary = {
        "seed":        seed,
        "val_ratio":   val_ratio,
        "test_ratio":  test_ratio,
        "n_total":     n,
        "n_train":     len(train_imgs),
        "n_val":       len(val_imgs),
        "n_test":      len(test_imgs),
        "train_image_ids": [img["id"] for img in train_imgs],
        "val_image_ids":   [img["id"] for img in val_imgs],
        "test_image_ids":  [img["id"] for img in test_imgs],
    }
    summary_path = os.path.join(output_dir, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[split] JSONs written to: {output_dir}")
    print(f"        train → {train_json_path}")
    print(f"        val   → {val_json_path}")
    print(f"        test  → {test_json_path}")
    print(f"        summary → {summary_path}\n")

    return train_json_path, val_json_path, test_json_path


# ---------------------------------------------------------------------------
# SwinT helpers
# ---------------------------------------------------------------------------

def load_train_net(swint_repo):
    train_net_path = os.path.join(swint_repo, "train_net.py")

    if not os.path.exists(train_net_path):
        raise FileNotFoundError(
            f"Could not find train_net.py at: {train_net_path}\n"
            "Please clone/place SwinT_detectron2 in the repository root."
        )

    sys.path.insert(0, swint_repo)

    spec = importlib.util.spec_from_file_location("train_net", train_net_path)
    train_net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_net)

    return train_net


def ensure_swins_config(swint_repo):
    """Create Swin-S config from Swin-T config if Swin-S config is missing."""

    src_cfg = os.path.join(
        swint_repo, "configs", "SwinT", "mask_rcnn_swint_T_FPN_3x.yaml"
    )
    dst_cfg = os.path.join(
        swint_repo, "configs", "SwinT", "mask_rcnn_swins_S_FPN_3x.yaml"
    )

    if os.path.exists(dst_cfg):
        return dst_cfg

    if not os.path.exists(src_cfg):
        raise FileNotFoundError(
            f"Could not find source Swin-T config at: {src_cfg}"
        )

    text = pathlib.Path(src_cfg).read_text()
    text = re.sub(
        r"(DEPTHS\s*:\s*\[)\s*2\s*,\s*2\s*,\s*6\s*,\s*2\s*(\])",
        r"\g<1>2, 2, 18, 2\g<2>",
        text,
    )
    pathlib.Path(dst_cfg).write_text(text)

    # Validate that depth substitution actually succeeded.
    if "18" not in pathlib.Path(dst_cfg).read_text():
        raise RuntimeError(
            "Swin-S config generation failed — depth substitution did not apply.\n"
            f"Please manually verify or edit: {dst_cfg}\n"
            "Expected DEPTHS: [2, 2, 18, 2] for Swin-S."
        )

    return dst_cfg


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

    # ── Determine image/annotation paths ──────────────────────────────────
    auto_split_mode = args.images is not None and args.json is not None
    manual_mode     = (
        args.train_images is not None
        and args.train_json is not None
        and args.val_images is not None
        and args.val_json   is not None
    )

    if not auto_split_mode and not manual_mode:
        raise ValueError(
            "You must provide either:\n"
            "  Auto-split:  --images <dir> --json <file>\n"
            "  Manual:      --train_images <dir> --train_json <file> "
            "--val_images <dir> --val_json <file>"
        )

    if auto_split_mode and manual_mode:
        print(
            "[warn] Both auto-split and manual arguments were supplied. "
            "Manual (pre-split) paths take precedence."
        )
        auto_split_mode = False

    if auto_split_mode:
        split_dir = os.path.join(args.output_dir, "splits")
        train_json, val_json, _ = split_coco_annotations(
            images_dir     = args.images,
            coco_json_path = args.json,
            output_dir     = split_dir,
            val_ratio      = args.val_ratio,
            test_ratio     = args.test_ratio,
            seed           = args.seed,
        )
        # Images live in the same folder for all three subsets
        train_images_dir = args.images
        val_images_dir   = args.images
    else:
        train_images_dir = args.train_images
        train_json       = args.train_json
        val_images_dir   = args.val_images
        val_json         = args.val_json

    # ── Load SwinT repo & config ───────────────────────────────────────────
    train_net   = load_train_net(args.swint_repo)
    config_file = ensure_swins_config(args.swint_repo)

    # ── Register datasets ──────────────────────────────────────────────────
    train_name, val_name = register_datasets(
        train_images_dir, train_json,
        val_images_dir,   val_json,
    )

    if not os.path.exists(args.pretrained_weights) and not args.resume:
        raise FileNotFoundError(
            f"Pretrained weights not found: {args.pretrained_weights}\n"
            "Please provide converted Swin-S Detectron2 weights via "
            "--pretrained_weights."
        )

    # ── Build Detectron2 CLI args ──────────────────────────────────────────
    cli_args = [
        "--config-file", config_file,
        "--num-gpus",    str(args.num_gpus),

        "OUTPUT_DIR",    args.output_dir,
    ]

    # When resuming, Detectron2 loads weights from the last checkpoint
    # automatically. Passing MODEL.WEIGHTS to a non-existent pretrained file
    # would cause a crash, so we skip it on resume.
    if not args.resume:
        cli_args += ["MODEL.WEIGHTS", args.pretrained_weights]

    cli_args += [

        # Explicitly override depths to Swin-S in case config file differs
        "MODEL.SWINT.DEPTHS", "(2, 2, 18, 2)",

        # Datasets
        "DATASETS.TRAIN", f"('{train_name}',)",
        "DATASETS.TEST",  f"('{val_name}',)",

        # Single class: word
        "MODEL.ROI_HEADS.NUM_CLASSES", "1",

        # Image resizing
        "INPUT.MAX_SIZE_TRAIN", "1333",
        "INPUT.MIN_SIZE_TRAIN", "(640, 672, 704, 736, 768, 800)",
        "INPUT.MAX_SIZE_TEST",  "1333",
        "INPUT.MIN_SIZE_TEST",  "800",

        # Optimizer and schedule
        "SOLVER.IMS_PER_BATCH", "2",
        "SOLVER.BASE_LR",       "0.0001",
        "SOLVER.MAX_ITER",      "5000",
        "SOLVER.STEPS",         "(3000, 4000)",
        "SOLVER.GAMMA",         "0.1",

        # Warmup
        "SOLVER.WARMUP_ITERS",  "200",
        "SOLVER.WARMUP_FACTOR", "0.001",

        # ROI pooling resolution
        "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION",  "14",
        "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION", "14",

        # Checkpointing and evaluation
        "SOLVER.CHECKPOINT_PERIOD", "2500",
        "TEST.EVAL_PERIOD",         "2500",

        # Dataloader
        "DATALOADER.NUM_WORKERS", "2",
    ]

    # --resume must be appended at the end (not inserted mid-list) to avoid
    # breaking positional argument parsing.
    if args.resume:
        cli_args.append("--resume")

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
