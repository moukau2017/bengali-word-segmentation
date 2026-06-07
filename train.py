"""
train.py

Training script for Bengali handwritten word segmentation using
Swin Transformer (Swin-S) + Mask R-CNN.

This script is the original training notebook. 

Expected dataset format:
- Images folder
- COCO-format annotation JSON
- Single category: "word"

Example usage:

python train.py \
    --train_images /path/to/train/images \
    --train_json /path/to/train_annotations.json \
    --val_images /path/to/val/images \
    --val_json /path/to/val_annotations.json \
    --output_dir outputs/training \
    --swint_repo SwinT_detectron2 \
    --pretrained_weights models/swin_small_patch4_window7_224_d2.pth
"""

import argparse
import importlib.util
import os
import pathlib
import re
import sys

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import launch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Swin-S + Mask R-CNN for Bengali handwritten word segmentation"
    )

    parser.add_argument(
        "--train_images",
        required=True,
        help="Path to training images"
    )

    parser.add_argument(
        "--train_json",
        required=True,
        help="Path to training COCO annotation JSON"
    )

    parser.add_argument(
        "--val_images",
        required=True,
        help="Path to validation images"
    )

    parser.add_argument(
        "--val_json",
        required=True,
        help="Path to validation COCO annotation JSON"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/training",
        help="Directory to save checkpoints and logs"
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
        swint_repo,
        "configs",
        "SwinT",
        "mask_rcnn_swint_T_FPN_3x.yaml"
    )

    dst_cfg = os.path.join(
        swint_repo,
        "configs",
        "SwinT",
        "mask_rcnn_swins_S_FPN_3x.yaml"
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
        text
    )

    pathlib.Path(dst_cfg).write_text(text)

    return dst_cfg


def register_datasets(train_images, train_json, val_images, val_json):
    train_name = "hw_bengali_train"
    val_name = "hw_bengali_val"

    for name in [train_name, val_name]:
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    register_coco_instances(train_name, {}, train_json, train_images)
    register_coco_instances(val_name, {}, val_json, val_images)

    return train_name, val_name


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_net = load_train_net(args.swint_repo)
    config_file = ensure_swins_config(args.swint_repo)

    train_name, val_name = register_datasets(
        args.train_images,
        args.train_json,
        args.val_images,
        args.val_json
    )

    if not os.path.exists(args.pretrained_weights) and not args.resume:
        raise FileNotFoundError(
            f"Pretrained weights not found: {args.pretrained_weights}\n"
            "Please provide converted Swin-S Detectron2 weights using --pretrained_weights."
        )

    cli_args = [
        "--config-file", config_file,
        "--num-gpus", str(args.num_gpus),

        "OUTPUT_DIR", args.output_dir,

        "MODEL.WEIGHTS", args.pretrained_weights,

        # Swin-S backbone
        "MODEL.SWINT.DEPTHS", "(2, 2, 18, 2)",

        # Datasets
        "DATASETS.TRAIN", f"('{train_name}',)",
        "DATASETS.TEST", f"('{val_name}',)",

        # Single class: word
        "MODEL.ROI_HEADS.NUM_CLASSES", "1",

        # Image resizing
        "INPUT.MAX_SIZE_TRAIN", "1333",
        "INPUT.MIN_SIZE_TRAIN", "(640, 672, 704, 736, 768, 800)",
        "INPUT.MAX_SIZE_TEST", "1333",
        "INPUT.MIN_SIZE_TEST", "800",

        # Optimizer and schedule
        "SOLVER.IMS_PER_BATCH", "2",
        "SOLVER.BASE_LR", "0.0001",
        "SOLVER.MAX_ITER", "5000",
        "SOLVER.STEPS", "(12000, 14000)",
        "SOLVER.GAMMA", "0.1",

        # Warmup
        "SOLVER.WARMUP_ITERS", "200",
        "SOLVER.WARMUP_FACTOR", "0.001",

        # ROI pooling resolution
        "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION", "14",
        "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION", "14",

        # Checkpointing and evaluation
        "SOLVER.CHECKPOINT_PERIOD", "3000",
        "TEST.EVAL_PERIOD", "3000",

        # Dataloader
        "DATALOADER.NUM_WORKERS", "2",
    ]

    if args.resume:
        cli_args.insert(3, "--resume")

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
