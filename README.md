# bengali-word-segmentation
## SwinWordSeg: Multi-Scale Word Segmentation from Handwritten Bengali Answer-book Images



# Key Features

- Handwritten word segmentation for Bengali handwritten images, using Swin Transformer and Mask R-CNN.
- Handles challenging scenarios such as varying handwriting styles, irregular spacing, touching words, superscripts, subscripts, and noise artifacts.
- Trained and evaluated on a manually annotated dataset collected from multiple Bengali-medium schools.
- Evaluated using standard COCO metrics (AP50 and AP75) along with word-level Precision, Recall, and F1-score.
- Visualization of input and segmentation outputs.
- Trained model, test set, and corresponding annotations are provided with evaluation and visualization pipelines for verification
- Provided the training pipeline and code for reproducibility and further research
---

# 📄 Citation & Resources

### Repository

https://github.com/moukau2017/bengali-word-segmentation

---


  # 🚀 Quick Start

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch
- Detectron2


# 📁 Project Structure

bengali-word-segmentation/
│
├── models/
│   └── .gitkeep
│
├── outputs/
│   └── example_predictions/
│
├── evaluate.py
├── inference_visualize.py
├── requirements.txt
├── README.md
└── LICENSE



# 🖼 Inference & Visualization

## The evaluation script reports:
bounding Box, segmentation and Word Detection Metrics

- AP50
- AP75
- Precision
- Recall
- F1-score
  
Run evaluation:
python evaluate.py \
    --model models/final_model.pth \
    --images /path/to/validation_images \
    --annotations /path/to/val_annotations.json



Generate segmentation visualizations:

python inference_visualize.py \
    --model models/final_model.pth \
    --images /path/to/validation_images

Prediction visualizations will be saved in:
outputs/


## Pretrained Model

The trained model weights can be downloaded from Google Drive:

Download Model

https://drive.google.com/file/d/1HKPQ8I7f7mZpVFkVkZtFhCyQ-7CdA4Vj/view?usp=sharing

Note: The model file is hosted externally due to GitHub file-size limitations.

After downloading, place the model in:
models/final_model.pth



## Validation Dataset

The validation images and COCO annotation file can be downloaded from:

Download Validation Dataset

https://drive.google.com/drive/folders/1yCpxaiwM3vrg5Tl-Ks6tuR0dRnHwxJVM?usp=sharing

Contents:
Validation images (30 images)
COCO annotation file (val_annotations.json)

# 🔬 Reproducibility

The repository provides:

- Trained model weights
- Validation dataset
- COCO annotations
- Evaluation script
- Inference visualization script

allowing independent verification of the model.

## Training

The training script used in this study is provided for transparency and reproducibility.

The in-house training dataset is not publicly available due to privacy and institutional restrictions, as it contains handwritten student answer-book images collected from multiple Bengali-medium schools. The dataset may be made available for research purposes upon reasonable request to the authors.

Researchers may use their own dataset following the same COCO annotation format. Image annotation tool VGG image annotator.
python train.py \
  --train_images /path/to/train/images \
  --train_json /path/to/train_annotations.json \
  --val_images /path/to/validation/images \
  --val_json /path/to/val_annotations.json \
  --output_dir outputs/training


# 📜 License

This repository is released under the MIT License.


# 🙏 Acknowledgments

- Detectron2
- Swin Transformer
- COCO API
- VIA

---





