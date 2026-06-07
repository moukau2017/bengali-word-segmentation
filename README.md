# bengali-word-segmentation
SwinWordSeg: Multi-Scale Word Segmentation from Handwritten Bengali Answer-book Images
# 🔬 Abstract
Word segmentation is a crucial phase for handwritten text recognition (HTR). When the input image contains multiple words written across several text lines, it is crucial to accurately segment individual words for precise text recognition. This task is challenging due to variations in handwriting styles and script-specific complexities, including differing stroke patterns, irregular spacing, inconsistent word shapes, and various noise elements. Segmenting handwritten school students' pages becomes more challenging. This study introduces an instance segmentation framework based on Mask R-CNN integrated with Swin Transformer, optimized for word segmentation from Bengali handwritten answer scripts. To train and test the proposed system, answer books from middle school students across multiple Bengali-medium schools were collected and annotated. The proposed model is compared with two widely used architectures: U-Net and Mask R-CNN with a ResNet backbone. Experimental results demonstrate that the proposed model outperformed those. The model was found to be superior to multiple pre-trained computer vision and large-language models, as tested on a word segmentation task using the same dataset. The model was also tested on publicly available handwritten Bengali and Devanagari datasets, demonstrating its robustness.


# ✨ Key Features

- Handwritten word segmentation using Swin Transformer and Mask R-CNN
- Designed for Bengali handwritten school answer-book images containing dense and irregular handwriting
- Handles challenging scenarios such as varying handwriting styles, irregular spacing, touching words, superscripts, subscripts, and noise artifacts
- Trained and evaluated on a manually annotated dataset collected from multiple Bengali-medium schools
- Demonstrates superior performance
- Evaluated using standard COCO metrics (AP50 and AP75) along with word-level Precision, Recall, and F1-score
- Visualization of segmentation outputs
- Provides reproducible evaluation and visualization pipelines for independent verification
- Provided trained model, test set and corresoponding annotation for model performance varification
- Supports COCO-format annotations


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



# 📊 Evaluation

The evaluation script reports:
## Bounding Box Metrics
- AP50
- AP75
## Segmentation Metrics
- AP50
- AP75
## Word Detection Metrics
- Precision
- Recall
- F1-score
Run evaluation:

python evaluate.py \
    --model models/final_model.pth \
    --images /path/to/validation_images \
    --annotations /path/to/val_annotations.json


# 🖼 Inference & Visualization
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



# 📂 Validation Dataset

The validation images and COCO annotation file can be downloaded from:

Download Validation Dataset

https://drive.google.com/drive/folders/1yCpxaiwM3vrg5Tl-Ks6tuR0dRnHwxJVM?usp=sharing

Contents:
Validation images (30 images)
COCO annotation file (val_annotations.json)

# 🔬 Reproducibility

The repository provides:

Trained model weights
Validation dataset
COCO annotations
Evaluation script
Inference visualization script

allowing independent verification of the model.
---
## Training

The training script used in this study is provided for transparency and reproducibility.

The original training dataset is not publicly distributed because it consists of handwritten student answer-book images collected from multiple Bengali-medium schools.

Researchers may use their own dataset following the same COCO annotation format. Annotated images from VGG image annotator.


# 📜 License

This repository is released under the MIT License.


# 🙏 Acknowledgments

- Detectron2
- Swin Transformer
- COCO API

---





