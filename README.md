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

## Installation

Clone repository

```bash
git clone https://github.com/moukau2017/bengali-word-segmentation.git
cd bengali-word-segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install Detectron2

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

---

# 📁 Project Structure

```text
bengali-word-segmentation/
│
├── models/
│   └── final_model.pth
│
├── sample_data/
│   ├── images/
│   └── annotations/
│       └── val_annotations.json
│
├── scripts/
│   ├── evaluate.py
│   ├── inference_visualize.py
│   └── via_to_coco_converter.py
│
├── notebooks/
│   └── evaluation_notebook.ipynb
│
├── outputs/
│   └── predictions/
│
├── requirements.txt
└── README.md



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
```bash
python scripts/evaluate.py
```

---
# 🖼 Inference & Visualization

## Pretrained Model

Download the trained model from:

(https://drive.google.com/file/d/1HKPQ8I7f7mZpVFkVkZtFhCyQ-7CdA4Vj/view?usp=sharing)

Use the model in inference.py 
Test on validation images with annotation

Generate segmentation visualizations:

```bash
python scripts/inference_visualize.py
```

Predicted images are saved in:

```text
outputs/predictions/
```

---

# 📂 Dataset

The model is evaluated on handwritten Bengali answer-book images.

### Validation Dataset

- Images: 30
- Annotation format: COCO JSON
- Task: Word-level instance segmentation

Dataset structure:

```text
sample_data/
├── images/
└── annotations/
    └── val_annotations.json
```
# 🔬 Reproducibility

The repository provides:

- Trained model weights
- Evaluation scripts
- Sample validation dataset
- Visualization utilities

---

# 📜 License

This repository is released under the MIT License.


# 🙏 Acknowledgments

- Detectron2
- Swin Transformer
- COCO API

---





