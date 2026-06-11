# bengali-word-segmentation
## Word Segmentation from Handwritten Bengali Answer-book Images



### Key Features

- Handwritten word segmentation for Bengali handwritten images, using Swin Transformer and Mask R-CNN.
- Handles challenging scenarios such as varying handwriting styles, irregular spacing, touching words, superscripts, subscripts, and noise artifacts.
- Trained and evaluated on a manually annotated dataset collected from multiple Bengali-medium schools.
- Evaluated using standard COCO metrics (AP50 and AP75) along with word-level Precision, Recall, and F1-score.
- Visualization of input and segmentation outputs.
- Trained model, test set, and corresponding annotations are provided with evaluation and visualization pipelines for verification
- Provided the training pipeline and code for reproducibility and further research
---

## Citation & Resources

### Repository

https://github.com/moukau2017/bengali-word-segmentation

---


  ## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch
- Detectron2
  
### Clone SwinT_detectron2
git clone https://github.com/xiaohu2015/SwinT_detectron2.git

### Clone SwinT_detectron2
git clone https://github.com/xiaohu2015/SwinT_detectron2.git

### Install dependencies
pip install -r requirements.txt

### Download pretrained Swin-S weights
Download the Detectron2-converted Swin-S weights and place them at models/swin_small_patch4_window7_224_d2.pth:



## Part A -- Quick prediction with trained model Model

Our trained model weights can be downloaded from Google Drive: Quick Prediction on Sample Images

Use the provided trained model and sample images to see results immediately. No training required.

Download Model

https://drive.google.com/file/d/1HKPQ8I7f7mZpVFkVkZtFhCyQ-7CdA4Vj/view?usp=sharing

plae it at:

models/final_model.pth

After download the trained model, use Sample Images in:

sample_input/

Run prediction

python predict.py \
    --model   models/final_model.pth \
    --images  sample_input/

Results are saved to outputs/predictions/ 

with coloured segmentation masks, bounding box and confidence scores on each word.



## Path B — Full Reproducibility (Train → Evaluate → Visualize)

Download the dataset first:

Dataset link: 

Place files as:
data/
├── images/           ← all images
└── annotations.json  ← single COCO-format annotation file

Download and convert pretrained Swin-S backbone weights

# Download original Swin-S ImageNet weights

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth

# Convert to Detectron2 format
mkdir -p models

cd SwinT_detectron2

python convert_to_d2.py \
    --source_model ../swin_small_patch4_window7_224.pth \
    --output_model ../models/swin_small_patch4_window7_224_d2.pth
cd ..

# 🔬 Reproducibility Steps
Train the model
python train.py \
    --images      data/images \
    --json        data/annotations.json \
    --output_dir  outputs/training \
    --swint_repo  SwinT_detectron2 \
    --pretrained_weights models/swin_small_patch4_window7_224_d2.pth

Evaluate the trained model
Print
- AP50
- AP75
- Precision
- Recall
- F1-score

Visualize predictions on the test set

Generate segmentation visualizations:

python inference_visualize.py \
    --model models/final_model.pth \
    --images /path/to/validation_images

The training script used in this study is provided for transparency and reproducibility.

Prediction visualizations will be saved in:
outputs/
"""
bengali-word-segmentation/
├── predict.py                 ← quick prediction on any images
├── train.py                   ← training with auto dataset split
├── evaluate.py                ← evaluation: AP50, AP75, P, R, F1
├── inference_visualize.py     ← visualize predictions on test set
├── requirements.txt
├── LICENSE
├── sample_images/             ← sample images for quick testing
├── models/                    ← place model weights here
│   └── final_model.pth
└── SwinT_detectron2/          ← cloned separately (see Installation)

"""

# 📜 License

This repository is released under the MIT License.


# 🙏 Acknowledgments

- Detectron2
- Swin Transformer
- COCO API
- VIA

---





