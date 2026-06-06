# bengali-word-segmentation
SwinWordSeg: Multi-Scale Word Segmentation from Handwritten Bengali Answer-book Images
# 🔬 Abstract
Word segmentation is a crucial phase for handwritten text recognition (HTR). When the input image contains multiple words written across several text lines, it is crucial to accurately segment individual words for precise text recognition. This task is challenging due to variations in handwriting styles and script-specific complexities, including differing stroke patterns, irregular spacing, inconsistent word shapes, and various noise elements. Segmenting handwritten school students' pages becomes more challenging. This study introduces an instance segmentation framework based on Mask R-CNN integrated with Swin Transformer, optimized for word segmentation from Bengali handwritten answer scripts. To train and test the proposed system, answer books from middle school students across multiple Bengali-medium schools were collected and annotated. The proposed model is compared with two widely used architectures: U-Net and Mask R-CNN with a ResNet backbone. Experimental results demonstrate that the proposed model outperformed those. The model was found to be superior to multiple pre-trained computer vision and large-language models, as tested on a word segmentation task using the same dataset. The model was also tested on publicly available handwritten Bengali and Devanagari datasets, demonstrating its robustness.


# ✨ Key Features

- Swin Transformer based feature extraction
- Mask R-CNN based word instance segmentation
- Handles dense handwritten answer scripts
- Supports Bengali handwritten word segmentation
- COCO-format evaluation
- Precision, Recall, and F1-score reporting
- Visualization of segmentation outputs
- Reproducible evaluation pipeline
