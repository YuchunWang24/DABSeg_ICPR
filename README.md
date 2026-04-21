# DABSeg

Official implementation of **DABSeg**, a degradation-aware joint blur-segmentation framework for multimodal 3D brain tumor MRI segmentation.

## Introduction

Brain tumor segmentation based on multimodal MRI plays an important role in radiotherapy planning, surgical assistance, and treatment evaluation.  
However, in real clinical acquisition, patient motion may introduce blur and degradation, which weakens boundary and texture cues and leads to unstable segmentation performance.

To address this issue, we propose **DABSeg**, an end-to-end joint blur-segmentation framework for degraded multimodal 3D MRI brain tumor segmentation.  
The framework mainly consists of:

- **FDMDS**: Feature-Domain Motion Deblurring Stem
- **DAMI**: Degradation-Aware Multi-Modal Interaction module
- **Joint Loss**: weighted Dice loss with reconstruction supervision

DABSeg takes four MRI modalities as input:

- T1
- T1ce
- T2
- FLAIR

and predicts three clinically important tumor subregions:

- **ET**: Enhancing Tumor
- **TC**: Tumor Core
- **WT**: Whole Tumor

---

## Repository Structure

```text
DABSeg_ICPR/
└── DABSeg/
    ├── dataset/
    │   ├── brats2020/
    │   └── brats2020_S0/
    │       ├── train/
    │       ├── val/
    │       └── test/
    ├── models/
    ├── main.py
    ├── test.py
    ├── utils.py
    └── README.md

Dataset Organization

This project uses a BraTS-style case-wise folder structure.

Expected directory tree

dataset/
├── brats2020/
└── brats2020_S0/
    ├── train/
    │   ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_002/
    │   └── ...
    ├── val/
    │   ├── BraTS20_Training_003/
    │   ├── BraTS20_Training_004/
    │   └── ...
    └── test/
        ├── BraTS20_Training_008/
        │   ├── BraTS20_Training_008_flair.nii.gz
        │   ├── BraTS20_Training_008_seg.nii.gz
        │   ├── BraTS20_Training_008_t1.nii.gz
        │   ├── BraTS20_Training_008_t1ce.nii.gz
        │   └── BraTS20_Training_008_t2.nii.gz
        ├── BraTS20_Training_026/
        ├── BraTS20_Training_029/
        └── ...



Required files for each case

Each case folder should contain the following five files:

*_t1.nii.gz
*_t1ce.nii.gz
*_t2.nii.gz
*_flair.nii.gz
*_seg.nii.gz
Example
BraTS20_Training_008/
├── BraTS20_Training_008_t1.nii.gz
├── BraTS20_Training_008_t1ce.nii.gz
├── BraTS20_Training_008_t2.nii.gz
├── BraTS20_Training_008_flair.nii.gz
└── BraTS20_Training_008_seg.nii.gz
Notes
brats2020/ can be used for the original BraTS2020 dataset.
brats2020_S0/ is used for the degraded / blurred dataset in this project.
If your degraded dataset folder uses another name, such as brats2020_S2, please modify the dataset path accordingly.
The repository assumes that train, val, and test are already split into separate folders.



Environment

Recommended environment:

Python 3.8+
PyTorch
MONAI
NumPy
TensorBoard
timm
einops

Example installation:

pip install torch torchvision torchaudio
pip install monai nibabel numpy tensorboard timm einops


Training

Please first prepare the dataset according to the folder structure above.

Example training command:

python main.py \
  --mode train \
  --dataset-folder dataset/brats2020_S0 \
  --exp-name DABSeg \
  --devices 0 \
  --batch-size 2 \
  --lr 1e-4 \
  --end-epoch 250


Testing

Example testing command:

python test.py \
  --mode test \
  --dataset-folder dataset/brats2020_S0 \
  --exp-name DABSeg \
  --devices 0 \
  --tta True


The testing script will:

load the trained best model
perform sliding-window inference
optionally use test-time augmentation (TTA)
save segmentation predictions
save evaluation metrics such as Dice / HD95 / sensitivity / specificity
