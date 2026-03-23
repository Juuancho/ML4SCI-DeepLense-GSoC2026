# Specific Test V — Lens Finding & Data Pipelines

## Overview

Solution for Specific Test V of the ML4SCI DeepLense evaluation for Google Summer of Code. The task is binary classification of observational astronomical images into lens and non-lens categories, with severe class imbalance as the central challenge.

## Results

| Metric | Value              |
| ---- |--------------------|
| AUC-ROC | 0.9841             |
|Sensitivity | (TPR)0.9487        |
|Specificity | 0.9422             |
|Optimal threshold | 0.488 (Youden's J) |
|Lenses detected | 185 / 195          |

The predefined train/test split provided by the dataset organizers was used as-is (train_lenses, train_nonlenses, test_lenses, test_nonlenses).

### Key Challenge — Class Imbalance 1:99

The test set contains 195 lenses vs 19,455 non-lenses
. Accuracy is therefore a misleading metric — a model predicting "non-lens" always would achieve 99% accuracy. AUC-ROC is the only meaningful metric for this task, as it evaluates performance across all classification thresholds.

## Approach

**Architecture**: ResNet18 pretrained on ImageNet, fine-tuned for binary lens classification.

### Imbalance handled at three levels:

- WeightedRandomSampler — balanced 50/50 batches during training
- BCEWithLogitsLoss(pos_weight=16.6) — penalizes missing a lens 16.6x more than a false positive
- Data augmentation — random flips and rotations on training set

**Training**: Two-phase fine-tuning identical to Common Test I — frozen backbone phase followed by full fine-tuning with differential learning rates (backbone lr=1e-5, head lr=1e-4) and CosineAnnealingLR scheduling.

**Threshold**: Optimal decision threshold of 0.488 selected via Youden's J statistic rather than the default 0.5.

## Repository Structure
```
Specific_Test_V/
├── ST_V.ipynb       # Full implementation notebook
├── best_model_test_V.pth        # Trained model weights (or Google Drive link)
├── evaluation_test_v.png        # ROC curve and confusion matrix
├── gradcam_test_v.png           # Grad-CAM activation maps
└── README.md
```

### Requirements

| torch>=2.0<br/>torchvision>=0.15<br/>numpy<br/>scikit-learn<br/>matplotlib |
|----------------------------------------------------------------------------| 

## How to Run

1. Download the dataset and place it under ./dataset_v/ with the original folder structure:

```
dataset_v/
├── train_lenses/
├── train_nonlenses/
├── test_lenses/
└── test_nonlenses/
```

2. Open DeepLense_Test_V.ipynb and run all cells in order.

## Trained Model Weights

Pre-trained weights available for download:  
https://drive.google.com/file/d/1cotHMb9-xRFplEsUw3Uquhp47CTlOrzm/view?usp=sharing