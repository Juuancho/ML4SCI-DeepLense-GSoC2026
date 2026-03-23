# DeepLense — Common Test I: Multi-Class Classification

## Overview

This repository contains the solution for Common Test I of the ML4SCI DeepLense evaluation for Google Summer of Code. 
The task consists of building a deep learning model to classify strong gravitational lensing images into three categories: no substructure, spherical subhalo substructure, and vortex substructure.

## Results

| Metric                | Score  |
|-----------------------|--------|
| Validation Accuracy   | 90.27% |
| AUC — no substructure | 0.9868 |
| AUC — sphere subhalo  | 0.9680 |
| AUC — vortex          | 0.9846 |
| AUC Macro Average     | 0.9798 |

Evaluation performed on a 90/10 stratified train-test split (33,750 train / 3,750 test) over 37,500 total samples.

## Repository Structure
```
deeplense-test-I/                                                        
├── DeepLense_Test_I.ipynb       # Main notebook with full implementation
├── best_model_test_I.pth        # Trained model weights
├── evaluation_roc_cm.png        # ROC curves and confusion matrix
├── gradcam_visualization.png    # Grad-CAM activation maps
└── README.md
```

## Approach

**Architecture**: ResNet18 pretrained on ImageNet, fine-tuned for astrophysical image classification.

### Key design decisions:

- Single-channel .npy arrays are converted to 3-channel input via channel stacking (tensor.repeat(3, 1, 1)), preserving pretrained weights intact.
- Dataset statistics (mean, std) computed directly from the training split rather than using ImageNet normalization, accounting for the domain gap between natural and astronomical images.
- Classification head replaced with BatchNorm1d → Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→3) for strong regularization.
- CrossEntropyLoss with label_smoothing=0.1 to prevent overconfident predictions.

### Training strategy — two phases:

1. **Phase 1 (frozen backbone)**: Only the classification head is trained for 10 epochs at lr=1e-3. This stabilizes the head before modifying backbone weights.
2. **Phase 2 (full fine-tuning)**: Entire network unfrozen with differential learning rates — backbone at lr=1e-5, head at lr=1e-4 — with early stopping (patience=8).

**Data augmentation**: Random horizontal flip, vertical flip, and rotation — exploiting the rotational symmetry of gravitational lensing images.

### Interpretability
Grad-CAM visualizations confirm that the model consistently activates the Einstein ring region across all three classes, demonstrating that the model learned physically meaningful features rather than spurious background correlations.

### Requirements

| torch>=2.0<br/>torchvision>=0.15<br/>numpy<br/>scikit-learn<br/>matplotlib |
|----------------------------------------------------------------------------|

## How to Run

1. Download the dataset from the official DeepLense link and place it under ./dataset/ with the following structure:

```text
dataset/
├── train/
│   ├── no/
│   ├── sphere/
│   └── vort/
└── val/
    ├── no/
    ├── sphere/
    └── vort/
``` 
2. Open DeepLense_Test_I.ipynb and run all cells in order.
3. The notebook automatically performs the 90/10 stratified split, trains the model, and generates all evaluation metrics and visualizations.

## Trained Model Weights

Pre-trained weights available for download:
https://drive.google.com/file/d/1cyOvqvMa3v3lD9uTeQVb3D4tQDlDaYER/view?usp=sharing