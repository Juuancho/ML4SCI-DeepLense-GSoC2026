# ML4SCI DeepLense — GSoC 2025

Solutions to the ML4SCI DeepLense evaluation tests for Google Summer of Code 2025.
Both tests are implemented in PyTorch using ResNet18 with transfer learning and
two-phase fine-tuning strategies.

## Results Summary

| Test | Task | Key Challenge | AUC | Accuracy |
|------|------|---------------|-----|----------|
| [Common Test I](./Common_Test_I/) | Multi-class gravitational lens classification | Domain gap (simulated data) | 0.9798 | 90.27% |
| [Specific Test V](./Specific_Test_V/) | Binary lens finding | Class imbalance 1:99 | 0.9841 | — |

> Accuracy is not reported for Specific Test V because it is a misleading metric
> under 1:99 class imbalance. AUC-ROC is the appropriate evaluation metric for both tasks.

## Repository Structure
```
ML4SCI-DeepLense-GSoC2025/
├── README.md
├── Common_Test_I/
│   ├── best_model_test_I.pth
│   ├── Multi-Class CF.ipynb
│   ├── evaluation_roc_cm.png
│   ├── gradcam_visualization.png
│   └── README.md
└── Specific_Test_V/
    ├── best_model_test_V.pth
    ├── ST_V.ipynb
    ├── evaluation_test_v.png
    ├── gradcam_test_v.png
    └── README.md
```

## Approach Overview

Both tests use **ResNet18 pretrained on ImageNet** with a two-phase fine-tuning strategy:

- **Phase 1** — backbone frozen, only the classification head trained
- **Phase 2** — full network unfrozen with differential learning rates
  (backbone `lr=1e-5`, head `lr=1e-4`) and CosineAnnealingLR scheduling

Each test required a different strategy to handle its specific challenge:

**Common Test I** addresses the domain gap between ImageNet features and
simulated astrophysical data through channel stacking, dataset-specific
normalization, and label smoothing.

**Specific Test V** addresses severe class imbalance (1:99) through a
three-layer strategy: WeightedRandomSampler for balanced batches,
BCEWithLogitsLoss with pos_weight for asymmetric loss, and Youden's J
statistic for optimal threshold selection.

## Trained Model Weights

| Model | Link |
|-------|------|
| Common Test I — ResNet18 | [Google Drive](https://drive.google.com/file/d/1cyOvqvMa3v3lD9uTeQVb3D4tQDlDaYER/view?usp=sharing) |
| Specific Test V — ResNet18 | [Google Drive](https://drive.google.com/file/d/1cotHMb9-xRFplEsUw3Uquhp47CTlOrzm/view?usp=sharing) |

## Requirements
```
torch>=2.0
torchvision>=0.15
numpy
scikit-learn
matplotlib
```

## Author

Juan Caraballo Nieves   
Systems Engineering Student   
GSoC 2025 Applicant — ML4SCI DeepLense