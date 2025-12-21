# Scientific Image Forgery Detection - Documentation

This documentation covers the training pipeline and inference system for detecting forged regions in scientific images.

## Table of Contents

1. [Quick Start Guide](quick_start.md) - How to run the pipeline
2. [System Overview](system_overview.md) - Architecture and workflow
3. [Script Reference](script_reference.md) - Detailed script documentation

## Project Structure

```
ScientificImageForgeryDetection/
├── bin/                          # Executable scripts
│   ├── run_pipeline.sh           # Main orchestrator
│   ├── clean.sh                  # Cleanup utility
│   ├── script_0_train_v3.py      # V3 segmentation training
│   ├── script_1_train_v4.py      # V4 segmentation fine-tuning
│   ├── script_2_train_binary_classifier.py  # Classifier training
│   ├── script_3_two_stage_submission.py     # Inference
│   ├── generate_forged_augmented.py         # Data augmentation
│   ├── generate_authentic_augmented.py      # Data augmentation
│   └── validate_test.py          # Validation utility
├── models/                       # Trained model weights
├── train_images/                 # Training data
│   ├── forged/                   # Original forged images
│   ├── authentic/                # Original authentic images
│   ├── forged_augmented/         # Augmented forged images
│   └── authentic_augmented/      # Augmented authentic images
├── train_masks/                  # Ground truth masks
├── log/                          # Pipeline logs
└── DOCUMENTATION/                # This documentation
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- Conda environment: `phi4`

## Key Metrics

The system optimizes for **Net Score = TP - FP** (True Positives minus False Positives).

| Metric | Target |
|--------|--------|
| Recall | >70% |
| FP Rate | <10% |
| Net Score | Maximize |
