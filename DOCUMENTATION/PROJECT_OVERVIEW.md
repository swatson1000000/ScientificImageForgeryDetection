# Scientific Image Forgery Detection - Project Overview

## Project Summary

This is a **scientific image forgery detection system** that uses deep learning to automatically identify tampered regions in scientific images. The system implements a sophisticated **two-stage pipeline** combining binary classification with ensemble segmentation to achieve high accuracy in detecting image forgeries.

### Key Features

- **Two-Stage Detection Pipeline**: Binary classifier (Pass 1) filters obvious authentic images; ensemble segmentation (Pass 2) detects forgery regions for remaining images
- **Ensemble Learning**: 4 specialized segmentation models working together with test-time augmentation (TTA)
- **Hard Negative Mining**: Incorporates challenging cases to improve robustness
- **Adaptive Thresholding**: Adjusts detection thresholds based on image brightness
- **Connected Component Filtering**: Removes noise and small false positive regions

## Problem Statement

Scientific images in papers are often subject to tampering for fraudulent purposes. This system addresses the need for automated, reliable detection of image forgeries, particularly in scientific publications where image integrity is critical.

### Detection Task

- **Input**: Scientific images (any resolution, RGB or grayscale)
- **Output**: 
  - Binary classification: Authentic vs. Forged
  - Segmentation mask: Pixel-level localization of forged regions
  - Confidence scores and statistics

## Architecture Overview

### Stage 1: Binary Classification (Pass 1)
- **Model**: EfficientNet B2 with binary classifier head
- **Purpose**: Fast filtering of obviously authentic images
- **Decision**: If probability ≥ 0.25 (default), proceed to Stage 2; otherwise, classify as authentic
- **Benefit**: Reduces computational cost by skipping expensive segmentation for clear negatives

### Stage 2: Segmentation (Pass 2)
- **Ensemble**: 4 specialized FPN-based models:
  - `highres_no_ela_v4`: Optimized for high-resolution analysis
  - `hard_negative_v4`: Specialized on challenging authentic images
  - `high_recall_v4`: Maximizes recall/sensitivity
  - `enhanced_aug_v4`: Trained with enhanced augmentation
  
- **Architecture**: FPN (Feature Pyramid Network) with attention gates
  - Base encoder: EfficientNet B2 (ImageNet pretrained)
  - Decoder: FPN architecture for multi-scale feature fusion
  - Attention Gate: Spatial attention mechanism focusing on forgery regions
  
- **Aggregation**: 
  - Individual predictions averaged across 4 models
  - Test-Time Augmentation (4x flips) with MAX aggregation
  - Adaptive threshold based on image brightness
  - Connected component filtering to remove noise

## Training Strategy

### Phase 1: Base Model Training (v3)
- **Data**: 
  - Forged: ~10,315 images (2,751 original + 7,564 augmented)
  - Authentic: ~12,200 images (balanced sampling)
- **Loss**: Combined Focal + Dice + FP penalty
- **Purpose**: Learn basic forgery detection patterns

### Phase 2: Hard Negative Mining (v4)
- **Key Insight**: FP errors often involve tricky authentic images
- **Strategy**: 
  - Identify authentic images producing false positives
  - Add with high sample weight (5x) during retraining
  - Increase FP penalty from 2.0 to 4.0
- **Result**: Significant FP reduction while maintaining recall

## Best Configuration Performance

| Metric | Value |
|--------|-------|
| Recall | 79% |
| False Positive Rate | 3.4% |
| Classifier Threshold | 0.25 |
| Segmentation Threshold | 0.35 |
| Min Area Filter | 300 pixels |
| TTA | 4x flips with MAX aggregation |

## Data Organization

```
/home/swatson/work/MachineLearning/kaggle/ScientificImageForgeryDetection/
├── train_images/
│   ├── forged/              # 2,751 forged training images
│   ├── authentic/           # ~2,400 authentic training images
│   ├── forged_augmented/    # 7,564 augmented forged images
│   ├── authentic_augmented/ # Additional augmented authentic
│   └── authentic_augmented_v2/
├── train_masks/             # Segmentation masks for forged images
├── train_masks_augmented/   # Masks for augmented forged images
├── validation_images/       # Validation dataset
├── test_authentic_100/      # Hard negatives: tricky authentic images
├── test_forged_100/         # Test set: forged images
├── test_forged_all/         # Full test forged set
├── models/                  # Trained model checkpoints
│   ├── binary_classifier_best.pth
│   ├── highres_no_ela_v4_best.pth
│   ├── hard_negative_v4_best.pth
│   ├── high_recall_v4_best.pth
│   └── enhanced_aug_v4_best.pth
└── bin/                     # Training and inference scripts
    ├── script_1_train_v3.py
    ├── script_2_train_binary_classifier.py
    ├── script_3_train_v4.py
    └── script_4_two_pass_pipeline.py
```

## Key Innovations

### 1. Two-Pass Pipeline
Instead of running expensive segmentation on every image, the classifier efficiently filters obvious authentic images first, reducing computational overhead.

### 2. Hard Negative Mining Strategy
Most false positives come from challenging but authentic images. By explicitly training on these cases, the model learns to properly reject them.

### 3. Attention-Augmented FPN
The Attention Gate mechanism allows the network to focus on likely forgery regions, improving precision and reducing false positives.

### 4. Adaptive Thresholding
Image brightness significantly affects forgery visibility:
- **Dark images** (brightness < 50): Lower threshold (0.20-0.25)
- **Bright images** (brightness > 200): Higher threshold (0.40-0.50)
- **Normal images**: Default threshold (0.35)

## Performance Considerations

### Training Time
- Script 1 (v3 base models): ~4-6 hours per model
- Script 2 (binary classifier): ~2-3 hours
- Script 3 (v4 fine-tuning): ~2-4 hours per model
- **Total**: ~20-30 hours for full pipeline

### Inference Speed
- **Pass 1 (Classifier)**: ~10 ms per image
- **Pass 2 (Segmentation)**: ~500-600 ms per image (4 models + TTA)
- **Average** (50% pass to seg): ~250-300 ms per image

### Memory Requirements
- GPU: 8GB+ VRAM recommended
- CPU: 16GB+ RAM
- Training batch size: 4 (segmentation), 16 (classifier)

## Hyperparameter Tuning

### Critical Parameters

| Parameter | Current | Impact | Notes |
|-----------|---------|--------|-------|
| `classifier_threshold` | 0.25 | Controls false positive rate | Higher = fewer to seg |
| `seg_threshold` | 0.35 | Pixel-level forgery detection | Adaptive based on brightness |
| `min_area` | 300 | Removes noise | Higher = fewer false positives |
| `tta_flips` | 4x | Test-time augmentation | 4x provides sweet spot |
| `focal_loss_gamma` | 2.0 | Loss weighting | Higher = more focus on hard examples |
| `fp_penalty` | 4.0 (v4) | False positive penalty | v3: 2.0, v4: 4.0 |

### Tuning Guidelines

1. **For higher precision** (fewer false positives):
   - Increase `classifier_threshold` to 0.35-0.5
   - Increase `seg_threshold` to 0.40-0.45
   - Increase `min_area` to 500+

2. **For higher recall** (fewer false negatives):
   - Decrease `classifier_threshold` to 0.1-0.15
   - Decrease `seg_threshold` to 0.25-0.30
   - Decrease or remove `min_area` filtering

3. **For balanced performance**:
   - Use default values or adjust slightly
   - Try values: 0.25, 0.35, 300

## Requirements & Dependencies

### Core Dependencies
- PyTorch: Deep learning framework
- OpenCV: Image processing
- NumPy: Numerical computing
- Albumentations: Data augmentation
- segmentation_models_pytorch: Pre-built segmentation architectures
- timm: PyTorch Image Models (EfficientNet backbone)

### System Requirements
- GPU: NVIDIA CUDA 11.0+ (for GPU acceleration)
- Python: 3.8+
- CUDA/cuDNN: For training

## File Structure Reference

```
bin/
├── script_1_train_v3.py              # Base model training (v3)
├── script_2_train_binary_classifier.py  # Classifier training
├── script_3_train_v4.py              # Hard negative fine-tuning (v4)
└── script_4_two_pass_pipeline.py      # Inference pipeline

DOCUMENTATION/
├── PROJECT_OVERVIEW.md                # This file
├── USAGE_GUIDE.md                     # How to run scripts
└── DETAILED_EXPLANATIONS.md           # Technical deep dives
```

## Next Steps

1. **Review [USAGE_GUIDE.md](USAGE_GUIDE.md)** for instructions on running the scripts
2. **Consult [DETAILED_EXPLANATIONS.md](DETAILED_EXPLANATIONS.md)** for technical details
3. **Check [CLAUDE.md](../CLAUDE.md)** for critical inference requirements

---

**Last Updated**: December 2025  
**Status**: Production-ready with hard negative mining (v4 models)
